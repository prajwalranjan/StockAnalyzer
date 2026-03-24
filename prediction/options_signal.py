"""
prediction/options_signal.py — Phase 4
========================================
Detects institutional positioning in the options market
before it shows up in the stock price.

Two signals:
  O1. Put/Call Ratio (PCR)      — market sentiment for this stock
  O2. Unusual Call OI           — abnormal institutional call buying

Key insight: institutions often position in options BEFORE making
large moves in the underlying stock. Rising call OI at near-the-money
strikes = someone expects upside. Unusual activity = smart money.

Note: Only FnO stocks have options data. Non-FnO stocks gracefully
return neutral scores.

STATUS: Shadow mode — scores logged, shown on dashboard,
        never block or boost trades until validated.
"""

import os
import logging
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)

NSE_CACHE = Path(os.path.dirname(__file__)) / ".." / "data" / "nse_cache"
NSE_CACHE.mkdir(parents=True, exist_ok=True)


def _get_nse():
    from nse import NSE

    is_server = os.environ.get("RAILWAY_ENVIRONMENT") is not None
    return NSE(download_folder=str(NSE_CACHE), server=is_server)


def _clean_symbol(sym: str) -> str:
    """Strip .NS/.BO suffix for NSE API calls."""
    return sym.replace(".NS", "").replace(".BO", "")


# ─── Signal O1 — Put/Call Ratio ───────────────────────────────────────────────


def get_pcr(sym: str) -> dict:
    """
    Fetches the Put/Call Ratio for a stock from its options chain.

    PCR = Total Put OI / Total Call OI

    Interpretation for breakout context:
      PCR < 0.6 and falling = aggressive call buying = bullish positioning
                               Institutions expecting upside move
      PCR 0.6-0.8            = slight bullish tilt
      PCR 0.8-1.2 (normal)   = neutral market
      PCR > 1.5              = heavy put buying = bearish/hedging
                               Could signal institutions hedging a position

    Note: India PCR thresholds differ from US markets.
    Indian options PCR normal range: 0.8-1.2 (vs 0.7-1.0 in US)

    Returns:
    {
        "pcr":        0.72,
        "total_call_oi": 150000,
        "total_put_oi":  108000,
        "verdict":    "BULLISH",   # BULLISH / NEUTRAL / BEARISH
        "score":      8,           # 0-10 contribution
        "detail":     "...",
        "source":     "nse"
    }
    """
    symbol = _clean_symbol(sym)

    try:
        with _get_nse() as nse:
            # Get nearest expiry date
            expiries = _get_expiry_dates(nse, symbol)
            if not expiries:
                return _pcr_fallback(f"{symbol} has no options (not FnO)")

            nearest_expiry = expiries[0]
            chain = nse.compileOptionChain(symbol, nearest_expiry)

        if not chain:
            return _pcr_fallback(f"Empty option chain for {symbol}")

        total_call_oi = int(chain.get("totalCallOI", 0) or 0)
        total_put_oi = int(chain.get("totalPutOI", 0) or 0)

        if total_call_oi == 0:
            return _pcr_fallback(f"No call OI data for {symbol}")

        pcr = round(total_put_oi / total_call_oi, 3)

        # Score based on PCR — bullish = low PCR (more calls than puts)
        if pcr < 0.6:
            score = 10
            verdict = "BULLISH"
            detail = f"PCR {pcr:.2f} — aggressive call buying (very bullish)"
        elif pcr < 0.8:
            score = 7
            verdict = "BULLISH"
            detail = f"PCR {pcr:.2f} — bullish options positioning"
        elif pcr <= 1.2:
            score = 4
            verdict = "NEUTRAL"
            detail = f"PCR {pcr:.2f} — neutral options market"
        elif pcr <= 1.5:
            score = 2
            verdict = "BEARISH"
            detail = f"PCR {pcr:.2f} — slight put bias (cautious)"
        else:
            score = 0
            verdict = "BEARISH"
            detail = f"PCR {pcr:.2f} — heavy put buying (bearish/hedging)"

        return {
            "pcr": pcr,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "verdict": verdict,
            "score": score,
            "detail": detail,
            "source": "nse",
        }

    except Exception as e:
        return _pcr_fallback(f"PCR fetch failed: {e}")


def _get_expiry_dates(nse, symbol: str) -> list:
    """Gets sorted list of upcoming expiry dates for a symbol."""
    try:
        raw = nse.optionChain(symbol)
        if not raw:
            return []

        records = raw.get("records", {})
        expiry_list = records.get("expiryDates", [])

        today = date.today()
        upcoming = []
        for exp_str in expiry_list:
            try:
                exp_date = datetime.strptime(exp_str, "%d-%b-%Y").date()
                if exp_date >= today:
                    upcoming.append(datetime.strptime(exp_str, "%d-%b-%Y"))
            except:
                continue

        return sorted(upcoming)
    except:
        return []


def _pcr_fallback(reason: str) -> dict:
    return {
        "pcr": None,
        "total_call_oi": 0,
        "total_put_oi": 0,
        "verdict": "NEUTRAL",
        "score": 4,
        "detail": reason,
        "source": "fallback",
    }


# ─── Signal O2 — Unusual Call OI ─────────────────────────────────────────────


def get_unusual_call_oi(sym: str) -> dict:
    """
    Detects unusually high call open interest at near-the-money strikes.

    When a large institution expects a stock to move up, they buy calls.
    This shows up as OI at near-money strikes that is 3x+ the normal
    OI at those strikes.

    We look at the 3 strikes nearest to current price on the call side:
      - ATM (at-the-money): closest to current price
      - 1 strike OTM: slightly above current price
      - 2 strikes OTM: further above current price

    If call OI at these strikes is 3x+ the average OI across all strikes,
    something unusual is happening.

    Returns:
    {
        "unusual":       True,
        "atm_call_oi":   250000,
        "avg_call_oi":   65000,
        "oi_ratio":      3.8,       # atm OI / avg OI
        "max_pain":      1450,      # strike price where max options expire worthless
        "score":         5,         # 0-5 contribution
        "iv_risk":       False,     # True if IV is abnormally high (event risk)
        "detail":        "...",
        "source":        "nse"
    }
    """
    symbol = _clean_symbol(sym)

    try:
        with _get_nse() as nse:
            expiries = _get_expiry_dates(nse, symbol)
            if not expiries:
                return _unusual_fallback(f"{symbol} has no options (not FnO)")

            nearest_expiry = expiries[0]
            chain = nse.compileOptionChain(symbol, nearest_expiry)

        if not chain:
            return _unusual_fallback(f"Empty option chain for {symbol}")

        atm_strike = chain.get("atmStrike")
        max_pain = chain.get("maxPain")
        strike_data = chain.get("oc", {})

        if not atm_strike or not strike_data:
            return _unusual_fallback("Missing ATM or strike data")

        # Collect all call OI values
        all_call_ois = []
        atm_region_ois = []

        # Get all available strikes sorted
        strikes = sorted([float(k) for k in strike_data.keys()])

        for strike in strikes:
            strike_info = strike_data.get(str(int(strike)), {})
            call_oi = int(strike_info.get("CE", {}).get("openInterest", 0) or 0)
            all_call_ois.append(call_oi)

            # ATM region: within 3 strikes of ATM
            if abs(strike - atm_strike) <= (strikes[1] - strikes[0]) * 3:
                atm_region_ois.append(call_oi)

        if not all_call_ois or sum(all_call_ois) == 0:
            return _unusual_fallback("No call OI data available")

        avg_call_oi = sum(all_call_ois) / len(all_call_ois)
        atm_call_oi = max(atm_region_ois) if atm_region_ois else 0

        oi_ratio = round(atm_call_oi / avg_call_oi, 2) if avg_call_oi > 0 else 0

        # Check IV risk (if ATM IV is very high, likely an event is priced in)
        atm_info = strike_data.get(str(int(atm_strike)), {})
        atm_iv = float(atm_info.get("CE", {}).get("impliedVolatility", 0) or 0)
        iv_risk = atm_iv > 50  # > 50% IV suggests event risk (earnings, news)

        # Score based on OI ratio
        if oi_ratio >= 4.0:
            score = 5
            unusual = True
            detail = f"very unusual call OI ({oi_ratio:.1f}x avg) — strong institutional positioning"
        elif oi_ratio >= 3.0:
            score = 4
            unusual = True
            detail = (
                f"unusual call OI ({oi_ratio:.1f}x avg) — institutional buying likely"
            )
        elif oi_ratio >= 2.0:
            score = 2
            unusual = False
            detail = f"above average call OI ({oi_ratio:.1f}x avg) — some interest"
        else:
            score = 0
            unusual = False
            detail = f"normal call OI ({oi_ratio:.1f}x avg) — no unusual activity"

        # Penalise if high IV risk (event = binary, don't trade)
        if iv_risk:
            score = max(0, score - 2)
            detail += f" | IV {atm_iv:.0f}% — elevated (event risk)"

        return {
            "unusual": unusual,
            "atm_call_oi": atm_call_oi,
            "avg_call_oi": round(avg_call_oi),
            "oi_ratio": oi_ratio,
            "max_pain": max_pain,
            "iv_risk": iv_risk,
            "atm_iv": round(atm_iv, 1),
            "score": score,
            "detail": detail,
            "source": "nse",
        }

    except Exception as e:
        return _unusual_fallback(f"Unusual OI check failed: {e}")


def _unusual_fallback(reason: str) -> dict:
    return {
        "unusual": False,
        "atm_call_oi": 0,
        "avg_call_oi": 0,
        "oi_ratio": 0,
        "max_pain": None,
        "iv_risk": False,
        "atm_iv": 0,
        "score": 0,
        "detail": reason,
        "source": "fallback",
    }


# ─── Main entry point ─────────────────────────────────────────────────────────


def compute(sym: str) -> dict:
    """
    Runs both options signals for a stock.

    Returns:
    {
        "total":        13,         # 0-15 contribution to quality score
        "summary":      "BULLISH",  # BULLISH / NEUTRAL / BEARISH
        "shadow_mode":  True,
        "signals": {
            "pcr":        {"score": 8, "detail": "...", "pcr": 0.72, ...},
            "unusual_oi": {"score": 5, "detail": "...", "unusual": True, ...},
        }
    }
    """
    pcr_result = get_pcr(sym)
    unusual_result = get_unusual_call_oi(sym)

    total = pcr_result["score"] + unusual_result["score"]

    if total >= 12:
        summary = "STRONGLY_BULLISH"
    elif total >= 8:
        summary = "BULLISH"
    elif total >= 5:
        summary = "NEUTRAL"
    else:
        summary = "BEARISH"

    return {
        "total": total,
        "summary": summary,
        "shadow_mode": True,
        "signals": {
            "pcr": pcr_result,
            "unusual_oi": unusual_result,
        },
    }
