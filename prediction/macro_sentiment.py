"""
prediction/macro_sentiment.py — Phase 2B
==========================================
Macro-level sentiment signals that affect the entire market
regardless of individual stock quality.

Five signals:
  M1. India VIX          — market fear gauge (most reliable)
  M2. Geopolitical risk  — GDELT global conflict tracker
  M3. Central bank proximity — RBI/Fed event calendar
  M4. Dollar + Oil       — macro headwinds for India
  M5. Overnight global   — S&P 500, Nikkei, Hang Seng

Key insight: company sentiment without macro context is dangerous.
A great stock in a geopolitical crisis is still a losing trade.
Both layers must be evaluated together.

Output: macro_score (0-100) + position_multiplier (0.5-1.0)
  The position multiplier adjusts trade size automatically:
  VIX 20+ → 75% of normal size
  VIX 25+ → 50% of normal size
  VIX 30+ → no new trades

STATUS: Shadow mode — scores logged, multiplier shown on dashboard
        but not yet applied to real position sizing.
"""

import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta


# ─── Signal M1 — India VIX ───────────────────────────────────────────────────


def india_vix() -> dict:
    """
    India VIX = NSE's fear gauge. Measures expected market volatility
    from Nifty options prices. Most reliable real-time macro signal.

    Historical significance:
      VIX > 25 has preceded or coincided with every major NSE
      correction since the index was introduced.

    Returns score contribution and position multiplier.
    """
    try:
        df = yf.download("^INDIAVIX", period="1mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return _vix_fallback("India VIX data unavailable")

        current = float(df["Close"].iloc[-1])
        prev_5d = float(df["Close"].iloc[-6:-1].mean()) if len(df) >= 6 else current
        change_5d = round(current - prev_5d, 2)

        # Score and multiplier
        if current >= 30:
            score, multiplier, level = 0, 0.0, "EXTREME — no new trades"
        elif current >= 25:
            score, multiplier, level = 15, 0.5, "HIGH — 50% position size"
        elif current >= 20:
            score, multiplier, level = 30, 0.75, "ELEVATED — 75% position size"
        elif current >= 15:
            score, multiplier, level = 45, 1.0, "MODERATE — normal trading"
        else:
            score, multiplier, level = 50, 1.0, "LOW — calm market"

        # Penalty if VIX is rising fast (trend matters as much as level)
        trend_penalty = 0
        if change_5d > 3:
            trend_penalty = 10
            level += " (rising fast — caution)"

        return {
            "score": max(0, score - trend_penalty),
            "multiplier": multiplier,
            "current": round(current, 2),
            "change_5d": change_5d,
            "level": level,
            "verdict": (
                "RED" if current >= 25 else "AMBER" if current >= 20 else "GREEN"
            ),
        }

    except Exception as e:
        return _vix_fallback(f"VIX fetch failed: {e}")


def _vix_fallback(reason):
    return {
        "score": 35,
        "multiplier": 0.75,
        "current": None,
        "change_5d": None,
        "level": reason,
        "verdict": "AMBER",
    }


# ─── Signal M2 — Geopolitical Risk (GDELT) ───────────────────────────────────


def geopolitical_risk() -> dict:
    """
    Uses GDELT (Global Database of Events, Language, and Tone) to
    measure global conflict and tension levels in real-time.

    GDELT tracks news events worldwide and scores geopolitical tension
    on a continuous scale. We focus on events involving India's key
    trading partners and oil suppliers.

    Free API — no key needed. Adds ~2 seconds to scan.

    India-relevant countries: USA, China, Pakistan, Middle East,
    Russia (oil/sanctions), European Union.
    """
    try:
        # GDELT GKG API — fetch last 24hrs of conflict events
        # Filter for tone (negative = conflict/tension)
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d%H%M%S")
        today = datetime.now().strftime("%Y%m%d%H%M%S")

        url = (
            "https://api.gdeltproject.org/api/v2/summary/summary?"
            f"d=today&t=summary&output=json"
        )

        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return _gdelt_fallback(f"GDELT API returned {resp.status_code}")

        data = resp.json()

        # Extract global conflict tone (-100 to +100, negative = conflict)
        # GDELT summary provides overall tone of global news
        tone = data.get("tone", {})
        avg_tone = float(tone.get("avgtone", 0)) if tone else 0

        # Convert tone to risk score (0-25 contribution to total)
        # GDELT tone: more negative = more conflict
        # Typical range: -5 to -2 (normal), < -7 (elevated tension)
        if avg_tone < -10:
            risk_score = 0
            risk_level = "EXTREME — severe global tension"
            verdict = "RED"
        elif avg_tone < -7:
            risk_score = 8
            risk_level = "HIGH — significant geopolitical tension"
            verdict = "RED"
        elif avg_tone < -5:
            risk_score = 15
            risk_level = "ELEVATED — notable global tension"
            verdict = "AMBER"
        elif avg_tone < -3:
            risk_score = 20
            risk_level = "MODERATE — some global tension (normal)"
            verdict = "GREEN"
        else:
            risk_score = 25
            risk_level = "LOW — relatively calm global environment"
            verdict = "GREEN"

        return {
            "score": risk_score,
            "avg_tone": round(avg_tone, 2),
            "risk_level": risk_level,
            "verdict": verdict,
            "source": "gdelt",
        }

    except requests.Timeout:
        return _gdelt_fallback("GDELT timeout — using neutral estimate")
    except Exception as e:
        return _gdelt_fallback(f"GDELT unavailable: {e}")


def _gdelt_fallback(reason):
    """Neutral fallback when GDELT is unavailable."""
    return {
        "score": 15,
        "avg_tone": None,
        "risk_level": reason,
        "verdict": "AMBER",
        "source": "fallback",
    }


# ─── Signal M3 — Central Bank Proximity ──────────────────────────────────────

# RBI MPC meeting dates — published 6 months in advance
# Update this list when RBI publishes new calendar
RBI_MEETING_DATES_2025_2026 = [
    date(2025, 4, 9),
    date(2025, 6, 6),
    date(2025, 8, 8),
    date(2025, 10, 8),
    date(2025, 12, 8),
    date(2026, 2, 7),
    date(2026, 4, 9),
    date(2026, 6, 6),
    date(2026, 8, 7),
    date(2026, 10, 7),
    date(2026, 12, 5),
]


def central_bank_proximity() -> dict:
    """
    Checks proximity to RBI MPC and US Fed meeting dates.

    Within 2 days of a central bank decision = no new trades.
    Binary event — market moves unpredictably on rate decisions
    even when the outcome is "expected".
    """
    today = date.today()

    # Find next RBI meeting
    rbi_days = None
    for meeting_date in sorted(RBI_MEETING_DATES_2025_2026):
        if meeting_date >= today:
            rbi_days = (meeting_date - today).days
            break

    # Fed meeting — fetch from yfinance economic calendar or estimate
    # Fed meets roughly every 6 weeks — approximate from known schedule
    # For simplicity, flag if within 2 days of any Wednesday
    # (Fed typically announces on Wednesdays)
    fed_days = _days_to_next_fed()

    # Determine verdict
    skip = False
    verdict = "GREEN"
    score = 25  # full score
    detail = ""

    if rbi_days is not None and rbi_days <= 2:
        skip = True
        verdict = "RED"
        score = 0
        detail = f"RBI MPC decision in {rbi_days} days — skip"
    elif fed_days <= 1:
        verdict = "AMBER"
        score = 15
        detail = f"Fed meeting today/tomorrow — reduce size"
    elif rbi_days is not None and rbi_days <= 5:
        verdict = "AMBER"
        score = 18
        detail = f"RBI MPC approaching in {rbi_days} days — caution"
    elif rbi_days is not None and rbi_days <= 10:
        score = 22
        detail = f"RBI MPC in {rbi_days} days — safe distance"
    else:
        detail = f"RBI in {rbi_days or '?'} days, Fed in {fed_days} days — clear"

    return {
        "score": score,
        "skip": skip,
        "verdict": verdict,
        "rbi_days": rbi_days,
        "fed_days": fed_days,
        "detail": detail,
    }


def _days_to_next_fed():
    """
    Approximate days to next Fed meeting.
    Fed meets ~8 times/year, roughly every 45 days.
    Known 2025-2026 approximate dates used as reference.
    """
    FED_DATES_2025_2026 = [
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 11, 5),
        date(2025, 12, 17),
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
    ]
    today = date.today()
    for d in sorted(FED_DATES_2025_2026):
        if d >= today:
            return (d - today).days
    return 30  # default if no upcoming date found


# ─── Signal M4 — Dollar Index + Oil ──────────────────────────────────────────


def dollar_oil_signal() -> dict:
    """
    Tracks two key macro variables for India:

    Dollar Index (DXY): Rising dollar → FII outflows from India
    (dollar assets become more attractive relative to rupee assets)

    Crude Oil (Brent): India imports 85% of oil needs.
    Oil spike → current account deficit widens → RBI tightens → negative.

    Both measured as 10-day momentum (rate of change matters more
    than absolute level).
    """
    try:
        # Dollar Index
        dxy = yf.download("DX-Y.NYB", period="1mo", progress=False, auto_adjust=True)
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy.columns = dxy.columns.get_level_values(0)

        # Crude Oil (WTI)
        oil = yf.download("CL=F", period="1mo", progress=False, auto_adjust=True)
        if isinstance(oil.columns, pd.MultiIndex):
            oil.columns = oil.columns.get_level_values(0)

        score = 25  # start at max, deduct for headwinds
        details = []

        # DXY momentum
        dxy_change = None
        if not dxy.empty and len(dxy) >= 11:
            dxy_change = round(
                (float(dxy["Close"].iloc[-1]) / float(dxy["Close"].iloc[-11]) - 1)
                * 100,
                2,
            )
            if dxy_change > 2.0:
                score -= 12
                details.append(f"dollar +{dxy_change:.1f}% (FII outflow risk)")
            elif dxy_change > 1.0:
                score -= 6
                details.append(f"dollar +{dxy_change:.1f}% (mild headwind)")
            else:
                details.append(f"dollar {dxy_change:+.1f}% (neutral)")

        # Oil momentum
        oil_change = None
        if not oil.empty and len(oil) >= 11:
            oil_change = round(
                (float(oil["Close"].iloc[-1]) / float(oil["Close"].iloc[-11]) - 1)
                * 100,
                2,
            )
            if oil_change > 8.0:
                score -= 10
                details.append(f"oil +{oil_change:.1f}% (inflation pressure)")
            elif oil_change > 4.0:
                score -= 5
                details.append(f"oil +{oil_change:.1f}% (mild pressure)")
            elif oil_change < -8.0:
                score += 3  # falling oil = good for India
                details.append(f"oil {oil_change:.1f}% (tailwind for India)")
            else:
                details.append(f"oil {oil_change:+.1f}% (neutral)")

        score = max(0, min(25, score))
        verdict = "RED" if score < 8 else "AMBER" if score < 16 else "GREEN"

        return {
            "score": score,
            "dxy_change": dxy_change,
            "oil_change": oil_change,
            "verdict": verdict,
            "detail": " | ".join(details) if details else "data unavailable",
        }

    except Exception as e:
        return {
            "score": 15,
            "dxy_change": None,
            "oil_change": None,
            "verdict": "AMBER",
            "detail": f"fetch failed: {e}",
        }


# ─── Signal M5 — Overnight Global Markets ────────────────────────────────────


def overnight_global() -> dict:
    """
    Checks how major global markets closed overnight.
    Indian market correlation with global markets: ~0.6.

    A severe overnight fall significantly raises Indian open-down
    probability. Used as a same-day entry filter.

    Sources: S&P 500 (US), Nikkei (Japan), Hang Seng (Hong Kong)
    """
    try:
        symbols = {
            "sp500": "^GSPC",
            "nikkei": "^N225",
            "hangseng": "^HSI",
        }

        changes = {}
        for name, sym in symbols.items():
            try:
                df = yf.download(sym, period="5d", progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty and len(df) >= 2:
                    chg = (
                        float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1
                    ) * 100
                    changes[name] = round(chg, 2)
            except:
                changes[name] = None

        # Composite: equal-weighted average of available markets
        valid = [v for v in changes.values() if v is not None]
        composite = round(sum(valid) / len(valid), 2) if valid else 0

        # Score based on composite overnight move
        if composite < -3.0:
            score = 0
            verdict = "RED"
            detail = f"severe overnight fall {composite:.1f}% — skip entries today"
        elif composite < -2.0:
            score = 8
            verdict = "RED"
            detail = f"significant overnight fall {composite:.1f}% — no new entries"
        elif composite < -1.0:
            score = 15
            verdict = "AMBER"
            detail = f"mild overnight weakness {composite:.1f}% — caution"
        elif composite > 1.0:
            score = 25
            verdict = "GREEN"
            detail = f"positive overnight {composite:.1f}% — supportive"
        else:
            score = 20
            verdict = "GREEN"
            detail = f"flat overnight {composite:.1f}% — neutral"

        return {
            "score": score,
            "composite": composite,
            "sp500": changes.get("sp500"),
            "nikkei": changes.get("nikkei"),
            "hangseng": changes.get("hangseng"),
            "verdict": verdict,
            "detail": detail,
        }

    except Exception as e:
        return {
            "score": 15,
            "composite": None,
            "sp500": None,
            "nikkei": None,
            "hangseng": None,
            "verdict": "AMBER",
            "detail": f"fetch failed: {e}",
        }


# ─── Main entry point ─────────────────────────────────────────────────────────


def compute() -> dict:
    """
    Runs all five macro signals and returns combined result.

    Returns:
    {
        "macro_score":         62,        # 0-100
        "position_multiplier": 0.75,      # 0.5 to 1.0
        "verdict":             "AMBER",   # GREEN / AMBER / RED
        "shadow_mode":         True,
        "signals": {
            "india_vix":    {...},
            "geopolitical": {...},
            "central_bank": {...},
            "dollar_oil":   {...},
            "overnight":    {...},
        }
    }
    """
    vix_result = india_vix()
    gdelt_result = geopolitical_risk()
    cb_result = central_bank_proximity()
    do_result = dollar_oil_signal()
    og_result = overnight_global()

    # Hard skip conditions — any one triggers RED
    hard_skip = (
        vix_result["verdict"] == "RED"
        or cb_result["skip"]
        or og_result["verdict"] == "RED"
    )

    # Aggregate score (each signal has its own max contribution)
    raw_score = (
        vix_result["score"]  # max 50
        + gdelt_result["score"]  # max 25
        + cb_result["score"]  # max 25
        + do_result["score"]  # max 25
        + og_result["score"]  # max 25
    )
    # Normalise to 0-100 (total possible = 150)
    macro_score = round(min(100, raw_score / 150 * 100))

    # Position multiplier from VIX (primary driver)
    position_multiplier = vix_result["multiplier"]

    # Further reduce if multiple AMBER/RED signals
    amber_count = sum(
        1
        for r in [vix_result, gdelt_result, cb_result, do_result, og_result]
        if r["verdict"] in ["AMBER", "RED"]
    )
    if amber_count >= 4:
        position_multiplier = min(position_multiplier, 0.5)
    elif amber_count >= 2:
        position_multiplier = min(position_multiplier, 0.75)

    # Overall verdict
    if hard_skip or macro_score < 30:
        verdict = "RED"
    elif macro_score < 55 or amber_count >= 2:
        verdict = "AMBER"
    else:
        verdict = "GREEN"

    return {
        "macro_score": macro_score,
        "position_multiplier": position_multiplier,
        "verdict": verdict,
        "hard_skip": hard_skip,
        "shadow_mode": True,
        "signals": {
            "india_vix": vix_result,
            "geopolitical": gdelt_result,
            "central_bank": cb_result,
            "dollar_oil": do_result,
            "overnight": og_result,
        },
    }
