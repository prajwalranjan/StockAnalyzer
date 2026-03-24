"""
prediction/nse_scraper.py — NSE Data Fetcher
=============================================
Fetches public data from NSE India using the `nse` package
(pip install nse[server]) which handles cookie management.

All data fetched is public — published daily by NSE/SEBI.

Functions:
  get_fii_dii()        — FII + DII net buy/sell for last 5 days
  get_delivery(sym)    — Delivery % for a stock (last 20 days)
  get_block_deals()    — Block deals from last 5 trading days
  get_insider_trades() — Insider/promoter filings from last 30 days

Each function returns a clean dict — no raw NSE objects exposed.
All functions fall back gracefully and log errors without crashing.
"""

import os
import logging
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Cookie/cache folder for the NSE package
NSE_CACHE = Path(os.path.dirname(__file__)) / ".." / "data" / "nse_cache"
NSE_CACHE.mkdir(parents=True, exist_ok=True)


def _get_nse():
    """Returns an NSE client instance. Uses server=True for Railway."""
    from nse import NSE

    is_server = os.environ.get("RAILWAY_ENVIRONMENT") is not None
    return NSE(download_folder=str(NSE_CACHE), server=is_server)


# ─── FII / DII flows ─────────────────────────────────────────────────────────


def get_fii_dii(days: int = 5) -> dict:
    """
    Fetches FII (Foreign) and DII (Domestic) net buy/sell activity
    for the last N trading days.

    FII net buyer for 3+ consecutive days = institutional accumulation.
    Both FII and DII buying simultaneously = strongest confirmation.

    Returns:
    {
        "fii_net_5d":    +1234.56,   # Rs Crore, positive = buying
        "dii_net_5d":    -234.56,
        "fii_trend":     "BUYING",   # BUYING / SELLING / MIXED
        "dii_trend":     "SELLING",
        "combined":      "FII_BUYING",
        "days":          [...],      # per-day breakdown
        "detail":        "FII bought Rs1234cr over 5 days",
        "source":        "nse"
    }
    """
    try:
        with _get_nse() as nse:
            data = nse.fiiDiiStats(days)

        if not data:
            return _fii_dii_fallback("No FII/DII data returned")

        fii_total = 0.0
        dii_total = 0.0
        days_data = []

        for row in data:
            # NSE returns: date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net
            try:
                fii_net = float(row.get("fiidiiNet", row.get("fii_net", 0)) or 0)
                dii_net = float(row.get("diiNet", row.get("dii_net", 0)) or 0)
                fii_total += fii_net
                dii_total += dii_net
                days_data.append(
                    {
                        "date": str(row.get("date", "")),
                        "fii_net": round(fii_net, 2),
                        "dii_net": round(dii_net, 2),
                    }
                )
            except (TypeError, ValueError):
                continue

        fii_trend = _trend(fii_total, days_data, "fii_net")
        dii_trend = _trend(dii_total, days_data, "dii_net")

        if fii_trend == "BUYING" and dii_trend == "BUYING":
            combined = "BOTH_BUYING"
        elif fii_trend == "BUYING":
            combined = "FII_BUYING"
        elif dii_trend == "BUYING":
            combined = "DII_BUYING"
        elif fii_trend == "SELLING":
            combined = "FII_SELLING"
        else:
            combined = "MIXED"

        detail = (
            f"FII {'+' if fii_total >= 0 else ''}{fii_total:.0f}cr, "
            f"DII {'+' if dii_total >= 0 else ''}{dii_total:.0f}cr over {len(days_data)} days"
        )

        return {
            "fii_net_5d": round(fii_total, 2),
            "dii_net_5d": round(dii_total, 2),
            "fii_trend": fii_trend,
            "dii_trend": dii_trend,
            "combined": combined,
            "days": days_data,
            "detail": detail,
            "source": "nse",
        }

    except Exception as e:
        return _fii_dii_fallback(f"FII/DII fetch failed: {e}")


def _trend(total, days_data, key):
    if not days_data:
        return "MIXED"
    consecutive_buy = sum(1 for d in days_data[-3:] if d.get(key, 0) > 0)
    if total > 0 and consecutive_buy >= 2:
        return "BUYING"
    elif total < 0:
        return "SELLING"
    return "MIXED"


def _fii_dii_fallback(reason):
    return {
        "fii_net_5d": 0,
        "dii_net_5d": 0,
        "fii_trend": "MIXED",
        "dii_trend": "MIXED",
        "combined": "MIXED",
        "days": [],
        "detail": reason,
        "source": "fallback",
    }


# ─── Delivery percentage ──────────────────────────────────────────────────────


def get_delivery(sym: str, days: int = 20) -> dict:
    """
    Fetches delivery percentage data for a stock.

    Delivery % = what % of traded shares were actually delivered
    (not squared off intraday). High delivery = conviction buying,
    not just speculation.

    Rising delivery % on increasing volume = strong institutional interest.

    Returns:
    {
        "delivery_pct_today":   72.5,     # today's delivery %
        "delivery_pct_avg20":   55.2,     # 20-day average
        "delta":                +17.3,    # today vs avg
        "trend":                "RISING", # RISING / FALLING / STABLE
        "detail":               "...",
        "source":               "nse"
    }
    """
    # Strip .NS suffix for NSE API
    symbol = sym.replace(".NS", "").replace(".BO", "")

    try:
        with _get_nse() as nse:
            end = date.today()
            start = end - timedelta(days=days + 10)  # extra buffer for holidays
            data = nse.deliveryBhavcopy(date=end)

        if data is None:
            return _delivery_fallback(f"No delivery data for {symbol}")

        # data is a DataFrame — filter for our symbol
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            row = data[data["SYMBOL"] == symbol]
            if row.empty:
                return _delivery_fallback(f"{symbol} not in delivery bhavcopy")

            today_pct = float(row["DELIV_PER"].iloc[0])

            # For 20-day average we'd need historical — use today's as proxy
            # and flag that historical avg is unavailable in this call
            return {
                "delivery_pct_today": round(today_pct, 2),
                "delivery_pct_avg20": None,  # requires historical loop
                "delta": None,
                "trend": "UNKNOWN",
                "detail": f"delivery {today_pct:.1f}% today (historical avg pending)",
                "source": "nse",
            }

        return _delivery_fallback("Unexpected data format from NSE")

    except Exception as e:
        return _delivery_fallback(f"Delivery fetch failed: {e}")


def _delivery_fallback(reason):
    return {
        "delivery_pct_today": None,
        "delivery_pct_avg20": None,
        "delta": None,
        "trend": "UNKNOWN",
        "detail": reason,
        "source": "fallback",
    }


# ─── Block deals ──────────────────────────────────────────────────────────────


def get_block_deals(sym: str, days: int = 5) -> dict:
    """
    Fetches block deal activity for a stock in last N trading days.
    Block deals = large trades > Rs10cr executed in a single transaction.

    Large buy block = institutional conviction.
    Large sell block = institutional exit.

    Returns:
    {
        "buy_deals":    2,          # count of buy block deals
        "sell_deals":   0,
        "net_value_cr": +245.6,     # Rs crore, positive = net buy
        "verdict":      "BUY",      # BUY / SELL / NEUTRAL
        "detail":       "...",
        "source":       "nse"
    }
    """
    symbol = sym.replace(".NS", "").replace(".BO", "")

    try:
        with _get_nse() as nse:
            data = nse.blockDeals()

        if data is None:
            return _block_fallback(f"No block deal data")

        import pandas as pd

        if isinstance(data, pd.DataFrame):
            # Filter for our symbol in last N days
            cutoff = date.today() - timedelta(days=days)
            sym_data = data[data["SYMBOL"] == symbol].copy()

            if sym_data.empty:
                return {
                    "buy_deals": 0,
                    "sell_deals": 0,
                    "net_value_cr": 0,
                    "verdict": "NEUTRAL",
                    "detail": "no block deals in last 5 days",
                    "source": "nse",
                }

            buy_deals = 0
            sell_deals = 0
            net_value = 0.0

            for _, row in sym_data.iterrows():
                qty = float(row.get("QTY_TRADED", 0) or 0)
                price = float(row.get("TRADE_PRICE", 0) or 0)
                value = qty * price / 1e7  # convert to crore
                txn_type = str(row.get("BUY_SELL", "")).upper()

                if txn_type == "BUY" or txn_type == "B":
                    buy_deals += 1
                    net_value += value
                elif txn_type == "SELL" or txn_type == "S":
                    sell_deals += 1
                    net_value -= value

            verdict = (
                "BUY" if net_value > 50 else "SELL" if net_value < -50 else "NEUTRAL"
            )
            detail = f"{buy_deals} buy, {sell_deals} sell block deals ({net_value:+.1f}cr net)"

            return {
                "buy_deals": buy_deals,
                "sell_deals": sell_deals,
                "net_value_cr": round(net_value, 2),
                "verdict": verdict,
                "detail": detail,
                "source": "nse",
            }

        return _block_fallback("Unexpected data format")

    except Exception as e:
        return _block_fallback(f"Block deal fetch failed: {e}")


def _block_fallback(reason):
    return {
        "buy_deals": 0,
        "sell_deals": 0,
        "net_value_cr": 0,
        "verdict": "NEUTRAL",
        "detail": reason,
        "source": "fallback",
    }


# ─── Insider / promoter trades ────────────────────────────────────────────────


def get_insider_trades(sym: str, days: int = 30) -> dict:
    """
    Fetches SEBI insider trading disclosures for a stock.
    Promoter buying their own stock = strongest possible signal.
    Required to be disclosed within 2 trading days of transaction.

    Returns:
    {
        "promoter_buying":  True,
        "promoter_selling": False,
        "net_shares":       50000,    # positive = net buy
        "veto":             False,    # True if large promoter sell
        "verdict":          "GREEN",
        "detail":           "...",
        "source":           "nse"
    }
    """
    symbol = sym.replace(".NS", "").replace(".BO", "")

    try:
        with _get_nse() as nse:
            data = nse.insider(symbol=symbol)

        if data is None:
            return _insider_fallback(f"No insider data for {symbol}")

        import pandas as pd

        if isinstance(data, pd.DataFrame) and not data.empty:
            cutoff = date.today() - timedelta(days=days)
            buy_qty = 0
            sell_qty = 0

            for _, row in data.iterrows():
                try:
                    txn_date = pd.Timestamp(row.get("acqfromDt", "")).date()
                    if txn_date < cutoff:
                        continue
                except:
                    continue

                qty = int(row.get("secAcq", 0) or 0)
                mode = str(row.get("tdpTransactionType", "")).lower()

                if "acqui" in mode or "buy" in mode or "purchase" in mode:
                    buy_qty += qty
                elif "disp" in mode or "sell" in mode or "transfer" in mode:
                    sell_qty += qty

            net = buy_qty - sell_qty

            # Large promoter sell = veto signal
            veto = sell_qty > 100000

            verdict = "GREEN"
            if buy_qty > 0 and sell_qty == 0:
                verdict = "GREEN"
            elif veto:
                verdict = "RED"
            elif sell_qty > buy_qty:
                verdict = "AMBER"

            detail = (
                f"promoter bought {buy_qty:,}, sold {sell_qty:,} shares in last {days} days"
                if (buy_qty > 0 or sell_qty > 0)
                else f"no insider activity in last {days} days"
            )

            return {
                "promoter_buying": buy_qty > 0,
                "promoter_selling": sell_qty > 0,
                "net_shares": net,
                "veto": veto,
                "verdict": verdict,
                "detail": detail,
                "source": "nse",
            }

        return {
            "promoter_buying": False,
            "promoter_selling": False,
            "net_shares": 0,
            "veto": False,
            "verdict": "GREEN",
            "detail": "no insider activity found",
            "source": "nse",
        }

    except Exception as e:
        return _insider_fallback(f"Insider fetch failed: {e}")


def _insider_fallback(reason):
    return {
        "promoter_buying": False,
        "promoter_selling": False,
        "net_shares": 0,
        "veto": False,
        "verdict": "GREEN",
        "detail": reason,
        "source": "fallback",
    }
