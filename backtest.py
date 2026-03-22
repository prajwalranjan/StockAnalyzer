"""
backtest.py — Adaptive Strategy Backtester v4
==============================================
5 years · 122 stocks · 3 strategies compared

New in v4:
  ✅ Sector momentum filter — only buy a stock when its sector index
     is also trending above its 20-day MA. Prevents buying into
     sector-wide downtrends even when one stock looks strong.

  Sector indices used (all from Yahoo Finance, free):
    Banks/Finance  → ^NSEBANK
    IT             → ^CNXIT
    Pharma         → ^CNXPHARMA
    Auto           → ^CNXAUTO
    Energy/Oil     → ^CNXENERGY
    Metals         → ^CNXMETAL
    FMCG/Consumer  → ^CNXFMCG
    Industrials    → ^NSEI (Nifty 50 as fallback for uncategorised)

Strategies compared:
  v2 — Fixed params + regime (our best so far)
  v3 — v2 + sector momentum filter
  v4 — v3 + adaptive self-tuning

Run with:  python backtest.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

# ─── Stock universe ────────────────────────────────────────────────────────────

STOCKS = [
    # Banks & Finance
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "FEDERALBNK.NS",
    "IDFCFIRSTB.NS",
    "BANDHANBNK.NS",
    "CHOLAFIN.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "MUTHOOTFIN.NS",
    "MANAPPURAM.NS",
    "PNB.NS",
    "CANBK.NS",
    "UNIONBANK.NS",
    "INDIANB.NS",
    "AUBANK.NS",
    "RBLBANK.NS",
    "IDBI.NS",
    "LICHSGFIN.NS",
    "RECLTD.NS",
    "PFC.NS",
    "IRFC.NS",
    # IT
    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "PERSISTENT.NS",
    "LTIM.NS",
    "TECHM.NS",
    "MPHASIS.NS",
    "COFORGE.NS",
    "OFSS.NS",
    "KPITTECH.NS",
    "ZENSARTECH.NS",
    "NIITLTD.NS",
    # Pharma
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    "LUPIN.NS",
    "AUROPHARMA.NS",
    "TORNTPHARM.NS",
    "ALKEM.NS",
    "ABBOTINDIA.NS",
    "PFIZER.NS",
    "GLAXO.NS",
    "SANOFI.NS",
    # Auto
    "MARUTI.NS",
    "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS",
    "EICHERMOT.NS",
    "TVSMOTOR.NS",
    "MOTHERSON.NS",
    "BOSCHLTD.NS",
    "BHARATFORG.NS",
    "MRF.NS",
    "APOLLOTYRE.NS",
    "BALKRISIND.NS",
    # Energy
    "ONGC.NS",
    "BPCL.NS",
    "GAIL.NS",
    "TATAPOWER.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "IOC.NS",
    "HINDPETRO.NS",
    "ADANIGREEN.NS",
    "TORNTPOWER.NS",
    "CESC.NS",
    "NHPC.NS",
    "SJVN.NS",
    # Metals
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",
    "VEDL.NS",
    "SAIL.NS",
    "NMDC.NS",
    "COALINDIA.NS",
    "NATIONALUM.NS",
    "WELCORP.NS",
    # Consumer / FMCG
    "TITAN.NS",
    "ASIANPAINT.NS",
    "NESTLEIND.NS",
    "HAVELLS.NS",
    "VOLTAS.NS",
    "PIDILITIND.NS",
    "BERGEPAINT.NS",
    "KANSAINER.NS",
    "RELAXO.NS",
    "BATAINDIA.NS",
    "PAGEIND.NS",
    "VIPIND.NS",
    "VBL.NS",
    "RADICO.NS",
    "UBL.NS",
    # Industrials / Others
    "RELIANCE.NS",
    "LT.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "SIEMENS.NS",
    "ABB.NS",
    "BHEL.NS",
    "THERMAX.NS",
    "CUMMINSIND.NS",
    "GRINDWELL.NS",
    "AIAENG.NS",
    "SCHAEFFLER.NS",
    # Cement
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "ACC.NS",
    "AMBUJACEM.NS",
    "SHREECEM.NS",
    "JKCEMENT.NS",
    "RAMCOCEM.NS",
    # Telecom / New-age
    "BHARTIARTL.NS",
    "INDIAMART.NS",
    "NYKAA.NS",
    # Retail
    "DMART.NS",
    "TRENT.NS",
    "SHOPERSTOP.NS",
]
STOCKS = list(dict.fromkeys(STOCKS))  # deduplicate

# ─── Sector index mapping ──────────────────────────────────────────────────────
# Maps each stock to its sector index (Yahoo Finance symbol).
# The sector momentum filter checks: is this index above its 20-day MA today?
# If not, skip the stock even if it looks like a breakout.

SECTOR_INDEX = {
    # Banks & Finance → Nifty Bank
    "HDFCBANK.NS": "^NSEBANK",
    "ICICIBANK.NS": "^NSEBANK",
    "SBIN.NS": "^NSEBANK",
    "AXISBANK.NS": "^NSEBANK",
    "KOTAKBANK.NS": "^NSEBANK",
    "FEDERALBNK.NS": "^NSEBANK",
    "IDFCFIRSTB.NS": "^NSEBANK",
    "BANDHANBNK.NS": "^NSEBANK",
    "AUBANK.NS": "^NSEBANK",
    "RBLBANK.NS": "^NSEBANK",
    "PNB.NS": "^NSEBANK",
    "CANBK.NS": "^NSEBANK",
    "UNIONBANK.NS": "^NSEBANK",
    "INDIANB.NS": "^NSEBANK",
    "IDBI.NS": "^NSEBANK",
    "CHOLAFIN.NS": "^NSEBANK",
    "BAJFINANCE.NS": "^NSEBANK",
    "BAJAJFINSV.NS": "^NSEBANK",
    "MUTHOOTFIN.NS": "^NSEBANK",
    "MANAPPURAM.NS": "^NSEBANK",
    "LICHSGFIN.NS": "^NSEBANK",
    "RECLTD.NS": "^NSEBANK",
    "PFC.NS": "^NSEBANK",
    "IRFC.NS": "^NSEBANK",
    # IT → Nifty IT
    "TCS.NS": "^CNXIT",
    "INFY.NS": "^CNXIT",
    "WIPRO.NS": "^CNXIT",
    "HCLTECH.NS": "^CNXIT",
    "PERSISTENT.NS": "^CNXIT",
    "LTIM.NS": "^CNXIT",
    "TECHM.NS": "^CNXIT",
    "MPHASIS.NS": "^CNXIT",
    "COFORGE.NS": "^CNXIT",
    "OFSS.NS": "^CNXIT",
    "KPITTECH.NS": "^CNXIT",
    "ZENSARTECH.NS": "^CNXIT",
    "NIITLTD.NS": "^CNXIT",
    # Pharma → Nifty Pharma
    "SUNPHARMA.NS": "^CNXPHARMA",
    "DRREDDY.NS": "^CNXPHARMA",
    "CIPLA.NS": "^CNXPHARMA",
    "DIVISLAB.NS": "^CNXPHARMA",
    "LUPIN.NS": "^CNXPHARMA",
    "AUROPHARMA.NS": "^CNXPHARMA",
    "TORNTPHARM.NS": "^CNXPHARMA",
    "ALKEM.NS": "^CNXPHARMA",
    "ABBOTINDIA.NS": "^CNXPHARMA",
    "PFIZER.NS": "^CNXPHARMA",
    "GLAXO.NS": "^CNXPHARMA",
    "SANOFI.NS": "^CNXPHARMA",
    # Auto → Nifty Auto
    "MARUTI.NS": "^CNXAUTO",
    "BAJAJ-AUTO.NS": "^CNXAUTO",
    "HEROMOTOCO.NS": "^CNXAUTO",
    "EICHERMOT.NS": "^CNXAUTO",
    "TVSMOTOR.NS": "^CNXAUTO",
    "MOTHERSON.NS": "^CNXAUTO",
    "BOSCHLTD.NS": "^CNXAUTO",
    "BHARATFORG.NS": "^CNXAUTO",
    "MRF.NS": "^CNXAUTO",
    "APOLLOTYRE.NS": "^CNXAUTO",
    "BALKRISIND.NS": "^CNXAUTO",
    # Energy → Nifty Energy
    "ONGC.NS": "^CNXENERGY",
    "BPCL.NS": "^CNXENERGY",
    "GAIL.NS": "^CNXENERGY",
    "TATAPOWER.NS": "^CNXENERGY",
    "NTPC.NS": "^CNXENERGY",
    "POWERGRID.NS": "^CNXENERGY",
    "IOC.NS": "^CNXENERGY",
    "HINDPETRO.NS": "^CNXENERGY",
    "ADANIGREEN.NS": "^CNXENERGY",
    "TORNTPOWER.NS": "^CNXENERGY",
    "CESC.NS": "^CNXENERGY",
    "NHPC.NS": "^CNXENERGY",
    "SJVN.NS": "^CNXENERGY",
    # Metals → Nifty Metal
    "TATASTEEL.NS": "^CNXMETAL",
    "JSWSTEEL.NS": "^CNXMETAL",
    "HINDALCO.NS": "^CNXMETAL",
    "VEDL.NS": "^CNXMETAL",
    "SAIL.NS": "^CNXMETAL",
    "NMDC.NS": "^CNXMETAL",
    "COALINDIA.NS": "^CNXMETAL",
    "NATIONALUM.NS": "^CNXMETAL",
    "WELCORP.NS": "^CNXMETAL",
    # Consumer / FMCG → Nifty FMCG
    "TITAN.NS": "^CNXFMCG",
    "ASIANPAINT.NS": "^CNXFMCG",
    "NESTLEIND.NS": "^CNXFMCG",
    "HAVELLS.NS": "^CNXFMCG",
    "VOLTAS.NS": "^CNXFMCG",
    "PIDILITIND.NS": "^CNXFMCG",
    "BERGEPAINT.NS": "^CNXFMCG",
    "KANSAINER.NS": "^CNXFMCG",
    "RELAXO.NS": "^CNXFMCG",
    "BATAINDIA.NS": "^CNXFMCG",
    "PAGEIND.NS": "^CNXFMCG",
    "VIPIND.NS": "^CNXFMCG",
    "VBL.NS": "^CNXFMCG",
    "RADICO.NS": "^CNXFMCG",
    "UBL.NS": "^CNXFMCG",
    "DMART.NS": "^CNXFMCG",
    "TRENT.NS": "^CNXFMCG",
    "SHOPERSTOP.NS": "^CNXFMCG",
    "NYKAA.NS": "^CNXFMCG",
}
# Everything not mapped above falls back to Nifty 50 (^NSEI)

SECTOR_SYMBOLS = list(set(SECTOR_INDEX.values())) + ["^NSEI"]
NIFTY_SYMBOL = "^NSEI"

# ─── Configs ───────────────────────────────────────────────────────────────────

BASE = {
    "starting_capital": 20_000,
    "capital_per_trade": 8_000,
    "max_open_positions": 2,
    "stop_loss_pct": 0.04,
    "profit_target_pct": 0.08,
    "max_hold_days": 15,
    "daily_loss_limit": 1_000,
    "brokerage_per_trade": 20,
    "stt_pct": 0.001,
    "exchange_charges_pct": 0.00035,
    "slippage_pct": 0.003,
    "rsi_min": 60,
    "rsi_max": 70,
    "volume_ratio_min": 2.0,
    "adx_min": 25,
    "use_regime": True,
    "adaptive": False,
    "sector_filter": False,  # key new flag
    "min_price": 50,
    "max_price": 2000,
}

CONFIG_V2 = {
    **BASE,
    "label": "v2 Best-so-far (no sector filter)",
    "confirmation_candle": False,
}

CONFIG_V3 = {
    **BASE,
    "label": "v3 + Sector momentum filter",
    "sector_filter": True,
    "confirmation_candle": False,
}

CONFIG_V4 = {
    **BASE,
    "label": "v4 + Sector filter + Adaptive",
    "sector_filter": True,
    "adaptive": True,
    "confirmation_candle": False,
}

CONFIG_V5 = {
    **BASE,
    "label": "v5 + Confirmation candle + Adaptive",
    "sector_filter": True,
    "adaptive": True,
    "confirmation_candle": True,
}  # NEW: wait for day-2 confirmation

# ─── Indicators ────────────────────────────────────────────────────────────────


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))


def compute_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    dm_p = (high - high.shift()).clip(lower=0)
    dm_m = (low.shift() - low).clip(lower=0)
    dm_p = dm_p.where(dm_p > dm_m, 0)
    dm_m = dm_m.where(dm_m > dm_p, 0)
    atr = tr.ewm(span=period, adjust=False).mean()
    di_p = 100 * dm_p.ewm(span=period, adjust=False).mean() / atr
    di_m = 100 * dm_m.ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-10)
    return dx.ewm(span=period, adjust=False).mean()


def sector_is_trending(sym, date, sector_data):
    """
    THE KEY NEW FILTER.

    Returns True if the stock's sector index is above its 20-day
    moving average on this date — meaning the whole sector has
    momentum, not just this one stock.

    Returns True also if we don't have sector data (fail-open,
    don't penalise stocks for missing sector data).

    Plain English: "Is the tide coming in for this sector?
    If yes, individual stocks are more likely to keep rising.
    If no, even a strong-looking stock is likely fighting the current."
    """
    sector_sym = SECTOR_INDEX.get(sym, NIFTY_SYMBOL)
    df = sector_data.get(sector_sym)
    if df is None or df.empty or date not in df.index:
        return True  # fail-open: allow trade if no data

    idx = df.index.get_loc(date)
    if idx < 20:
        return True  # not enough history yet

    close = df["Close"]
    current = float(close.iloc[idx])
    ma20 = float(close.iloc[idx - 20 : idx].mean())

    return current > ma20  # True = sector trending up = allow entry


def compute_regime(nifty_df, date):
    """
    Nifty 50 market regime:
    BULL  = above 50-day MA  → trade normally
    CHOP  = between 50 and 200-day MA → 1 position max
    BEAR  = below 200-day MA → no new trades
    """
    if nifty_df.empty or date not in nifty_df.index:
        return "BULL"
    idx = nifty_df.index.get_loc(date)
    if idx < 200:
        return "BULL"
    close = nifty_df["Close"]
    current = float(close.iloc[idx])
    ma50 = float(close.iloc[idx - 50 : idx].mean())
    ma200 = float(close.iloc[idx - 200 : idx].mean())
    if current > ma50:
        return "BULL"
    elif current > ma200:
        return "CHOP"
    else:
        return "BEAR"


def has_gap_history(df, idx, lookback=60, threshold=0.03, max_allowed=1):
    """Skip stocks that have gapped down badly more than once recently."""
    if idx < lookback:
        return False
    window = df.iloc[idx - lookback : idx]
    gaps = (window["Open"] - window["Close"].shift(1)) / window["Close"].shift(1)
    return (gaps < -threshold).sum() > max_allowed


# ─── Costs ─────────────────────────────────────────────────────────────────────


def tcost(price, qty, side, cfg):
    v = price * qty
    return round(
        cfg["brokerage_per_trade"]
        + v * cfg["exchange_charges_pct"]
        + (v * cfg["stt_pct"] if side == "SELL" else 0),
        2,
    )


def slip(price, side, cfg):
    s = cfg["slippage_pct"]
    return price * (1 + s) if side == "BUY" else price * (1 - s)


# ─── Entry signal ──────────────────────────────────────────────────────────────


def buy_signal(sym, df, idx, cfg, sector_data):
    """
    All conditions must pass to get a BUY signal:
    1. Price in range
    2. Breakout above 20-day high
    3. Volume spike (2x)
    4. RSI 60-70
    5. ADX > 25 (strong trend)
    6. No gap-up chase
    7. No gap-down history
    8. [NEW] Sector index above its 20-day MA
    """
    if idx < 30:
        return False, "need more history"

    today = df.iloc[idx]
    price = float(today["Close"])

    if not (cfg["min_price"] <= price <= cfg["max_price"]):
        return False, "price out of range"

    if price <= float(df["Close"].iloc[idx - 21 : idx].max()):
        return False, "no breakout"

    avg_vol = float(df["Volume"].iloc[idx - 21 : idx].mean())
    vol_ratio = float(today["Volume"]) / avg_vol if avg_vol > 0 else 0
    if vol_ratio < cfg["volume_ratio_min"]:
        return False, f"low volume {vol_ratio:.1f}x"

    r = float(compute_rsi(df["Close"]).iloc[idx])
    if np.isnan(r) or not (cfg["rsi_min"] <= r <= cfg["rsi_max"]):
        return False, f"RSI {r:.1f}"

    if cfg["adx_min"] > 0:
        adx = float(compute_adx(df).iloc[idx])
        if np.isnan(adx) or adx < cfg["adx_min"]:
            return False, f"ADX {adx:.1f} weak"

    gap = (price - float(df["Close"].iloc[idx - 1])) / float(df["Close"].iloc[idx - 1])
    if gap > 0.05:
        return False, f"gap-up chase {gap*100:.1f}%"

    if has_gap_history(df, idx):
        return False, "gap-prone stock"

    # Sector momentum filter
    if cfg.get("sector_filter"):
        if not sector_is_trending(sym, df.index[idx], sector_data):
            sector_sym = SECTOR_INDEX.get(sym, NIFTY_SYMBOL)
            return False, f"sector {sector_sym} below MA20"

    return True, {
        "rsi": round(r, 1),
        "vol_ratio": round(vol_ratio, 2),
        "breakout_close": float(df["Close"].iloc[idx]),
    }


# ─── Exit logic ────────────────────────────────────────────────────────────────


def check_exit(pos, row, days_held, cfg):
    low, high, close = float(row["Low"]), float(row["High"]), float(row["Close"])
    bp = pos["actual_buy"]
    sl = bp * (1 - cfg["stop_loss_pct"])
    tp = bp * (1 + cfg["profit_target_pct"])

    # Gap down overnight
    if float(row["Open"]) < bp * (1 - cfg["stop_loss_pct"] * 1.5):
        return [("FULL", float(row["Open"]), "gap-down")]

    if low <= sl:
        return [("FULL", sl, "stop-loss")]
    if high >= tp:
        return [("FULL", tp, "profit-target")]
    if days_held >= cfg["max_hold_days"]:
        return [("FULL", close, f"time-stop({days_held}d)")]
    return []


# ─── Adaptive tuning ───────────────────────────────────────────────────────────


def self_tune(cfg, recent):
    if len(recent) < 3:
        return cfg, None
    cfg = deepcopy(cfg)
    wins = [t for t in recent if t["pnl"] > 0]
    wr = len(wins) / len(recent)
    n = len(recent)

    if wr < 0.35 and cfg["volume_ratio_min"] < 3.0:
        cfg["volume_ratio_min"] = round(min(cfg["volume_ratio_min"] + 0.25, 3.0), 2)
        return cfg, f"↑ vol min → {cfg['volume_ratio_min']}x  (win rate: {wr:.0%})"
    elif n < 2 and cfg["volume_ratio_min"] > 1.5:
        cfg["volume_ratio_min"] = round(max(cfg["volume_ratio_min"] - 0.25, 1.5), 2)
        return cfg, f"↓ vol min → {cfg['volume_ratio_min']}x  (too few signals)"
    elif wr > 0.60:
        return cfg, f"✓ no change  (win rate good: {wr:.0%})"
    return cfg, None


# ─── Helpers ───────────────────────────────────────────────────────────────────


def open_val(positions, data, date):
    return sum(
        float(data[s].loc[date]["Close"]) * p["qty"]
        for s, p in positions.items()
        if s in data and date in data[s].index
    )


def do_sell(pos, price, qty, date, sym, reason, cfg):
    actual = slip(price, "SELL", cfg)
    fee = tcost(actual, qty, "SELL", cfg)
    pnl = (actual - pos["actual_buy"]) * qty - fee
    return (
        pnl,
        fee,
        {
            "symbol": sym,
            "buy_date": pos["buy_date"].strftime("%d %b %Y"),
            "sell_date": date.strftime("%d %b %Y"),
            "buy_price": round(pos["actual_buy"], 2),
            "sell_price": round(actual, 2),
            "qty": qty,
            "pnl": round(pnl, 2),
            "reason": reason,
            "days_held": (date - pos["buy_date"]).days,
            "_date": date,
        },
    )


# ─── Simulation ────────────────────────────────────────────────────────────────


def simulate(data, nifty_df, sector_data, cfg, trading_days):
    active = deepcopy(cfg)
    capital = cfg["starting_capital"]
    positions = {}
    pending = {}  # sym -> breakout_close (waiting for day-2 confirmation)
    trades, daily_log, tune_log = [], [], []
    total_fees = 0
    day_loss, prev_date, last_tune = 0, None, None

    for date in trading_days:
        if prev_date is None or date.date() != prev_date.date():
            day_loss = 0
        prev_date = date

        if active.get("adaptive") and (
            last_tune is None or (date - last_tune).days >= 60
        ):
            recent = [t for t in trades if (date - t["_date"]).days <= 60]
            active, note = self_tune(active, recent)
            if note:
                tune_log.append({"date": date.strftime("%b %Y"), "change": note})
            last_tune = date

        regime = compute_regime(nifty_df, date) if active.get("use_regime") else "BULL"

        # Exits (always, even in Bear)
        for sym in list(positions.keys()):
            if sym not in data or date not in data[sym].index:
                continue
            pos = positions[sym]
            row = data[sym].loc[date]
            held = (date - pos["buy_date"]).days
            for action, price, reason in check_exit(pos, row, held, active):
                tag = reason + (" [BEAR]" if regime == "BEAR" else "")
                pnl, fee, rec = do_sell(pos, price, pos["qty"], date, sym, tag, active)
                capital += slip(price, "SELL", active) * pos["qty"] - fee
                total_fees += fee
                day_loss += pnl
                trades.append(rec)
                del positions[sym]
                break

        # Entries (Bull/Chop only)
        if regime != "BEAR":
            paused = day_loss <= -active["daily_loss_limit"]
            max_slots = 1 if regime == "CHOP" else active["max_open_positions"]
            slots = max_slots - len(positions)

            if not paused and slots > 0:

                # ── Step A: Check pending signals for confirmation ────────
                # These are stocks that broke out YESTERDAY.
                # If they close above their breakout close today → buy.
                confirmed = []
                if active.get("confirmation_candle"):
                    for sym, info in list(pending.items()):
                        if sym in positions or sym not in data:
                            del pending[sym]
                            continue
                        if date not in data[sym].index:
                            del pending[sym]
                            continue
                        today_close = float(data[sym].loc[date]["Close"])
                        breakout_close = info["breakout_close"]
                        if today_close > breakout_close:
                            # Confirmed — add to buy list
                            confirmed.append((sym, data[sym].loc[date], info))
                        # Whether confirmed or not, clear from pending
                        del pending[sym]

                # ── Step B: Scan for fresh breakout signals today ─────────
                fresh = []
                for sym, df in data.items():
                    if sym in positions or date not in df.index:
                        continue
                    if active.get("confirmation_candle") and sym in pending:
                        continue  # already waiting for confirmation
                    idx = df.index.get_loc(date)
                    ok, info = buy_signal(sym, df, idx, active, sector_data)
                    if ok:
                        if active.get("confirmation_candle"):
                            # Don't buy today — park in pending for tomorrow
                            pending[sym] = info
                        else:
                            fresh.append((sym, df.loc[date], info))

                # With confirmation: buy confirmed; without: buy fresh
                candidates = confirmed if active.get("confirmation_candle") else fresh
                candidates.sort(key=lambda x: x[2]["vol_ratio"], reverse=True)

                for sym, row, info in candidates[:slots]:
                    price = float(row["Close"])
                    actual = slip(price, "BUY", active)
                    qty = int(active["capital_per_trade"] / actual)
                    if qty < 1:
                        continue
                    cost = actual * qty + tcost(actual, qty, "BUY", active)
                    if capital < cost:
                        continue
                    capital -= cost
                    total_fees += tcost(actual, qty, "BUY", active)
                    positions[sym] = {
                        "buy_date": date,
                        "actual_buy": actual,
                        "qty": qty,
                    }
        else:
            # In BEAR regime — clear any pending signals (don't accumulate stale ones)
            if active.get("confirmation_candle"):
                pending.clear()

        ov = open_val(positions, data, date)
        daily_log.append(
            {"date": date, "value": round(capital + ov, 2), "regime": regime}
        )

    # Close remaining
    for sym, pos in list(positions.items()):
        if sym not in data:
            continue
        last = float(data[sym]["Close"].iloc[-1])
        actual = slip(last, "SELL", active)
        fee = tcost(actual, pos["qty"], "SELL", active)
        pnl = (actual - pos["actual_buy"]) * pos["qty"] - fee
        capital += actual * pos["qty"] - fee
        total_fees += fee
        trades.append(
            {
                "symbol": sym,
                "buy_date": pos["buy_date"].strftime("%d %b %Y"),
                "sell_date": "end",
                "buy_price": round(pos["actual_buy"], 2),
                "sell_price": round(actual, 2),
                "qty": pos["qty"],
                "pnl": round(pnl, 2),
                "reason": "backtest-end",
                "days_held": (trading_days[-1] - pos["buy_date"]).days,
                "_date": trading_days[-1],
            }
        )

    return trades, daily_log, tune_log, total_fees


# ─── Metrics ───────────────────────────────────────────────────────────────────


def metrics(trades, daily_log, fees, cfg):
    if not trades or not daily_log:
        return None
    sc = cfg["starting_capital"]
    final = daily_log[-1]["value"]
    pnl = final - sc
    ret = pnl / sc * 100
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / len(trades) * 100
    avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_l = np.mean([t["pnl"] for t in losses]) if losses else 0
    rr = abs(avg_w / avg_l) if avg_l else 0
    ev = (wr / 100 * avg_w) + ((1 - wr / 100) * avg_l)

    vals = [d["value"] for d in daily_log]
    peak, mdd = sc, 0
    for v in vals:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak * 100)

    reasons = {}
    for t in trades:
        r = t["reason"].split("(")[0].split("[")[0].strip()
        reasons[r] = reasons.get(r, 0) + 1

    # Count how many trades were blocked by sector filter
    # (we can estimate from trade count difference vs v2)
    return {
        "final": final,
        "pnl": pnl,
        "ret": ret,
        "ann": ret / 5,
        "n": len(trades),
        "wr": wr,
        "avg_w": avg_w,
        "avg_l": avg_l,
        "rr": rr,
        "ev": ev,
        "mdd": mdd,
        "fees": fees,
        "wins": len(wins),
        "losses": len(losses),
        "best": max(t["pnl"] for t in trades),
        "worst": min(t["pnl"] for t in trades),
        "reasons": reasons,
        "time_stops": sum(1 for t in trades if "time-stop" in t["reason"]),
        "gap_downs": sum(1 for t in trades if "gap-down" in t["reason"]),
        "whipsaws": sum(1 for t in trades if t["days_held"] <= 2 and t["pnl"] < 0),
    }


# ─── Reports ───────────────────────────────────────────────────────────────────


def print_comparison(results):
    S = "─" * 74
    print(f"\n{'='*74}")
    print("  COMPARISON  (same ₹20,000 · same 122 stocks · 5 years)")
    print(f"{'='*74}\n")
    labels = [r["cfg"]["label"] for r in results]
    mets = [metrics(r["trades"], r["daily"], r["fees"], r["cfg"]) for r in results]

    def row(label, vals, fmt):
        print(f"  {label:<28} " + " ".join(fmt(v) for v in vals))

    print(f"  {'':28} " + " ".join(f"{l:>22}" for l in labels))
    print(f"  {S}")
    row("Final Capital", [m["final"] for m in mets], lambda v: f"₹{v:>20,.0f}")
    row("Total Return", [m["ret"] for m in mets], lambda v: f"{v:>+21.1f}%")
    row("Annualised Return", [m["ann"] for m in mets], lambda v: f"{v:>+21.1f}%")
    row("Max Drawdown", [m["mdd"] for m in mets], lambda v: f"{v:>20.1f}%")
    print(f"  {S}")
    row("Total Trades", [m["n"] for m in mets], lambda v: f"{v:>22.0f}")
    row("Win Rate", [m["wr"] for m in mets], lambda v: f"{v:>21.1f}%")
    row("Avg Win", [m["avg_w"] for m in mets], lambda v: f"₹{v:>+20,.0f}")
    row("Avg Loss", [m["avg_l"] for m in mets], lambda v: f"₹{v:>+20,.0f}")
    row("Reward : Risk", [m["rr"] for m in mets], lambda v: f"{v:>21.2f}x")
    row("Expected / trade", [m["ev"] for m in mets], lambda v: f"₹{v:>+20,.0f}")
    print(f"  {S}")
    row("Fees Paid", [m["fees"] for m in mets], lambda v: f"₹{v:>20,.0f}")
    row("Gap Downs", [m["gap_downs"] for m in mets], lambda v: f"{v:>22.0f}")
    row("Time Stops", [m["time_stops"] for m in mets], lambda v: f"{v:>22.0f}")
    print()


def print_detail(result, tune_log, sector_blocks=None):
    cfg = result["cfg"]
    trades = result["trades"]
    daily = result["daily"]
    m = metrics(trades, daily, result["fees"], cfg)
    if not m:
        return
    S = "─" * 60

    print(f"\n{'='*60}")
    print(f"  DETAILED REPORT — {cfg['label']}")
    print(f"{'='*60}")

    # Monthly returns
    df_log = pd.DataFrame(daily).set_index("date")
    monthly = df_log["value"].resample("ME").last().pct_change().dropna() * 100
    good = bad = 0
    print(f"\nMONTHLY RETURNS\n{S}")
    for month, ret in monthly.items():
        bar = (
            "+" * min(int(abs(ret) * 1.5), 28)
            if ret >= 0
            else "-" * min(int(abs(ret) * 1.5), 28)
        )
        good += ret >= 0
        bad += ret < 0
        print(f"  {month.strftime('%b %Y')}   {ret:+6.2f}%  {bar}")
    print(
        f"\n  Positive: {good}  Negative: {bad}  ({good/(good+bad)*100:.0f}% positive months)"
    )

    # Exit breakdown
    print(f"\nHOW TRADES ENDED\n{S}")
    for r, c in sorted(m["reasons"].items(), key=lambda x: -x[1]):
        bar = "#" * int(c / max(m["reasons"].values()) * 25)
        print(f"  {r:30s} {c:3d}  ({c/m['n']*100:.0f}%)  {bar}")

    # Regime
    if cfg.get("use_regime"):
        regimes = {}
        for d in daily:
            regimes[d.get("regime", "BULL")] = (
                regimes.get(d.get("regime", "BULL"), 0) + 1
            )
        tot = len(daily)
        print(f"\nMARKET REGIME\n{S}")
        for reg in ["BULL", "CHOP", "BEAR"]:
            c = regimes.get(reg, 0)
            bar = "#" * int(c / tot * 30)
            print(f"  {reg:5s}  {c:4d} days ({c/tot*100:.0f}%)  {bar}")

    # Sector filter stats
    if cfg.get("sector_filter") and sector_blocks is not None:
        print(f"\nSECTOR FILTER IMPACT\n{S}")
        print(f"  Trades blocked by sector filter: {sector_blocks}")
        print(f"  (These were valid breakouts, but sector was in a downtrend)")
        top = (
            sorted(sector_blocks.items(), key=lambda x: -x[1])[:5]
            if isinstance(sector_blocks, dict)
            else []
        )
        if top:
            print(f"  Most blocked sectors:")
            for sym, count in top:
                print(f"    {sym:15s} {count} blocked")

    # Tuning log
    if tune_log:
        print(f"\nADAPTIVE TUNING LOG\n{S}")
        for e in tune_log[:15]:
            print(f"  {e['date']:10s}  {e['change']}")

    # Last 15 trades
    print(f"\nLAST 15 TRADES\n{S}")
    print(f"  {'Stock':12s}  {'Buy':>7}  {'Sell':>7}  {'Qty':>3}  {'P&L':>8}  Reason")
    for t in trades[-15:]:
        sym = t["symbol"].replace(".NS", "")
        print(
            f"  {sym:12s}  Rs{t['buy_price']:>5.0f}  Rs{t['sell_price']:>5.0f}"
            f"  {t['qty']:>3}  Rs{t['pnl']:>+7.0f}  {t['reason']}"
        )

    # Verdict
    print(f"\nVERDICT\n{S}")
    print(
        f"  Win rate {m['wr']:.0f}%  x  Reward:Risk {m['rr']:.2f}x  =  Rs{m['ev']:+.0f} per trade"
    )
    print(f"  {m['n']} trades / 5 years = ~{m['n']//60} trades per month")
    print(f"  Worst stretch: Rs20,000 could drop to Rs{20000*(1-m['mdd']/100):,.0f}\n")

    if m["ev"] > 0 and m["rr"] >= 1.5 and m["wr"] >= 45:
        v = "GREEN — Strategy viable. Paper trade 4 weeks, then go live."
    elif m["ev"] > 0:
        v = "YELLOW — Positive expected value. Paper trade 6+ weeks first."
    elif m["pnl"] > -4000:
        v = "ORANGE — Still losing but much less. Keep refining."
    else:
        v = "RED — Needs more work. Don't deploy real money yet."

    print(f"  {v}")
    print(f"\n{'='*60}")
    print("  Past performance does not guarantee future results.")
    print(f"{'='*60}\n")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "=" * 60)
    print("  TRADING BOT — BACKTESTER v5")
    print("  5 Years | 122 stocks | Confirmation candle added")
    print("=" * 60)

    end = datetime.today()
    start = end - timedelta(days=365 * 5 + 30)

    # Download stock data
    print(f"\nDownloading 5 years of data for {len(STOCKS)} stocks...\n")
    data, skipped = {}, []
    for sym in STOCKS:
        try:
            df = yf.download(
                sym, start=start, end=end, progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= 100:
                data[sym] = df
                print(f"  OK {sym}")
            else:
                skipped.append(sym)
                print(f"  SKIP {sym} ({len(df)} days)")
        except Exception as e:
            skipped.append(sym)
            print(f"  FAIL {sym}: {e}")

    # Download sector indices
    print(f"\nDownloading {len(SECTOR_SYMBOLS)} sector indices...")
    sector_data = {}
    for sym in SECTOR_SYMBOLS:
        try:
            df = yf.download(
                sym, start=start, end=end, progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= 50:
                sector_data[sym] = df
                print(f"  OK {sym} ({len(df)} days)")
            else:
                print(f"  SKIP {sym}")
        except Exception as e:
            print(f"  FAIL {sym}: {e}")

    nifty_df = sector_data.get(NIFTY_SYMBOL, pd.DataFrame())

    print(f"\nLoaded {len(data)}/{len(STOCKS)} stocks")
    print(f"Loaded {len(sector_data)}/{len(SECTOR_SYMBOLS)} sector indices")

    trading_days = sorted(set(d for df in data.values() for d in df.index))
    print(f"Trading days: {len(trading_days)}\n")

    # Run v3, v4, v5 — v2 kept as reference but not re-run (saves time)
    all_results = []
    for cfg in [CONFIG_V3, CONFIG_V4, CONFIG_V5]:
        print(f"Simulating {cfg['label']}...")
        trades, daily, tune_log, fees = simulate(
            data, nifty_df, sector_data, cfg, trading_days
        )
        pnl = (daily[-1]["value"] - cfg["starting_capital"]) if daily else 0
        print(f"  -> {len(trades)} trades | P&L: Rs{pnl:+,.0f}\n")
        all_results.append(
            {
                "cfg": cfg,
                "trades": trades,
                "daily": daily,
                "tune_log": tune_log,
                "fees": fees,
            }
        )

    print_comparison(all_results)

    # Detailed report for best version
    best = max(all_results, key=lambda r: r["daily"][-1]["value"] if r["daily"] else 0)
    print(f"(Detailed report for best: {best['cfg']['label']})")
    print_detail(best, best["tune_log"])


if __name__ == "__main__":
    t0 = datetime.now()
    main()
    elapsed = (datetime.now() - t0).seconds
    print(f"Done in {elapsed}s\n")
