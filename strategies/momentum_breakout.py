"""
strategy.py — V5 Strategy Engine
Implements the exact strategy proven in backtesting:
  Breakout + Volume 2x + RSI 60-70 + ADX>25 + Sector trending + Confirmation candle
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import database

STOCKS = [
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
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    "LUPIN.NS",
    "AUROPHARMA.NS",
    "TORNTPHARM.NS",
    "ALKEM.NS",
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
    "ONGC.NS",
    "BPCL.NS",
    "GAIL.NS",
    "TATAPOWER.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "IOC.NS",
    "NHPC.NS",
    "SJVN.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",
    "VEDL.NS",
    "SAIL.NS",
    "NMDC.NS",
    "COALINDIA.NS",
    "NATIONALUM.NS",
    "WELCORP.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "NESTLEIND.NS",
    "HAVELLS.NS",
    "VOLTAS.NS",
    "PIDILITIND.NS",
    "BERGEPAINT.NS",
    "RELAXO.NS",
    "BATAINDIA.NS",
    "VBL.NS",
    "RELIANCE.NS",
    "LT.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "SIEMENS.NS",
    "ABB.NS",
    "BHEL.NS",
    "THERMAX.NS",
    "CUMMINSIND.NS",
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "ACC.NS",
    "AMBUJACEM.NS",
    "SHREECEM.NS",
    "BHARTIARTL.NS",
    "DMART.NS",
    "TRENT.NS",
]

SECTOR_INDEX = {
    "HDFCBANK.NS": "^NSEBANK",
    "ICICIBANK.NS": "^NSEBANK",
    "SBIN.NS": "^NSEBANK",
    "AXISBANK.NS": "^NSEBANK",
    "KOTAKBANK.NS": "^NSEBANK",
    "FEDERALBNK.NS": "^NSEBANK",
    "IDFCFIRSTB.NS": "^NSEBANK",
    "BANDHANBNK.NS": "^NSEBANK",
    "CHOLAFIN.NS": "^NSEBANK",
    "BAJFINANCE.NS": "^NSEBANK",
    "BAJAJFINSV.NS": "^NSEBANK",
    "MUTHOOTFIN.NS": "^NSEBANK",
    "MANAPPURAM.NS": "^NSEBANK",
    "PNB.NS": "^NSEBANK",
    "CANBK.NS": "^NSEBANK",
    "UNIONBANK.NS": "^NSEBANK",
    "INDIANB.NS": "^NSEBANK",
    "AUBANK.NS": "^NSEBANK",
    "RBLBANK.NS": "^NSEBANK",
    "IDBI.NS": "^NSEBANK",
    "LICHSGFIN.NS": "^NSEBANK",
    "RECLTD.NS": "^NSEBANK",
    "PFC.NS": "^NSEBANK",
    "IRFC.NS": "^NSEBANK",
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
    "SUNPHARMA.NS": "^CNXPHARMA",
    "DRREDDY.NS": "^CNXPHARMA",
    "CIPLA.NS": "^CNXPHARMA",
    "DIVISLAB.NS": "^CNXPHARMA",
    "LUPIN.NS": "^CNXPHARMA",
    "AUROPHARMA.NS": "^CNXPHARMA",
    "TORNTPHARM.NS": "^CNXPHARMA",
    "ALKEM.NS": "^CNXPHARMA",
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
    "ONGC.NS": "^CNXENERGY",
    "BPCL.NS": "^CNXENERGY",
    "GAIL.NS": "^CNXENERGY",
    "TATAPOWER.NS": "^CNXENERGY",
    "NTPC.NS": "^CNXENERGY",
    "POWERGRID.NS": "^CNXENERGY",
    "IOC.NS": "^CNXENERGY",
    "NHPC.NS": "^CNXENERGY",
    "SJVN.NS": "^CNXENERGY",
    "TATASTEEL.NS": "^CNXMETAL",
    "JSWSTEEL.NS": "^CNXMETAL",
    "HINDALCO.NS": "^CNXMETAL",
    "VEDL.NS": "^CNXMETAL",
    "SAIL.NS": "^CNXMETAL",
    "NMDC.NS": "^CNXMETAL",
    "COALINDIA.NS": "^CNXMETAL",
    "NATIONALUM.NS": "^CNXMETAL",
    "WELCORP.NS": "^CNXMETAL",
    "TITAN.NS": "^CNXFMCG",
    "ASIANPAINT.NS": "^CNXFMCG",
    "NESTLEIND.NS": "^CNXFMCG",
    "HAVELLS.NS": "^CNXFMCG",
    "VOLTAS.NS": "^CNXFMCG",
    "PIDILITIND.NS": "^CNXFMCG",
    "BERGEPAINT.NS": "^CNXFMCG",
    "RELAXO.NS": "^CNXFMCG",
    "BATAINDIA.NS": "^CNXFMCG",
    "VBL.NS": "^CNXFMCG",
    "DMART.NS": "^CNXFMCG",
    "TRENT.NS": "^CNXFMCG",
}
NIFTY = "^NSEI"

CFG = {
    "rsi_min": 60,
    "rsi_max": 70,
    "volume_ratio_min": 2.0,
    "adx_min": 25,
    "stop_loss_pct": 0.04,
    "profit_target_pct": 0.08,
    "max_hold_days": 15,
    "capital_per_trade": 8000,
    "max_positions": 2,
    "daily_loss_limit": 1000,
    "min_price": 50,
    "max_price": 2000,
}

_cache = {}


def _fetch(sym, period="3mo"):
    key = f"{sym}_{period}"
    if key not in _cache:
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            _cache[key] = df if len(df) >= 25 else pd.DataFrame()
        except:
            _cache[key] = pd.DataFrame()
    return _cache[key]


def clear_cache():
    _cache.clear()


def _rsi(prices, p=14):
    d = prices.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))


def _adx(df, p=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(
        axis=1
    )
    dp = (h - h.shift()).clip(lower=0)
    dm = (l.shift() - l).clip(lower=0)
    dp = dp.where(dp > dm, 0)
    dm = dm.where(dm > dp, 0)
    atr = tr.ewm(span=p, adjust=False).mean()
    dip = 100 * dp.ewm(span=p, adjust=False).mean() / atr
    dim = 100 * dm.ewm(span=p, adjust=False).mean() / atr
    dx = 100 * (dip - dim).abs() / (dip + dim + 1e-10)
    return dx.ewm(span=p, adjust=False).mean()


def get_market_regime():
    df = _fetch(NIFTY, "1y")
    if df.empty or len(df) < 200:
        return "BULL"
    c = df["Close"]
    cur, ma50, ma200 = (
        float(c.iloc[-1]),
        float(c.iloc[-50:].mean()),
        float(c.iloc[-200:].mean()),
    )
    if cur > ma50:
        return "BULL"
    elif cur > ma200:
        return "CHOP"
    else:
        return "BEAR"


def get_nifty_info():
    df = _fetch(NIFTY, "5d")
    if df.empty:
        return {"price": 0, "change_pct": 0}
    price = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else price
    return {
        "price": round(price, 2),
        "change_pct": round((price - prev) / prev * 100, 2),
    }


def _sector_trending(sym):
    sector_sym = SECTOR_INDEX.get(sym, NIFTY)
    df = _fetch(sector_sym, "3mo")
    if df.empty or len(df) < 22:
        return True
    c = df["Close"]
    return float(c.iloc[-1]) > float(c.iloc[-20:].mean())


def get_current_price(sym):
    df = _fetch(sym, "5d")
    if df.empty:
        return 0.0
    return round(float(df["Close"].iloc[-1]), 2)


def check_signal(sym):
    """Run all v5 entry conditions. Returns (signal_dict, None) or (None, reason)."""
    df = _fetch(sym, "3mo")
    if df.empty or len(df) < 32:
        return None, "insufficient data"

    price = float(df["Close"].iloc[-1])
    if not (CFG["min_price"] <= price <= CFG["max_price"]):
        return None, "price out of range"

    prev_high = float(df["Close"].iloc[-22:-1].max())
    if price <= prev_high:
        return None, "no breakout"

    avg_vol = float(df["Volume"].iloc[-22:-1].mean())
    vol_ratio = float(df["Volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 0
    if vol_ratio < CFG["volume_ratio_min"]:
        return None, f"low volume {vol_ratio:.1f}x"

    rsi_val = float(_rsi(df["Close"]).iloc[-1])
    if np.isnan(rsi_val) or not (CFG["rsi_min"] <= rsi_val <= CFG["rsi_max"]):
        return None, f"RSI {rsi_val:.1f}"

    adx_val = float(_adx(df).iloc[-1])
    if np.isnan(adx_val) or adx_val < CFG["adx_min"]:
        return None, f"ADX {adx_val:.1f} weak"

    gap = (price - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2])
    if gap > 0.05:
        return None, f"gap-up chase {gap*100:.1f}%"

    if not _sector_trending(sym):
        return None, "sector not trending"

    qty = int(CFG["capital_per_trade"] / price)
    if qty < 1:
        return None, "price too high"

    return {
        "symbol": sym,
        "display_name": sym.replace(".NS", ""),
        "price": round(price, 2),
        "rsi": round(rsi_val, 1),
        "adx": round(adx_val, 1),
        "volume_ratio": round(vol_ratio, 2),
        "prev_high": round(prev_high, 2),
        "quantity": qty,
        "invested": round(qty * price, 2),
        "stop_loss": round(price * (1 - CFG["stop_loss_pct"]), 2),
        "target": round(price * (1 + CFG["profit_target_pct"]), 2),
        "sector": SECTOR_INDEX.get(sym, NIFTY),
    }, None


def scan_for_signals():
    """
    Full scan. Returns dict with:
      - pending: Day 1 breakouts detected today (need confirmation tomorrow)
      - confirmed: Yesterday's pending signals that held above breakout close
      - regime: current market state
    """
    clear_cache()
    regime = get_market_regime()
    open_syms = {p["symbol"] for p in database.get_open_positions()}

    confirmed = []
    if regime != "BEAR":
        for p in database.get_pending_signals():
            df = _fetch(p["symbol"], "5d")
            confirmed_flag = False
            if not df.empty and len(df) >= 2:
                today_close = float(df["Close"].iloc[-1])
                if today_close > p["breakout_close"]:
                    confirmed_flag = True
                    confirmed.append(
                        {**p, "current_price": round(today_close, 2), "confirmed": True}
                    )
            database.mark_signal_processed(p["id"], confirmed=confirmed_flag)

    pending = []
    if regime != "BEAR":
        for sym in STOCKS:
            if sym in open_syms:
                continue
            signal, _ = check_signal(sym)
            if signal:
                pending.append(signal)
                database.save_pending_signal(signal)

    return {"pending": pending, "confirmed": confirmed, "regime": regime}


def check_exits():
    """Check all open positions for stop loss / profit target / time stop."""
    positions = database.get_open_positions()
    exits = []
    for pos in positions:
        df = _fetch(pos["symbol"], "5d")
        if df.empty:
            continue
        current = float(df["Close"].iloc[-1])
        bp = pos["buy_price"]
        chg = (current - bp) / bp * 100
        sl, tp = bp * (1 - CFG["stop_loss_pct"]), bp * (1 + CFG["profit_target_pct"])
        reason = None
        if current <= sl:
            reason = f"Stop loss ({chg:+.1f}%)"
        elif current >= tp:
            reason = f"Profit target ({chg:+.1f}%)"
        elif pos["days_held"] >= CFG["max_hold_days"]:
            reason = f"Time stop — {pos['days_held']} days held"
        if reason:
            exits.append(
                {
                    **pos,
                    "current_price": round(current, 2),
                    "pnl": round((current - bp) * pos["quantity"], 2),
                    "pnl_pct": round(chg, 2),
                    "reason": reason,
                }
            )
    return exits
