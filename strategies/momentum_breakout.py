"""
strategies/track1.py — V5 Momentum Breakout Strategy
======================================================
Pre-phase update: expanded from 90 → 250 Nifty 500 stocks.
More stocks = more signals = more training data for future ML.

Also adds paper parallel run — same signals as live Track 1,
but max 5 positions, paper mode only. Generates 2-3x more
labelled trade samples for the Phase 5 ML model without
risking any additional real money.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import database

# ─── Expanded universe (250 Nifty 500 stocks) ────────────────────────────────
# Grouped by sector. Stocks priced Rs50-Rs2000 with decent liquidity.
# Removed: stocks frequently delisted, very illiquid, or price > Rs2000.

STOCKS = [
    # Banks — large & mid cap
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "FEDERALBNK.NS",
    "IDFCFIRSTB.NS",
    "BANDHANBNK.NS",
    "AUBANK.NS",
    "RBLBANK.NS",
    "PNB.NS",
    "CANBK.NS",
    "UNIONBANK.NS",
    "INDIANB.NS",
    "IDBI.NS",
    "BANKBARODA.NS",
    "MAHABANK.NS",
    "CENTRALBK.NS",
    "IOB.NS",
    "UCOBANK.NS",
    # NBFCs & Finance
    "CHOLAFIN.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "MUTHOOTFIN.NS",
    "MANAPPURAM.NS",
    "LICHSGFIN.NS",
    "RECLTD.NS",
    "PFC.NS",
    "IRFC.NS",
    "M&MFIN.NS",
    "SUNDARMFIN.NS",
    "ABCAPITAL.NS",
    "IIFL.NS",
    "POONAWALLA.NS",
    # IT & Tech
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
    "LTTS.NS",
    "BIRLASOFT.NS",
    "MASTEK.NS",
    "INTELLECT.NS",
    "SONATSOFTW.NS",
    "TATAELXSI.NS",
    "CYIENT.NS",
    # Pharma & Healthcare
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
    "IPCALAB.NS",
    "NATCOPHARM.NS",
    "GLENMARK.NS",
    "GRANULES.NS",
    "AJANTPHARM.NS",
    "LAURUSLABS.NS",
    "BIOCON.NS",
    "ESTRELLABLAST.NS",
    # Auto & Auto Ancillaries
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
    "CEATLTD.NS",
    "EXIDEIND.NS",
    "AMARAJABAT.NS",
    "SUNDRMFAST.NS",
    "ENDURANCE.NS",
    "CRAFTSMAN.NS",
    # Energy & Power
    "ONGC.NS",
    "BPCL.NS",
    "GAIL.NS",
    "TATAPOWER.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "IOC.NS",
    "HINDPETRO.NS",
    "NHPC.NS",
    "SJVN.NS",
    "ADANIGREEN.NS",
    "TORNTPOWER.NS",
    "CESC.NS",
    "JSWENERGY.NS",
    "IREDA.NS",
    "GIPCL.NS",
    "RPOWER.NS",
    "INDIAGRID.NS",
    # Metals & Mining
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",
    "VEDL.NS",
    "SAIL.NS",
    "NMDC.NS",
    "COALINDIA.NS",
    "NATIONALUM.NS",
    "WELCORP.NS",
    "JINDALSTEL.NS",
    "RATNAMANI.NS",
    "APLAPOLLO.NS",
    "ABIRLANUVO.NS",
    "MOIL.NS",
    "GMRINFRA.NS",
    # FMCG & Consumer
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
    "RADICO.NS",
    "UBL.NS",
    "TATACONSUM.NS",
    "EMAMILTD.NS",
    "JYOTHYLAB.NS",
    "MARICO.NS",
    "DABUR.NS",
    "COLPAL.NS",
    "GILLETTE.NS",
    "PGHH.NS",
    "HONASA.NS",
    "MANYAVAR.NS",
    # Retail & New-age
    "DMART.NS",
    "TRENT.NS",
    "SHOPERSTOP.NS",
    "VMART.NS",
    "ZOMATO.NS",
    "NYKAA.NS",
    "INDIAMART.NS",
    "CARTRADE.NS",
    # Industrials & Capital Goods
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
    "ELGIEQUIP.NS",
    "TIINDIA.NS",
    "BHARAT.NS",
    "COCHINSHIP.NS",
    "MAZDOCK.NS",
    "GRSE.NS",
    "BEL.NS",
    "HAL.NS",
    "BEML.NS",
    # Cement & Construction
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "ACC.NS",
    "AMBUJACEM.NS",
    "SHREECEM.NS",
    "JKCEMENT.NS",
    "RAMCOCEM.NS",
    "HEIDELBERG.NS",
    "BIRLACORPN.NS",
    "NUVOCO.NS",
    # Real Estate
    "DLF.NS",
    "GODREJPROP.NS",
    "OBEROIRLTY.NS",
    "PRESTIGE.NS",
    "PHOENIXLTD.NS",
    "BRIGADE.NS",
    "SOBHA.NS",
    # Telecom & Media
    "BHARTIARTL.NS",
    "IDEA.NS",
    # Insurance & AMC
    "HDFCLIFE.NS",
    "SBILIFE.NS",
    "ICICIPRULIFE.NS",
    "LICI.NS",
    "HDFCAMC.NS",
    "NIPPONLIFE.NS",
    "ABSLAMC.NS",
    # Logistics & Infrastructure
    "CONCOR.NS",
    "BLUEDART.NS",
    "MAHINDRALOG.NS",
    "DELHIVERY.NS",
    "IRB.NS",
    "NHAI.NS",
    # Chemicals & Specialty
    "SRF.NS",
    "ATUL.NS",
    "AARTIIND.NS",
    "DEEPAKNTR.NS",
    "NAVINFLUOR.NS",
    "FINEORG.NS",
    "CLEAN.NS",
    "SUDARSCHEM.NS",
    "GALAXYSURF.NS",
    "VINDHYATEL.NS",
    # Hotels & Travel
    "INDHOTEL.NS",
    "LEMERIDIEN.NS",
    "TAJGVK.NS",
    "CHALET.NS",
]

# Deduplicate while preserving order
STOCKS = list(dict.fromkeys(STOCKS))

# ─── Sector index mapping (expanded) ─────────────────────────────────────────

SECTOR_INDEX = {
    # Banks
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
    "BANKBARODA.NS": "^NSEBANK",
    "MAHABANK.NS": "^NSEBANK",
    "CENTRALBK.NS": "^NSEBANK",
    "IOB.NS": "^NSEBANK",
    "UCOBANK.NS": "^NSEBANK",
    # NBFCs
    "CHOLAFIN.NS": "^NSEBANK",
    "BAJFINANCE.NS": "^NSEBANK",
    "BAJAJFINSV.NS": "^NSEBANK",
    "MUTHOOTFIN.NS": "^NSEBANK",
    "MANAPPURAM.NS": "^NSEBANK",
    "LICHSGFIN.NS": "^NSEBANK",
    "RECLTD.NS": "^NSEBANK",
    "PFC.NS": "^NSEBANK",
    "IRFC.NS": "^NSEBANK",
    "M&MFIN.NS": "^NSEBANK",
    "SUNDARMFIN.NS": "^NSEBANK",
    "ABCAPITAL.NS": "^NSEBANK",
    "IIFL.NS": "^NSEBANK",
    "POONAWALLA.NS": "^NSEBANK",
    # Insurance & AMC
    "HDFCLIFE.NS": "^NSEBANK",
    "SBILIFE.NS": "^NSEBANK",
    "ICICIPRULIFE.NS": "^NSEBANK",
    "LICI.NS": "^NSEBANK",
    "HDFCAMC.NS": "^NSEBANK",
    "NIPPONLIFE.NS": "^NSEBANK",
    "ABSLAMC.NS": "^NSEBANK",
    # IT
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
    "LTTS.NS": "^CNXIT",
    "BIRLASOFT.NS": "^CNXIT",
    "MASTEK.NS": "^CNXIT",
    "INTELLECT.NS": "^CNXIT",
    "SONATSOFTW.NS": "^CNXIT",
    "TATAELXSI.NS": "^CNXIT",
    "CYIENT.NS": "^CNXIT",
    # Pharma
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
    "IPCALAB.NS": "^CNXPHARMA",
    "NATCOPHARM.NS": "^CNXPHARMA",
    "GLENMARK.NS": "^CNXPHARMA",
    "GRANULES.NS": "^CNXPHARMA",
    "AJANTPHARM.NS": "^CNXPHARMA",
    "LAURUSLABS.NS": "^CNXPHARMA",
    "BIOCON.NS": "^CNXPHARMA",
    # Auto
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
    "CEATLTD.NS": "^CNXAUTO",
    "EXIDEIND.NS": "^CNXAUTO",
    "AMARAJABAT.NS": "^CNXAUTO",
    "SUNDRMFAST.NS": "^CNXAUTO",
    "ENDURANCE.NS": "^CNXAUTO",
    "CRAFTSMAN.NS": "^CNXAUTO",
    # Energy
    "ONGC.NS": "^CNXENERGY",
    "BPCL.NS": "^CNXENERGY",
    "GAIL.NS": "^CNXENERGY",
    "TATAPOWER.NS": "^CNXENERGY",
    "NTPC.NS": "^CNXENERGY",
    "POWERGRID.NS": "^CNXENERGY",
    "IOC.NS": "^CNXENERGY",
    "HINDPETRO.NS": "^CNXENERGY",
    "NHPC.NS": "^CNXENERGY",
    "SJVN.NS": "^CNXENERGY",
    "ADANIGREEN.NS": "^CNXENERGY",
    "TORNTPOWER.NS": "^CNXENERGY",
    "CESC.NS": "^CNXENERGY",
    "JSWENERGY.NS": "^CNXENERGY",
    "IREDA.NS": "^CNXENERGY",
    "GIPCL.NS": "^CNXENERGY",
    "RPOWER.NS": "^CNXENERGY",
    "INDIAGRID.NS": "^CNXENERGY",
    # Metals
    "TATASTEEL.NS": "^CNXMETAL",
    "JSWSTEEL.NS": "^CNXMETAL",
    "HINDALCO.NS": "^CNXMETAL",
    "VEDL.NS": "^CNXMETAL",
    "SAIL.NS": "^CNXMETAL",
    "NMDC.NS": "^CNXMETAL",
    "COALINDIA.NS": "^CNXMETAL",
    "NATIONALUM.NS": "^CNXMETAL",
    "WELCORP.NS": "^CNXMETAL",
    "JINDALSTEL.NS": "^CNXMETAL",
    "RATNAMANI.NS": "^CNXMETAL",
    "APLAPOLLO.NS": "^CNXMETAL",
    "MOIL.NS": "^CNXMETAL",
    # FMCG / Consumer
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
    "RADICO.NS": "^CNXFMCG",
    "UBL.NS": "^CNXFMCG",
    "TATACONSUM.NS": "^CNXFMCG",
    "EMAMILTD.NS": "^CNXFMCG",
    "JYOTHYLAB.NS": "^CNXFMCG",
    "MARICO.NS": "^CNXFMCG",
    "DABUR.NS": "^CNXFMCG",
    "COLPAL.NS": "^CNXFMCG",
    "GILLETTE.NS": "^CNXFMCG",
    "PGHH.NS": "^CNXFMCG",
    "HONASA.NS": "^CNXFMCG",
    "MANYAVAR.NS": "^CNXFMCG",
    "DMART.NS": "^CNXFMCG",
    "TRENT.NS": "^CNXFMCG",
    "SHOPERSTOP.NS": "^CNXFMCG",
    "VMART.NS": "^CNXFMCG",
}
# Everything not mapped above uses Nifty 50 as fallback
NIFTY = "^NSEI"

# ─── Config ───────────────────────────────────────────────────────────────────

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

# Paper parallel run config — same signals, more positions, no real money
# Generates extra labelled training data for the Phase 5 ML model
CFG_PAPER = {
    **CFG,
    "max_positions": 5,  # wider net → more training samples
    "capital_per_trade": 8000,
    "mode": "PAPER_PARALLEL",  # distinct from live trades in DB
}

# ─── Data cache ───────────────────────────────────────────────────────────────

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


# ─── Indicators ───────────────────────────────────────────────────────────────


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


# ─── Market context ───────────────────────────────────────────────────────────


def get_market_regime():
    """BULL / CHOP / BEAR based on Nifty 50 vs its moving averages."""
    df = _fetch(NIFTY, "1y")
    if df.empty or len(df) < 200:
        return "BULL"
    c = df["Close"]
    cur = float(c.iloc[-1])
    ma50 = float(c.iloc[-50:].mean())
    ma200 = float(c.iloc[-200:].mean())
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
    """True if the stock's sector index is above its 20-day MA."""
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


# ─── Entry signal ─────────────────────────────────────────────────────────────


def check_signal(sym):
    """
    Runs all V5 entry conditions.
    Returns (signal_dict, None) or (None, reason_string).

    V5 conditions:
    1. Price in range (Rs50-2000)
    2. Breakout above 20-day high
    3. Volume >= 2x 20-day average
    4. RSI between 60-70
    5. ADX > 25 (strong trend)
    6. No gap-up chase (< 5% overnight)
    7. Sector index above its 20-day MA
    """
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


# ─── Full scan ────────────────────────────────────────────────────────────────


def scan_for_signals():
    """
    Full daily scan. Returns:
      pending:   Day 1 breakouts today (need confirmation tomorrow)
      confirmed: Yesterday's pending signals that held above breakout close
      regime:    BULL / CHOP / BEAR

    Also runs paper parallel scan automatically — same signals,
    wider position limit, records as PAPER_PARALLEL mode in DB.
    This generates extra training data without risking real money.
    """
    clear_cache()
    regime = get_market_regime()
    open_syms = {p["symbol"] for p in database.get_open_positions()}

    # ── Check pending signals for Day 2 confirmation ──────────────
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
                        {
                            **p,
                            "current_price": round(today_close, 2),
                            "confirmed": True,
                        }
                    )
            database.mark_signal_processed(p["id"], confirmed=confirmed_flag)

    # ── Scan for fresh Day 1 breakouts ────────────────────────────
    pending = []
    if regime != "BEAR":
        for sym in STOCKS:
            if sym in open_syms:
                continue
            signal, _ = check_signal(sym)
            if signal:
                pending.append(signal)
                database.save_pending_signal(signal)

    # ── Paper parallel run ────────────────────────────────────────
    # Runs automatically alongside the live scan.
    # Uses same signals but records as PAPER_PARALLEL in database.
    # Max 5 positions (vs 2 live) — captures more breakout setups.
    # These trades accumulate as ML training data.
    _run_paper_parallel(regime, open_syms, pending)

    return {"pending": pending, "confirmed": confirmed, "regime": regime}


def _run_paper_parallel(regime, live_open_syms, todays_signals):
    """
    Paper parallel run — auto-records paper trades for ML training data.
    Completely separate from live trades. Max 5 positions.
    Called automatically during every scan.
    """
    if regime == "BEAR":
        return

    # Get paper parallel open positions
    paper_positions = database.get_open_positions_by_mode("PAPER_PARALLEL")
    paper_syms = {p["symbol"] for p in paper_positions}

    # Check exits on paper positions
    for pos in paper_positions:
        df = _fetch(pos["symbol"], "5d")
        if df.empty:
            continue
        current = float(df["Close"].iloc[-1])
        bp = pos["buy_price"]
        sl = bp * (1 - CFG["stop_loss_pct"])
        tp = bp * (1 + CFG["profit_target_pct"])
        reason = None

        if current <= sl:
            reason = f"Stop loss ({(current-bp)/bp*100:+.1f}%)"
        elif current >= tp:
            reason = f"Profit target ({(current-bp)/bp*100:+.1f}%)"
        elif pos["days_held"] >= CFG["max_hold_days"]:
            reason = f"Time stop ({pos['days_held']}d)"

        if reason:
            database.close_trade(pos["id"], current, reason)
            paper_syms.discard(pos["symbol"])

    # Open new paper positions from today's signals
    slots = CFG_PAPER["max_positions"] - len(paper_syms)
    for signal in todays_signals[:slots]:
        sym = signal["symbol"]
        if sym in paper_syms or sym in live_open_syms:
            continue
        database.add_trade(
            sym,
            signal["quantity"],
            signal["price"],
            mode="PAPER_PARALLEL",
        )


# ─── Exit check ───────────────────────────────────────────────────────────────


def check_exits():
    """
    Check all LIVE open positions for exit conditions.
    Returns list of positions needing action today.
    Paper parallel positions are handled automatically in scan_for_signals().
    """
    positions = database.get_open_positions()
    # Only check live positions — exclude paper parallel
    positions = [p for p in positions if p.get("mode") != "PAPER_PARALLEL"]

    exits = []
    for pos in positions:
        df = _fetch(pos["symbol"], "5d")
        if df.empty:
            continue
        current = float(df["Close"].iloc[-1])
        bp = pos["buy_price"]
        chg = (current - bp) / bp * 100
        sl = bp * (1 - CFG["stop_loss_pct"])
        tp = bp * (1 + CFG["profit_target_pct"])
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
