"""
track2_strategy.py — Track 2: Mean Reversion + Quality Filter
==============================================================
Completely different from Track 1 (breakout momentum).

PHILOSOPHY:
  Find fundamentally strong stocks that have dipped 10-20% from
  their recent highs without any fundamental reason, then wait
  for them to recover. This is the opposite of Track 1 — it buys
  weakness, not strength.

WHY THIS COMPLEMENTS TRACK 1:
  - Track 1 fires in bull markets (breakouts need momentum)
  - Track 2 fires in choppy/sideways markets (quality stocks dip + recover)
  - Together they are active across more market conditions

ENTRY CONDITIONS (all must pass):
  1. Quality filter: ROE > 12%, Debt/Equity < 1.5, P/E < 40
  2. Dip filter: Price is 10-20% below its 52-week high
  3. Not in freefall: stock hasn't fallen more than 25% (could be broken)
  4. RSI < 45: oversold, selling pressure easing
  5. Sector not in a downtrend (same sector filter as Track 1)
  6. Market not in Bear regime (same regime filter)

EXIT CONDITIONS:
  - Profit target: +12% (larger than Track 1 — recovery moves are bigger)
  - Stop loss: -6% (slightly wider — these stocks need room to breathe)
  - Time stop: 30 days (longer hold — recovery takes time)

AUTOMATION:
  Fully automated paper trading. Runs daily, places/closes paper trades
  automatically. No human approval needed (it's paper money).

CAPITAL: Rs50,000 simulated
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import os

# ─── Config ───────────────────────────────────────────────────────────────────

CFG = {
    "starting_capital": 50_000,
    "capital_per_trade": 10_000,
    "max_positions": 5,
    "stop_loss_pct": 0.06,
    "profit_target_pct": 0.12,
    "max_hold_days": 30,
    "dip_min": 0.10,
    "dip_max": 0.25,
    "rsi_max": 45,
    "roe_min": 12,
    "debt_equity_max": 1.5,
    "pe_max": 40,
    "min_price": 50,
    "max_price": 5000,
}

# ─── DB connection (SQLite locally, PostgreSQL on Railway) ────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL")
USE_PG = DATABASE_URL is not None

if USE_PG:
    import psycopg2
    import psycopg2.extras

SQLITE_FILE = os.path.join(os.path.dirname(__file__), "data", "track2.db")


def _conn():
    if USE_PG:
        return psycopg2.connect(DATABASE_URL)
    else:
        import sqlite3

        os.makedirs(os.path.dirname(SQLITE_FILE), exist_ok=True)
        conn = __import__("sqlite3").connect(SQLITE_FILE)
        conn.row_factory = __import__("sqlite3").Row
        return conn


def _ex(conn, sql, params=()):
    """Execute SQL — normalises ? vs %s between SQLite and PostgreSQL."""
    if USE_PG:
        sql = sql.replace("?", "%s")
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return cur
    return conn.execute(sql, params)


def _all(cur):
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def _one(cur):
    row = cur.fetchone()
    return dict(row) if row else None


def _days_held(rows):
    today = date.today()
    for r in rows:
        try:
            r["days_held"] = (today - date.fromisoformat(r["buy_date"])).days
        except:
            r["days_held"] = 0
    return rows


# Quality large-cap universe — Nifty 100 stocks known for fundamentals
STOCKS = [
    # Banks (quality ones with strong ROE)
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "CHOLAFIN.NS",
    # IT (high ROE, cash-rich, low debt)
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "PERSISTENT.NS",
    "LTIM.NS",
    "TECHM.NS",
    "MPHASIS.NS",
    "COFORGE.NS",
    # Pharma
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    "LUPIN.NS",
    "TORNTPHARM.NS",
    "ABBOTINDIA.NS",
    # Auto
    "MARUTI.NS",
    "BAJAJ-AUTO.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "TVSMOTOR.NS",
    # Consumer / FMCG (high ROE, brand moats)
    "TITAN.NS",
    "PIDILITIND.NS",
    "ASIANPAINT.NS",
    "NESTLEIND.NS",
    "HAVELLS.NS",
    "BERGEPAINT.NS",
    "VBL.NS",
    "DMART.NS",
    "TRENT.NS",
    # Industrials
    "LT.NS",
    "SIEMENS.NS",
    "ABB.NS",
    "THERMAX.NS",
    "CUMMINSIND.NS",
    "SCHAEFFLER.NS",
    "AIAENG.NS",
    # Cement
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "SHREECEM.NS",
    "AMBUJACEM.NS",
    # Telecom
    "BHARTIARTL.NS",
    # Energy (selective — avoid PSU traps)
    "RELIANCE.NS",
    "TATAPOWER.NS",
    "POWERGRID.NS",
    "NTPC.NS",
]
STOCKS = list(dict.fromkeys(STOCKS))

SECTOR_INDEX = {
    "HDFCBANK.NS": "^NSEBANK",
    "ICICIBANK.NS": "^NSEBANK",
    "KOTAKBANK.NS": "^NSEBANK",
    "AXISBANK.NS": "^NSEBANK",
    "SBIN.NS": "^NSEBANK",
    "BAJFINANCE.NS": "^NSEBANK",
    "BAJAJFINSV.NS": "^NSEBANK",
    "CHOLAFIN.NS": "^NSEBANK",
    "TCS.NS": "^CNXIT",
    "INFY.NS": "^CNXIT",
    "HCLTECH.NS": "^CNXIT",
    "WIPRO.NS": "^CNXIT",
    "PERSISTENT.NS": "^CNXIT",
    "LTIM.NS": "^CNXIT",
    "TECHM.NS": "^CNXIT",
    "MPHASIS.NS": "^CNXIT",
    "COFORGE.NS": "^CNXIT",
    "SUNPHARMA.NS": "^CNXPHARMA",
    "DRREDDY.NS": "^CNXPHARMA",
    "CIPLA.NS": "^CNXPHARMA",
    "DIVISLAB.NS": "^CNXPHARMA",
    "LUPIN.NS": "^CNXPHARMA",
    "TORNTPHARM.NS": "^CNXPHARMA",
    "ABBOTINDIA.NS": "^CNXPHARMA",
    "MARUTI.NS": "^CNXAUTO",
    "BAJAJ-AUTO.NS": "^CNXAUTO",
    "EICHERMOT.NS": "^CNXAUTO",
    "HEROMOTOCO.NS": "^CNXAUTO",
    "TVSMOTOR.NS": "^CNXAUTO",
    "TITAN.NS": "^CNXFMCG",
    "PIDILITIND.NS": "^CNXFMCG",
    "ASIANPAINT.NS": "^CNXFMCG",
    "NESTLEIND.NS": "^CNXFMCG",
    "HAVELLS.NS": "^CNXFMCG",
    "BERGEPAINT.NS": "^CNXFMCG",
    "VBL.NS": "^CNXFMCG",
    "DMART.NS": "^CNXFMCG",
    "TRENT.NS": "^CNXFMCG",
    "RELIANCE.NS": "^NSEI",
    "LT.NS": "^NSEI",
    "SIEMENS.NS": "^NSEI",
    "ABB.NS": "^NSEI",
    "THERMAX.NS": "^NSEI",
    "CUMMINSIND.NS": "^NSEI",
    "SCHAEFFLER.NS": "^NSEI",
    "AIAENG.NS": "^NSEI",
    "ULTRACEMCO.NS": "^NSEI",
    "GRASIM.NS": "^NSEI",
    "SHREECEM.NS": "^NSEI",
    "AMBUJACEM.NS": "^NSEI",
    "BHARTIARTL.NS": "^NSEI",
    "TATAPOWER.NS": "^CNXENERGY",
    "POWERGRID.NS": "^CNXENERGY",
    "NTPC.NS": "^CNXENERGY",
}
NIFTY = "^NSEI"

# ─── Schema ───────────────────────────────────────────────────────────────────


def init_db():
    conn = _conn()
    if USE_PG:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS t2_trades (
                id              SERIAL PRIMARY KEY,
                symbol          TEXT NOT NULL,
                quantity        INTEGER NOT NULL,
                buy_price       REAL NOT NULL,
                sell_price      REAL,
                buy_date        TEXT NOT NULL,
                sell_date       TEXT,
                status          TEXT DEFAULT 'OPEN',
                pnl             REAL DEFAULT 0,
                exit_reason     TEXT,
                roe             REAL,
                debt_equity     REAL,
                pe_ratio        REAL,
                dip_pct         REAL,
                rsi_at_entry    REAL
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS t2_portfolio_history (
                id      SERIAL PRIMARY KEY,
                date    TEXT NOT NULL UNIQUE,
                value   REAL NOT NULL,
                cash    REAL NOT NULL,
                n_open  INTEGER NOT NULL
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS t2_scan_log (
                id          SERIAL PRIMARY KEY,
                scan_date   TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                action      TEXT NOT NULL,
                reason      TEXT,
                price       REAL,
                pnl         REAL
            )
        """
        )
        conn.commit()
        cur.close()
    else:
        import sqlite3

        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS t2_trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                quantity        INTEGER NOT NULL,
                buy_price       REAL NOT NULL,
                sell_price      REAL,
                buy_date        TEXT NOT NULL,
                sell_date       TEXT,
                status          TEXT DEFAULT 'OPEN',
                pnl             REAL DEFAULT 0,
                exit_reason     TEXT,
                roe             REAL,
                debt_equity     REAL,
                pe_ratio        REAL,
                dip_pct         REAL,
                rsi_at_entry    REAL
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS t2_portfolio_history (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                date    TEXT NOT NULL UNIQUE,
                value   REAL NOT NULL,
                cash    REAL NOT NULL,
                n_open  INTEGER NOT NULL
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS t2_scan_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date   TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                action      TEXT NOT NULL,
                reason      TEXT,
                price       REAL,
                pnl         REAL
            )
        """
        )
        conn.commit()
    conn.close()


def get_open_positions():
    conn = _conn()
    cur = _ex(
        conn, "SELECT * FROM t2_trades WHERE status='OPEN' ORDER BY buy_date DESC"
    )
    rows = _all(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return _days_held(rows)


def get_closed_trades():
    conn = _conn()
    cur = _ex(
        conn, "SELECT * FROM t2_trades WHERE status='CLOSED' ORDER BY sell_date DESC"
    )
    rows = _all(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return rows


def get_stats():
    conn = _conn()
    cur = _ex(
        conn,
        """
        SELECT COUNT(*) as n,
               COALESCE(SUM(pnl),0) as total_pnl,
               SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pnl<=0 THEN 1 ELSE 0 END) as losses,
               COALESCE(AVG(CASE WHEN pnl>0 THEN pnl END),0) as avg_win,
               COALESCE(AVG(CASE WHEN pnl<=0 THEN pnl END),0) as avg_loss
        FROM t2_trades WHERE status='CLOSED'
    """,
    )
    row = _one(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return row or {
        "n": 0,
        "total_pnl": 0,
        "wins": 0,
        "losses": 0,
        "avg_win": 0,
        "avg_loss": 0,
    }


def _add_trade(symbol, qty, price, roe, de, pe, dip, rsi):
    conn = _conn()
    _ex(
        conn,
        """
        INSERT INTO t2_trades
        (symbol, quantity, buy_price, buy_date, status, roe, debt_equity, pe_ratio, dip_pct, rsi_at_entry)
        VALUES (?,?,?,?,'OPEN',?,?,?,?,?)
    """,
        (symbol, qty, price, date.today().isoformat(), roe, de, pe, dip, rsi),
    )
    conn.commit()
    conn.close()


def _close_trade(trade_id, sell_price, reason):
    conn = _conn()
    cur = _ex(conn, "SELECT * FROM t2_trades WHERE id=?", (trade_id,))
    trade = _one(cur)
    if USE_PG:
        cur.close()
    if not trade:
        conn.close()
        return 0
    pnl = round((sell_price - trade["buy_price"]) * trade["quantity"], 2)
    _ex(
        conn,
        "UPDATE t2_trades SET sell_price=?, sell_date=?, status='CLOSED', pnl=?, exit_reason=? WHERE id=?",
        (sell_price, date.today().isoformat(), pnl, reason, trade_id),
    )
    conn.commit()
    conn.close()
    return pnl


def _log_scan(symbol, action, reason, price=None, pnl=None):
    conn = _conn()
    _ex(
        conn,
        "INSERT INTO t2_scan_log (scan_date, symbol, action, reason, price, pnl) VALUES (?,?,?,?,?,?)",
        (date.today().isoformat(), symbol, action, reason, price, pnl),
    )
    conn.commit()
    conn.close()


def save_portfolio_snapshot(value, cash, n_open):
    conn = _conn()
    today = date.today().isoformat()
    if USE_PG:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO t2_portfolio_history (date, value, cash, n_open) VALUES (%s,%s,%s,%s)
            ON CONFLICT (date) DO UPDATE SET value=EXCLUDED.value, cash=EXCLUDED.cash, n_open=EXCLUDED.n_open
        """,
            (today, round(value, 2), round(cash, 2), n_open),
        )
        cur.close()
    else:
        conn.execute(
            """
            INSERT INTO t2_portfolio_history (date, value, cash, n_open) VALUES (?,?,?,?)
            ON CONFLICT(date) DO UPDATE SET value=excluded.value, cash=excluded.cash, n_open=excluded.n_open
        """,
            (today, round(value, 2), round(cash, 2), n_open),
        )
    conn.commit()
    conn.close()


def get_portfolio_history():
    conn = _conn()
    cur = _ex(
        conn,
        "SELECT date, value, cash, n_open FROM t2_portfolio_history ORDER BY date ASC",
    )
    rows = _all(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return rows


def get_invested_capital():
    conn = _conn()
    cur = _ex(
        conn,
        "SELECT COALESCE(SUM(buy_price * quantity), 0) as invested FROM t2_trades WHERE status='OPEN'",
    )
    row = _one(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return row["invested"] if row else 0


def get_realised_pnl():
    conn = _conn()
    cur = _ex(
        conn,
        "SELECT COALESCE(SUM(pnl),0) as total FROM t2_trades WHERE status='CLOSED'",
    )
    row = _one(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return row["total"] if row else 0


# ─── Data cache ───────────────────────────────────────────────────────────────

_cache = {}


def _fetch(sym, period="6mo"):
    key = f"{sym}_{period}"
    if key not in _cache:
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            _cache[key] = df if len(df) >= 30 else pd.DataFrame()
        except:
            _cache[key] = pd.DataFrame()
    return _cache[key]


def _fundamentals(sym):
    """Fetch fundamentals via yfinance .info. Returns dict or None."""
    try:
        info = yf.Ticker(sym).info
        roe = info.get("returnOnEquity")  # decimal, e.g. 0.18 = 18%
        de = info.get("debtToEquity")  # e.g. 45 = 0.45x (yfinance reports as %)
        pe = info.get("trailingPE")
        if roe is not None:
            roe = round(roe * 100, 2)  # convert to %
        if de is not None:
            de = round(de / 100, 2)  # convert to ratio
        return {"roe": roe, "de": de, "pe": pe}
    except:
        return {"roe": None, "de": None, "pe": None}


def clear_cache():
    _cache.clear()


# ─── Indicators ───────────────────────────────────────────────────────────────


def _rsi(prices, p=14):
    d = prices.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))


def _get_market_regime():
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


def _sector_trending(sym):
    sector_sym = SECTOR_INDEX.get(sym, NIFTY)
    df = _fetch(sector_sym, "3mo")
    if df.empty or len(df) < 22:
        return True
    c = df["Close"]
    return float(c.iloc[-1]) > float(c.iloc[-20:].mean())


def _current_price(sym):
    df = _fetch(sym, "5d")
    if df.empty:
        return None
    return round(float(df["Close"].iloc[-1]), 2)


# ─── Entry check ──────────────────────────────────────────────────────────────


def _check_entry(sym):
    """
    Run all Track 2 entry conditions.
    Returns (signal_dict, None) or (None, reason_string).
    """
    df = _fetch(sym, "6mo")
    if df.empty or len(df) < 60:
        return None, "insufficient data"

    price = float(df["Close"].iloc[-1])
    if not (CFG["min_price"] <= price <= CFG["max_price"]):
        return None, "price out of range"

    # 52-week high and dip calculation
    high_52w = float(df["High"].max())
    dip_pct = (high_52w - price) / high_52w

    if dip_pct < CFG["dip_min"]:
        return None, f"not enough dip ({dip_pct*100:.1f}% < {CFG['dip_min']*100:.0f}%)"
    if dip_pct > CFG["dip_max"]:
        return None, f"dip too large — possible breakdown ({dip_pct*100:.1f}%)"

    # RSI — must be oversold
    rsi_val = float(_rsi(df["Close"]).iloc[-1])
    if np.isnan(rsi_val) or rsi_val > CFG["rsi_max"]:
        return None, f"RSI {rsi_val:.1f} not oversold"

    # Sector check — don't buy into a sector downtrend
    if not _sector_trending(sym):
        return None, "sector not trending"

    # Fundamentals
    fund = _fundamentals(sym)
    roe, de, pe = fund["roe"], fund["de"], fund["pe"]

    if roe is not None and roe < CFG["roe_min"]:
        return None, f"ROE {roe:.1f}% too low"
    if de is not None and de > CFG["debt_equity_max"]:
        return None, f"Debt/Equity {de:.2f} too high"
    if pe is not None and pe > CFG["pe_max"]:
        return None, f"P/E {pe:.1f} too expensive"

    qty = int(CFG["capital_per_trade"] / price)
    if qty < 1:
        return None, "price too high for position size"

    return {
        "symbol": sym,
        "display_name": sym.replace(".NS", ""),
        "price": round(price, 2),
        "high_52w": round(high_52w, 2),
        "dip_pct": round(dip_pct * 100, 1),
        "rsi": round(rsi_val, 1),
        "roe": roe,
        "de": de,
        "pe": pe,
        "quantity": qty,
        "invested": round(qty * price, 2),
        "stop_loss": round(price * (1 - CFG["stop_loss_pct"]), 2),
        "target": round(price * (1 + CFG["profit_target_pct"]), 2),
        "sector": SECTOR_INDEX.get(sym, NIFTY),
        "strategy": "MEAN_REVERSION",
    }, None


# ─── Exit check ───────────────────────────────────────────────────────────────


def _check_exit(pos):
    """Check one open position for exit conditions. Returns reason or None."""
    price = _current_price(pos["symbol"])
    if not price:
        return None, None
    bp = pos["buy_price"]
    sl = bp * (1 - CFG["stop_loss_pct"])
    tp = bp * (1 + CFG["profit_target_pct"])
    chg = (price - bp) / bp * 100

    if price <= sl:
        return price, f"Stop loss ({chg:+.1f}%)"
    if price >= tp:
        return price, f"Profit target ({chg:+.1f}%)"
    if pos["days_held"] >= CFG["max_hold_days"]:
        return price, f"Time stop ({pos['days_held']} days)"
    return None, None


# ─── Main daily runner ────────────────────────────────────────────────────────


def run_daily():
    """
    The core automation. Call this once per day (or on-demand).
    1. Check all open positions for exits → close them automatically
    2. Scan for new entry signals → buy automatically if capital available
    3. Save portfolio snapshot

    Returns a summary dict for the dashboard to display.
    """
    clear_cache()
    init_db()

    regime = _get_market_regime()
    log = []
    closed_today = []
    bought_today = []

    # ── Step 1: Check exits ────────────────────────────────────────
    for pos in get_open_positions():
        exit_price, reason = _check_exit(pos)
        if exit_price and reason:
            pnl = _close_trade(pos["id"], exit_price, reason)
            _log_scan(pos["symbol"], "SOLD", reason, exit_price, pnl)
            closed_today.append(
                {
                    "symbol": pos["symbol"].replace(".NS", ""),
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "reason": reason,
                }
            )
            log.append(
                f"SOLD {pos['symbol'].replace('.NS','')} @ Rs{exit_price} — {reason} (P&L: Rs{pnl:+.0f})"
            )

    # ── Step 2: Find entries (only if market not Bear) ─────────────
    signals_found = []
    if regime != "BEAR":
        open_syms = {p["symbol"] for p in get_open_positions()}
        for sym in STOCKS:
            if sym in open_syms:
                continue
            signal, reason = _check_entry(sym)
            if signal:
                signals_found.append(signal)
            else:
                _log_scan(sym, "SIGNAL_SKIPPED", reason)

        # Sort by deepest dip (most oversold quality stock first)
        signals_found.sort(key=lambda x: x["dip_pct"], reverse=True)

        # Buy top signals if capital allows
        invested = get_invested_capital()
        realised = get_realised_pnl()
        cash = CFG["starting_capital"] - invested + realised
        open_count = len(get_open_positions())

        for sig in signals_found:
            if open_count >= CFG["max_open_positions"]:
                break
            cost = sig["price"] * sig["quantity"]
            if cash < cost:
                continue

            _add_trade(
                sig["symbol"],
                sig["quantity"],
                sig["price"],
                sig["roe"],
                sig["de"],
                sig["pe"],
                sig["dip_pct"],
                sig["rsi"],
            )
            _log_scan(
                sig["symbol"],
                "BOUGHT",
                f"Dip {sig['dip_pct']}%, RSI {sig['rsi']}, ROE {sig['roe']}%",
                sig["price"],
            )
            cash -= cost
            open_count += 1
            bought_today.append(sig)
            log.append(
                f"BOUGHT {sig['display_name']} @ Rs{sig['price']} — dip {sig['dip_pct']}%, RSI {sig['rsi']}"
            )
    else:
        log.append("BEAR market — no new entries. Watching existing positions only.")

    # ── Step 3: Portfolio snapshot ─────────────────────────────────
    invested = get_invested_capital()
    realised = get_realised_pnl()
    open_pos = get_open_positions()

    # Current market value of open positions
    open_mkt_val = 0
    for pos in open_pos:
        p = _current_price(pos["symbol"])
        open_mkt_val += (p or pos["buy_price"]) * pos["quantity"]

    cash = CFG["starting_capital"] - invested + realised
    portfolio_val = cash + open_mkt_val
    save_portfolio_snapshot(portfolio_val, cash, len(open_pos))

    return {
        "regime": regime,
        "closed_today": closed_today,
        "bought_today": bought_today,
        "signals_found": len(signals_found),
        "log": log,
        "portfolio_value": round(portfolio_val, 2),
        "cash": round(cash, 2),
        "open_positions": len(open_pos),
        "run_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }


def get_summary():
    """Current portfolio summary for dashboard."""
    init_db()
    open_pos = get_open_positions()
    stats = get_stats()
    invested = get_invested_capital()
    realised = get_realised_pnl()

    open_mkt_val = 0
    open_pnl = 0
    for pos in open_pos:
        p = _current_price(pos["symbol"]) or pos["buy_price"]
        open_mkt_val += p * pos["quantity"]
        open_pnl += (p - pos["buy_price"]) * pos["quantity"]

    cash = CFG["starting_capital"] - invested + realised
    portfolio = cash + open_mkt_val
    total_pnl = realised + open_pnl

    n = stats["n"] or 0
    wins = stats["wins"] or 0

    return {
        "portfolio_value": round(portfolio, 2),
        "starting_capital": CFG["starting_capital"],
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / CFG["starting_capital"] * 100, 2),
        "realised_pnl": round(realised, 2),
        "open_pnl": round(open_pnl, 2),
        "cash": round(cash, 2),
        "invested": round(invested, 2),
        "open_positions": len(open_pos),
        "total_trades": n,
        "wins": wins,
        "losses": stats["losses"] or 0,
        "win_rate": round(wins / n * 100, 1) if n > 0 else 0,
        "avg_win": round(stats["avg_win"] or 0, 2),
        "avg_loss": round(stats["avg_loss"] or 0, 2),
        "max_positions": CFG["max_positions"],
    }
