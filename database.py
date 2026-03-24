"""
database.py — Storage layer for Track 1.

Auto-detects environment:
  - Local (no DATABASE_URL env var) → SQLite at data/trading_bot.db
  - Railway (DATABASE_URL set)       → PostgreSQL

This means the same code runs identically on your laptop and in the cloud.
The only difference is which database it connects to.
"""

import os
import sqlite3
from datetime import datetime, date

# ─── Connection setup ─────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL")  # Set automatically by Railway
USE_PG = DATABASE_URL is not None

if USE_PG:
    import psycopg2
    import psycopg2.extras

SQLITE_FILE = os.path.join(os.path.dirname(__file__), "data", "trading_bot.db")


def _conn():
    """Returns a database connection — SQLite locally, PostgreSQL on Railway."""
    if USE_PG:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        os.makedirs(os.path.dirname(SQLITE_FILE), exist_ok=True)
        conn = sqlite3.connect(SQLITE_FILE)
        conn.row_factory = sqlite3.Row
        return conn


def _execute(conn, sql, params=()):
    """
    Unified execute that handles differences between SQLite and PostgreSQL.
    PostgreSQL uses %s placeholders, SQLite uses ? — we normalise here.
    """
    if USE_PG:
        sql = sql.replace("?", "%s")
        # PostgreSQL doesn't support AUTOINCREMENT — uses SERIAL
        # PostgreSQL uses ON CONFLICT DO UPDATE differently — handled in init_db
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return cur
    else:
        return conn.execute(sql, params)


def _fetchall(cursor):
    if USE_PG:
        return [dict(r) for r in cursor.fetchall()]
    else:
        return [dict(r) for r in cursor.fetchall()]


def _fetchone(cursor):
    if USE_PG:
        row = cursor.fetchone()
        return dict(row) if row else None
    else:
        row = cursor.fetchone()
        return dict(row) if row else None


# ─── Schema ───────────────────────────────────────────────────────────────────


def init_db():
    conn = _conn()

    if USE_PG:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id          SERIAL PRIMARY KEY,
                symbol      TEXT NOT NULL,
                quantity    INTEGER NOT NULL,
                buy_price   REAL NOT NULL,
                sell_price  REAL,
                buy_date    TEXT NOT NULL,
                sell_date   TEXT,
                status      TEXT DEFAULT 'OPEN',
                pnl         REAL DEFAULT 0,
                exit_reason TEXT,
                mode        TEXT DEFAULT 'PAPER'
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id    SERIAL PRIMARY KEY,
                date  TEXT NOT NULL UNIQUE,
                value REAL NOT NULL
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id              SERIAL PRIMARY KEY,
                symbol          TEXT NOT NULL,
                signal_date     TEXT NOT NULL,
                breakout_close  REAL NOT NULL,
                rsi             REAL,
                adx             REAL,
                volume_ratio    REAL,
                stop_loss       REAL,
                target          REAL,
                quantity        INTEGER,
                status          TEXT DEFAULT 'PENDING',
                sector          TEXT
            )
        """
        )
        conn.commit()
        cur.close()
    else:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                quantity    INTEGER NOT NULL,
                buy_price   REAL NOT NULL,
                sell_price  REAL,
                buy_date    TEXT NOT NULL,
                sell_date   TEXT,
                status      TEXT DEFAULT 'OPEN',
                pnl         REAL DEFAULT 0,
                exit_reason TEXT,
                mode        TEXT DEFAULT 'PAPER'
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                date  TEXT NOT NULL UNIQUE,
                value REAL NOT NULL
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                signal_date     TEXT NOT NULL,
                breakout_close  REAL NOT NULL,
                rsi             REAL,
                adx             REAL,
                volume_ratio    REAL,
                stop_loss       REAL,
                target          REAL,
                quantity        INTEGER,
                status          TEXT DEFAULT 'PENDING',
                sector          TEXT
            )
        """
        )
        conn.commit()
    conn.close()


# ─── Helper: days held ────────────────────────────────────────────────────────


def _add_days_held(rows):
    """Compute days_held in Python — works for both SQLite and PostgreSQL."""
    today = date.today()
    for r in rows:
        try:
            buy = date.fromisoformat(r["buy_date"])
            r["days_held"] = (today - buy).days
        except:
            r["days_held"] = 0
    return rows


# ─── Trades ───────────────────────────────────────────────────────────────────


def get_open_positions():
    conn = _conn()
    cur = _execute(
        conn, "SELECT * FROM trades WHERE status='OPEN' ORDER BY buy_date DESC"
    )
    rows = _fetchall(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return _add_days_held(rows)


def get_open_positions_by_mode(mode):
    """
    Returns open positions filtered by mode.
    Used to separate live trades from PAPER_PARALLEL trades.
    mode: 'LIVE', 'PAPER', or 'PAPER_PARALLEL'
    """
    conn = _conn()
    cur = _execute(
        conn,
        "SELECT * FROM trades WHERE status='OPEN' AND mode=? ORDER BY buy_date DESC",
        (mode,),
    )
    rows = _fetchall(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return _add_days_held(rows)


def get_closed_trades():
    conn = _conn()
    cur = _execute(
        conn, "SELECT * FROM trades WHERE status='CLOSED' ORDER BY sell_date DESC"
    )
    rows = _fetchall(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return rows


def get_trade_by_id(trade_id):
    conn = _conn()
    cur = _execute(conn, "SELECT * FROM trades WHERE id=?", (trade_id,))
    row = _fetchone(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return row


def add_trade(symbol, quantity, buy_price, mode="PAPER"):
    conn = _conn()
    _execute(
        conn,
        "INSERT INTO trades (symbol, quantity, buy_price, buy_date, status, mode) VALUES (?,?,?,?,'OPEN',?)",
        (symbol, quantity, buy_price, date.today().isoformat(), mode),
    )
    conn.commit()
    conn.close()


def close_trade(trade_id, sell_price, reason="manual"):
    conn = _conn()
    cur = _execute(conn, "SELECT * FROM trades WHERE id=?", (trade_id,))
    trade = _fetchone(cur)
    if USE_PG:
        cur.close()
    if not trade:
        conn.close()
        return 0
    pnl = round((sell_price - trade["buy_price"]) * trade["quantity"], 2)
    _execute(
        conn,
        "UPDATE trades SET sell_price=?, sell_date=?, status='CLOSED', pnl=?, exit_reason=? WHERE id=?",
        (sell_price, date.today().isoformat(), pnl, reason, trade_id),
    )
    conn.commit()
    conn.close()
    return pnl


def get_todays_pnl():
    conn = _conn()
    cur = _execute(
        conn,
        "SELECT COALESCE(SUM(pnl),0) as total FROM trades WHERE status='CLOSED' AND sell_date=?",
        (date.today().isoformat(),),
    )
    row = _fetchone(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return row["total"] if row else 0


def get_stats():
    conn = _conn()
    cur = _execute(
        conn,
        """
        SELECT COUNT(*) as n,
               COALESCE(SUM(pnl),0) as total_pnl,
               SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pnl<=0 THEN 1 ELSE 0 END) as losses
        FROM trades WHERE status='CLOSED'
    """,
    )
    row = _fetchone(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return row or {"n": 0, "total_pnl": 0, "wins": 0, "losses": 0}


# ─── Portfolio history ────────────────────────────────────────────────────────


def save_portfolio_value(value):
    conn = _conn()
    today = date.today().isoformat()
    if USE_PG:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO portfolio_history (date, value) VALUES (%s, %s)
            ON CONFLICT (date) DO UPDATE SET value = EXCLUDED.value
        """,
            (today, round(value, 2)),
        )
        cur.close()
    else:
        conn.execute(
            """
            INSERT INTO portfolio_history (date, value) VALUES (?,?)
            ON CONFLICT(date) DO UPDATE SET value=excluded.value
        """,
            (today, round(value, 2)),
        )
    conn.commit()
    conn.close()


def get_portfolio_history():
    conn = _conn()
    cur = _execute(conn, "SELECT date, value FROM portfolio_history ORDER BY date ASC")
    rows = _fetchall(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return rows


# ─── Signals ──────────────────────────────────────────────────────────────────


def save_pending_signal(signal):
    conn = _conn()
    today = date.today().isoformat()
    cur = _execute(
        conn,
        "SELECT id FROM signals WHERE symbol=? AND signal_date=? AND status='PENDING'",
        (signal["symbol"], today),
    )
    exists = _fetchone(cur)
    if USE_PG:
        cur.close()
    if not exists:
        _execute(
            conn,
            """
            INSERT INTO signals
            (symbol, signal_date, breakout_close, rsi, adx, volume_ratio,
             stop_loss, target, quantity, status, sector)
            VALUES (?,?,?,?,?,?,?,?,?,'PENDING',?)
        """,
            (
                signal["symbol"],
                today,
                signal["price"],
                signal.get("rsi"),
                signal.get("adx"),
                signal.get("volume_ratio"),
                signal.get("stop_loss"),
                signal.get("target"),
                signal.get("quantity"),
                signal.get("sector", ""),
            ),
        )
        conn.commit()
    conn.close()


def get_pending_signals():
    conn = _conn()
    cur = _execute(
        conn,
        """
        SELECT * FROM signals WHERE status='PENDING' AND signal_date < ?
        ORDER BY signal_date DESC
    """,
        (date.today().isoformat(),),
    )
    rows = _fetchall(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return rows


def get_todays_signals():
    conn = _conn()
    cur = _execute(
        conn,
        "SELECT * FROM signals WHERE signal_date=? ORDER BY volume_ratio DESC",
        (date.today().isoformat(),),
    )
    rows = _fetchall(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return rows


def mark_signal_processed(signal_id, confirmed=False):
    status = "CONFIRMED" if confirmed else "REJECTED"
    conn = _conn()
    _execute(conn, "UPDATE signals SET status=? WHERE id=?", (status, signal_id))
    conn.commit()
    conn.close()


def get_recent_signals(limit=30):
    conn = _conn()
    cur = _execute(
        conn,
        "SELECT * FROM signals ORDER BY signal_date DESC, id DESC LIMIT ?",
        (limit,),
    )
    rows = _fetchall(cur)
    if USE_PG:
        cur.close()
    conn.close()
    return rows
