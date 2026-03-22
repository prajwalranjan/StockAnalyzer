"""
database.py — All data storage for the trading bot.
Single SQLite file: trading_bot.db (no server, no setup needed).
"""

import sqlite3
from datetime import datetime, date

DB_FILE = "trading_bot.db"


def _conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _conn()
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


# ─── Trades ───────────────────────────────────────────────────────────────────


def get_open_positions():
    conn = _conn()
    rows = conn.execute(
        """
        SELECT *, CAST(julianday('now') - julianday(buy_date) AS INTEGER) as days_held
        FROM trades WHERE status='OPEN' ORDER BY buy_date DESC
    """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_closed_trades():
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE status='CLOSED' ORDER BY sell_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_trade(symbol, quantity, buy_price, mode="PAPER"):
    conn = _conn()
    conn.execute(
        """
        INSERT INTO trades (symbol, quantity, buy_price, buy_date, status, mode)
        VALUES (?,?,?,?,'OPEN',?)
    """,
        (symbol, quantity, buy_price, date.today().isoformat(), mode),
    )
    conn.commit()
    conn.close()


def close_trade(trade_id, sell_price, reason="manual"):
    conn = _conn()
    trade = conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
    if not trade:
        conn.close()
        return 0
    pnl = round((sell_price - trade["buy_price"]) * trade["quantity"], 2)
    conn.execute(
        """
        UPDATE trades SET sell_price=?, sell_date=?, status='CLOSED', pnl=?, exit_reason=?
        WHERE id=?
    """,
        (sell_price, date.today().isoformat(), pnl, reason, trade_id),
    )
    conn.commit()
    conn.close()
    return pnl


def get_todays_pnl():
    conn = _conn()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(pnl),0) as total FROM trades
        WHERE status='CLOSED' AND sell_date=?
    """,
        (date.today().isoformat(),),
    ).fetchone()
    conn.close()
    return row["total"]


def get_stats():
    conn = _conn()
    row = conn.execute(
        """
        SELECT COUNT(*) as n,
               COALESCE(SUM(pnl),0) as total_pnl,
               SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pnl<=0 THEN 1 ELSE 0 END) as losses
        FROM trades WHERE status='CLOSED'
    """
    ).fetchone()
    conn.close()
    return dict(row)


# ─── Portfolio history ────────────────────────────────────────────────────────


def save_portfolio_value(value):
    conn = _conn()
    conn.execute(
        """
        INSERT INTO portfolio_history (date, value) VALUES (?,?)
        ON CONFLICT(date) DO UPDATE SET value=excluded.value
    """,
        (date.today().isoformat(), round(value, 2)),
    )
    conn.commit()
    conn.close()


def get_portfolio_history():
    conn = _conn()
    rows = conn.execute(
        "SELECT date, value FROM portfolio_history ORDER BY date ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Signals ──────────────────────────────────────────────────────────────────


def save_pending_signal(signal):
    conn = _conn()
    existing = conn.execute(
        """
        SELECT id FROM signals WHERE symbol=? AND signal_date=? AND status='PENDING'
    """,
        (signal["symbol"], date.today().isoformat()),
    ).fetchone()
    if not existing:
        conn.execute(
            """
            INSERT INTO signals
            (symbol, signal_date, breakout_close, rsi, adx, volume_ratio,
             stop_loss, target, quantity, status, sector)
            VALUES (?,?,?,?,?,?,?,?,?,'PENDING',?)
        """,
            (
                signal["symbol"],
                date.today().isoformat(),
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
    rows = conn.execute(
        """
        SELECT * FROM signals WHERE status='PENDING' AND signal_date < date('now')
        ORDER BY signal_date DESC
    """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_todays_signals():
    conn = _conn()
    rows = conn.execute(
        """
        SELECT * FROM signals WHERE signal_date=? ORDER BY volume_ratio DESC
    """,
        (date.today().isoformat(),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_signal_processed(signal_id, confirmed=False):
    status = "CONFIRMED" if confirmed else "REJECTED"
    conn = _conn()
    conn.execute("UPDATE signals SET status=? WHERE id=?", (status, signal_id))
    conn.commit()
    conn.close()


def get_recent_signals(limit=30):
    conn = _conn()
    rows = conn.execute(
        """
        SELECT * FROM signals ORDER BY signal_date DESC, id DESC LIMIT ?
    """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
