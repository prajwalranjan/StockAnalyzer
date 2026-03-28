"""
automation/logger.py — Automation Task Logger
===============================================
Logs every scheduled task run to the automation_log table.
Tracks status (RUNNING/SUCCESS/FAILED), timing, result summary,
and any error messages.

Used by:
  automation/scheduler.py  — writes logs for each task
  app.py                   — exposes /api/automation/log endpoint
  dashboard.html           — Automation tab reads and displays logs
"""

import os
import sqlite3
from datetime import datetime, timezone

_DB_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")


def _conn():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import psycopg2

            conn = psycopg2.connect(db_url)
            return conn, True
        except Exception:
            pass
    conn = sqlite3.connect(_DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn, False


def init_log_table():
    """Create automation_log table if it doesn't exist."""
    conn, use_pg = _conn()
    cur = conn.cursor()

    if use_pg:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS automation_log (
                id              SERIAL PRIMARY KEY,
                task_name       TEXT NOT NULL,
                status          TEXT NOT NULL,
                started_at      TIMESTAMP,
                completed_at    TIMESTAMP,
                duration_ms     INTEGER,
                result_summary  TEXT,
                error_message   TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
    else:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS automation_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name       TEXT NOT NULL,
                status          TEXT NOT NULL,
                started_at      TEXT,
                completed_at    TEXT,
                duration_ms     INTEGER,
                result_summary  TEXT,
                error_message   TEXT,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    conn.commit()
    cur.close()
    conn.close()


def log_start(task_name: str) -> int:
    """
    Log that a task has started.
    Returns the log entry ID for updating later.
    """
    init_log_table()
    conn, use_pg = _conn()
    cur = conn.cursor()
    ph = "%s" if use_pg else "?"
    now = datetime.now(timezone.utc).isoformat()

    if use_pg:
        cur.execute(
            f"""
            INSERT INTO automation_log (task_name, status, started_at)
            VALUES ({ph}, {ph}, {ph}) RETURNING id
        """,
            (task_name, "RUNNING", now),
        )
        log_id = cur.fetchone()[0]
    else:
        cur.execute(
            f"""
            INSERT INTO automation_log (task_name, status, started_at)
            VALUES ({ph}, {ph}, {ph})
        """,
            (task_name, "RUNNING", now),
        )
        log_id = cur.lastrowid

    conn.commit()
    cur.close()
    conn.close()
    return log_id


def log_success(log_id: int, result_summary: str, duration_ms: int = None):
    """Mark a task as successfully completed."""
    _update_log(log_id, "SUCCESS", result_summary, None, duration_ms)


def log_failure(log_id: int, error_message: str, duration_ms: int = None):
    """Mark a task as failed with an error message."""
    _update_log(log_id, "FAILED", None, error_message, duration_ms)


def log_skip(log_id: int, reason: str, duration_ms: int = None):
    """Mark a task as skipped (market closed / weekend / holiday)."""
    _update_log(log_id, "SKIPPED", reason, None, duration_ms)


def _update_log(
    log_id: int, status: str, result_summary: str, error_message: str, duration_ms: int
):
    conn, use_pg = _conn()
    cur = conn.cursor()
    ph = "%s" if use_pg else "?"
    now = datetime.now(timezone.utc).isoformat()

    cur.execute(
        f"""
        UPDATE automation_log
        SET status={ph}, completed_at={ph}, duration_ms={ph},
            result_summary={ph}, error_message={ph}
        WHERE id={ph}
    """,
        (status, now, duration_ms, result_summary, error_message, log_id),
    )

    conn.commit()
    cur.close()
    conn.close()


def get_recent_logs(days: int = 7) -> list:
    """
    Returns automation logs for the last N days.
    Grouped by date for dashboard display.
    """
    init_log_table()
    conn, use_pg = _conn()
    cur = conn.cursor()
    ph = "%s" if use_pg else "?"

    if use_pg:
        cur.execute(
            f"""
            SELECT * FROM automation_log
            WHERE created_at >= NOW() - INTERVAL '{days} days'
            ORDER BY created_at DESC
            LIMIT 200
        """
        )
    else:
        cur.execute(
            f"""
            SELECT * FROM automation_log
            WHERE created_at >= datetime('now', '-{days} days')
            ORDER BY created_at DESC
            LIMIT 200
        """
        )

    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def get_last_run(task_name: str) -> dict:
    """Returns the most recent log entry for a specific task."""
    init_log_table()
    conn, use_pg = _conn()
    cur = conn.cursor()
    ph = "%s" if use_pg else "?"

    cur.execute(
        f"""
        SELECT * FROM automation_log
        WHERE task_name={ph}
        ORDER BY id DESC
        LIMIT 1
    """,
        (task_name,),
    )

    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def get_task_summary() -> list:
    """
    Returns latest status for each task type.
    Used for the dashboard status overview.
    """
    init_log_table()
    conn, use_pg = _conn()
    cur = conn.cursor()

    if use_pg:
        cur.execute(
            """
            SELECT DISTINCT ON (task_name)
                task_name, status, started_at,
                completed_at, duration_ms, result_summary, error_message
            FROM automation_log
            ORDER BY task_name, created_at DESC
        """
        )
    else:
        cur.execute(
            """
            SELECT task_name, status, started_at,
                   completed_at, duration_ms, result_summary, error_message
            FROM automation_log
            WHERE id IN (
                SELECT MAX(id) FROM automation_log GROUP BY task_name
            )
            ORDER BY task_name
        """
        )

    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows
