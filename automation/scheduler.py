"""
automation/scheduler.py — Task Scheduler
==========================================
Runs automated tasks on a schedule using APScheduler.
All tasks run in IST (Asia/Kolkata, UTC+5:30).

Schedule:
  09:00 AM IST — Track 2 daily automation (exits + entries)
  09:15 AM IST — Daily scan (momentum breakout signals)
  03:30 PM IST — Portfolio snapshot (save today's value)

Each task:
  1. Logs start to automation_log
  2. Executes the task
  3. Logs success or failure with result summary

Only starts on production (gunicorn) or when explicitly enabled.
Skipped on pytest runs to avoid interference with tests.
"""

import os
import sys
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _is_test_run() -> bool:
    return "pytest" in sys.modules


def _log_task(task_name: str, task_fn, *args, **kwargs) -> dict:
    """
    Wrapper that logs start/success/failure for any task function.
    Returns the task result dict.
    """
    from automation.logger import log_start, log_success, log_failure

    log_id = log_start(task_name)
    started = time.time()
    result = {}

    try:
        result = task_fn(*args, **kwargs) or {}
        duration_ms = int((time.time() - started) * 1000)
        summary = _summarise(task_name, result)
        log_success(log_id, summary, duration_ms)
        logger.info(f"[SCHEDULER] {task_name} completed in {duration_ms}ms — {summary}")
    except Exception as e:
        duration_ms = int((time.time() - started) * 1000)
        log_failure(log_id, str(e), duration_ms)
        logger.error(f"[SCHEDULER] {task_name} FAILED — {e}")
        result = {"error": str(e)}

    return result


def _summarise(task_name: str, result: dict) -> str:
    """Generates a human-readable summary from task result."""
    if task_name == "track2_automation":
        exits = len(result.get("exits", []))
        entries = len(result.get("entries", []))
        return f"{exits} exits, {entries} entries"

    elif task_name == "daily_scan":
        pending = len(result.get("pending", []))
        confirmed = len(result.get("confirmed", []))
        regime = result.get("regime", "UNKNOWN")
        return f"regime: {regime}, {pending} pending, {confirmed} confirmed"

    elif task_name == "portfolio_snapshot":
        value = result.get("value", 0)
        return f"Rs{value:,.0f} saved"

    return str(result)[:120]


# ─── Task functions ───────────────────────────────────────────────────────────


def task_track2_automation():
    """Runs Track 2 daily automation — check exits, find entries."""
    from strategies import mean_reversion as t2

    return t2.run_daily()


def task_daily_scan():
    """Runs the momentum breakout daily scan."""
    from strategies import momentum_breakout as strategy

    return strategy.scan_for_signals()


def task_portfolio_snapshot():
    """Saves today's portfolio value to history."""
    import database
    from strategies import momentum_breakout as strategy

    positions = [
        p for p in database.get_open_positions() if p.get("mode") != "PAPER_PARALLEL"
    ]
    stats = database.get_stats()
    open_pnl = 0
    invested = 0

    for pos in positions:
        price = strategy.get_current_price(pos["symbol"])
        open_pnl += price * pos["quantity"] - pos["buy_price"] * pos["quantity"]
        invested += pos["buy_price"] * pos["quantity"]

    closed_pnl = stats["total_pnl"] or 0
    total_pnl = closed_pnl + open_pnl
    portfolio_value = 20_000 + total_pnl

    database.save_portfolio_value(round(portfolio_value, 2))
    return {"value": round(portfolio_value, 2)}


# ─── Scheduler setup ─────────────────────────────────────────────────────────

_scheduler = None


def start_scheduler(app=None):
    """
    Initialises and starts the APScheduler.
    Called once at app startup.

    Skipped during pytest runs to avoid interference with tests.
    Skipped if DISABLE_SCHEDULER=true environment variable is set.
    """
    global _scheduler

    if _is_test_run():
        logger.info("[SCHEDULER] Skipped — pytest environment detected")
        return

    if os.environ.get("DISABLE_SCHEDULER", "").lower() == "true":
        logger.info("[SCHEDULER] Skipped — DISABLE_SCHEDULER=true")
        return

    if _scheduler is not None and _scheduler.running:
        logger.info("[SCHEDULER] Already running")
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        _scheduler = BackgroundScheduler(
            timezone="Asia/Kolkata",
            job_defaults={"misfire_grace_time": 600},  # 10 min grace for late fires
        )

        # 09:00 AM IST — Track 2 automation
        _scheduler.add_job(
            func=lambda: _log_task("track2_automation", task_track2_automation),
            trigger=CronTrigger(hour=9, minute=0, timezone="Asia/Kolkata"),
            id="track2_automation",
            name="Track 2 Daily Automation",
            replace_existing=True,
        )

        # 09:15 AM IST — Daily scan
        _scheduler.add_job(
            func=lambda: _log_task("daily_scan", task_daily_scan),
            trigger=CronTrigger(hour=9, minute=15, timezone="Asia/Kolkata"),
            id="daily_scan",
            name="Daily Breakout Scan",
            replace_existing=True,
        )

        # 03:30 PM IST — Portfolio snapshot
        _scheduler.add_job(
            func=lambda: _log_task("portfolio_snapshot", task_portfolio_snapshot),
            trigger=CronTrigger(hour=15, minute=30, timezone="Asia/Kolkata"),
            id="portfolio_snapshot",
            name="Portfolio Snapshot",
            replace_existing=True,
        )

        _scheduler.start()
        logger.info("[SCHEDULER] Started — Track2@9:00, Scan@9:15, Snapshot@3:30 IST")

    except Exception as e:
        logger.error(f"[SCHEDULER] Failed to start: {e}")


def get_scheduler_status() -> list:
    """Returns next run times for all scheduled jobs."""
    if _scheduler is None or not _scheduler.running:
        return []

    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append(
            {
                "id": job.id,
                "name": job.name,
                "next_run": (
                    job.next_run_time.isoformat() if job.next_run_time else None
                ),
            }
        )
    return jobs


def trigger_task_now(task_name: str) -> dict:
    """
    Manually trigger a scheduled task immediately.
    Used by the /api/automation/trigger endpoint.
    """
    tasks = {
        "track2_automation": task_track2_automation,
        "daily_scan": task_daily_scan,
        "portfolio_snapshot": task_portfolio_snapshot,
    }

    if task_name not in tasks:
        return {"error": f"Unknown task: {task_name}"}

    return _log_task(task_name, tasks[task_name])
