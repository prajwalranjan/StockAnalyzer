"""
automation/scheduler.py — Task Scheduler
==========================================
Runs automated tasks on a schedule using APScheduler.
All tasks run in IST (Asia/Kolkata, UTC+5:30).

Schedule:
  09:00 AM IST — Track 2 daily automation (exits + entries)
  09:15 AM IST — Daily scan (momentum breakout signals)
  03:30 PM IST — Portfolio snapshot (save today's value)

Market hours check:
  NSE trades Monday-Friday, 9:15 AM - 3:30 PM IST.
  Weekends, public holidays, after hours -> tasks skip silently.
  Skipped runs are logged with status SKIPPED.
"""

import os
import sys
import time
import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)


# NSE holiday list - update annually
NSE_HOLIDAYS_2025_2026 = {
    date(2025, 1, 26),
    date(2025, 2, 26),
    date(2025, 3, 14),
    date(2025, 3, 31),
    date(2025, 4, 14),
    date(2025, 4, 18),
    date(2025, 5, 1),
    date(2025, 8, 15),
    date(2025, 8, 27),
    date(2025, 10, 2),
    date(2025, 10, 21),
    date(2025, 10, 22),
    date(2025, 11, 5),
    date(2025, 12, 25),
    date(2026, 1, 26),
    date(2026, 2, 26),
    date(2026, 3, 20),
    date(2026, 4, 3),
    date(2026, 4, 14),
    date(2026, 5, 1),
    date(2026, 8, 15),
    date(2026, 10, 2),
    date(2026, 12, 25),
}


def _ist_now():
    """Returns current datetime in IST."""
    try:
        import pytz

        return datetime.now(pytz.timezone("Asia/Kolkata"))
    except ImportError:
        from datetime import timezone, timedelta

        return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def is_market_open() -> tuple:
    """
    Returns (is_open: bool, reason: str).
    NSE open: Mon-Fri 9:15 AM - 3:30 PM IST, excluding holidays.
    """
    now = _ist_now()
    today = now.date()

    if now.weekday() == 5:
        return False, "Saturday - market closed"
    if now.weekday() == 6:
        return False, "Sunday - market closed"
    if today in NSE_HOLIDAYS_2025_2026:
        return False, "NSE holiday"

    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now < open_t:
        return False, "Pre-market - opens at 9:15 AM IST"
    if now > close_t:
        return False, "Market closed at 3:30 PM IST"

    return True, "Market open"


def is_trading_day() -> tuple:
    """Returns (is_trading: bool, reason: str) for today."""
    now = _ist_now()
    today = now.date()

    if now.weekday() >= 5:
        return False, "Weekend"
    if today in NSE_HOLIDAYS_2025_2026:
        return False, "NSE holiday"

    return True, "Trading day"


def _is_test_run() -> bool:
    return "pytest" in sys.modules


def _log_task(task_name, task_fn, *args, **kwargs):
    from automation.logger import log_start, log_success, log_failure, log_skip

    log_id = log_start(task_name)
    started = time.time()
    result = {}

    try:
        result = task_fn(*args, **kwargs) or {}
        duration_ms = int((time.time() - started) * 1000)
        summary = _summarise(task_name, result)

        if result.get("skipped"):
            log_skip(log_id, result.get("reason", "skipped"), duration_ms)
            logger.info(f"[SCHEDULER] {task_name} skipped - {result.get('reason')}")
        else:
            log_success(log_id, summary, duration_ms)
            logger.info(f"[SCHEDULER] {task_name} done in {duration_ms}ms - {summary}")
    except Exception as e:
        duration_ms = int((time.time() - started) * 1000)
        log_failure(log_id, str(e), duration_ms)
        logger.error(f"[SCHEDULER] {task_name} FAILED - {e}")
        result = {"error": str(e)}

    return result


def _summarise(task_name, result):
    if result.get("skipped"):
        return f"skipped - {result.get('reason', '')}"
    if task_name == "track2_automation":
        return f"{len(result.get('exits',[]))} exits, {len(result.get('entries',[]))} entries"
    elif task_name == "daily_scan":
        return f"regime:{result.get('regime','?')} {len(result.get('pending',[]))} pending {len(result.get('confirmed',[]))} confirmed"
    elif task_name == "portfolio_snapshot":
        return f"Rs{result.get('value',0):,.0f} saved"
    return str(result)[:120]


# Task functions


def task_track2_automation():
    trading, reason = is_trading_day()
    if not trading:
        return {"skipped": True, "reason": reason}
    from strategies import mean_reversion as t2

    return t2.run_daily()


def task_daily_scan():
    open_, reason = is_market_open()
    if not open_:
        return {"skipped": True, "reason": reason}
    from strategies import momentum_breakout as strategy

    return strategy.scan_for_signals()


def task_portfolio_snapshot():
    trading, reason = is_trading_day()
    if not trading:
        return {"skipped": True, "reason": reason}

    import database
    from strategies import momentum_breakout as strategy

    positions = [
        p for p in database.get_open_positions() if p.get("mode") != "PAPER_PARALLEL"
    ]
    stats = database.get_stats()
    open_pnl = 0
    for pos in positions:
        price = strategy.get_current_price(pos["symbol"])
        open_pnl += price * pos["quantity"] - pos["buy_price"] * pos["quantity"]

    value = round(20_000 + (stats["total_pnl"] or 0) + open_pnl, 2)
    database.save_portfolio_value(value)
    return {"value": value}


# Scheduler setup

_scheduler = None


def start_scheduler(app=None):
    global _scheduler

    if _is_test_run():
        logger.info("[SCHEDULER] Skipped - pytest")
        return
    if os.environ.get("DISABLE_SCHEDULER", "").lower() == "true":
        logger.info("[SCHEDULER] Disabled via env")
        return
    if _scheduler is not None and _scheduler.running:
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        _scheduler = BackgroundScheduler(
            timezone="Asia/Kolkata",
            job_defaults={"misfire_grace_time": 600},
        )

        _scheduler.add_job(
            func=lambda: _log_task("track2_automation", task_track2_automation),
            trigger=CronTrigger(hour=9, minute=0, timezone="Asia/Kolkata"),
            id="track2_automation",
            name="Track 2 Daily Automation",
            replace_existing=True,
        )
        _scheduler.add_job(
            func=lambda: _log_task("daily_scan", task_daily_scan),
            trigger=CronTrigger(hour=9, minute=15, timezone="Asia/Kolkata"),
            id="daily_scan",
            name="Daily Breakout Scan",
            replace_existing=True,
        )
        _scheduler.add_job(
            func=lambda: _log_task("portfolio_snapshot", task_portfolio_snapshot),
            trigger=CronTrigger(hour=15, minute=30, timezone="Asia/Kolkata"),
            id="portfolio_snapshot",
            name="Portfolio Snapshot",
            replace_existing=True,
        )

        _scheduler.start()
        logger.info("[SCHEDULER] Started - Track2@9:00, Scan@9:15, Snapshot@3:30 IST")
        logger.info(
            "[SCHEDULER] Market hours check active - skips weekends + NSE holidays"
        )

    except Exception as e:
        logger.error(f"[SCHEDULER] Failed to start: {e}")


def get_scheduler_status() -> list:
    if _scheduler is None or not _scheduler.running:
        return []
    return [
        {
            "id": j.id,
            "name": j.name,
            "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
        }
        for j in _scheduler.get_jobs()
    ]


def trigger_task_now(task_name: str) -> dict:
    """Manual trigger - bypasses market hours check."""
    if task_name not in ["track2_automation", "daily_scan", "portfolio_snapshot"]:
        return {"error": f"Unknown task: {task_name}"}

    def bypass():
        if task_name == "daily_scan":
            from strategies import momentum_breakout as s

            return s.scan_for_signals()
        elif task_name == "track2_automation":
            from strategies import mean_reversion as t2

            return t2.run_daily()
        elif task_name == "portfolio_snapshot":
            import database
            from strategies import momentum_breakout as s

            positions = [
                p
                for p in database.get_open_positions()
                if p.get("mode") != "PAPER_PARALLEL"
            ]
            stats = database.get_stats()
            open_pnl = sum(
                s.get_current_price(p["symbol"]) * p["quantity"]
                - p["buy_price"] * p["quantity"]
                for p in positions
            )
            value = round(20_000 + (stats["total_pnl"] or 0) + open_pnl, 2)
            database.save_portfolio_value(value)
            return {"value": value}

    return _log_task(task_name, bypass)
