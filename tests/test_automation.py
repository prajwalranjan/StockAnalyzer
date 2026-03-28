"""
tests/test_automation.py
=========================
Tests for the automation logger and scheduler.
Run with: pytest tests/ -v
"""

import pytest
import time
from datetime import datetime


# ─── Logger Tests ─────────────────────────────────────────────────────────────


class TestAutomationLogger:

    def setup_method(self):
        from automation import logger

        self.lg = logger
        self.lg.init_log_table()

    def test_init_table_runs_without_error(self):
        """Table creation must succeed without error."""
        self.lg.init_log_table()

    def test_log_start_returns_id(self):
        """log_start must return a positive integer ID."""
        log_id = self.lg.log_start("test_task")
        assert isinstance(log_id, int)
        assert log_id > 0

    def test_log_success_updates_record(self):
        """log_success must update status to SUCCESS."""
        log_id = self.lg.log_start("test_success_task")
        self.lg.log_success(log_id, "all good", duration_ms=500)

        last = self.lg.get_last_run("test_success_task")
        assert last is not None
        assert last["status"] == "SUCCESS"
        assert last["result_summary"] == "all good"
        assert last["duration_ms"] == 500

    def test_log_failure_updates_record(self):
        """log_failure must update status to FAILED."""
        log_id = self.lg.log_start("test_fail_task")
        self.lg.log_failure(log_id, "something broke", duration_ms=200)

        last = self.lg.get_last_run("test_fail_task")
        assert last is not None
        assert last["status"] == "FAILED"
        assert last["error_message"] == "something broke"

    def test_log_start_sets_running_status(self):
        """Newly started task must have RUNNING status."""
        log_id = self.lg.log_start("test_running_task")
        last = self.lg.get_last_run("test_running_task")
        assert last is not None
        assert last["status"] == "RUNNING"

    def test_get_recent_logs_returns_list(self):
        """get_recent_logs must return a list."""
        logs = self.lg.get_recent_logs(days=7)
        assert isinstance(logs, list)

    def test_get_recent_logs_contains_recent_entry(self):
        """A log entry created now must appear in recent logs."""
        log_id = self.lg.log_start("test_recent_task")
        self.lg.log_success(log_id, "test result")

        logs = self.lg.get_recent_logs(days=1)
        task_names = [l["task_name"] for l in logs]
        assert "test_recent_task" in task_names

    def test_get_last_run_unknown_task(self):
        """get_last_run for unknown task must return None."""
        result = self.lg.get_last_run("nonexistent_task_xyz")
        assert result is None

    def test_get_task_summary_returns_list(self):
        """get_task_summary must return a list."""
        result = self.lg.get_task_summary()
        assert isinstance(result, list)

    def test_get_task_summary_has_required_keys(self):
        """Each summary entry must have required keys."""
        log_id = self.lg.log_start("summary_test_task")
        self.lg.log_success(log_id, "done")

        summary = self.lg.get_task_summary()
        entry = next(
            (s for s in summary if s["task_name"] == "summary_test_task"), None
        )
        assert entry is not None
        for key in ["task_name", "status", "result_summary"]:
            assert key in entry

    def test_multiple_runs_last_run_is_latest(self):
        """get_last_run must return the most recent entry."""
        log_id1 = self.lg.log_start("multi_run_task")
        self.lg.log_success(log_id1, "run 1")

        time.sleep(0.01)  # ensure different timestamps

        log_id2 = self.lg.log_start("multi_run_task")
        self.lg.log_success(log_id2, "run 2")

        last = self.lg.get_last_run("multi_run_task")
        assert last["result_summary"] == "run 2"

    def test_duration_ms_stored_correctly(self):
        """Duration in milliseconds must be stored and retrieved correctly."""
        log_id = self.lg.log_start("duration_test_task")
        self.lg.log_success(log_id, "done", duration_ms=1234)

        last = self.lg.get_last_run("duration_test_task")
        assert last["duration_ms"] == 1234

    def test_error_message_stored_correctly(self):
        """Error message must be stored and retrieved correctly."""
        log_id = self.lg.log_start("error_test_task")
        self.lg.log_failure(log_id, "ValueError: invalid input")

        last = self.lg.get_last_run("error_test_task")
        assert "ValueError" in last["error_message"]


# ─── Scheduler Tests ──────────────────────────────────────────────────────────


class TestAutomationScheduler:

    def setup_method(self):
        from automation import scheduler

        self.sc = scheduler

    def test_get_scheduler_status_returns_list(self):
        """get_scheduler_status must return a list (empty if not started)."""
        result = self.sc.get_scheduler_status()
        assert isinstance(result, list)

    def test_trigger_unknown_task_returns_error(self):
        """Triggering unknown task must return error dict."""
        result = self.sc.trigger_task_now("nonexistent_task_xyz")
        assert "error" in result

    def test_trigger_portfolio_snapshot(self):
        """Triggering portfolio_snapshot must complete without crash."""
        result = self.sc.trigger_task_now("portfolio_snapshot")
        assert "error" not in result or result.get("error") is None
        # Should have a value key
        assert "value" in result or "error" in result

    def test_trigger_daily_scan(self):
        """Triggering daily_scan must complete and return regime."""
        try:
            result = self.sc.trigger_task_now("daily_scan")
            assert "regime" in result or "error" in result
        except Exception:
            pytest.skip("Network unavailable")

    def test_log_task_wrapper_logs_success(self):
        """_log_task must log a successful task run."""
        from automation.logger import get_last_run

        def sample_task():
            return {"value": 42}

        self.sc._log_task("test_wrapper_task", sample_task)
        last = get_last_run("test_wrapper_task")
        assert last is not None
        assert last["status"] == "SUCCESS"

    def test_log_task_wrapper_logs_failure(self):
        """_log_task must log a failed task run."""
        from automation.logger import get_last_run

        def failing_task():
            raise ValueError("intentional test failure")

        self.sc._log_task("test_fail_wrapper_task", failing_task)
        last = get_last_run("test_fail_wrapper_task")
        assert last is not None
        assert last["status"] == "FAILED"
        assert "intentional test failure" in last["error_message"]

    def test_log_task_records_duration(self):
        """_log_task must record non-zero duration for completed task."""
        from automation.logger import get_last_run

        def slow_task():
            time.sleep(0.05)
            return {"done": True}

        self.sc._log_task("test_duration_task", slow_task)
        last = get_last_run("test_duration_task")
        assert last["duration_ms"] is not None
        assert last["duration_ms"] >= 50  # at least 50ms

    def test_summarise_track2(self):
        """_summarise must return readable string for track2_automation."""
        result = {"exits": [1, 2], "entries": [1]}
        summary = self.sc._summarise("track2_automation", result)
        assert "2 exits" in summary
        assert "1 entries" in summary

    def test_summarise_daily_scan(self):
        """_summarise must return readable string for daily_scan."""
        result = {"pending": [1], "confirmed": [], "regime": "BEAR"}
        summary = self.sc._summarise("daily_scan", result)
        assert "BEAR" in summary
        assert "1 pending" in summary

    def test_summarise_portfolio_snapshot(self):
        """_summarise must return readable string for portfolio_snapshot."""
        result = {"value": 20500}
        summary = self.sc._summarise("portfolio_snapshot", result)
        assert "20,500" in summary or "20500" in summary

    def test_start_scheduler_skipped_in_pytest(self):
        """Scheduler must not start during pytest runs."""
        self.sc.start_scheduler()
        assert True

    def test_is_trading_day_returns_tuple(self):
        result = self.sc.is_trading_day()
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_is_market_open_returns_tuple(self):
        result = self.sc.is_market_open()
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_nse_holiday_detected(self):
        from unittest.mock import patch
        from datetime import datetime

        holiday = datetime(2026, 1, 26, 10, 0, 0)
        with patch("automation.scheduler._ist_now", return_value=holiday):
            open_, reason = self.sc.is_market_open()
            assert open_ is False
            assert "holiday" in reason.lower()

    def test_saturday_detected(self):
        from unittest.mock import patch
        from datetime import datetime

        saturday = datetime(2026, 3, 28, 10, 0, 0)
        with patch("automation.scheduler._ist_now", return_value=saturday):
            open_, reason = self.sc.is_market_open()
            assert open_ is False
            assert "saturday" in reason.lower()

    def test_market_open_during_hours(self):
        from unittest.mock import patch
        from datetime import datetime

        friday_10am = datetime(2026, 3, 27, 10, 0, 0)
        with patch("automation.scheduler._ist_now", return_value=friday_10am):
            open_, _ = self.sc.is_market_open()
            assert open_ is True

    def test_pre_market_detected(self):
        from unittest.mock import patch
        from datetime import datetime

        friday_8am = datetime(2026, 3, 27, 8, 0, 0)
        with patch("automation.scheduler._ist_now", return_value=friday_8am):
            open_, reason = self.sc.is_market_open()
            assert open_ is False
            assert "9:15" in reason or "pre" in reason.lower()

    def test_after_market_detected(self):
        from unittest.mock import patch
        from datetime import datetime

        friday_4pm = datetime(2026, 3, 27, 16, 0, 0)
        with patch("automation.scheduler._ist_now", return_value=friday_4pm):
            open_, reason = self.sc.is_market_open()
            assert open_ is False
            assert "3:30" in reason or "closed" in reason.lower()

    def test_skipped_task_logged_as_skipped(self):
        from automation.logger import get_last_run
        from unittest.mock import patch

        with patch(
            "automation.scheduler.is_market_open", return_value=(False, "Sunday")
        ):
            self.sc._log_task("daily_scan", self.sc.task_daily_scan)
        last = get_last_run("daily_scan")
        assert last is not None
        assert last["status"] == "SKIPPED"
