"""
prediction/sentiment_validator.py — Phase 2A Validator
========================================================
Tracks sentiment verdicts against actual trade outcomes to
validate whether the classifier is actually predictive.

Without this, shadow mode is just logging — we'd have no
systematic way to know if sentiment is adding value.

Validation logic:
  Every time a trade closes, log:
    - What was the sentiment verdict at entry? (GREEN/AMBER/RED)
    - What was the outcome? (WIN/LOSS/TIME)

  Monthly correlation check:
    - RED verdicts → what % became losses?
    - GREEN verdicts → what % became wins?

Promotion criteria (from design doc):
  RED → LOSS correlation > 65% → enable live filtering
  RED → LOSS correlation < 55% → tune or discard

Run via:
  from prediction.sentiment_validator import monthly_report
  monthly_report()
"""

import os
import sqlite3
from datetime import date, timedelta

# Store validation data in the same DB location as trades
DB_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")


def _conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_validator_table():
    """Create the sentiment validation log table if it doesn't exist."""
    conn = _conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_validation_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            trade_date      TEXT NOT NULL,
            close_date      TEXT,
            sentiment_verdict TEXT,          -- GREEN / AMBER / RED
            sentiment_score   REAL,
            headline_score    REAL,
            announcement_verdict TEXT,
            earnings_days     INTEGER,
            analyst_action    INTEGER,
            trade_outcome     TEXT,          -- WIN / LOSS / TIME (filled on close)
            pnl               REAL,
            logged_at         TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def log_entry(symbol: str, sentiment_result: dict, trade_date: str = None):
    """
    Log a sentiment verdict at the time of trade entry.
    Called from momentum_breakout.py when a signal is detected.
    """
    init_validator_table()
    conn = _conn()

    signals = sentiment_result.get("signals", {})
    headline = signals.get("headlines", {})
    earnings = signals.get("earnings", {})
    analyst = signals.get("analyst", {})
    announce = signals.get("announcements", {})

    conn.execute(
        """
        INSERT INTO sentiment_validation_log
        (symbol, trade_date, sentiment_verdict, sentiment_score,
         headline_score, announcement_verdict, earnings_days, analyst_action)
        VALUES (?,?,?,?,?,?,?,?)
    """,
        (
            symbol,
            trade_date or date.today().isoformat(),
            sentiment_result.get("verdict", "GREEN"),
            sentiment_result.get("score", 0),
            headline.get("avg_score"),
            announce.get("verdict", "GREEN"),
            earnings.get("days_to_earnings"),
            analyst.get("action", 0),
        ),
    )
    conn.commit()
    conn.close()


def log_outcome(symbol: str, trade_date: str, outcome: str, pnl: float):
    """
    Update the validation log with the trade outcome when it closes.
    outcome: "WIN" / "LOSS" / "TIME"
    Called from app.py when a trade is closed.
    """
    conn = _conn()
    conn.execute(
        """
        UPDATE sentiment_validation_log
        SET trade_outcome = ?, pnl = ?, close_date = ?
        WHERE symbol = ? AND trade_date = ? AND trade_outcome IS NULL
    """,
        (outcome, pnl, date.today().isoformat(), symbol, trade_date),
    )
    conn.commit()
    conn.close()


def monthly_report() -> dict:
    """
    Computes sentiment accuracy statistics for the last 90 days.
    Run this monthly to validate the classifier.

    Returns a report dict with:
      - per-verdict win rates
      - overall accuracy
      - recommendation: KEEP / TUNE / UPGRADE / DISCARD
    """
    init_validator_table()
    conn = _conn()

    cutoff = (date.today() - timedelta(days=90)).isoformat()
    rows = conn.execute(
        """
        SELECT sentiment_verdict, trade_outcome, pnl
        FROM sentiment_validation_log
        WHERE trade_outcome IS NOT NULL AND trade_date >= ?
    """,
        (cutoff,),
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "Need at least 1 closed trade with sentiment logged.",
            "samples": 0,
        }

    rows = [dict(r) for r in rows]
    total = len(rows)

    # Break down by verdict
    stats = {"GREEN": [], "AMBER": [], "RED": []}
    for r in rows:
        verdict = r["sentiment_verdict"] or "GREEN"
        outcome = r["trade_outcome"]
        if verdict in stats:
            stats[verdict].append(outcome)

    def win_rate(outcomes):
        if not outcomes:
            return None
        wins = sum(1 for o in outcomes if o == "WIN")
        return round(wins / len(outcomes) * 100, 1)

    green_wr = win_rate(stats["GREEN"])
    amber_wr = win_rate(stats["AMBER"])
    red_wr = win_rate(stats["RED"])

    # RED → LOSS correlation (key metric)
    red_outcomes = stats["RED"]
    red_loss_rate = None
    if red_outcomes:
        losses = sum(1 for o in red_outcomes if o == "LOSS")
        red_loss_rate = round(losses / len(red_outcomes) * 100, 1)

    # Baseline win rate (all trades regardless of sentiment)
    all_outcomes = [r["trade_outcome"] for r in rows]
    baseline_wr = win_rate(all_outcomes)

    # Recommendation
    recommendation = "INSUFFICIENT_DATA"
    recommendation_detail = ""

    if total >= 10:
        if red_loss_rate is not None:
            if red_loss_rate >= 65:
                recommendation = "ENABLE_LIVE_FILTERING"
                recommendation_detail = (
                    f"RED verdicts → {red_loss_rate}% losses. "
                    f"Classifier validated. Enable live filtering."
                )
            elif red_loss_rate >= 55:
                recommendation = "KEEP_SHADOW"
                recommendation_detail = (
                    f"RED verdicts → {red_loss_rate}% losses. "
                    f"Borderline — continue shadow mode, need more data."
                )
            else:
                recommendation = "UPGRADE_TO_COHERE"
                recommendation_detail = (
                    f"RED verdicts → only {red_loss_rate}% losses. "
                    f"Classifier not reliably predicting bad trades. "
                    f"Consider upgrading sentiment model."
                )
        else:
            recommendation = "KEEP_SHADOW"
            recommendation_detail = "No RED verdicts yet. Continue shadow mode."

    report = {
        "period": "last 90 days",
        "total_samples": total,
        "baseline_win_rate": baseline_wr,
        "by_verdict": {
            "GREEN": {
                "count": len(stats["GREEN"]),
                "win_rate": green_wr,
            },
            "AMBER": {
                "count": len(stats["AMBER"]),
                "win_rate": amber_wr,
            },
            "RED": {
                "count": len(stats["RED"]),
                "win_rate": red_wr,
                "loss_rate": red_loss_rate,
            },
        },
        "recommendation": recommendation,
        "recommendation_detail": recommendation_detail,
        "promotion_threshold": "RED loss rate > 65% = enable live filtering",
    }

    return report


def get_recent_logs(limit: int = 20) -> list:
    """Returns recent sentiment validation entries for dashboard display."""
    init_validator_table()
    conn = _conn()
    rows = conn.execute(
        """
        SELECT * FROM sentiment_validation_log
        ORDER BY logged_at DESC LIMIT ?
    """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
