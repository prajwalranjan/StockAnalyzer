"""
ml/collector.py — ML Data Collector
=====================================
Logs feature vectors at signal time and outcomes at trade close.
Every signal that fires gets a full snapshot of all 45 features.
Every trade that closes gets its outcome attached to that snapshot.

This runs from day one — before the ML model is trained.
By the time we have 200 trades, we have 200 labelled training samples.

Called from:
  momentum_breakout.py → log_signal_features() on every confirmed signal
  app.py               → log_trade_outcome() when a trade closes

Shadow mode: collecting data, not yet making predictions.
Active mode (Phase 5): predictor.py uses trained model on new signals.
"""

import os
import json
import sqlite3
from datetime import date
from pathlib import Path

# Use same DB as rest of app
_DB_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "trading_bot.db")


def _conn():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        # PostgreSQL on Railway
        try:
            import psycopg2

            conn = psycopg2.connect(db_url)
            return conn, True
        except Exception:
            pass
    conn = sqlite3.connect(_DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn, False


def init_ml_table():
    """Create ml_signal_log table if it doesn't exist."""
    conn, use_pg = _conn()
    cur = conn.cursor()

    if use_pg:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_signal_log (
                id                  SERIAL PRIMARY KEY,
                signal_id           INTEGER,
                symbol              TEXT,
                signal_date         TEXT,
                close_date          TEXT,
                mode                TEXT DEFAULT 'LIVE',

                -- Technical
                rsi                 REAL,
                adx                 REAL,
                volume_ratio        REAL,
                breakout_pct        REAL,
                gap_pct             REAL,

                -- Volume Intelligence
                vol_buildup_score   INTEGER,
                vol_buildup_days    INTEGER,
                consolidation_score INTEGER,
                consolidation_ratio REAL,
                rel_strength_score  INTEGER,
                rel_strength_delta  REAL,

                -- Company Sentiment
                sentiment_verdict   TEXT,
                sentiment_avg_score REAL,
                analyst_action      INTEGER,
                days_to_earnings    INTEGER,

                -- Macro
                india_vix           REAL,
                vix_5d_change       REAL,
                geopolitical_risk   REAL,
                rbi_days            INTEGER,
                dxy_10d_change      REAL,
                oil_10d_change      REAL,
                overnight_composite REAL,
                macro_score         INTEGER,

                -- Smart Money
                fii_net_5d          REAL,
                fii_trend           TEXT,
                delivery_pct        REAL,
                promoter_buying     INTEGER,
                insider_veto        INTEGER,

                -- Options
                pcr                 REAL,
                unusual_call_oi     INTEGER,
                iv_risk             INTEGER,

                -- Market Context
                regime              TEXT,
                sector_trending     INTEGER,
                nifty_20d_return    REAL,
                quality_score       INTEGER,
                quality_grade       TEXT,
                position_multiplier REAL,

                -- Raw score JSON (for debugging/auditing)
                score_json          TEXT,

                -- Outcome (filled on close)
                outcome             TEXT,
                pnl                 REAL,
                pnl_pct             REAL,
                days_held           INTEGER,
                exit_reason         TEXT,

                logged_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
    else:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_signal_log (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id           INTEGER,
                symbol              TEXT,
                signal_date         TEXT,
                close_date          TEXT,
                mode                TEXT DEFAULT 'LIVE',

                rsi                 REAL,
                adx                 REAL,
                volume_ratio        REAL,
                breakout_pct        REAL,
                gap_pct             REAL,

                vol_buildup_score   INTEGER,
                vol_buildup_days    INTEGER,
                consolidation_score INTEGER,
                consolidation_ratio REAL,
                rel_strength_score  INTEGER,
                rel_strength_delta  REAL,

                sentiment_verdict   TEXT,
                sentiment_avg_score REAL,
                analyst_action      INTEGER,
                days_to_earnings    INTEGER,

                india_vix           REAL,
                vix_5d_change       REAL,
                geopolitical_risk   REAL,
                rbi_days            INTEGER,
                dxy_10d_change      REAL,
                oil_10d_change      REAL,
                overnight_composite REAL,
                macro_score         INTEGER,

                fii_net_5d          REAL,
                fii_trend           TEXT,
                delivery_pct        REAL,
                promoter_buying     INTEGER,
                insider_veto        INTEGER,

                pcr                 REAL,
                unusual_call_oi     INTEGER,
                iv_risk             INTEGER,

                regime              TEXT,
                sector_trending     INTEGER,
                nifty_20d_return    REAL,
                quality_score       INTEGER,
                quality_grade       TEXT,
                position_multiplier REAL,

                score_json          TEXT,

                outcome             TEXT,
                pnl                 REAL,
                pnl_pct             REAL,
                days_held           INTEGER,
                exit_reason         TEXT,

                logged_at           TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    conn.commit()
    cur.close()
    conn.close()


def log_signal_features(
    symbol: str,
    signal: dict,
    score_result: dict,
    signal_id: int = None,
    mode: str = "LIVE",
):
    """
    Log a full feature vector when a signal fires.
    Called from momentum_breakout.py on every confirmed signal.

    signal:       the signal dict from check_signal()
    score_result: the full result from score.compute_score()
    signal_id:    the DB id of the signal record (optional)
    mode:         LIVE or PAPER_PARALLEL
    """
    init_ml_table()

    modules = score_result.get("modules", {})

    # Extract sub-module data safely
    vi = modules.get("volume_intelligence", {})
    cs = modules.get("company_sentiment", {})
    ms = modules.get("macro_sentiment", {})
    sm = modules.get("smart_money", {})
    opts = modules.get("options_signal", {})

    vi_sigs = vi.get("signals", vi)  # handle both flat and nested
    cs_sigs = cs.get("signals", {})
    ms_sigs = ms.get("signals", {})
    sm_sigs = sm.get("signals", {})
    opt_sigs = opts.get("signals", {})

    # Volume intelligence sub-signals
    vb = vi_sigs.get("volume_buildup", {})
    co = vi_sigs.get("consolidation", {})
    rs = vi_sigs.get("relative_strength", {})

    # Company sentiment sub-signals
    hl = cs_sigs.get("headlines", {})
    ea = cs_sigs.get("earnings", {})
    an = cs_sigs.get("analyst", {})

    # Macro sub-signals
    vix_sig = ms_sigs.get("india_vix", {})
    geo_sig = ms_sigs.get("geopolitical", {})
    cb_sig = ms_sigs.get("central_bank", {})
    do_sig = ms_sigs.get("dollar_oil", {})
    og_sig = ms_sigs.get("overnight", {})

    # Smart money sub-signals
    fii_sig = sm_sigs.get("fii_dii", {}).get("raw", {})
    del_sig = sm_sigs.get("delivery", {}).get("raw", {})
    ins_sig = sm_sigs.get("insider", {}).get("raw", {})

    # Options sub-signals
    pcr_sig = opt_sigs.get("pcr", {})
    uoi_sig = opt_sigs.get("unusual_oi", {})

    # Compute breakout_pct and gap_pct from signal
    price = signal.get("price", 0)
    prev_high = signal.get("prev_high", price)
    breakout_pct = round((price - prev_high) / prev_high * 100, 3) if prev_high else 0

    conn, use_pg = _conn()
    cur = conn.cursor()

    placeholder = "%s" if use_pg else "?"

    cur.execute(
        f"""
        INSERT INTO ml_signal_log (
            signal_id, symbol, signal_date, mode,
            rsi, adx, volume_ratio, breakout_pct,
            vol_buildup_score, vol_buildup_days,
            consolidation_score, consolidation_ratio,
            rel_strength_score, rel_strength_delta,
            sentiment_verdict, sentiment_avg_score,
            analyst_action, days_to_earnings,
            india_vix, vix_5d_change, geopolitical_risk,
            rbi_days, dxy_10d_change, oil_10d_change,
            overnight_composite, macro_score,
            fii_net_5d, fii_trend, delivery_pct,
            promoter_buying, insider_veto,
            pcr, unusual_call_oi, iv_risk,
            regime, sector_trending, nifty_20d_return,
            quality_score, quality_grade, position_multiplier,
            score_json
        ) VALUES (
            {placeholder},{placeholder},{placeholder},{placeholder},
            {placeholder},{placeholder},{placeholder},{placeholder},
            {placeholder},{placeholder},
            {placeholder},{placeholder},
            {placeholder},{placeholder},
            {placeholder},{placeholder},
            {placeholder},{placeholder},
            {placeholder},{placeholder},{placeholder},
            {placeholder},{placeholder},{placeholder},
            {placeholder},{placeholder},
            {placeholder},{placeholder},{placeholder},
            {placeholder},{placeholder},
            {placeholder},{placeholder},{placeholder},
            {placeholder},{placeholder},{placeholder},
            {placeholder},{placeholder},{placeholder},
            {placeholder}
        )
    """,
        (
            signal_id,
            symbol,
            date.today().isoformat(),
            mode,
            signal.get("rsi"),
            signal.get("adx"),
            signal.get("volume_ratio"),
            breakout_pct,
            vb.get("score"),
            vi.get("volume_buildup_days"),
            co.get("score"),
            co.get("consolidation_ratio")
            or vi_sigs.get("consolidation", {}).get("ratio"),
            rs.get("score"),
            vi_sigs.get("relative_strength", {}).get("delta"),
            cs.get("verdict"),
            hl.get("avg_score"),
            an.get("action"),
            ea.get("days_to_earnings"),
            vix_sig.get("current"),
            vix_sig.get("change_5d"),
            geo_sig.get("avg_tone"),
            cb_sig.get("rbi_days"),
            do_sig.get("dxy_change"),
            do_sig.get("oil_change"),
            og_sig.get("composite"),
            ms.get("macro_score"),
            fii_sig.get("fii_net_5d"),
            fii_sig.get("fii_trend"),
            del_sig.get("delivery_pct_today"),
            int(ins_sig.get("promoter_buying", False)),
            int(ins_sig.get("veto", False)),
            pcr_sig.get("pcr"),
            int(uoi_sig.get("unusual", False)),
            int(uoi_sig.get("iv_risk", False)),
            signal.get("sector", ""),
            int(True),  # sector_trending is always True if signal fires
            None,  # nifty_20d_return — filled from separate fetch if needed
            score_result.get("score"),
            score_result.get("grade"),
            score_result.get("position_multiplier"),
            json.dumps(score_result, default=str),
        ),
    )

    conn.commit()
    cur.close()
    conn.close()


def log_trade_outcome(
    symbol: str,
    signal_date: str,
    outcome: str,
    pnl: float,
    pnl_pct: float,
    days_held: int,
    exit_reason: str,
):
    """
    Update the ml_signal_log with trade outcome when a trade closes.
    Called from app.py in api_close_trade().

    outcome: WIN / LOSS / TIME
    """
    conn, use_pg = _conn()
    cur = conn.cursor()
    ph = "%s" if use_pg else "?"

    cur.execute(
        f"""
        UPDATE ml_signal_log
        SET outcome={ph}, pnl={ph}, pnl_pct={ph},
            days_held={ph}, exit_reason={ph}, close_date={ph}
        WHERE symbol={ph} AND signal_date={ph} AND outcome IS NULL
    """,
        (
            outcome,
            pnl,
            pnl_pct,
            days_held,
            exit_reason,
            date.today().isoformat(),
            symbol,
            signal_date,
        ),
    )

    conn.commit()
    cur.close()
    conn.close()


def get_sample_count() -> dict:
    """Returns count of total and labelled samples."""
    conn, use_pg = _conn()
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM ml_signal_log")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM ml_signal_log WHERE outcome IS NOT NULL")
        labelled = cur.fetchone()[0]
        cur.execute(
            """
            SELECT outcome, COUNT(*) as cnt
            FROM ml_signal_log
            WHERE outcome IS NOT NULL
            GROUP BY outcome
        """
        )
        breakdown = {row[0]: row[1] for row in cur.fetchall()}
    except Exception:
        total, labelled, breakdown = 0, 0, {}
    finally:
        cur.close()
        conn.close()

    return {
        "total": total,
        "labelled": labelled,
        "breakdown": breakdown,
        "ml_ready": labelled >= 50,
        "ml_reliable": labelled >= 200,
    }
