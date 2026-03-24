"""
prediction/score.py — Breakout Quality Score Aggregator
=========================================================
Combines scores from all prediction modules into a single
Breakout Quality Score (0-100) for each signal.

Current modules and their max contribution:
  Phase 1 — Volume Intelligence    30 pts  ← ACTIVE (shadow mode)
  Phase 2A — Company Sentiment     15 pts  ← NOT YET BUILT
  Phase 2B — Macro Sentiment       15 pts  ← NOT YET BUILT
  Phase 3 — Smart Money            25 pts  ← NOT YET BUILT
  Phase 4 — Options Signal         15 pts  ← NOT YET BUILT
  ─────────────────────────────────────────
  TOTAL                           100 pts

Score interpretation:
  0-40   → SKIP (too many unknowns)
  41-60  → WEAK (reduce position or skip)
  61-80  → GOOD (standard Rs8,000 position)
  81-100 → STRONG (consider Rs10,000 position)

SHADOW MODE:
  All scores are computed and logged.
  No score currently blocks or filters any trade.
  Scores are shown on signal cards for human review only.
  Shadow period: 2 weeks minimum before considering live filtering.
"""

from prediction import volume_intelligence
from prediction import company_sentiment
from prediction import macro_sentiment
from prediction import smart_money
from prediction import options_signal


# ─── Module weights (max points per module) ───────────────────────────────────
# These will be recalibrated by the Phase 6 self-improving loop.
# For now they are hand-coded based on our design document.

MODULE_MAX = {
    "volume_intelligence": 30,
    "company_sentiment": 15,  # Phase 2A — not yet built
    "macro_sentiment": 15,  # Phase 2B — not yet built
    "smart_money": 25,  # Phase 3  — not yet built
    "options_signal": 15,  # Phase 4  — not yet built
}

TOTAL_MAX = sum(MODULE_MAX.values())  # 100


def compute_score(sym, stock_df, sector_df) -> dict:
    """
    Computes the full Breakout Quality Score for a signal.

    Only runs modules that are currently built.
    Future modules slot in here as they are built.

    Returns a score dict attached to the signal for dashboard display
    and logged to database for ML training data later.
    """
    modules = {}
    raw_total = 0

    # ── Phase 1: Volume Intelligence (ACTIVE) ──────────────────────
    vi = volume_intelligence.compute(stock_df, sector_df)
    modules["volume_intelligence"] = vi
    raw_total += vi["total"]

    # ── Phase 2A: Company Sentiment (ACTIVE — shadow mode) ──────────
    cs = company_sentiment.compute(sym)
    modules["company_sentiment"] = cs
    raw_total += max(0, cs.get("score", 0))  # only add positive contribution

    # ── Phase 2B: Macro Sentiment (ACTIVE — shadow mode) ────────────
    ms = macro_sentiment.compute()
    modules["macro_sentiment"] = ms
    # Macro score contributes 0-15 pts based on environment quality
    macro_contribution = round(ms["macro_score"] / 100 * MODULE_MAX["macro_sentiment"])
    raw_total += macro_contribution

    # ── Phase 3: Smart Money (ACTIVE — shadow mode) ──────────────────
    sm = smart_money.compute(sym)
    modules["smart_money"] = sm
    raw_total += sm["total"]

    # Hard veto from insider selling overrides everything
    if sm.get("veto"):
        return {
            "symbol": sym,
            "score": 0,
            "grade": "VETO",
            "raw_total": 0,
            "active_max": active_max if "active_max" in dir() else 70,
            "modules": modules,
            "shadow_mode": True,
            "phases_active": [
                "volume_intelligence",
                "company_sentiment",
                "macro_sentiment",
                "smart_money",
            ],
            "position_multiplier": 0,
            "macro_verdict": "RED",
            "veto_reason": "large promoter selling detected",
        }

    # ── Phase 4: Options Signal (ACTIVE — shadow mode) ──────────────
    os_result = options_signal.compute(sym)
    modules["options_signal"] = os_result
    raw_total += os_result["total"]

    # ── Score normalisation ────────────────────────────────────────
    active_max = (
        MODULE_MAX["volume_intelligence"]
        + MODULE_MAX["company_sentiment"]
        + MODULE_MAX["macro_sentiment"]
        + MODULE_MAX["smart_money"]
        + MODULE_MAX["options_signal"]
    )
    if active_max > 0:
        normalised = round((raw_total / active_max) * 100)
    else:
        normalised = 0

    normalised = max(0, min(100, normalised))

    if normalised >= 81:
        grade = "STRONG"
    elif normalised >= 61:
        grade = "GOOD"
    elif normalised >= 41:
        grade = "FAIR"
    else:
        grade = "WEAK"

    macro_multiplier = ms.get("position_multiplier", 1.0)
    macro_verdict = ms.get("verdict", "GREEN")

    return {
        "symbol": sym,
        "score": normalised,
        "grade": grade,
        "raw_total": raw_total,
        "active_max": active_max,
        "modules": modules,
        "shadow_mode": True,
        "phases_active": [
            "volume_intelligence",
            "company_sentiment",
            "macro_sentiment",
            "smart_money",
            "options_signal",
        ],
        "position_multiplier": macro_multiplier,
        "macro_verdict": macro_verdict,
    }


def score_label(score: int) -> str:
    """Human-readable label for a score."""
    if score >= 81:
        return "STRONG"
    if score >= 61:
        return "GOOD"
    if score >= 41:
        return "FAIR"
    return "WEAK"


def score_color(score: int) -> str:
    """CSS variable name for score colour."""
    if score >= 81:
        return "var(--green)"
    if score >= 61:
        return "var(--cyan)"
    if score >= 41:
        return "var(--yellow)"
    return "var(--muted)"
