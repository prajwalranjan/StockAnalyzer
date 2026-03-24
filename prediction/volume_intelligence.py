"""
prediction/volume_intelligence.py — Phase 1
============================================
Detects institutional accumulation BEFORE a breakout occurs.

Three signals:
  1. Volume buildup    — rising volume for 3-5 days pre-breakout
  2. Consolidation     — price coiling in tight range pre-breakout
  3. Relative strength — stock outperforming its sector recently

All three return a score (0-10) and a human-readable detail string.
The score aggregator in score.py combines them into the final
Breakout Quality Score contribution from this module (0-30 points).

STATUS: Shadow mode — scores logged and shown on dashboard,
        but never block or filter trades.
"""

import numpy as np
import pandas as pd


def volume_buildup(df) -> tuple[int, str]:
    """
    Checks if volume was consistently above average in the days
    leading up to today's breakout.

    Genuine breakouts are often preceded by 3-5 days of quietly
    rising volume as institutions accumulate before the move.
    A single-day volume spike with no prior buildup is more likely
    to be noise or a retail reaction.

    Scoring:
      0 days above avg  → 0 pts
      1-2 days above avg → 3 pts
      3-4 days above avg → 6 pts
      5 days above avg   → 10 pts

    Returns: (score 0-10, detail string)
    """
    if df.empty or len(df) < 30:
        return 0, "insufficient data"

    # 20-day average volume (same window as breakout check)
    avg_vol = float(df["Volume"].iloc[-22:-1].mean())
    if avg_vol <= 0:
        return 0, "no volume data"

    # Check last 5 days BEFORE today (not including today's spike)
    recent_vols = df["Volume"].iloc[-6:-1]
    days_above = int((recent_vols > avg_vol).sum())

    if days_above == 0:
        return 0, "no volume buildup (single-day spike)"
    elif days_above <= 2:
        return 3, f"mild buildup ({days_above}/5 days above avg)"
    elif days_above <= 4:
        return 6, f"good buildup ({days_above}/5 days above avg)"
    else:
        return 10, f"strong buildup (5/5 days above avg — institutional accumulation)"


def consolidation_tightness(df) -> tuple[int, str]:
    """
    Measures how tightly the stock has been trading in the days
    before the breakout — the "coiling spring" pattern.

    A stock that trades in an increasingly narrow range for 1-3 weeks
    before breaking out tends to follow through much more reliably.
    The tighter the range, the more compressed the energy.

    ratio = (10-day high-low range) / (30-day high-low range)
    Lower ratio = tighter coil = more compressed energy

    Scoring:
      ratio >= 0.65  → 0 pts  (no consolidation)
      ratio < 0.65   → 3 pts  (slight tightening)
      ratio < 0.50   → 6 pts  (moderate coil)
      ratio < 0.35   → 10 pts (very tight coil — high energy)

    Returns: (score 0-10, detail string)
    """
    if df.empty or len(df) < 35:
        return 0, "insufficient data"

    # Use High/Low for true range measurement
    hi_10 = float(df["High"].iloc[-11:-1].max())
    lo_10 = float(df["Low"].iloc[-11:-1].min())
    hi_30 = float(df["High"].iloc[-31:-1].max())
    lo_30 = float(df["Low"].iloc[-31:-1].min())

    range_10 = hi_10 - lo_10
    range_30 = hi_30 - lo_30

    if range_30 <= 0:
        return 0, "no range data"

    ratio = round(range_10 / range_30, 3)

    if ratio >= 0.65:
        return 0, f"no consolidation (range ratio {ratio:.2f})"
    elif ratio >= 0.50:
        return 3, f"slight tightening (range ratio {ratio:.2f})"
    elif ratio >= 0.35:
        return 6, f"good coil (range ratio {ratio:.2f})"
    else:
        return 10, f"very tight coil (range ratio {ratio:.2f} — compressed energy)"


def relative_strength(stock_df, sector_df) -> tuple[int, str]:
    """
    Compares the stock's recent return vs its sector index.
    A stock that holds up better than its sector during dips,
    or rises faster during rallies, shows hidden institutional demand.

    When the sector recovers, these "relative strength leaders"
    tend to break out first and run the hardest.

    delta = stock_return_20d - sector_return_20d

    Scoring:
      delta <= 0%   → 0 pts  (underperforming — avoid)
      delta > 0%    → 3 pts  (slight outperformance)
      delta > 4%    → 6 pts  (clear outperformance)
      delta > 8%    → 10 pts (strong leadership — institutional buying)

    Returns: (score 0-10, detail string)
    """
    if stock_df.empty or len(stock_df) < 22:
        return 3, "insufficient stock data (neutral)"

    stock_ret = (
        float(stock_df["Close"].iloc[-1]) / float(stock_df["Close"].iloc[-22]) - 1
    ) * 100

    # If no sector data, return neutral score
    if sector_df is None or sector_df.empty or len(sector_df) < 22:
        return 3, f"stock +{stock_ret:.1f}% (no sector data for comparison)"

    sector_ret = (
        float(sector_df["Close"].iloc[-1]) / float(sector_df["Close"].iloc[-22]) - 1
    ) * 100

    delta = round(stock_ret - sector_ret, 2)

    if delta <= 0:
        return (
            0,
            f"underperforming sector by {abs(delta):.1f}% (stock {stock_ret:+.1f}% vs sector {sector_ret:+.1f}%)",
        )
    elif delta <= 4:
        return 3, f"slight outperformance +{delta:.1f}% vs sector"
    elif delta <= 8:
        return 6, f"clear outperformance +{delta:.1f}% vs sector"
    else:
        return 10, f"strong RS leader +{delta:.1f}% vs sector — institutional demand"


def compute(stock_df, sector_df) -> dict:
    """
    Runs all three volume intelligence signals and returns
    a combined result dict ready for score.py to aggregate.

    Returns:
    {
        "volume_buildup":     {"score": 6, "detail": "..."},
        "consolidation":      {"score": 10, "detail": "..."},
        "relative_strength":  {"score": 3, "detail": "..."},
        "total":              19,          # 0-30
        "summary":            "GOOD",      # WEAK / FAIR / GOOD / STRONG
    }
    """
    vb_score, vb_detail = volume_buildup(stock_df)
    ct_score, ct_detail = consolidation_tightness(stock_df)
    rs_score, rs_detail = relative_strength(stock_df, sector_df)

    total = vb_score + ct_score + rs_score  # max 30

    if total <= 8:
        summary = "WEAK"
    elif total <= 16:
        summary = "FAIR"
    elif total <= 23:
        summary = "GOOD"
    else:
        summary = "STRONG"

    return {
        "volume_buildup": {"score": vb_score, "detail": vb_detail},
        "consolidation": {"score": ct_score, "detail": ct_detail},
        "relative_strength": {"score": rs_score, "detail": rs_detail},
        "total": total,
        "summary": summary,
    }
