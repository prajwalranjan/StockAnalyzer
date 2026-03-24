"""
prediction/smart_money.py — Phase 3
=====================================
Aggregates institutional footprint signals into a Smart Money
score (0-25) for each breakout candidate.

Four signals:
  S1. FII/DII flows     — sector-level institutional direction
  S2. Delivery %        — conviction vs speculation
  S3. Block deals       — large institutional transactions
  S4. Insider activity  — promoter/insider buying (strongest signal)

These signals detect what institutions are doing BEFORE the
price fully reflects it. When multiple signals align, the
probability of a genuine breakout following through is higher.

STATUS: Shadow mode — scores logged, shown on dashboard,
        never block trades until validated.
"""

from prediction import nse_scraper


# ─── Signal weights within this module (max 25 total) ────────────────────────

SIGNAL_MAX = {
    "fii_dii": 8,  # sector flow direction
    "delivery": 7,  # conviction buying
    "block_deal": 5,  # large institutional trades
    "insider": 5,  # promoter/insider activity (highest quality signal)
}


# ─── S1 — FII/DII Flow Score ─────────────────────────────────────────────────


def score_fii_dii(fii_dii_data: dict) -> tuple[int, str]:
    """
    Scores the FII/DII flow signal.

    Both buying simultaneously = strongest signal (8 pts)
    FII buying alone = good signal (5 pts)
    DII buying alone = moderate signal (3 pts)
    Mixed/unknown = neutral (2 pts)
    FII selling = penalty (0 pts)

    Returns: (score 0-8, detail string)
    """
    if fii_dii_data.get("source") == "fallback":
        return 2, f"FII/DII data unavailable — neutral"

    combined = fii_dii_data.get("combined", "MIXED")
    fii_net = fii_dii_data.get("fii_net_5d", 0)
    detail = fii_dii_data.get("detail", "")

    if combined == "BOTH_BUYING":
        return 8, f"both FII + DII buying ({detail})"
    elif combined == "FII_BUYING":
        return 5, f"FII net buying ({detail})"
    elif combined == "DII_BUYING":
        return 3, f"DII net buying, FII neutral ({detail})"
    elif combined == "FII_SELLING":
        return 0, f"FII net selling — institutional exit ({detail})"
    else:
        return 2, f"mixed institutional flows ({detail})"


# ─── S2 — Delivery % Score ───────────────────────────────────────────────────


def score_delivery(delivery_data: dict) -> tuple[int, str]:
    """
    Scores the delivery percentage signal.

    High delivery % relative to average = conviction buying
    Low delivery % = speculative, noise traders

    Returns: (score 0-7, detail string)
    """
    if delivery_data.get("source") == "fallback":
        return 2, "delivery data unavailable — neutral"

    today_pct = delivery_data.get("delivery_pct_today")
    delta = delivery_data.get("delta")
    detail = delivery_data.get("detail", "")

    if today_pct is None:
        return 2, "delivery % not available"

    # Score based on absolute delivery % (higher = more conviction)
    if today_pct >= 70:
        score = 7
        msg = f"very high delivery {today_pct:.1f}% — strong conviction"
    elif today_pct >= 55:
        score = 5
        msg = f"high delivery {today_pct:.1f}% — good conviction"
    elif today_pct >= 40:
        score = 3
        msg = f"moderate delivery {today_pct:.1f}%"
    else:
        score = 1
        msg = f"low delivery {today_pct:.1f}% — mostly speculative"

    # Boost if significantly above average (delta available)
    if delta is not None and delta > 15:
        score = min(7, score + 1)
        msg += f" (+{delta:.1f}% above avg — rising conviction)"

    return score, msg


# ─── S3 — Block Deal Score ────────────────────────────────────────────────────


def score_block_deals(block_data: dict) -> tuple[int, str]:
    """
    Scores block deal activity.

    Large institutional buy block = 5 pts
    No block activity = neutral (2 pts)
    Large sell block = 0 pts

    Returns: (score 0-5, detail string)
    """
    if block_data.get("source") == "fallback":
        return 2, "block deal data unavailable — neutral"

    verdict = block_data.get("verdict", "NEUTRAL")
    detail = block_data.get("detail", "")
    net_value = block_data.get("net_value_cr", 0)

    if verdict == "BUY":
        score = 5
        msg = f"institutional buy block deals ({detail})"
    elif verdict == "SELL":
        score = 0
        msg = f"institutional sell block deals — exit signal ({detail})"
    else:
        score = 2
        msg = "no significant block deal activity"

    return score, msg


# ─── S4 — Insider Activity Score ─────────────────────────────────────────────


def score_insider(insider_data: dict) -> tuple[int, str]:
    """
    Scores insider/promoter trading activity.
    This is the highest-quality signal in the module — promoters
    buying their own stock is the strongest possible bullish signal.

    Promoter buying = 5 pts (maximum)
    No activity = neutral (2 pts)
    Promoter selling > 1% stake = VETO (0 pts, flags RED)

    Returns: (score 0-5, detail string)
    """
    if insider_data.get("source") == "fallback":
        return 2, "insider data unavailable — neutral"

    buying = insider_data.get("promoter_buying", False)
    selling = insider_data.get("promoter_selling", False)
    veto = insider_data.get("veto", False)
    detail = insider_data.get("detail", "")

    if veto:
        return 0, f"VETO: large promoter sell — {detail}"
    elif buying and not selling:
        return 5, f"promoter buying own stock — {detail}"
    elif buying and selling:
        return 3, f"mixed insider activity — {detail}"
    elif selling:
        return 1, f"promoter selling — {detail}"
    else:
        return 2, "no insider activity (neutral)"


# ─── Main entry point ─────────────────────────────────────────────────────────


def compute(sym: str) -> dict:
    """
    Fetches all smart money data for a stock and computes
    the combined score (0-25).

    Fetches:
      - FII/DII sector flows (last 5 days)
      - Delivery % (today's bhavcopy)
      - Block deals (last 5 days)
      - Insider trades (last 30 days)

    Returns:
    {
        "total":        18,          # 0-25
        "summary":      "GOOD",      # WEAK / FAIR / GOOD / STRONG
        "veto":         False,       # True if hard veto triggered
        "shadow_mode":  True,
        "signals": {
            "fii_dii":    {"score": 5, "detail": "...", "raw": {...}},
            "delivery":   {"score": 5, "detail": "...", "raw": {...}},
            "block_deal": {"score": 3, "detail": "...", "raw": {...}},
            "insider":    {"score": 5, "detail": "...", "raw": {...}},
        }
    }
    """
    # Fetch all data sources
    fii_dii_data = nse_scraper.get_fii_dii(days=5)
    delivery_data = nse_scraper.get_delivery(sym)
    block_data = nse_scraper.get_block_deals(sym, days=5)
    insider_data = nse_scraper.get_insider_trades(sym, days=30)

    # Score each signal
    fii_score, fii_detail = score_fii_dii(fii_dii_data)
    del_score, del_detail = score_delivery(delivery_data)
    blk_score, blk_detail = score_block_deals(block_data)
    ins_score, ins_detail = score_insider(insider_data)

    # Check for hard veto (large promoter selling)
    veto = insider_data.get("veto", False)

    total = fii_score + del_score + blk_score + ins_score

    if veto:
        total = 0
        summary = "VETO"
    elif total >= 20:
        summary = "STRONG"
    elif total >= 14:
        summary = "GOOD"
    elif total >= 8:
        summary = "FAIR"
    else:
        summary = "WEAK"

    return {
        "total": total,
        "summary": summary,
        "veto": veto,
        "shadow_mode": True,
        "signals": {
            "fii_dii": {
                "score": fii_score,
                "detail": fii_detail,
                "raw": fii_dii_data,
            },
            "delivery": {
                "score": del_score,
                "detail": del_detail,
                "raw": delivery_data,
            },
            "block_deal": {
                "score": blk_score,
                "detail": blk_detail,
                "raw": block_data,
            },
            "insider": {
                "score": ins_score,
                "detail": ins_detail,
                "raw": insider_data,
            },
        },
    }
