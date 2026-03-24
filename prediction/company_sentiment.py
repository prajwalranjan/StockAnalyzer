"""
prediction/company_sentiment.py — Phase 2A
===========================================
Classifies company-specific news sentiment for a stock using
the Cohere API. Runs in shadow mode — logs verdicts but never
blocks trades until validated by sentiment_validator.py.

Signals:
  2A.1 News headline sentiment     (Cohere classify)
  2A.2 NSE corporate announcements (keyword rules — reliable for filings)
  2A.3 Earnings proximity filter   (yfinance calendar)
  2A.4 Analyst actions             (yfinance recommendations)

Verdict scale:
  GREEN  → neutral/positive environment → proceed
  AMBER  → mild negative signals → flag but don't block (shadow)
  RED    → clearly negative → veto (once validated and enabled)

STATUS: Shadow mode — verdicts logged, never block trades yet.
"""

import os
import re
from datetime import date, timedelta
import yfinance as yf
import cohere

# ─── Cohere client ────────────────────────────────────────────────────────────


def _get_client():
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        return None
    return cohere.Client(api_key)


# ─── NSE announcement keywords ───────────────────────────────────────────────
# These are reliable enough for rule-based classification.
# NSE filings use standardised language so keyword matching works well.

NSE_POSITIVE_KEYWORDS = [
    "buyback",
    "buy back",
    "dividend",
    "bonus shares",
    "stock split",
    "order win",
    "capacity expansion",
    "debt free",
    "debt reduction",
    "record revenue",
    "record profit",
    "upgrade",
    "new contract",
    "joint venture",
    "acquisition completed",
]

NSE_NEGATIVE_KEYWORDS = [
    "sebi",
    "penalty",
    "show cause",
    "fraud",
    "investigation",
    "audit qualification",
    "going concern",
    "default",
    "npa",
    "promoter pledge",
    "pledge increase",
    "insider trading",
    "regulatory action",
    "nclt",
    "insolvency",
    "winding up",
    "resignation of md",
    "resignation of ceo",
    "resignation of cfo",
    "related party",
    "forensic audit",
    "qualified opinion",
]


# ─── Signal 2A.1 — News headline sentiment (Cohere) ──────────────────────────


def classify_headlines(headlines: list[str]) -> dict:
    """
    Classifies a list of news headlines using Cohere's classify endpoint.
    Falls back to rule-based if Cohere is unavailable.

    Returns:
    {
        "avg_score":   0.42,       # -1.0 to +1.0
        "min_score":  -0.12,       # worst single headline
        "count":       3,          # number of headlines classified
        "verdict":    "GREEN",     # GREEN / AMBER / RED
        "detail":     "...",       # human readable
        "source":     "cohere"     # or "fallback"
    }
    """
    if not headlines:
        return {
            "avg_score": 0,
            "min_score": 0,
            "count": 0,
            "verdict": "GREEN",
            "detail": "no recent headlines",
            "source": "none",
        }

    client = _get_client()
    if client:
        return _classify_with_cohere(client, headlines)
    else:
        return _classify_fallback(headlines)


def _classify_with_cohere(client, headlines: list[str]) -> dict:
    """Uses Cohere's classify API to score each headline."""
    try:
        # Cohere classify needs examples to work.
        # We provide a small set of financial examples as training signal.
        examples = [
            cohere.ClassifyExample(
                "Company wins Rs500 crore order from government", "positive"
            ),
            cohere.ClassifyExample("Board approves buyback of shares", "positive"),
            cohere.ClassifyExample(
                "Company reports record quarterly profit", "positive"
            ),
            cohere.ClassifyExample("Promoter increases stake in company", "positive"),
            cohere.ClassifyExample("Dividend declared for shareholders", "positive"),
            cohere.ClassifyExample(
                "Company expands capacity with new plant", "positive"
            ),
            cohere.ClassifyExample(
                "SEBI issues show cause notice to company", "negative"
            ),
            cohere.ClassifyExample("Promoter pledges additional shares", "negative"),
            cohere.ClassifyExample("MD resigns amid controversy", "negative"),
            cohere.ClassifyExample("Company defaults on debt repayment", "negative"),
            cohere.ClassifyExample("Audit firm qualifies annual report", "negative"),
            cohere.ClassifyExample(
                "Fraud investigation launched by regulators", "negative"
            ),
            cohere.ClassifyExample(
                "Quarterly results in line with estimates", "neutral"
            ),
            cohere.ClassifyExample(
                "Company to hold board meeting next week", "neutral"
            ),
            cohere.ClassifyExample("Annual general meeting scheduled", "neutral"),
            cohere.ClassifyExample("Management to discuss expansion plans", "neutral"),
        ]

        response = client.classify(
            inputs=headlines[:10],  # Cohere free tier: max 10 per call
            examples=examples,
        )

        scores = []
        for classification in response.classifications:
            label = classification.prediction
            confidence = classification.confidence
            # Convert to -1 to +1 scale
            if label == "positive":
                scores.append(confidence)
            elif label == "negative":
                scores.append(-confidence)
            else:
                scores.append(0.0)

        avg_score = round(sum(scores) / len(scores), 3)
        min_score = round(min(scores), 3)

        verdict, detail = _score_to_verdict(avg_score, min_score, len(headlines))

        return {
            "avg_score": avg_score,
            "min_score": min_score,
            "count": len(headlines),
            "verdict": verdict,
            "detail": detail,
            "source": "cohere",
        }

    except Exception as e:
        # Cohere call failed — fall back to rule-based
        return _classify_fallback(headlines, error=str(e))


def _classify_fallback(headlines: list[str], error: str = None) -> dict:
    """
    Rule-based fallback when Cohere is unavailable.
    Scans for strong positive/negative financial keywords.
    Less accurate (~65%) but catches obvious red flags.
    """
    NEGATIVE = [
        "fraud",
        "penalty",
        "sebi",
        "investigation",
        "default",
        "loss",
        "recall",
        "resign",
        "downgrade",
        "insolvency",
        "pledge",
        "npa",
        "winding",
        "nclt",
        "show cause",
        "forensic",
        "going concern",
    ]
    POSITIVE = [
        "order",
        "profit",
        "buyback",
        "dividend",
        "expansion",
        "upgrade",
        "record",
        "growth",
        "acquisition",
        "contract",
        "surplus",
        "bonus",
    ]

    scores = []
    for h in headlines:
        h_lower = h.lower()
        neg_hits = sum(1 for w in NEGATIVE if w in h_lower)
        pos_hits = sum(1 for w in POSITIVE if w in h_lower)
        if neg_hits > pos_hits:
            scores.append(-0.6)
        elif pos_hits > neg_hits:
            scores.append(0.6)
        else:
            scores.append(0.0)

    avg_score = round(sum(scores) / len(scores), 3) if scores else 0
    min_score = round(min(scores), 3) if scores else 0

    verdict, detail = _score_to_verdict(avg_score, min_score, len(headlines))
    source = f"fallback (cohere error: {error})" if error else "fallback"

    return {
        "avg_score": avg_score,
        "min_score": min_score,
        "count": len(headlines),
        "verdict": verdict,
        "detail": detail,
        "source": source,
    }


def _score_to_verdict(
    avg_score: float, min_score: float, count: int
) -> tuple[str, str]:
    """Convert numeric scores to a verdict and human-readable detail."""
    if min_score < -0.7:
        return (
            "RED",
            f"hard veto — one very negative headline (min score {min_score:.2f})",
        )
    elif avg_score < -0.3:
        return (
            "RED",
            f"overall negative environment (avg {avg_score:.2f} across {count} headlines)",
        )
    elif avg_score < 0.0:
        return "AMBER", f"slightly negative news environment (avg {avg_score:.2f})"
    else:
        return (
            "GREEN",
            f"neutral/positive news environment (avg {avg_score:.2f} across {count} headlines)",
        )


# ─── Signal 2A.2 — NSE corporate announcements ───────────────────────────────


def check_nse_announcements(sym: str) -> dict:
    """
    Scans recent news from yfinance for NSE-filing type announcements.
    Uses keyword rules — NSE filings use standardised language
    so keyword matching is reliable here (unlike free-form news).

    Returns:
    {
        "score":    +5 / 0 / -5,
        "veto":     False,          # True if hard RED veto triggered
        "verdict":  "GREEN",
        "detail":   "..."
    }
    """
    try:
        ticker = yf.Ticker(sym)
        news = ticker.news or []

        # Combine title + summary for each article
        texts = []
        for item in news[:15]:  # last 15 news items
            title = item.get("title", "")
            summary = item.get("summary", "") or item.get("description", "")
            texts.append(f"{title} {summary}".lower())

        combined = " ".join(texts)

        # Check for hard veto triggers first
        veto_triggers = [
            "sebi",
            "show cause",
            "fraud",
            "investigation",
            "going concern",
            "audit qualification",
            "forensic",
            "insolvency",
            "nclt",
            "insider trading",
            "winding up",
        ]
        for trigger in veto_triggers:
            if trigger in combined:
                return {
                    "score": -10,
                    "veto": True,
                    "verdict": "RED",
                    "detail": f"hard veto — '{trigger}' detected in recent filings",
                }

        # Check for negative signals
        neg_hits = [kw for kw in NSE_NEGATIVE_KEYWORDS if kw in combined]
        pos_hits = [kw for kw in NSE_POSITIVE_KEYWORDS if kw in combined]

        if neg_hits:
            return {
                "score": -5,
                "veto": False,
                "verdict": "AMBER",
                "detail": f"negative filing signals: {', '.join(neg_hits[:3])}",
            }
        elif pos_hits:
            return {
                "score": +5,
                "veto": False,
                "verdict": "GREEN",
                "detail": f"positive filing signals: {', '.join(pos_hits[:3])}",
            }
        else:
            return {
                "score": 0,
                "veto": False,
                "verdict": "GREEN",
                "detail": "no significant announcements detected",
            }

    except Exception as e:
        return {
            "score": 0,
            "veto": False,
            "verdict": "GREEN",
            "detail": f"announcement check failed: {e}",
        }


# ─── Signal 2A.3 — Earnings proximity filter ─────────────────────────────────


def earnings_proximity(sym: str) -> dict:
    """
    Checks how many days until the next earnings date.
    Within 3 trading days = SKIP regardless of other signals.
    """
    try:
        ticker = yf.Ticker(sym)
        calendar = ticker.calendar

        if not calendar:
            return {
                "days_to_earnings": None,
                "skip": False,
                "verdict": "GREEN",
                "detail": "earnings date unavailable",
            }

        # yfinance returns calendar as dict: {"Earnings Date": [date1, date2]}
        # or sometimes as a DataFrame — handle both
        earnings_dates = []

        if isinstance(calendar, dict):
            raw = calendar.get("Earnings Date", [])
            if not isinstance(raw, list):
                raw = [raw]
            earnings_dates = raw
        else:
            # DataFrame format (older yfinance)
            if not calendar.empty and "Earnings Date" in calendar.index:
                for val in calendar.loc["Earnings Date"]:
                    earnings_dates.append(val)

        # Find next upcoming earnings date
        next_date = None
        for val in earnings_dates:
            try:
                import pandas as pd

                d = pd.Timestamp(val).date()
                if d >= date.today():
                    next_date = d
                    break
            except:
                continue

        if next_date:
            days = (next_date - date.today()).days
            if days <= 3:
                return {
                    "days_to_earnings": days,
                    "skip": True,
                    "verdict": "RED",
                    "detail": f"earnings in {days} days — skip (binary event risk)",
                }
            elif days <= 7:
                return {
                    "days_to_earnings": days,
                    "skip": False,
                    "verdict": "AMBER",
                    "detail": f"earnings approaching in {days} days — caution",
                }
            else:
                return {
                    "days_to_earnings": days,
                    "skip": False,
                    "verdict": "GREEN",
                    "detail": f"next earnings in {days} days",
                }

        return {
            "days_to_earnings": None,
            "skip": False,
            "verdict": "GREEN",
            "detail": "no upcoming earnings found",
        }

    except Exception as e:
        return {
            "days_to_earnings": None,
            "skip": False,
            "verdict": "GREEN",
            "detail": f"earnings check failed: {e}",
        }


# ─── Signal 2A.4 — Analyst actions ───────────────────────────────────────────


def analyst_action(sym: str) -> dict:
    """
    Checks for recent analyst upgrades or downgrades (last 7 days).

    Returns:
    {
        "action":   +1 / 0 / -1,   # +1 upgrade, 0 none, -1 downgrade
        "verdict":  "GREEN",
        "detail":   "..."
    }
    """
    try:
        ticker = yf.Ticker(sym)
        recs = ticker.recommendations

        if recs is None or recs.empty:
            return {"action": 0, "verdict": "GREEN", "detail": "no analyst data"}

        cutoff = date.today() - timedelta(days=7)
        recent = (
            recs[recs.index.date >= cutoff]
            if hasattr(recs.index, "date")
            else recs.tail(3)
        )

        if recent.empty:
            return {
                "action": 0,
                "verdict": "GREEN",
                "detail": "no recent analyst actions",
            }

        # Check for upgrades/downgrades in 'To Grade' or 'Action' column
        actions = []
        for col in ["Action", "To Grade", "toGrade"]:
            if col in recent.columns:
                for val in recent[col].str.lower().values:
                    if any(
                        w in str(val)
                        for w in ["upgrade", "buy", "outperform", "overweight"]
                    ):
                        actions.append(+1)
                    elif any(
                        w in str(val)
                        for w in ["downgrade", "sell", "underperform", "underweight"]
                    ):
                        actions.append(-1)

        if not actions:
            return {
                "action": 0,
                "verdict": "GREEN",
                "detail": "no directional analyst actions",
            }

        net = sum(actions)
        if net > 0:
            return {
                "action": +1,
                "verdict": "GREEN",
                "detail": f"analyst upgrade in last 7 days",
            }
        elif net < 0:
            return {
                "action": -1,
                "verdict": "AMBER",
                "detail": f"analyst downgrade in last 7 days",
            }
        else:
            return {"action": 0, "verdict": "GREEN", "detail": "mixed analyst actions"}

    except Exception as e:
        return {"action": 0, "verdict": "GREEN", "detail": f"analyst check failed: {e}"}


# ─── Main entry point ─────────────────────────────────────────────────────────


def compute(sym: str, headlines: list[str] = None) -> dict:
    """
    Runs all four company sentiment signals and returns a combined result.

    If headlines not provided, fetches them from yfinance automatically.

    Returns:
    {
        "verdict":        "GREEN",   # overall: GREEN / AMBER / RED
        "score":          +8,        # -15 to +15 contribution to quality score
        "shadow_mode":    True,
        "signals": {
            "headlines":      {...},
            "announcements":  {...},
            "earnings":       {...},
            "analyst":        {...},
        }
    }
    """
    # Fetch headlines if not provided
    if headlines is None:
        try:
            ticker = yf.Ticker(sym)
            news = ticker.news or []
            headlines = [
                item.get("title", "") for item in news[:10] if item.get("title")
            ]
        except:
            headlines = []

    # Run all four signals
    headline_result = classify_headlines(headlines)
    announcement_result = check_nse_announcements(sym)
    earnings_result = earnings_proximity(sym)
    analyst_result = analyst_action(sym)

    # Determine overall verdict
    # Hard veto conditions
    if (
        headline_result["verdict"] == "RED"
        or announcement_result["veto"]
        or earnings_result["skip"]
    ):
        overall_verdict = "RED"
    elif (
        headline_result["verdict"] == "AMBER"
        or announcement_result["verdict"] == "AMBER"
        or earnings_result["verdict"] == "AMBER"
        or analyst_result["verdict"] == "AMBER"
    ):
        overall_verdict = "AMBER"
    else:
        overall_verdict = "GREEN"

    # Compute score contribution (-15 to +15)
    score = 0
    if overall_verdict == "GREEN":
        score = min(15, 5 + announcement_result["score"] + analyst_result["action"] * 3)
    elif overall_verdict == "AMBER":
        score = 0
    else:  # RED
        score = -15

    return {
        "verdict": overall_verdict,
        "score": score,
        "shadow_mode": True,
        "signals": {
            "headlines": headline_result,
            "announcements": announcement_result,
            "earnings": earnings_result,
            "analyst": analyst_result,
        },
    }
