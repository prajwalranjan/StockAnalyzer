"""
ml/dataset.py — ML Dataset Preparation
========================================
Loads feature vectors from ml_signal_log, cleans them,
encodes categoricals, and splits into train/test sets
using walk-forward validation.

Key design decisions:
  - TIME stops treated as LOSS (slight negative outcome)
  - Categorical features label-encoded (regime, sentiment_verdict etc.)
  - Missing values filled with column median (not dropped)
  - Exponential sample weighting (recent trades matter more)
  - Walk-forward splits respect temporal order (no future leakage)
"""

import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
from ml.collector import _conn


# ─── Feature columns ─────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "rsi",
    "adx",
    "volume_ratio",
    "breakout_pct",
    "vol_buildup_score",
    "vol_buildup_days",
    "consolidation_score",
    "consolidation_ratio",
    "rel_strength_score",
    "rel_strength_delta",
    "sentiment_avg_score",
    "analyst_action",
    "days_to_earnings",
    "india_vix",
    "vix_5d_change",
    "geopolitical_risk",
    "rbi_days",
    "dxy_10d_change",
    "oil_10d_change",
    "overnight_composite",
    "macro_score",
    "fii_net_5d",
    "delivery_pct",
    "promoter_buying",
    "insider_veto",
    "pcr",
    "unusual_call_oi",
    "iv_risk",
    "sector_trending",
    "quality_score",
    "position_multiplier",
]

CATEGORICAL_FEATURES = [
    "sentiment_verdict",  # GREEN=1, AMBER=0, RED=-1
    "fii_trend",  # BUYING=1, MIXED=0, SELLING=-1
    "regime",  # BULL=2, CHOP=1, BEAR=0
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Encoding maps for categorical features
ENCODINGS = {
    "sentiment_verdict": {"GREEN": 1, "AMBER": 0, "RED": -1},
    "fii_trend": {"BUYING": 1, "MIXED": 0, "SELLING": -1},
    "regime": {"BULL": 2, "CHOP": 1, "BEAR": 0},
}

LABEL_MAP = {"WIN": 1, "LOSS": 0, "TIME": 0}  # TIME = LOSS for binary classifier


# ─── Load data ────────────────────────────────────────────────────────────────


def load_raw(min_samples: int = 10, include_paper: bool = True) -> pd.DataFrame:
    """
    Loads all labelled samples from ml_signal_log.

    Returns raw DataFrame with outcome column.
    Returns empty DataFrame if insufficient data.
    """
    conn, use_pg = _conn()
    ph = "%s" if use_pg else "?"

    modes = "('LIVE', 'PAPER_PARALLEL')" if include_paper else "('LIVE')"
    query = f"""
        SELECT * FROM ml_signal_log
        WHERE outcome IS NOT NULL
        AND mode IN {modes}
        ORDER BY signal_date ASC
    """

    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if len(df) < min_samples:
        return pd.DataFrame()

    return df


def prepare(df: pd.DataFrame, half_life_days: int = 90) -> tuple:
    """
    Prepares a raw DataFrame for model training.

    Steps:
      1. Encode categorical features
      2. Fill missing values with column median
      3. Compute sample weights (exponential decay — recent matters more)
      4. Encode outcome labels

    Returns: (X, y, sample_weights, feature_names)
    """
    if df.empty:
        return None, None, None, []

    df = df.copy()

    # Encode categoricals
    for col, mapping in ENCODINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Select feature columns that exist in this DataFrame
    available = [f for f in ALL_FEATURES if f in df.columns]

    X = df[available].copy()

    # Fill missing values with column median
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Binary labels: WIN=1, LOSS/TIME=0
    y = df["outcome"].map(LABEL_MAP).fillna(0).astype(int)

    # Exponential sample weights — recent trades matter more
    if "signal_date" in df.columns:
        today = pd.Timestamp.today()
        ages = df["signal_date"].apply(lambda d: (today - pd.Timestamp(d)).days)
        weights = np.exp(-np.log(2) * ages / half_life_days)
        weights = weights / weights.sum() * len(weights)  # normalise
    else:
        weights = np.ones(len(df))

    return X.values, y.values, weights, available


# ─── Walk-forward splits ──────────────────────────────────────────────────────


def walk_forward_splits(
    df: pd.DataFrame, train_months: int = 6, test_months: int = 3, step_months: int = 1
) -> list:
    """
    Generates walk-forward train/test index pairs.

    Each split:
      Train: months 1 to N
      Test:  months N+1 to N+test_months

    Roll forward by step_months each iteration.

    Returns list of (train_indices, test_indices) tuples.
    """
    if df.empty or "signal_date" not in df.columns:
        return []

    df = df.copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df.sort_values("signal_date").reset_index(drop=True)

    start = df["signal_date"].min()
    end = df["signal_date"].max()
    splits = []

    train_end = start + pd.DateOffset(months=train_months)

    while train_end + pd.DateOffset(months=test_months) <= end:
        test_end = train_end + pd.DateOffset(months=test_months)
        train_idx = df[df["signal_date"] < train_end].index.tolist()
        test_idx = df[
            (df["signal_date"] >= train_end) & (df["signal_date"] < test_end)
        ].index.tolist()

        if len(train_idx) >= 20 and len(test_idx) >= 5:
            splits.append((train_idx, test_idx))

        train_end += pd.DateOffset(months=step_months)

    return splits


# ─── Quick stats ──────────────────────────────────────────────────────────────


def dataset_summary(df: pd.DataFrame) -> dict:
    """Returns summary statistics for the training dataset."""
    if df.empty:
        return {"status": "no data"}

    total = len(df)
    wins = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()
    times = (df["outcome"] == "TIME").sum()
    win_rate = round(wins / total * 100, 1) if total > 0 else 0

    date_range = ""
    if "signal_date" in df.columns:
        date_range = f"{df['signal_date'].min()} to {df['signal_date'].max()}"

    return {
        "total": total,
        "wins": int(wins),
        "losses": int(losses),
        "times": int(times),
        "win_rate": win_rate,
        "date_range": date_range,
        "ml_ready": total >= 50,
        "ml_reliable": total >= 200,
        "status": "ready" if total >= 50 else f"need {50 - total} more samples",
    }
