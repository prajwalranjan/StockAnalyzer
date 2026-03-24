"""
ml/evaluator.py — Model Evaluator
===================================
Walk-forward validation and performance reporting.

Run this before deploying any model to production.
The model is only deployed if it beats the current baseline
by 2%+ on the holdout set across multiple windows.

Also provides feature importance analysis — tells you which
signals are actually predicting outcomes in Indian markets.
"""

import json
import numpy as np
from ml.dataset import load_raw, prepare, walk_forward_splits, dataset_summary
from ml.trainer import train_bootstrap, train_single, evaluate_model, load_model
from ml.predictor import get_feature_importance


def run_walk_forward_validation(min_samples: int = 50) -> dict:
    """
    Runs walk-forward validation on all available data.

    Returns a full validation report:
    {
        "windows":    [...],     # per-window results
        "avg_win_rate": 57.2,
        "baseline":   47.0,
        "improvement": 10.2,
        "pass":       True,      # True if consistently beats baseline
        "deploy_recommendation": "DEPLOY" / "WAIT" / "INSUFFICIENT_DATA"
    }
    """
    df = load_raw(min_samples=min_samples)
    if df.empty:
        return {
            "pass": False,
            "deploy_recommendation": "INSUFFICIENT_DATA",
            "message": f"Need {min_samples} labelled samples. Collect more trade data.",
        }

    summary = dataset_summary(df)
    baseline = summary.get("win_rate", 47.0)
    splits = walk_forward_splits(df, train_months=6, test_months=3)

    if not splits:
        return {
            "pass": False,
            "deploy_recommendation": "INSUFFICIENT_DATA",
            "message": "Not enough time range for walk-forward validation.",
        }

    window_results = []

    for i, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train, y_train, w_train, features = prepare(train_df)
        X_test, y_test, _, _ = prepare(test_df)

        if X_train is None or len(X_train) < 20 or len(X_test) < 5:
            continue

        n = len(train_df)
        model = (
            train_bootstrap(X_train, y_train, w_train)
            if n < 200
            else train_single(X_train, y_train, w_train)
        )

        metrics = evaluate_model(model, X_test, y_test)

        window_results.append(
            {
                "window": i + 1,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "win_rate": metrics["win_rate"],
                "accuracy": metrics["accuracy"],
                "beats_baseline": metrics["win_rate"] > baseline + 2.0,
            }
        )

    if not window_results:
        return {
            "pass": False,
            "deploy_recommendation": "INSUFFICIENT_DATA",
            "message": "No valid walk-forward windows found.",
        }

    avg_win_rate = round(
        sum(w["win_rate"] for w in window_results) / len(window_results), 1
    )
    windows_passing = sum(1 for w in window_results if w["beats_baseline"])
    pass_rate = windows_passing / len(window_results)

    # Pass criteria: beats baseline in majority of windows
    passes = pass_rate >= 0.6 and avg_win_rate > baseline + 2.0

    if passes:
        deploy_rec = "DEPLOY"
    elif avg_win_rate > baseline:
        deploy_rec = "MARGINAL — monitor more"
    else:
        deploy_rec = "WAIT — not consistently beating baseline"

    return {
        "windows": window_results,
        "n_windows": len(window_results),
        "windows_passing": windows_passing,
        "avg_win_rate": avg_win_rate,
        "baseline": baseline,
        "improvement": round(avg_win_rate - baseline, 1),
        "pass": passes,
        "pass_rate": round(pass_rate * 100, 1),
        "deploy_recommendation": deploy_rec,
        "total_samples": summary["total"],
    }


def feature_importance_report() -> dict:
    """
    Returns feature importance ranked by predictive power.
    Shows which signals are actually working in Indian markets.
    """
    importance = get_feature_importance()
    if not importance:
        return {
            "status": "no model trained yet",
            "message": "Train a model first using trainer.train_and_validate()",
        }

    top_10 = list(importance.items())[:10]
    bottom_5 = list(importance.items())[-5:]

    return {
        "status": "ok",
        "top_signals": [
            {"feature": k, "importance": round(float(v), 4)} for k, v in top_10
        ],
        "weak_signals": [
            {"feature": k, "importance": round(float(v), 4)} for k, v in bottom_5
        ],
        "total_features": len(importance),
        "interpretation": (
            "Features with high importance are strong predictors. "
            "Features with near-zero importance add noise — consider removing them."
        ),
    }


def full_report() -> dict:
    """
    Generates a complete evaluation report.
    Called from the /api/ml/report endpoint.
    """
    df = load_raw(min_samples=1)
    summary = dataset_summary(df)

    validation = run_walk_forward_validation()
    importance = feature_importance_report()

    from ml.trainer import get_model_metadata

    model_meta = get_model_metadata()

    return {
        "data_summary": summary,
        "model_metadata": model_meta,
        "validation": validation,
        "feature_importance": importance,
    }
