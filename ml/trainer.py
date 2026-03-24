"""
ml/trainer.py — Model Trainer
===============================
Trains an XGBoost classifier on the collected signal data.

Two training modes:
  bootstrap_ensemble — for <200 samples (safe, wide confidence)
  single_model       — for 200+ samples (more precise)

Walk-forward validation is mandatory before any model is deployed.
A model only replaces the current one if it beats it by 2%+ on holdout.

Usage:
  from ml.trainer import train_and_validate
  result = train_and_validate()
  if result["deploy"]:
      save_model(result["model"])
"""

import os
import pickle
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from ml.dataset import load_raw, prepare, walk_forward_splits, dataset_summary

MODEL_DIR = Path(os.path.dirname(__file__)) / ".." / "data" / "ml_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "current_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def train_bootstrap(X, y, weights, n_estimators: int = 100) -> object:
    """
    Bootstrap ensemble — 100 XGBoost models on random subsamples.
    More robust than single model on small datasets.
    Confidence interval narrows as data grows.

    Use when: n_samples < 200
    """
    from sklearn.ensemble import BaggingClassifier
    from xgboost import XGBClassifier

    base = XGBClassifier(
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )

    model = BaggingClassifier(
        estimator=base,
        n_estimators=n_estimators,
        max_samples=0.8,
        bootstrap=True,
        random_state=42,
    )

    model.fit(X, y, **{"sample_weight": weights} if weights is not None else {})
    return model


def train_single(X, y, weights) -> object:
    """
    Single XGBoost model — use when n_samples >= 200.
    Shallow tree depth prevents overfitting.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(
        max_depth=4,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )

    fit_params = {}
    if weights is not None:
        fit_params["sample_weight"] = weights

    model.fit(X, y, **fit_params)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluates model on a holdout set."""
    from sklearn.metrics import accuracy_score, classification_report

    if len(X_test) == 0:
        return {"win_rate": 0, "accuracy": 0, "n_samples": 0}

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    accuracy = round(accuracy_score(y_test, preds) * 100, 1)
    win_preds = preds[preds == 1]
    win_rate = round(y_test[preds == 1].mean() * 100, 1) if len(win_preds) > 0 else 0

    return {
        "accuracy": accuracy,
        "win_rate": win_rate,
        "n_samples": len(y_test),
        "n_predicted_wins": int(preds.sum()),
    }


def train_and_validate(min_samples: int = 50) -> dict:
    """
    Full training pipeline with walk-forward validation.

    Returns:
    {
        "deploy":     True/False,    # whether new model beats current
        "model":      model_object,
        "win_rate":   58.3,          # on holdout set
        "baseline":   47.2,          # current strategy win rate
        "n_samples":  120,
        "message":    "..."
    }
    """
    df = load_raw(min_samples=min_samples)
    if df.empty:
        return {
            "deploy": False,
            "message": f"Insufficient data. Need {min_samples} labelled samples.",
            "n_samples": 0,
        }

    summary = dataset_summary(df)
    n = summary["total"]

    # Walk-forward validation
    splits = walk_forward_splits(df, train_months=6, test_months=3)

    if not splits:
        # Not enough time range — use simple 80/20 split as fallback
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        splits = [(train_df.index.tolist(), test_df.index.tolist())]

    # Train on all data, validate on last walk-forward window
    train_idx, test_idx = splits[-1]
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train, y_train, w_train, features = prepare(train_df)
    X_test, y_test, _, _ = prepare(test_df)

    if X_train is None or len(X_train) < 20:
        return {"deploy": False, "message": "Insufficient training data after split."}

    # Choose model type based on sample count
    if n < 200:
        model = train_bootstrap(X_train, y_train, w_train)
        model_type = "bootstrap_ensemble"
    else:
        model = train_single(X_train, y_train, w_train)
        model_type = "single_xgboost"

    # Evaluate on holdout
    metrics = evaluate_model(model, X_test, y_test)

    # Current baseline win rate (from DB stats)
    baseline = summary.get("win_rate", 47.0)

    # Deploy only if meaningfully better
    improvement = metrics["win_rate"] - baseline
    deploy = improvement >= 2.0 and metrics["win_rate"] > 50.0

    return {
        "deploy": deploy,
        "model": model,
        "model_type": model_type,
        "win_rate": metrics["win_rate"],
        "baseline": baseline,
        "improvement": round(improvement, 1),
        "n_samples": n,
        "n_test": metrics["n_samples"],
        "features": features,
        "message": (
            f"Model trained. Win rate {metrics['win_rate']}% vs baseline {baseline}% "
            f"({'deploy' if deploy else 'not deploying — insufficient improvement'})"
        ),
    }


def save_model(model, feature_names: list, metadata: dict = None):
    """Saves model to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": feature_names}, f)

    meta = {
        "trained_at": datetime.now().isoformat(),
        "features": feature_names,
        **(metadata or {}),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_model() -> tuple:
    """
    Loads saved model from disk.
    Returns (model, feature_names) or (None, []) if no model saved yet.
    """
    if not MODEL_PATH.exists():
        return None, []
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["features"]
    except Exception:
        return None, []


def get_model_metadata() -> dict:
    """Returns metadata about the currently saved model."""
    if not METADATA_PATH.exists():
        return {"status": "no model trained yet"}
    try:
        with open(METADATA_PATH) as f:
            return json.load(f)
    except Exception:
        return {"status": "metadata unreadable"}
