"""
tests/test_ml.py
=================
Tests for the ML module (collector, dataset, trainer, predictor, evaluator).
Run with: pytest tests/ -v

These tests use synthetic data so they run without network access
and without needing real trade history.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta


# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_synthetic_df(n: int = 60, win_rate: float = 0.5) -> pd.DataFrame:
    """
    Creates a synthetic ml_signal_log DataFrame for testing.
    All required columns present with realistic values.
    """
    np.random.seed(42)
    today = date.today()
    dates = [(today - timedelta(days=n - i)).isoformat() for i in range(n)]
    outcomes = ["WIN" if np.random.random() < win_rate else "LOSS" for _ in range(n)]

    return pd.DataFrame(
        {
            "id": range(n),
            "symbol": ["RELIANCE.NS"] * n,
            "signal_date": dates,
            "mode": ["LIVE"] * n,
            "outcome": outcomes,
            "pnl": np.where(
                np.array(outcomes) == "WIN",
                np.random.uniform(300, 700, n),
                np.random.uniform(-350, -100, n),
            ),
            # Technical
            "rsi": np.random.uniform(60, 70, n),
            "adx": np.random.uniform(25, 40, n),
            "volume_ratio": np.random.uniform(2.0, 4.0, n),
            "breakout_pct": np.random.uniform(0.5, 3.0, n),
            "vol_buildup_score": np.random.randint(0, 11, n),
            "vol_buildup_days": np.random.randint(0, 6, n),
            "consolidation_score": np.random.randint(0, 11, n),
            "consolidation_ratio": np.random.uniform(0.2, 0.8, n),
            "rel_strength_score": np.random.randint(0, 11, n),
            "rel_strength_delta": np.random.uniform(-5, 15, n),
            # Company sentiment
            "sentiment_verdict": np.random.choice(["GREEN", "AMBER", "RED"], n),
            "sentiment_avg_score": np.random.uniform(-0.5, 0.8, n),
            "analyst_action": np.random.choice([-1, 0, 1], n),
            "days_to_earnings": np.random.randint(5, 90, n),
            # Macro
            "india_vix": np.random.uniform(12, 28, n),
            "vix_5d_change": np.random.uniform(-3, 5, n),
            "geopolitical_risk": np.random.uniform(0.1, 0.8, n),
            "rbi_days": np.random.randint(5, 60, n),
            "dxy_10d_change": np.random.uniform(-2, 3, n),
            "oil_10d_change": np.random.uniform(-5, 8, n),
            "overnight_composite": np.random.uniform(-2, 2, n),
            "macro_score": np.random.randint(30, 80, n),
            # Smart money
            "fii_net_5d": np.random.uniform(-500, 800, n),
            "fii_trend": np.random.choice(["BUYING", "MIXED", "SELLING"], n),
            "delivery_pct": np.random.uniform(30, 80, n),
            "promoter_buying": np.random.randint(0, 2, n),
            "insider_veto": np.zeros(n, dtype=int),
            # Options
            "pcr": np.random.uniform(0.5, 1.5, n),
            "unusual_call_oi": np.random.randint(0, 2, n),
            "iv_risk": np.zeros(n, dtype=int),
            # Context
            "regime": np.random.choice(["BULL", "CHOP"], n),
            "sector_trending": np.ones(n, dtype=int),
            "quality_score": np.random.randint(40, 90, n),
            "quality_grade": np.random.choice(["FAIR", "GOOD", "STRONG"], n),
            "position_multiplier": np.random.uniform(0.75, 1.0, n),
        }
    )


# ─── Collector Tests ──────────────────────────────────────────────────────────


class TestCollector:

    def setup_method(self):
        from ml import collector

        self.col = collector

    def test_init_table_runs_without_error(self):
        """init_ml_table should run without error."""
        self.col.init_ml_table()

    def test_get_sample_count_returns_dict(self):
        """get_sample_count must return a dict with required keys."""
        result = self.col.get_sample_count()
        assert isinstance(result, dict)
        for key in ["total", "labelled", "ml_ready", "ml_reliable"]:
            assert key in result

    def test_get_sample_count_types(self):
        """Sample count values must be correct types."""
        result = self.col.get_sample_count()
        assert isinstance(result["total"], int)
        assert isinstance(result["labelled"], int)
        assert isinstance(result["ml_ready"], bool)
        assert isinstance(result["ml_reliable"], bool)

    def test_ml_ready_threshold(self):
        """ml_ready should be True only when labelled >= 50."""
        result = self.col.get_sample_count()
        if result["labelled"] >= 50:
            assert result["ml_ready"] is True
        else:
            assert result["ml_ready"] is False

    def test_ml_reliable_threshold(self):
        """ml_reliable should be True only when labelled >= 200."""
        result = self.col.get_sample_count()
        if result["labelled"] >= 200:
            assert result["ml_reliable"] is True
        else:
            assert result["ml_reliable"] is False


# ─── Dataset Tests ────────────────────────────────────────────────────────────


class TestDataset:

    def setup_method(self):
        from ml import dataset

        self.ds = dataset

    def test_prepare_returns_correct_types(self):
        """prepare() must return arrays of correct types."""
        df = make_synthetic_df(60)
        X, y, weights, features = self.ds.prepare(df)
        assert X is not None
        assert y is not None
        assert weights is not None
        assert isinstance(features, list)

    def test_prepare_shapes_match(self):
        """X, y, and weights must have same number of rows."""
        df = make_synthetic_df(60)
        X, y, weights, features = self.ds.prepare(df)
        assert X.shape[0] == len(y) == len(weights)

    def test_prepare_no_nan(self):
        """prepare() must fill all NaN values."""
        df = make_synthetic_df(60)
        # Introduce some NaN values
        df.loc[0:5, "rsi"] = np.nan
        df.loc[10:15, "pcr"] = np.nan
        X, y, weights, _ = self.ds.prepare(df)
        assert not np.isnan(X).any()

    def test_prepare_binary_labels(self):
        """Labels must be binary (0 or 1)."""
        df = make_synthetic_df(60)
        _, y, _, _ = self.ds.prepare(df)
        assert set(y).issubset({0, 1})

    def test_prepare_win_label(self):
        """WIN outcome must map to 1."""
        df = make_synthetic_df(10)
        df["outcome"] = "WIN"
        _, y, _, _ = self.ds.prepare(df)
        assert all(y == 1)

    def test_prepare_loss_label(self):
        """LOSS outcome must map to 0."""
        df = make_synthetic_df(10)
        df["outcome"] = "LOSS"
        _, y, _, _ = self.ds.prepare(df)
        assert all(y == 0)

    def test_prepare_time_label(self):
        """TIME outcome must map to 0 (treated as loss)."""
        df = make_synthetic_df(10)
        df["outcome"] = "TIME"
        _, y, _, _ = self.ds.prepare(df)
        assert all(y == 0)

    def test_prepare_empty_df(self):
        """prepare() must handle empty DataFrame gracefully."""
        X, y, weights, features = self.ds.prepare(pd.DataFrame())
        assert X is None
        assert y is None

    def test_sample_weights_recent_higher(self):
        """Recent trades must have higher weights than old ones."""
        df = make_synthetic_df(60)
        _, _, weights, _ = self.ds.prepare(df, half_life_days=30)
        # Last 10 trades should have higher avg weight than first 10
        assert weights[-10:].mean() > weights[:10].mean()

    def test_categorical_encoding(self):
        """Categorical features must be numerically encoded."""
        df = make_synthetic_df(20)
        df["sentiment_verdict"] = "GREEN"
        df["fii_trend"] = "BUYING"
        df["regime"] = "BULL"
        X, _, _, features = self.ds.prepare(df)
        # Should not raise — categoricals properly encoded
        assert X is not None

    def test_walk_forward_splits_temporal_order(self):
        """Walk-forward splits must preserve temporal order."""
        df = make_synthetic_df(120)
        splits = self.ds.walk_forward_splits(df, train_months=3, test_months=2)
        if splits:
            for train_idx, test_idx in splits:
                # All train dates must be before test dates
                train_dates = df.iloc[train_idx]["signal_date"].values
                test_dates = df.iloc[test_idx]["signal_date"].values
                assert max(train_dates) <= min(test_dates)

    def test_walk_forward_no_data_overlap(self):
        """Train and test indices must not overlap."""
        df = make_synthetic_df(120)
        splits = self.ds.walk_forward_splits(df, train_months=3, test_months=2)
        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_dataset_summary_structure(self):
        """dataset_summary must return all required keys."""
        df = make_synthetic_df(60)
        summary = self.ds.dataset_summary(df)
        for key in ["total", "wins", "losses", "win_rate", "ml_ready"]:
            assert key in summary

    def test_dataset_summary_win_rate_calculation(self):
        """Win rate must match actual win count / total."""
        df = make_synthetic_df(100, win_rate=0.6)
        summary = self.ds.dataset_summary(df)
        assert 55 <= summary["win_rate"] <= 65  # ~60% with random seed variance

    def test_dataset_summary_empty(self):
        """dataset_summary must handle empty DataFrame."""
        summary = self.ds.dataset_summary(pd.DataFrame())
        assert summary.get("status") == "no data"

    def test_ml_ready_at_50_samples(self):
        """ml_ready must be True at exactly 50 samples."""
        df = make_synthetic_df(50)
        summary = self.ds.dataset_summary(df)
        assert summary["ml_ready"] is True

    def test_ml_not_ready_below_50(self):
        """ml_ready must be False below 50 samples."""
        df = make_synthetic_df(30)
        summary = self.ds.dataset_summary(df)
        assert summary["ml_ready"] is False


# ─── Trainer Tests ────────────────────────────────────────────────────────────


class TestTrainer:

    def setup_method(self):
        from ml import trainer

        self.tr = trainer

    def test_train_bootstrap_returns_model(self):
        """train_bootstrap must return a fitted model."""
        df = make_synthetic_df(60)
        from ml.dataset import prepare

        X, y, weights, _ = prepare(df)
        model = self.tr.train_bootstrap(X, y, weights, n_estimators=10)
        assert model is not None
        assert hasattr(model, "predict_proba")

    def test_train_single_returns_model(self):
        """train_single must return a fitted model."""
        df = make_synthetic_df(200)
        from ml.dataset import prepare

        X, y, weights, _ = prepare(df)
        model = self.tr.train_single(X, y, weights)
        assert model is not None
        assert hasattr(model, "predict_proba")

    def test_bootstrap_predict_proba_shape(self):
        """Bootstrap model predict_proba must return (n, 2) array."""
        df = make_synthetic_df(60)
        from ml.dataset import prepare

        X, y, weights, _ = prepare(df)
        model = self.tr.train_bootstrap(X, y, weights, n_estimators=5)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)

    def test_probabilities_sum_to_one(self):
        """All predicted probabilities must sum to 1.0 per row."""
        df = make_synthetic_df(60)
        from ml.dataset import prepare

        X, y, weights, _ = prepare(df)
        model = self.tr.train_bootstrap(X, y, weights, n_estimators=5)
        probs = model.predict_proba(X)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_probabilities_in_range(self):
        """All predicted probabilities must be between 0 and 1."""
        df = make_synthetic_df(60)
        from ml.dataset import prepare

        X, y, weights, _ = prepare(df)
        model = self.tr.train_bootstrap(X, y, weights, n_estimators=5)
        probs = model.predict_proba(X)[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_evaluate_model_returns_dict(self):
        """evaluate_model must return dict with required keys."""
        df = make_synthetic_df(60)
        from ml.dataset import prepare

        X, y, weights, _ = prepare(df)
        model = self.tr.train_bootstrap(X, y, weights, n_estimators=5)
        result = self.tr.evaluate_model(model, X[:20], y[:20])
        for key in ["accuracy", "win_rate", "n_samples"]:
            assert key in result

    def test_evaluate_model_empty_test(self):
        """evaluate_model must handle empty test set."""
        df = make_synthetic_df(60)
        from ml.dataset import prepare

        X, y, weights, _ = prepare(df)
        model = self.tr.train_bootstrap(X, y, weights, n_estimators=5)
        result = self.tr.evaluate_model(
            model, np.array([]).reshape(0, X.shape[1]), np.array([])
        )
        assert result["n_samples"] == 0

    def test_train_and_validate_insufficient_data(self):
        """train_and_validate must return no-deploy for insufficient data."""
        result = self.tr.train_and_validate(min_samples=10000)
        assert result["deploy"] is False
        assert "message" in result

    def test_save_and_load_model(self):
        """Save and load model must be consistent."""
        df = make_synthetic_df(60)
        from ml.dataset import prepare

        X, y, weights, features = prepare(df)
        model = self.tr.train_bootstrap(X, y, weights, n_estimators=5)
        self.tr.save_model(model, features, {"test": True})

        loaded_model, loaded_features = self.tr.load_model()
        assert loaded_model is not None
        assert loaded_features == features

    def test_loaded_model_predicts_same(self):
        """Loaded model must produce same predictions as original."""
        df = make_synthetic_df(60)
        from ml.dataset import prepare

        X, y, weights, features = prepare(df)
        model = self.tr.train_bootstrap(X, y, weights, n_estimators=5)
        self.tr.save_model(model, features)

        loaded_model, _ = self.tr.load_model()
        original_preds = model.predict_proba(X[:5])
        loaded_preds = loaded_model.predict_proba(X[:5])
        assert np.allclose(original_preds, loaded_preds)

    def test_model_metadata_structure(self):
        """Model metadata must contain required keys."""
        meta = self.tr.get_model_metadata()
        assert isinstance(meta, dict)
        # Either has real metadata or status message
        assert "trained_at" in meta or "status" in meta


# ─── Predictor Tests ──────────────────────────────────────────────────────────


class TestPredictor:

    def setup_method(self):
        from ml import predictor

        self.pr = predictor

    def _make_signal(self):
        return {
            "symbol": "RELIANCE.NS",
            "price": 1250.0,
            "prev_high": 1238.0,
            "rsi": 63.2,
            "adx": 28.4,
            "volume_ratio": 2.6,
            "sector": "^NSEI",
        }

    def _make_score_result(self):
        return {
            "score": 72,
            "grade": "GOOD",
            "position_multiplier": 0.75,
            "macro_verdict": "AMBER",
            "modules": {
                "volume_intelligence": {
                    "signals": {
                        "volume_buildup": {"score": 6},
                        "consolidation": {"score": 8},
                        "relative_strength": {"score": 6},
                    }
                },
                "company_sentiment": {
                    "verdict": "GREEN",
                    "signals": {
                        "headlines": {"avg_score": 0.3},
                        "earnings": {"days_to_earnings": 45},
                        "analyst": {"action": 0},
                        "announcements": {"verdict": "GREEN"},
                    },
                },
                "macro_sentiment": {
                    "macro_score": 58,
                    "signals": {
                        "india_vix": {"current": 18.3, "change_5d": 1.2},
                        "geopolitical": {"avg_tone": -4.2},
                        "central_bank": {"rbi_days": 12},
                        "dollar_oil": {"dxy_change": 0.8, "oil_change": 2.1},
                        "overnight": {"composite": -0.3},
                    },
                },
                "smart_money": {
                    "signals": {
                        "fii_dii": {"raw": {"fii_net_5d": 342, "fii_trend": "BUYING"}},
                        "delivery": {"raw": {"delivery_pct_today": 72.5}},
                        "insider": {"raw": {"promoter_buying": False, "veto": False}},
                        "block_deal": {"raw": {}},
                    }
                },
                "options_signal": {
                    "signals": {
                        "pcr": {"pcr": 0.82},
                        "unusual_oi": {"unusual": False, "iv_risk": False},
                    }
                },
            },
        }

    def test_predict_no_model_returns_no_model(self):
        """predict() without a trained model must return NO_MODEL."""
        # Only if no model has been saved yet
        model, _ = self.pr.load_model()
        if model is None:
            result = self.pr.predict(self._make_signal(), self._make_score_result())
            assert result["recommendation"] == "NO_MODEL"
            assert result["model_available"] is False
        else:
            pytest.skip("Model already trained — skip no-model test")

    def test_predict_returns_required_keys(self):
        """predict() must always return all required keys."""
        result = self.pr.predict(self._make_signal(), self._make_score_result())
        for key in [
            "p_win",
            "recommendation",
            "confidence",
            "model_available",
            "shadow_mode",
            "detail",
        ]:
            assert key in result

    def test_predict_shadow_mode_always_true(self):
        """shadow_mode must always be True until explicitly disabled."""
        result = self.pr.predict(self._make_signal(), self._make_score_result())
        assert result["shadow_mode"] is True

    def test_predict_p_win_in_range(self):
        """p_win must be between 0 and 1 when model available."""
        result = self.pr.predict(self._make_signal(), self._make_score_result())
        if result["p_win"] is not None:
            assert 0.0 <= result["p_win"] <= 1.0

    def test_predict_recommendation_valid(self):
        """recommendation must be one of the valid values."""
        result = self.pr.predict(self._make_signal(), self._make_score_result())
        assert result["recommendation"] in [
            "STRONG_BUY",
            "BUY",
            "NEUTRAL",
            "SKIP",
            "STRONG_SKIP",
            "NO_MODEL",
            "ERROR",
        ]

    def test_build_feature_vector_correct_length(self):
        """Feature vector must match expected feature count."""
        model, features = self.pr.load_model()
        if model is None:
            pytest.skip("No model trained yet")
        vector = self.pr._build_feature_vector(
            self._make_signal(), self._make_score_result(), features
        )
        assert vector is not None
        assert len(vector) == len(features)

    def test_get_feature_importance_returns_dict(self):
        """get_feature_importance must return a dict."""
        result = self.pr.get_feature_importance()
        assert isinstance(result, dict)

    def test_get_feature_importance_values_in_range(self):
        """Feature importance values must be non-negative."""
        result = self.pr.get_feature_importance()
        for val in result.values():
            assert float(val) >= 0


# ─── Evaluator Tests ──────────────────────────────────────────────────────────


class TestEvaluator:

    def setup_method(self):
        from ml import evaluator

        self.ev = evaluator

    def test_validation_insufficient_data(self):
        """Validation with no data must return INSUFFICIENT_DATA."""
        result = self.ev.run_walk_forward_validation(min_samples=10000)
        assert result["deploy_recommendation"] == "INSUFFICIENT_DATA"
        assert result["pass"] is False

    def test_full_report_structure(self):
        """full_report must return all required sections."""
        result = self.ev.full_report()
        for key in [
            "data_summary",
            "model_metadata",
            "validation",
            "feature_importance",
        ]:
            assert key in result

    def test_feature_importance_report_structure(self):
        """feature_importance_report must return required keys."""
        result = self.ev.feature_importance_report()
        assert isinstance(result, dict)
        assert "status" in result

    def test_validation_with_synthetic_data(self):
        """Walk-forward validation must run cleanly on synthetic data."""
        from unittest.mock import patch

        df = make_synthetic_df(120)

        # Mock load_raw to return our synthetic data
        with patch("ml.evaluator.load_raw", return_value=df):
            result = self.ev.run_walk_forward_validation(min_samples=10)

        assert "pass" in result
        assert "deploy_recommendation" in result

    def test_validation_windows_have_no_future_leak(self):
        """Each validation window must not use future data in training."""
        from ml.dataset import walk_forward_splits

        df = make_synthetic_df(120)
        splits = walk_forward_splits(df, train_months=3, test_months=2)

        for train_idx, test_idx in splits:
            train_dates = df.iloc[train_idx]["signal_date"].values
            test_dates = df.iloc[test_idx]["signal_date"].values
            if len(train_dates) > 0 and len(test_dates) > 0:
                assert max(train_dates) <= min(
                    test_dates
                ), "Future data leaked into training set!"
