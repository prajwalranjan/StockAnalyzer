"""
ml/predictor.py — ML Predictor
================================
Loads the trained model and predicts win probability for new signals.

Currently in shadow mode — predictions are logged and shown on the
dashboard but do NOT block or filter trades.

Activation threshold (once validated):
  P(WIN) >= 0.65 → strong buy recommendation
  P(WIN) 0.50-0.65 → proceed with caution
  P(WIN) < 0.50 → model disagrees — review manually

The predictor gracefully returns None if no model is trained yet,
so the rest of the pipeline is unaffected during the data collection phase.
"""

import numpy as np
from ml.trainer import load_model
from ml.dataset import ENCODINGS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES


def predict(signal: dict, score_result: dict) -> dict:
    """
    Predicts win probability for a signal using the trained model.

    Returns:
    {
        "p_win":          0.72,         # probability 0.0-1.0
        "recommendation": "STRONG_BUY", # STRONG_BUY/BUY/NEUTRAL/SKIP
        "confidence":     "HIGH",       # HIGH/MEDIUM/LOW
        "model_available": True,
        "shadow_mode":    True,         # True until model validated
        "detail":         "..."
    }
    """
    model, feature_names = load_model()

    if model is None:
        return {
            "p_win": None,
            "recommendation": "NO_MODEL",
            "confidence": "NONE",
            "model_available": False,
            "shadow_mode": True,
            "detail": "ML model not yet trained — need 50+ labelled trades",
        }

    # Build feature vector from signal + score_result
    feature_vector = _build_feature_vector(signal, score_result, feature_names)

    if feature_vector is None:
        return {
            "p_win": None,
            "recommendation": "ERROR",
            "confidence": "NONE",
            "model_available": True,
            "shadow_mode": True,
            "detail": "Feature extraction failed",
        }

    try:
        X = np.array([feature_vector])
        p_win = float(model.predict_proba(X)[0][1])
        p_win = round(p_win, 3)
    except Exception as e:
        return {
            "p_win": None,
            "recommendation": "ERROR",
            "confidence": "NONE",
            "model_available": True,
            "shadow_mode": True,
            "detail": f"Prediction failed: {e}",
        }

    # Recommendation thresholds
    if p_win >= 0.70:
        recommendation = "STRONG_BUY"
        confidence = "HIGH"
        detail = f"Model strongly favours this trade (P(WIN)={p_win:.0%})"
    elif p_win >= 0.60:
        recommendation = "BUY"
        confidence = "MEDIUM"
        detail = f"Model favours this trade (P(WIN)={p_win:.0%})"
    elif p_win >= 0.50:
        recommendation = "NEUTRAL"
        confidence = "LOW"
        detail = f"Model is neutral (P(WIN)={p_win:.0%})"
    elif p_win >= 0.40:
        recommendation = "SKIP"
        confidence = "MEDIUM"
        detail = f"Model leans against this trade (P(WIN)={p_win:.0%})"
    else:
        recommendation = "STRONG_SKIP"
        confidence = "HIGH"
        detail = f"Model strongly against this trade (P(WIN)={p_win:.0%})"

    return {
        "p_win": p_win,
        "recommendation": recommendation,
        "confidence": confidence,
        "model_available": True,
        "shadow_mode": True,  # flip to False once model validated
        "detail": detail,
    }


def _build_feature_vector(
    signal: dict, score_result: dict, feature_names: list
) -> list:
    """
    Builds a feature vector from signal + score_result dict,
    matching the exact feature order the model was trained on.
    """
    try:
        modules = score_result.get("modules", {})
        vi = modules.get("volume_intelligence", {})
        cs = modules.get("company_sentiment", {})
        ms = modules.get("macro_sentiment", {})
        sm = modules.get("smart_money", {})
        opts = modules.get("options_signal", {})

        vi_sigs = vi.get("signals", vi)
        cs_sigs = cs.get("signals", {})
        ms_sigs = ms.get("signals", {})
        sm_sigs = sm.get("signals", {})
        opt_sigs = opts.get("signals", {})

        vb = vi_sigs.get("volume_buildup", {})
        co = vi_sigs.get("consolidation", {})
        rs = vi_sigs.get("relative_strength", {})
        hl = cs_sigs.get("headlines", {})
        ea = cs_sigs.get("earnings", {})
        an = cs_sigs.get("analyst", {})

        vix_sig = ms_sigs.get("india_vix", {})
        geo_sig = ms_sigs.get("geopolitical", {})
        cb_sig = ms_sigs.get("central_bank", {})
        do_sig = ms_sigs.get("dollar_oil", {})
        og_sig = ms_sigs.get("overnight", {})

        fii_raw = sm_sigs.get("fii_dii", {}).get("raw", {})
        del_raw = sm_sigs.get("delivery", {}).get("raw", {})
        ins_raw = sm_sigs.get("insider", {}).get("raw", {})
        pcr_sig = opt_sigs.get("pcr", {})
        uoi_sig = opt_sigs.get("unusual_oi", {})

        price = signal.get("price", 0)
        prev_high = signal.get("prev_high", price)
        breakout_pct = (
            round((price - prev_high) / prev_high * 100, 3) if prev_high else 0
        )

        raw_features = {
            "rsi": signal.get("rsi"),
            "adx": signal.get("adx"),
            "volume_ratio": signal.get("volume_ratio"),
            "breakout_pct": breakout_pct,
            "vol_buildup_score": vb.get("score"),
            "vol_buildup_days": vi.get("volume_buildup_days"),
            "consolidation_score": co.get("score"),
            "consolidation_ratio": co.get("consolidation_ratio"),
            "rel_strength_score": rs.get("score"),
            "rel_strength_delta": rs.get("delta"),
            "sentiment_avg_score": hl.get("avg_score"),
            "analyst_action": an.get("action"),
            "days_to_earnings": ea.get("days_to_earnings"),
            "india_vix": vix_sig.get("current"),
            "vix_5d_change": vix_sig.get("change_5d"),
            "geopolitical_risk": geo_sig.get("avg_tone"),
            "rbi_days": cb_sig.get("rbi_days"),
            "dxy_10d_change": do_sig.get("dxy_change"),
            "oil_10d_change": do_sig.get("oil_change"),
            "overnight_composite": og_sig.get("composite"),
            "macro_score": ms.get("macro_score"),
            "fii_net_5d": fii_raw.get("fii_net_5d"),
            "delivery_pct": del_raw.get("delivery_pct_today"),
            "promoter_buying": int(ins_raw.get("promoter_buying", False)),
            "insider_veto": int(ins_raw.get("veto", False)),
            "pcr": pcr_sig.get("pcr"),
            "unusual_call_oi": int(uoi_sig.get("unusual", False)),
            "iv_risk": int(uoi_sig.get("iv_risk", False)),
            "sector_trending": 1,
            "quality_score": score_result.get("score"),
            "position_multiplier": score_result.get("position_multiplier"),
            # Categorical (encoded)
            "sentiment_verdict": ENCODINGS["sentiment_verdict"].get(
                cs.get("verdict", "GREEN"), 1
            ),
            "fii_trend": ENCODINGS["fii_trend"].get(
                fii_raw.get("fii_trend", "MIXED"), 0
            ),
            "regime": ENCODINGS["regime"].get(signal.get("regime", "BULL"), 2),
        }

        # Build vector in exact feature_names order
        vector = []
        for fname in feature_names:
            val = raw_features.get(fname)
            vector.append(float(val) if val is not None else 0.0)

        return vector

    except Exception:
        return None


def get_feature_importance() -> dict:
    """Returns feature importance from the trained model."""
    model, features = load_model()
    if model is None:
        return {}

    try:
        # Works for single XGBoost model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        # Works for BaggingClassifier ensemble
        elif hasattr(model, "estimators_"):
            all_imp = np.array(
                [
                    e.feature_importances_
                    for e in model.estimators_
                    if hasattr(e, "feature_importances_")
                ]
            )
            importances = all_imp.mean(axis=0)
        else:
            return {}

        return dict(
            sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        )
    except Exception:
        return {}
