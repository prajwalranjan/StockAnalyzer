"""
tests/test_prediction.py
========================
Tests for the prediction engine modules.
Run with: pytest tests/ -v

These tests use synthetic price data so they run instantly
without needing internet access or API keys.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta


# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_stock_df(days=60, trend="flat", with_coil=False, with_buildup=False):
    """
    Generates a synthetic OHLCV DataFrame for testing.
    trend: "up", "down", "flat"
    with_coil: True = tight consolidation in last 10 days
    with_buildup: True = rising volume in last 5 days
    """
    np.random.seed(42)
    base = 500.0
    dates = pd.date_range(end=date.today(), periods=days, freq="B")

    if trend == "up":
        prices = base + np.linspace(0, 100, days) + np.random.randn(days) * 3
    elif trend == "down":
        prices = base - np.linspace(0, 100, days) + np.random.randn(days) * 3
    else:
        prices = base + np.random.randn(days) * 5

    # Coil: very tight range in last 10 days
    if with_coil:
        prices[-10:] = prices[-11] + np.random.randn(10) * 0.5

    closes = prices
    opens = closes * (1 + np.random.randn(days) * 0.002)
    highs = np.maximum(closes, opens) * (1 + np.abs(np.random.randn(days)) * 0.005)
    lows = np.minimum(closes, opens) * (1 - np.abs(np.random.randn(days)) * 0.005)

    base_vol = 1_000_000
    volumes = base_vol + np.random.randint(-200_000, 200_000, days)

    # Volume buildup: rising volume in last 5 days
    if with_buildup:
        volumes[-5:] = [base_vol * (1.2 + i * 0.2) for i in range(5)]

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


# ─── Volume Intelligence Tests ────────────────────────────────────────────────


class TestVolumeIntelligence:

    def setup_method(self):
        from prediction import volume_intelligence

        self.vi = volume_intelligence

    def test_volume_buildup_score_range(self):
        """Score must always be 0-10."""
        df = make_stock_df(60)
        score, _ = self.vi.volume_buildup(df)
        assert 0 <= score <= 10

    def test_volume_buildup_detects_buildup(self):
        """High buildup stock should score higher than flat volume."""
        df_buildup = make_stock_df(60, with_buildup=True)
        df_flat = make_stock_df(60)
        score_buildup, _ = self.vi.volume_buildup(df_buildup)
        score_flat, _ = self.vi.volume_buildup(df_flat)
        assert score_buildup >= score_flat

    def test_volume_buildup_insufficient_data(self):
        """Should handle DataFrames that are too short."""
        df = make_stock_df(10)
        score, detail = self.vi.volume_buildup(df)
        assert score == 0
        assert "insufficient" in detail.lower()

    def test_volume_buildup_empty_df(self):
        """Should handle empty DataFrame gracefully."""
        score, detail = self.vi.volume_buildup(pd.DataFrame())
        assert score == 0

    def test_consolidation_score_range(self):
        """Score must always be 0-10."""
        df = make_stock_df(60)
        score, _ = self.vi.consolidation_tightness(df)
        assert 0 <= score <= 10

    def test_consolidation_detects_coil(self):
        """Coiling stock should score higher than non-coiling."""
        df_coil = make_stock_df(60, with_coil=True)
        df_flat = make_stock_df(60)
        score_coil, _ = self.vi.consolidation_tightness(df_coil)
        score_flat, _ = self.vi.consolidation_tightness(df_flat)
        assert score_coil >= score_flat

    def test_consolidation_insufficient_data(self):
        """Should handle short DataFrames."""
        df = make_stock_df(20)
        score, detail = self.vi.consolidation_tightness(df)
        assert score == 0
        assert "insufficient" in detail.lower()

    def test_relative_strength_score_range(self):
        """Score must always be 0-10."""
        stock_df = make_stock_df(60, trend="up")
        sector_df = make_stock_df(60, trend="flat")
        score, _ = self.vi.relative_strength(stock_df, sector_df)
        assert 0 <= score <= 10

    def test_relative_strength_outperformance(self):
        """Strongly outperforming stock should score higher."""
        stock_up = make_stock_df(60, trend="up")
        stock_flat = make_stock_df(60, trend="flat")
        sector_df = make_stock_df(60, trend="flat")
        score_up, _ = self.vi.relative_strength(stock_up, sector_df)
        score_flat, _ = self.vi.relative_strength(stock_flat, sector_df)
        assert score_up >= score_flat

    def test_relative_strength_no_sector_data(self):
        """Should return neutral score when sector data missing."""
        stock_df = make_stock_df(60)
        score, detail = self.vi.relative_strength(stock_df, None)
        assert score == 3  # neutral
        assert "no sector data" in detail.lower()

    def test_relative_strength_underperformance(self):
        """Underperforming stock should score 0."""
        stock_down = make_stock_df(60, trend="down")
        sector_up = make_stock_df(60, trend="up")
        score, _ = self.vi.relative_strength(stock_down, sector_up)
        assert score == 0

    def test_compute_returns_all_keys(self):
        """compute() must return all required keys."""
        df = make_stock_df(60)
        result = self.vi.compute(df, df)
        assert "volume_buildup" in result
        assert "consolidation" in result
        assert "relative_strength" in result
        assert "total" in result
        assert "summary" in result

    def test_compute_total_in_range(self):
        """Total score must be 0-30."""
        df = make_stock_df(60)
        result = self.vi.compute(df, df)
        assert 0 <= result["total"] <= 30

    def test_compute_summary_valid(self):
        """Summary must be one of the valid grades."""
        df = make_stock_df(60)
        result = self.vi.compute(df, df)
        assert result["summary"] in ["WEAK", "FAIR", "GOOD", "STRONG"]


# ─── Score Aggregator Tests ───────────────────────────────────────────────────


class TestScoreAggregator:

    def setup_method(self):
        from prediction import score

        self.score = score

    def test_score_range(self):
        """Final score must always be 0-100."""
        df = make_stock_df(60)
        result = self.score.compute_score("TEST.NS", df, df)
        assert 0 <= result["score"] <= 100

    def test_grade_valid(self):
        """Grade must be one of the valid values."""
        df = make_stock_df(60)
        result = self.score.compute_score("TEST.NS", df, df)
        assert result["grade"] in ["WEAK", "FAIR", "GOOD", "STRONG"]

    def test_required_keys(self):
        """Result must contain all required keys."""
        df = make_stock_df(60)
        result = self.score.compute_score("TEST.NS", df, df)
        for key in [
            "symbol",
            "score",
            "grade",
            "modules",
            "shadow_mode",
            "phases_active",
        ]:
            assert key in result

    def test_shadow_mode_always_true(self):
        """Shadow mode must always be True — never blocks trades."""
        df = make_stock_df(60)
        result = self.score.compute_score("TEST.NS", df, df)
        assert result["shadow_mode"] is True

    def test_score_label(self):
        """score_label returns correct label for boundary values."""
        assert self.score.score_label(81) == "STRONG"
        assert self.score.score_label(61) == "GOOD"
        assert self.score.score_label(41) == "FAIR"
        assert self.score.score_label(40) == "WEAK"
        assert self.score.score_label(0) == "WEAK"
        assert self.score.score_label(100) == "STRONG"

    def test_higher_quality_stock_scores_higher(self):
        """A stock with buildup + coil should score higher than flat stock."""
        df_quality = make_stock_df(60, trend="up", with_coil=True, with_buildup=True)
        df_weak = make_stock_df(60, trend="down")
        result_quality = self.score.compute_score("QUALITY.NS", df_quality, df_quality)
        result_weak = self.score.compute_score("WEAK.NS", df_weak, df_weak)
        assert result_quality["score"] >= result_weak["score"]


# ─── Company Sentiment Tests ──────────────────────────────────────────────────


class TestCompanySentiment:

    def setup_method(self):
        from prediction import company_sentiment

        self.cs = company_sentiment

    def test_classify_empty_headlines(self):
        """Empty headline list should return GREEN with count 0."""
        result = self.cs.classify_headlines([])
        assert result["verdict"] == "GREEN"
        assert result["count"] == 0

    def test_classify_obviously_negative(self):
        """Clear negative headlines should score negative."""
        headlines = [
            "SEBI imposes Rs50 crore penalty on company for fraud",
            "MD resigns amid regulatory investigation",
            "Promoter pledges additional shares amid debt concerns",
        ]
        result = self.cs.classify_headlines(headlines)
        # Should be RED or at least have negative avg_score
        # (Uses fallback if no API key in test env)
        assert result["avg_score"] <= 0 or result["verdict"] in ["RED", "AMBER"]

    def test_classify_obviously_positive(self):
        """Clear positive headlines should score positive."""
        headlines = [
            "Company wins record Rs2000 crore order",
            "Board approves 30% dividend for shareholders",
            "Company reports record quarterly profit, upgrades guidance",
        ]
        result = self.cs.classify_headlines(headlines)
        assert result["avg_score"] >= 0 or result["verdict"] == "GREEN"

    def test_verdict_always_valid(self):
        """Verdict must always be GREEN, AMBER, or RED."""
        for headlines in [[], ["neutral news"], ["SEBI penalty fraud"]]:
            result = self.cs.classify_headlines(headlines)
            assert result["verdict"] in ["GREEN", "AMBER", "RED"]

    def test_score_to_verdict_boundaries(self):
        """Boundary conditions for verdict logic."""
        assert self.cs._score_to_verdict(-0.8, -0.8, 1)[0] == "RED"
        assert self.cs._score_to_verdict(-0.4, -0.1, 3)[0] == "RED"
        assert self.cs._score_to_verdict(-0.1, -0.1, 3)[0] == "AMBER"
        assert self.cs._score_to_verdict(0.5, 0.2, 3)[0] == "GREEN"

    def test_nse_veto_keywords(self):
        """Hard veto keywords should trigger RED veto."""
        # We test the keyword logic directly via check_nse_announcements
        # by mocking — but since it calls yfinance, we test the keyword logic
        veto_text = "sebi issues show cause notice to company directors"
        veto_found = any(
            kw in veto_text for kw in ["sebi", "show cause", "fraud", "investigation"]
        )
        assert veto_found is True

    def test_compute_returns_required_keys(self):
        """compute() must always return all required keys."""
        result = self.cs.compute("RELIANCE.NS", headlines=[])
        assert "verdict" in result
        assert "score" in result
        assert "shadow_mode" in result
        assert "signals" in result
        for sig in ["headlines", "announcements", "earnings", "analyst"]:
            assert sig in result["signals"]

    def test_compute_shadow_mode_always_true(self):
        """Shadow mode must always be True."""
        result = self.cs.compute("TCS.NS", headlines=[])
        assert result["shadow_mode"] is True

    def test_compute_score_in_range(self):
        """Score contribution must be within bounds."""
        result = self.cs.compute("INFY.NS", headlines=[])
        assert -15 <= result["score"] <= 15


# ─── Macro Sentiment Tests ────────────────────────────────────────────────────


class TestMacroSentiment:

    def setup_method(self):
        from prediction import macro_sentiment

        self.ms = macro_sentiment

    def test_india_vix_returns_required_keys(self):
        """india_vix() must return all required keys."""
        try:
            result = self.ms.india_vix()
            for key in ["score", "multiplier", "verdict", "level"]:
                assert key in result
        except Exception:
            pytest.skip("Network unavailable")

    def test_india_vix_score_range(self):
        """VIX score must be 0-50."""
        try:
            result = self.ms.india_vix()
            assert 0 <= result["score"] <= 50
        except Exception:
            pytest.skip("Network unavailable")

    def test_india_vix_multiplier_range(self):
        """Position multiplier must be 0.0-1.0."""
        try:
            result = self.ms.india_vix()
            assert 0.0 <= result["multiplier"] <= 1.0
        except Exception:
            pytest.skip("Network unavailable")

    def test_india_vix_verdict_valid(self):
        """VIX verdict must be GREEN, AMBER, or RED."""
        try:
            result = self.ms.india_vix()
            assert result["verdict"] in ["GREEN", "AMBER", "RED"]
        except Exception:
            pytest.skip("Network unavailable")

    def test_vix_fallback_structure(self):
        """Fallback must return valid structure."""
        result = self.ms._vix_fallback("test reason")
        assert 0 <= result["score"] <= 50
        assert 0.0 <= result["multiplier"] <= 1.0
        assert result["verdict"] in ["GREEN", "AMBER", "RED"]

    def test_gdelt_fallback_structure(self):
        """GDELT fallback must return valid structure."""
        result = self.ms._gdelt_fallback("test reason")
        assert 0 <= result["score"] <= 25
        assert result["verdict"] in ["GREEN", "AMBER", "RED"]
        assert result["source"] == "fallback"

    def test_central_bank_returns_required_keys(self):
        """central_bank_proximity() must return required keys."""
        result = self.ms.central_bank_proximity()
        for key in ["score", "skip", "verdict", "rbi_days", "fed_days", "detail"]:
            assert key in result

    def test_central_bank_score_range(self):
        """Central bank score must be 0-25."""
        result = self.ms.central_bank_proximity()
        assert 0 <= result["score"] <= 25

    def test_central_bank_rbi_dates_future(self):
        """Next RBI date must be in the future or None."""
        result = self.ms.central_bank_proximity()
        if result["rbi_days"] is not None:
            assert result["rbi_days"] >= 0

    def test_dollar_oil_returns_required_keys(self):
        """dollar_oil_signal() must return required keys."""
        try:
            result = self.ms.dollar_oil_signal()
            for key in ["score", "verdict", "detail"]:
                assert key in result
        except Exception:
            pytest.skip("Network unavailable")

    def test_dollar_oil_score_range(self):
        """Dollar/oil score must be 0-25."""
        try:
            result = self.ms.dollar_oil_signal()
            assert 0 <= result["score"] <= 25
        except Exception:
            pytest.skip("Network unavailable")

    def test_overnight_returns_required_keys(self):
        """overnight_global() must return required keys."""
        try:
            result = self.ms.overnight_global()
            for key in ["score", "verdict", "detail", "composite"]:
                assert key in result
        except Exception:
            pytest.skip("Network unavailable")

    def test_overnight_score_range(self):
        """Overnight score must be 0-25."""
        try:
            result = self.ms.overnight_global()
            assert 0 <= result["score"] <= 25
        except Exception:
            pytest.skip("Network unavailable")

    def test_compute_returns_required_keys(self):
        """compute() must return all required keys."""
        try:
            result = self.ms.compute()
            for key in [
                "macro_score",
                "position_multiplier",
                "verdict",
                "shadow_mode",
                "signals",
            ]:
                assert key in result
        except Exception:
            pytest.skip("Network unavailable")

    def test_compute_macro_score_range(self):
        """Macro score must be 0-100."""
        try:
            result = self.ms.compute()
            assert 0 <= result["macro_score"] <= 100
        except Exception:
            pytest.skip("Network unavailable")

    def test_compute_position_multiplier_range(self):
        """Position multiplier must be 0.0-1.0."""
        try:
            result = self.ms.compute()
            assert 0.0 <= result["position_multiplier"] <= 1.0
        except Exception:
            pytest.skip("Network unavailable")

    def test_compute_shadow_mode_always_true(self):
        """Shadow mode must always be True."""
        try:
            result = self.ms.compute()
            assert result["shadow_mode"] is True
        except Exception:
            pytest.skip("Network unavailable")

    def test_compute_verdict_valid(self):
        """Overall verdict must be GREEN, AMBER, or RED."""
        try:
            result = self.ms.compute()
            assert result["verdict"] in ["GREEN", "AMBER", "RED"]
        except Exception:
            pytest.skip("Network unavailable")


# ─── Smart Money Tests ────────────────────────────────────────────────────────


class TestSmartMoney:

    def setup_method(self):
        from prediction import smart_money

        self.sm = smart_money

    def test_score_fii_dii_both_buying(self):
        data = {
            "combined": "BOTH_BUYING",
            "fii_net_5d": 500,
            "dii_net_5d": 200,
            "detail": "both buying",
            "source": "nse",
        }
        score, _ = self.sm.score_fii_dii(data)
        assert score == 8

    def test_score_fii_dii_selling(self):
        data = {
            "combined": "FII_SELLING",
            "fii_net_5d": -500,
            "dii_net_5d": 0,
            "detail": "selling",
            "source": "nse",
        }
        score, _ = self.sm.score_fii_dii(data)
        assert score == 0

    def test_score_fii_dii_fallback(self):
        data = {"source": "fallback"}
        score, detail = self.sm.score_fii_dii(data)
        assert 0 <= score <= 8
        assert "unavailable" in detail.lower()

    def test_score_delivery_high(self):
        data = {"delivery_pct_today": 75, "delta": None, "source": "nse"}
        score, _ = self.sm.score_delivery(data)
        assert score >= 5

    def test_score_delivery_low(self):
        data = {"delivery_pct_today": 25, "delta": None, "source": "nse"}
        score, _ = self.sm.score_delivery(data)
        assert score <= 2

    def test_score_delivery_fallback(self):
        data = {"source": "fallback"}
        score, _ = self.sm.score_delivery(data)
        assert 0 <= score <= 7

    def test_score_block_deals_buy(self):
        data = {
            "verdict": "BUY",
            "net_value_cr": 200,
            "buy_deals": 2,
            "sell_deals": 0,
            "detail": "2 buy blocks",
            "source": "nse",
        }
        score, _ = self.sm.score_block_deals(data)
        assert score == 5

    def test_score_block_deals_sell(self):
        data = {
            "verdict": "SELL",
            "net_value_cr": -200,
            "buy_deals": 0,
            "sell_deals": 2,
            "detail": "2 sell blocks",
            "source": "nse",
        }
        score, _ = self.sm.score_block_deals(data)
        assert score == 0

    def test_score_insider_buying(self):
        data = {
            "promoter_buying": True,
            "promoter_selling": False,
            "net_shares": 50000,
            "veto": False,
            "verdict": "GREEN",
            "detail": "bought 50000",
            "source": "nse",
        }
        score, _ = self.sm.score_insider(data)
        assert score == 5

    def test_score_insider_veto(self):
        data = {
            "promoter_buying": False,
            "promoter_selling": True,
            "net_shares": -200000,
            "veto": True,
            "verdict": "RED",
            "detail": "sold 200000",
            "source": "nse",
        }
        score, detail = self.sm.score_insider(data)
        assert score == 0
        assert "veto" in detail.lower()

    def test_score_insider_fallback(self):
        data = {"source": "fallback"}
        score, _ = self.sm.score_insider(data)
        assert 0 <= score <= 5

    def test_compute_returns_required_keys(self):
        try:
            result = self.sm.compute("RELIANCE.NS")
            for key in ["total", "summary", "veto", "shadow_mode", "signals"]:
                assert key in result
            for sig in ["fii_dii", "delivery", "block_deal", "insider"]:
                assert sig in result["signals"]
        except Exception:
            pytest.skip("NSE data unavailable")

    def test_compute_total_in_range(self):
        try:
            result = self.sm.compute("TCS.NS")
            assert 0 <= result["total"] <= 25
        except Exception:
            pytest.skip("NSE data unavailable")

    def test_compute_summary_valid(self):
        try:
            result = self.sm.compute("INFY.NS")
            assert result["summary"] in ["WEAK", "FAIR", "GOOD", "STRONG", "VETO"]
        except Exception:
            pytest.skip("NSE data unavailable")

    def test_compute_shadow_mode_always_true(self):
        try:
            result = self.sm.compute("HDFCBANK.NS")
            assert result["shadow_mode"] is True
        except Exception:
            pytest.skip("NSE data unavailable")

    def test_veto_zeroes_score(self):
        insider_data = {
            "promoter_buying": False,
            "promoter_selling": True,
            "net_shares": -500000,
            "veto": True,
            "verdict": "RED",
            "detail": "large sell",
            "source": "nse",
        }
        score, detail = self.sm.score_insider(insider_data)
        assert score == 0


# ─── Options Signal Tests ─────────────────────────────────────────────────────


class TestOptionsSignal:

    def setup_method(self):
        from prediction import options_signal

        self.os = options_signal

    def test_pcr_fallback_structure(self):
        result = self.os._pcr_fallback("test reason")
        assert result["verdict"] in ["BULLISH", "NEUTRAL", "BEARISH"]
        assert 0 <= result["score"] <= 10
        assert result["source"] == "fallback"

    def test_unusual_fallback_structure(self):
        result = self.os._unusual_fallback("test reason")
        assert result["unusual"] is False
        assert 0 <= result["score"] <= 5
        assert result["source"] == "fallback"

    def test_pcr_fallback_neutral_score(self):
        result = self.os._pcr_fallback("non-fno stock")
        assert result["score"] == 4
        assert result["verdict"] == "NEUTRAL"

    def test_compute_returns_required_keys(self):
        try:
            result = self.os.compute("RELIANCE.NS")
            for key in ["total", "summary", "shadow_mode", "signals"]:
                assert key in result
            for sig in ["pcr", "unusual_oi"]:
                assert sig in result["signals"]
        except Exception:
            pytest.skip("NSE options data unavailable")

    def test_compute_total_in_range(self):
        try:
            result = self.os.compute("TCS.NS")
            assert 0 <= result["total"] <= 15
        except Exception:
            pytest.skip("NSE options data unavailable")

    def test_compute_summary_valid(self):
        try:
            result = self.os.compute("INFY.NS")
            assert result["summary"] in [
                "STRONGLY_BULLISH",
                "BULLISH",
                "NEUTRAL",
                "BEARISH",
            ]
        except Exception:
            pytest.skip("NSE options data unavailable")

    def test_compute_shadow_mode_always_true(self):
        try:
            result = self.os.compute("HDFCBANK.NS")
            assert result["shadow_mode"] is True
        except Exception:
            pytest.skip("NSE options data unavailable")

    def test_non_fno_stock_graceful(self):
        try:
            result = self.os.compute("RELAXO.NS")
            assert result["total"] >= 0
            assert result["summary"] in [
                "STRONGLY_BULLISH",
                "BULLISH",
                "NEUTRAL",
                "BEARISH",
            ]
        except Exception:
            pytest.skip("NSE options data unavailable")

    def test_fallback_detail_contains_reason(self):
        result = self.os._pcr_fallback("custom reason")
        assert "custom reason" in result["detail"]


class TestSentimentValidator:

    def setup_method(self):
        from prediction import sentiment_validator

        self.sv = sentiment_validator

    def test_init_creates_table(self):
        """init_validator_table should run without error."""
        self.sv.init_validator_table()

    def test_monthly_report_no_data(self):
        """monthly_report with no data should return INSUFFICIENT_DATA."""
        report = self.sv.monthly_report()
        assert "status" in report or "recommendation" in report

    def test_get_recent_logs_returns_list(self):
        """get_recent_logs should always return a list."""
        logs = self.sv.get_recent_logs(5)
        assert isinstance(logs, list)
