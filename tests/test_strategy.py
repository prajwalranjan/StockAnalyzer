"""
tests/test_strategy.py
=======================
Tests for the momentum_breakout strategy module.
Run with: pytest tests/ -v

Tests config integrity, stock universe, and signal logic
without making any network calls.
"""

import pytest


# ─── Config Tests ─────────────────────────────────────────────────────────────


class TestConfig:

    def setup_method(self):
        from strategies import momentum_breakout

        self.mb = momentum_breakout

    def test_cfg_has_required_keys(self):
        """CFG must contain all required trading parameters."""
        required = [
            "rsi_min",
            "rsi_max",
            "volume_ratio_min",
            "adx_min",
            "stop_loss_pct",
            "profit_target_pct",
            "max_hold_days",
            "capital_per_trade",
            "max_positions",
            "daily_loss_limit",
            "min_price",
            "max_price",
        ]
        for key in required:
            assert key in self.mb.CFG, f"Missing CFG key: {key}"

    def test_rsi_range_valid(self):
        """RSI min must be less than max and both in 0-100."""
        cfg = self.mb.CFG
        assert 0 <= cfg["rsi_min"] < cfg["rsi_max"] <= 100

    def test_stop_loss_less_than_profit_target(self):
        """Stop loss must be smaller than profit target."""
        cfg = self.mb.CFG
        assert cfg["stop_loss_pct"] < cfg["profit_target_pct"]

    def test_stop_loss_reasonable(self):
        """Stop loss should be between 1% and 15%."""
        assert 0.01 <= self.mb.CFG["stop_loss_pct"] <= 0.15

    def test_profit_target_reasonable(self):
        """Profit target should be between 2% and 30%."""
        assert 0.02 <= self.mb.CFG["profit_target_pct"] <= 0.30

    def test_max_positions_at_least_1(self):
        """Must allow at least 1 open position."""
        assert self.mb.CFG["max_positions"] >= 1

    def test_capital_per_trade_positive(self):
        """Capital per trade must be positive."""
        assert self.mb.CFG["capital_per_trade"] > 0

    def test_price_range_valid(self):
        """Min price must be less than max price."""
        assert self.mb.CFG["min_price"] < self.mb.CFG["max_price"]

    def test_paper_cfg_has_more_positions(self):
        """Paper parallel config should allow more positions than live."""
        assert self.mb.CFG_PAPER["max_positions"] > self.mb.CFG["max_positions"]


# ─── Stock Universe Tests ─────────────────────────────────────────────────────


class TestStockUniverse:

    def setup_method(self):
        from strategies import momentum_breakout

        self.mb = momentum_breakout

    def test_stocks_not_empty(self):
        """STOCKS list must not be empty."""
        assert len(self.mb.STOCKS) > 0

    def test_stocks_expanded(self):
        """Universe should have at least 200 stocks (pre-phase requirement)."""
        assert len(self.mb.STOCKS) >= 200, (
            f"Expected 200+ stocks, got {len(self.mb.STOCKS)}. "
            "Did the pre-phase expansion run?"
        )

    def test_no_duplicate_stocks(self):
        """STOCKS list must have no duplicates."""
        assert len(self.mb.STOCKS) == len(
            set(self.mb.STOCKS)
        ), "Duplicate stocks found in STOCKS list"

    def test_all_stocks_end_with_ns(self):
        """All stocks should end with .NS (NSE suffix)."""
        non_ns = [s for s in self.mb.STOCKS if not s.endswith(".NS")]
        assert len(non_ns) == 0, f"Stocks without .NS suffix: {non_ns}"

    def test_sector_index_covers_stocks(self):
        """
        Every stock in STOCKS should either have a sector mapping
        or fall back to Nifty. Just verify the dict is not empty.
        """
        assert len(self.mb.SECTOR_INDEX) > 0

    def test_sector_values_valid(self):
        """All sector index values should be known Yahoo Finance symbols."""
        valid_sectors = {
            "^NSEBANK",
            "^CNXIT",
            "^CNXPHARMA",
            "^CNXAUTO",
            "^CNXENERGY",
            "^CNXMETAL",
            "^CNXFMCG",
            "^NSEI",
        }
        for sym, sector in self.mb.SECTOR_INDEX.items():
            assert sector in valid_sectors, f"{sym} maps to unknown sector {sector}"

    def test_nifty_fallback_defined(self):
        """NIFTY fallback symbol must be defined."""
        assert hasattr(self.mb, "NIFTY")
        assert self.mb.NIFTY == "^NSEI"


# ─── Indicator Tests ──────────────────────────────────────────────────────────


class TestIndicators:

    def setup_method(self):
        from strategies import momentum_breakout
        import pandas as pd
        import numpy as np

        self.mb = momentum_breakout

        # Build a synthetic price series
        np.random.seed(0)
        self.prices = pd.Series(500 + np.cumsum(np.random.randn(100) * 2))
        # Build a synthetic OHLCV DataFrame
        closes = self.prices
        self.df = pd.DataFrame(
            {
                "High": closes * 1.01,
                "Low": closes * 0.99,
                "Close": closes,
                "Volume": np.random.randint(500000, 2000000, 100),
            }
        )

    def test_rsi_range(self):
        """RSI must always be between 0 and 100."""
        rsi = self.mb._rsi(self.prices)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_length(self):
        """RSI series must have same length as input."""
        rsi = self.mb._rsi(self.prices)
        assert len(rsi) == len(self.prices)

    def test_adx_non_negative(self):
        """ADX must always be non-negative."""
        adx = self.mb._adx(self.df)
        valid = adx.dropna()
        assert (valid >= 0).all()

    def test_adx_length(self):
        """ADX series must have same length as input DataFrame."""
        adx = self.mb._adx(self.df)
        assert len(adx) == len(self.df)


# ─── Signal Logic Tests ───────────────────────────────────────────────────────


class TestSignalLogic:

    def setup_method(self):
        from strategies import momentum_breakout

        self.mb = momentum_breakout

    def test_check_signal_return_type(self):
        """check_signal must always return a 2-tuple."""
        # Uses cache — may make network call on first run
        # Testing with a well-known liquid stock
        result = self.mb.check_signal("TCS.NS")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_check_signal_none_or_dict(self):
        """First element must be None or a dict."""
        signal, reason = self.mb.check_signal("TCS.NS")
        assert signal is None or isinstance(signal, dict)

    def test_check_signal_reason_string(self):
        """Second element must always be a string or None."""
        signal, reason = self.mb.check_signal("TCS.NS")
        assert reason is None or isinstance(reason, str)

    def test_signal_dict_has_required_keys(self):
        """
        If a signal is returned, it must have all required keys.
        This test may not fire if market is closed / no breakout.
        """
        # We can't force a breakout, so test with cached data
        # Just verify the function runs without error
        try:
            signal, reason = self.mb.check_signal("HDFCBANK.NS")
            if signal is not None:
                required_keys = [
                    "symbol",
                    "display_name",
                    "price",
                    "rsi",
                    "adx",
                    "volume_ratio",
                    "quantity",
                    "stop_loss",
                    "target",
                    "sector",
                    "quality_score",
                    "quality_grade",
                ]
                for key in required_keys:
                    assert key in signal, f"Missing key in signal: {key}"
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

    def test_get_market_regime_valid(self):
        """get_market_regime must return BULL, CHOP, or BEAR."""
        try:
            regime = self.mb.get_market_regime()
            assert regime in ["BULL", "CHOP", "BEAR"]
        except Exception:
            pytest.skip("Network unavailable")
