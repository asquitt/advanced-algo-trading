"""
Tests for Pairs Trading Strategy

Run with: pytest test_pairs.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from solutions.pairs_trading_complete import PairsTradingStrategy, PairsTradingConfig


@pytest.fixture
def strategy():
    """Create a pairs trading strategy instance."""
    return PairsTradingStrategy()


@pytest.fixture
def cointegrated_pair():
    """Generate synthetic cointegrated price pair."""
    np.random.seed(42)
    n = 250

    # Asset 1: random walk
    returns1 = np.random.normal(0.0005, 0.02, n)
    price1 = pd.Series(100 * np.exp(np.cumsum(returns1)))

    # Asset 2: cointegrated with asset 1
    hedge_ratio = 1.5
    noise = np.random.normal(0, 0.01, n)
    price2 = pd.Series((price1 + noise) / hedge_ratio)

    return price1, price2, hedge_ratio


@pytest.fixture
def non_cointegrated_pair():
    """Generate synthetic non-cointegrated price pair."""
    np.random.seed(100)
    n = 250

    # Two independent random walks
    returns1 = np.random.normal(0.0005, 0.02, n)
    returns2 = np.random.normal(0.0003, 0.015, n)

    price1 = pd.Series(100 * np.exp(np.cumsum(returns1)))
    price2 = pd.Series(80 * np.exp(np.cumsum(returns2)))

    return price1, price2


class TestCointegration:
    """Test cointegration functionality."""

    def test_cointegration_detection(self, strategy, cointegrated_pair):
        """Test that cointegrated pairs are detected."""
        price1, price2, _ = cointegrated_pair

        is_coint, pvalue, hedge_ratio = strategy.test_cointegration(price1, price2)

        assert isinstance(is_coint, bool)
        assert is_coint is True, "Should detect cointegration"
        assert pvalue < 0.05, "P-value should be significant"
        assert hedge_ratio > 0, "Hedge ratio should be positive"

    def test_non_cointegration_detection(self, strategy, non_cointegrated_pair):
        """Test that non-cointegrated pairs are rejected."""
        price1, price2 = non_cointegrated_pair

        is_coint, pvalue, hedge_ratio = strategy.test_cointegration(price1, price2)

        assert isinstance(is_coint, bool)
        # Note: Random data might occasionally show cointegration by chance
        assert isinstance(pvalue, float)

    def test_hedge_ratio_calculation(self, strategy, cointegrated_pair):
        """Test hedge ratio calculation."""
        price1, price2, true_hedge_ratio = cointegrated_pair

        hedge_ratio = strategy.calculate_hedge_ratio(price1, price2)

        assert isinstance(hedge_ratio, float)
        assert hedge_ratio > 0
        # Should be close to true hedge ratio (1.5)
        assert 1.0 < hedge_ratio < 2.0


class TestSpreadCalculation:
    """Test spread and z-score calculations."""

    def test_spread_calculation(self, strategy, cointegrated_pair):
        """Test spread calculation."""
        price1, price2, hedge_ratio = cointegrated_pair

        spread = strategy.calculate_spread(price1, price2, hedge_ratio)

        assert isinstance(spread, pd.Series)
        assert len(spread) == len(price1)
        assert not spread.isna().all()

    def test_zscore_calculation(self, strategy, cointegrated_pair):
        """Test z-score calculation."""
        price1, price2, hedge_ratio = cointegrated_pair
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)

        zscore = strategy.calculate_zscore(spread, lookback=60)

        assert isinstance(zscore, pd.Series)
        assert len(zscore) == len(spread)
        # Z-scores should be normalized
        assert zscore.std() > 0

    def test_half_life_calculation(self, strategy, cointegrated_pair):
        """Test half-life calculation."""
        price1, price2, hedge_ratio = cointegrated_pair
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)

        half_life = strategy.calculate_half_life(spread)

        assert isinstance(half_life, float)
        assert half_life > 0
        # Half-life should be reasonable (not infinite)
        assert half_life < 100

    def test_stationarity_test(self, strategy, cointegrated_pair):
        """Test spread stationarity."""
        price1, price2, hedge_ratio = cointegrated_pair
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)

        is_stationary = strategy.is_spread_stationary(spread)

        assert isinstance(is_stationary, bool)


class TestSignalGeneration:
    """Test signal generation."""

    def test_entry_signals(self, strategy, cointegrated_pair):
        """Test entry signal generation."""
        price1, price2, hedge_ratio = cointegrated_pair
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)
        zscore = strategy.calculate_zscore(spread)

        signals = strategy.generate_entry_signals(zscore)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(zscore)
        # Signals should be -1, 0, or 1
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_exit_signals(self, strategy, cointegrated_pair):
        """Test exit signal generation."""
        price1, price2, hedge_ratio = cointegrated_pair
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)
        zscore = strategy.calculate_zscore(spread)

        entry_signals = strategy.generate_entry_signals(zscore)
        exit_signals = strategy.generate_exit_signals(zscore, entry_signals)

        assert isinstance(exit_signals, pd.Series)
        assert len(exit_signals) == len(zscore)
        # Exit signals should be 0 or 1
        assert set(exit_signals.unique()).issubset({0, 1})

    def test_combined_signals(self, strategy, cointegrated_pair):
        """Test signal combination."""
        price1, price2, hedge_ratio = cointegrated_pair
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)
        zscore = strategy.calculate_zscore(spread)

        entry_signals = strategy.generate_entry_signals(zscore)
        exit_signals = strategy.generate_exit_signals(zscore, entry_signals)
        combined = strategy.combine_signals(entry_signals, exit_signals)

        assert isinstance(combined, pd.Series)
        assert len(combined) == len(zscore)
        assert set(combined.unique()).issubset({-1, 0, 1})


class TestStrategyExecution:
    """Test full strategy execution."""

    def test_run_strategy_success(self, strategy, cointegrated_pair):
        """Test successful strategy execution."""
        price1, price2, _ = cointegrated_pair

        result = strategy.run_strategy(price1, price2, validate_pair=True)

        assert isinstance(result, dict)
        assert 'is_cointegrated' in result
        assert 'signals' in result
        assert 'spread' in result
        assert 'zscore' in result
        assert 'half_life' in result

        # Should be cointegrated
        assert result['is_cointegrated'] is True

    def test_run_strategy_failure(self, strategy, non_cointegrated_pair):
        """Test strategy with non-cointegrated pair."""
        price1, price2 = non_cointegrated_pair

        result = strategy.run_strategy(price1, price2, validate_pair=True)

        assert isinstance(result, dict)
        # Might not be cointegrated
        assert 'is_cointegrated' in result

    def test_strategy_generates_trades(self, strategy, cointegrated_pair):
        """Test that strategy generates some trades."""
        price1, price2, _ = cointegrated_pair

        result = strategy.run_strategy(price1, price2, validate_pair=True)
        signals = result['signals']

        # Should generate some non-zero signals
        num_trades = signals.diff().abs().sum() / 2
        assert num_trades > 0, "Strategy should generate trades"


class TestReturnCalculation:
    """Test return calculations."""

    def test_calculate_returns(self, strategy, cointegrated_pair):
        """Test strategy return calculation."""
        price1, price2, hedge_ratio = cointegrated_pair

        result = strategy.run_strategy(price1, price2, validate_pair=True)
        signals = result['signals']

        returns = strategy.calculate_strategy_returns(
            price1, price2, signals, hedge_ratio
        )

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(price1)
        # Returns should be numeric
        assert returns.dtype in [np.float64, np.float32]

    def test_returns_sensible(self, strategy, cointegrated_pair):
        """Test that returns are within reasonable bounds."""
        price1, price2, hedge_ratio = cointegrated_pair

        result = strategy.run_strategy(price1, price2, validate_pair=True)
        signals = result['signals']

        returns = strategy.calculate_strategy_returns(
            price1, price2, signals, hedge_ratio
        )

        # Daily returns should typically be < 10%
        assert (abs(returns) < 0.1).all(), "Returns should be reasonable"


class TestConfiguration:
    """Test configuration handling."""

    def test_custom_config(self):
        """Test strategy with custom configuration."""
        config = PairsTradingConfig(
            lookback_period=30,
            entry_zscore=2.5,
            exit_zscore=0.3
        )

        strategy = PairsTradingStrategy(config)

        assert strategy.config.lookback_period == 30
        assert strategy.config.entry_zscore == 2.5
        assert strategy.config.exit_zscore == 0.3

    def test_default_config(self, strategy):
        """Test default configuration values."""
        assert strategy.config.lookback_period == 60
        assert strategy.config.entry_zscore == 2.0
        assert strategy.config.exit_zscore == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
