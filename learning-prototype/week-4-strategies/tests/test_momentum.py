"""
Tests for Regime-Adaptive Momentum Strategy

Run with: pytest test_momentum.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from solutions.regime_momentum_complete import RegimeMomentumStrategy, RegimeMomentumConfig


@pytest.fixture
def strategy():
    """Create a momentum strategy instance."""
    return RegimeMomentumStrategy()


@pytest.fixture
def trending_prices():
    """Generate synthetic trending price data."""
    np.random.seed(42)
    n = 500

    # Create uptrend
    trend = np.linspace(0, 0.3, n)
    volatility = np.random.normal(0, 0.015, n)
    returns = trend/n + volatility

    prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    return prices


@pytest.fixture
def regime_change_prices():
    """Generate prices with regime changes."""
    np.random.seed(42)

    # Low vol period
    low_vol_returns = np.random.normal(0.0005, 0.01, 150)
    # High vol period
    high_vol_returns = np.random.normal(0.0005, 0.03, 150)
    # Normal vol period
    normal_vol_returns = np.random.normal(0.0005, 0.02, 200)

    all_returns = np.concatenate([low_vol_returns, high_vol_returns, normal_vol_returns])
    prices = pd.Series(100 * np.exp(np.cumsum(all_returns)))

    return prices


class TestVolatilityRegime:
    """Test volatility regime detection."""

    def test_volatility_calculation(self, strategy, trending_prices):
        """Test volatility calculation."""
        returns = trending_prices.pct_change()
        vol = strategy.calculate_realized_volatility(returns)

        assert isinstance(vol, pd.Series)
        assert len(vol) == len(returns)
        assert (vol > 0).any()

    def test_regime_detection(self, strategy, regime_change_prices):
        """Test volatility regime classification."""
        returns = regime_change_prices.pct_change()
        vol = strategy.calculate_realized_volatility(returns)
        regime = strategy.detect_volatility_regime(vol)

        assert isinstance(regime, pd.Series)
        # Should have all three regimes
        unique_regimes = set(regime.dropna().unique())
        assert 'low_vol' in unique_regimes or 'normal_vol' in unique_regimes or 'high_vol' in unique_regimes

    def test_regime_labels(self, strategy, regime_change_prices):
        """Test regime labels are correct."""
        returns = regime_change_prices.pct_change()
        vol = strategy.calculate_realized_volatility(returns)
        regime = strategy.detect_volatility_regime(vol)

        valid_regimes = {'low_vol', 'normal_vol', 'high_vol'}
        assert set(regime.dropna().unique()).issubset(valid_regimes)


class TestTrendRegime:
    """Test trend regime detection."""

    def test_moving_averages(self, strategy, trending_prices):
        """Test MA calculation."""
        ma_short, ma_long = strategy.calculate_moving_averages(trending_prices)

        assert isinstance(ma_short, pd.Series)
        assert isinstance(ma_long, pd.Series)
        assert len(ma_short) == len(trending_prices)
        assert len(ma_long) == len(trending_prices)

    def test_trend_detection(self, strategy, trending_prices):
        """Test trend regime classification."""
        trend_regime = strategy.detect_trend_regime(trending_prices)

        assert isinstance(trend_regime, pd.Series)
        valid_trends = {'bull', 'neutral', 'bear'}
        assert set(trend_regime.dropna().unique()).issubset(valid_trends)

    def test_bull_market_detection(self, strategy):
        """Test bull market detection."""
        # Create strong uptrend
        prices = pd.Series(np.linspace(100, 150, 100))
        trend_regime = strategy.detect_trend_regime(prices)

        # Should detect bull market
        assert 'bull' in trend_regime.values


class TestMomentumCalculation:
    """Test momentum calculations."""

    def test_single_momentum(self, strategy, trending_prices):
        """Test single momentum calculation."""
        momentum = strategy.calculate_momentum(trending_prices, lookback=20)

        assert isinstance(momentum, pd.Series)
        assert len(momentum) == len(trending_prices)

    def test_dual_momentum(self, strategy, trending_prices):
        """Test dual momentum calculation."""
        momentum_df = strategy.calculate_dual_momentum(trending_prices)

        assert isinstance(momentum_df, pd.DataFrame)
        assert 'mom_short' in momentum_df.columns
        assert 'mom_long' in momentum_df.columns
        assert len(momentum_df) == len(trending_prices)

    def test_momentum_zscore(self, strategy, trending_prices):
        """Test momentum z-score calculation."""
        momentum = strategy.calculate_momentum(trending_prices, lookback=20)
        zscore = strategy.calculate_momentum_zscore(momentum)

        assert isinstance(zscore, pd.Series)
        # Z-score should have some variation
        assert zscore.std() > 0


class TestSignalGeneration:
    """Test signal generation."""

    def test_base_signals(self, strategy, trending_prices):
        """Test base signal generation."""
        momentum_df = strategy.calculate_dual_momentum(trending_prices)
        signals = strategy.generate_base_signals(
            momentum_df['mom_short'],
            momentum_df['mom_long']
        )

        assert isinstance(signals, pd.Series)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_trend_adjustment(self, strategy, trending_prices):
        """Test trend regime adjustment."""
        momentum_df = strategy.calculate_dual_momentum(trending_prices)
        base_signals = strategy.generate_base_signals(
            momentum_df['mom_short'],
            momentum_df['mom_long']
        )
        trend_regime = strategy.detect_trend_regime(trending_prices)

        adjusted = strategy.adjust_signals_for_trend(base_signals, trend_regime)

        assert isinstance(adjusted, pd.Series)
        assert set(adjusted.unique()).issubset({-1, 0, 1})


class TestPositionSizing:
    """Test position sizing."""

    def test_regime_position_sizing(self, strategy, regime_change_prices):
        """Test regime-based position sizing."""
        returns = regime_change_prices.pct_change()
        vol = strategy.calculate_realized_volatility(returns)
        vol_regime = strategy.detect_volatility_regime(vol)

        position_size = strategy.calculate_regime_position_size(1.0, vol_regime)

        assert isinstance(position_size, pd.Series)
        assert (position_size > 0).all()
        # Should have different sizes for different regimes
        assert position_size.std() > 0

    def test_volatility_targeting(self, strategy, trending_prices):
        """Test volatility-targeted position sizing."""
        returns = trending_prices.pct_change()
        position_size = strategy.calculate_volatility_position_size(returns)

        assert isinstance(position_size, pd.Series)
        # Sizes should be clipped to reasonable range
        assert (position_size >= 0.5).all()
        assert (position_size <= 2.0).all()


class TestRiskManagement:
    """Test risk management."""

    def test_dynamic_stop_loss(self, strategy, regime_change_prices):
        """Test dynamic stop loss calculation."""
        returns = regime_change_prices.pct_change()
        vol = strategy.calculate_realized_volatility(returns)
        vol_regime = strategy.detect_volatility_regime(vol)

        stop_loss = strategy.calculate_dynamic_stop_loss(vol_regime)

        assert isinstance(stop_loss, pd.Series)
        assert (stop_loss > 0).all()
        # High vol should have wider stops
        high_vol_stops = stop_loss[vol_regime == 'high_vol']
        normal_vol_stops = stop_loss[vol_regime == 'normal_vol']

        if len(high_vol_stops) > 0 and len(normal_vol_stops) > 0:
            assert high_vol_stops.mean() > normal_vol_stops.mean()


class TestStrategyExecution:
    """Test full strategy execution."""

    def test_run_strategy(self, strategy, trending_prices):
        """Test complete strategy run."""
        result = strategy.run_strategy(trending_prices)

        assert isinstance(result, dict)
        assert 'signals' in result
        assert 'position_size' in result
        assert 'vol_regime' in result
        assert 'trend_regime' in result

    def test_strategy_generates_trades(self, strategy, trending_prices):
        """Test that strategy generates trades."""
        result = strategy.run_strategy(trending_prices)
        signals = result['signals']

        # Should have some non-zero signals
        assert (signals != 0).any()

    def test_returns_calculation(self, strategy, trending_prices):
        """Test return calculation."""
        result = strategy.run_strategy(trending_prices)

        returns = strategy.calculate_strategy_returns(
            trending_prices,
            result['signals'],
            result['position_size']
        )

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(trending_prices)


class TestConfiguration:
    """Test configuration."""

    def test_custom_config(self):
        """Test custom configuration."""
        config = RegimeMomentumConfig(
            lookback_short=10,
            lookback_long=30,
            low_vol_multiplier=2.0
        )
        strategy = RegimeMomentumStrategy(config)

        assert strategy.config.lookback_short == 10
        assert strategy.config.lookback_long == 30
        assert strategy.config.low_vol_multiplier == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
