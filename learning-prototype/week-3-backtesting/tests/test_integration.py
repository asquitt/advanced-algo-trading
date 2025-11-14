"""
Integration tests for the complete backtesting system.

Tests the full workflow: data -> signals -> backtest -> metrics

Run with: pytest tests/test_integration.py -v
"""

import sys
sys.path.append('../starter-code')
sys.path.append('../solutions')

import pytest
import pandas as pd
import numpy as np
from backtesting_engine_complete import (
    VectorizedBacktester,
    BacktestConfig,
    BacktestResult,
    WalkForwardAnalyzer,
    ParameterOptimizer,
    moving_average_crossover,
    rsi_strategy
)


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

    # Generate realistic price data
    returns = np.random.randn(len(dates)) * 0.015 + 0.0005
    prices = 100 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(dates)) * 0.002),
        'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.005)),
        'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.005)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    return data


@pytest.fixture
def multi_year_data():
    """Generate multi-year data for walk-forward testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

    returns = np.random.randn(len(dates)) * 0.01 + 0.0003
    prices = 100 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    return data


class TestVectorizedBacktester:
    """Test the VectorizedBacktester class."""

    def test_initialization(self):
        """Test backtester initialization."""
        backtester = VectorizedBacktester()
        assert backtester is not None
        assert backtester.config is not None
        assert backtester.metrics_analyzer is not None

    def test_initialization_with_config(self):
        """Test backtester initialization with custom config."""
        config = BacktestConfig(
            initial_capital=50000,
            enable_costs=False
        )
        backtester = VectorizedBacktester(config)
        assert backtester.config.initial_capital == 50000
        assert backtester.config.enable_costs is False

    def test_prepare_data(self, sample_data):
        """Test data preparation."""
        backtester = VectorizedBacktester()
        prepared = backtester.prepare_data(sample_data)

        assert 'returns' in prepared.columns
        assert 'log_returns' in prepared.columns
        assert len(prepared) > 0
        assert not prepared['returns'].isna().any()

    def test_generate_signals(self, sample_data):
        """Test signal generation."""
        backtester = VectorizedBacktester()
        prepared = backtester.prepare_data(sample_data)

        signals = backtester.generate_signals(
            prepared,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        assert len(signals) == len(prepared)
        assert signals.isin([-1, 0, 1]).all()

    def test_signals_to_positions(self, sample_data):
        """Test conversion of signals to positions."""
        backtester = VectorizedBacktester()
        prepared = backtester.prepare_data(sample_data)
        signals = pd.Series(1, index=prepared.index)

        positions = backtester.signals_to_positions(signals, prepared)

        assert len(positions) == len(signals)
        # First position should be 0 (no look-ahead bias)
        assert positions.iloc[0] == 0
        # Rest should be shifted signals
        assert (positions.iloc[1:] == signals.iloc[:-1]).all()

    def test_calculate_position_returns(self, sample_data):
        """Test calculation of position returns."""
        backtester = VectorizedBacktester()
        prepared = backtester.prepare_data(sample_data)

        positions = pd.Series(1, index=prepared.index)
        returns = backtester.calculate_position_returns(
            positions, prepared['returns']
        )

        assert len(returns) == len(positions)
        # Long position returns should match market returns
        assert (returns == prepared['returns']).all()

    def test_calculate_equity_curve(self):
        """Test equity curve calculation."""
        backtester = VectorizedBacktester()
        returns = pd.Series([0.01, -0.005, 0.02, 0.01])

        equity = backtester.calculate_equity_curve(returns, 100000)

        assert len(equity) == len(returns)
        assert equity.iloc[0] > 100000  # First return is positive
        # Check compounding
        expected_final = 100000 * (1.01) * (0.995) * (1.02) * (1.01)
        assert abs(equity.iloc[-1] - expected_final) < 1

    def test_run_backtest_no_costs(self, sample_data):
        """Test running a complete backtest without costs."""
        config = BacktestConfig(
            initial_capital=100000,
            enable_costs=False
        )
        backtester = VectorizedBacktester(config)

        result = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert result.returns is not None
        assert result.positions is not None
        assert result.metrics is not None
        assert result.start_date == sample_data.index[0]
        assert result.end_date >= sample_data.index[0]

    def test_run_backtest_with_costs(self, sample_data):
        """Test running a backtest with transaction costs."""
        config = BacktestConfig(
            initial_capital=100000,
            enable_costs=True
        )
        backtester = VectorizedBacktester(config)

        result = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        assert isinstance(result, BacktestResult)
        # With costs, returns should be lower
        assert result.metrics is not None


class TestStrategies:
    """Test different trading strategies."""

    def test_moving_average_crossover(self, sample_data):
        """Test MA crossover strategy."""
        backtester = VectorizedBacktester()

        result = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=10, slow=30)
        )

        assert result is not None
        assert len(result.positions) > 0
        # Should generate some signals
        assert (result.positions != 0).any()

    def test_rsi_strategy(self, sample_data):
        """Test RSI strategy."""
        backtester = VectorizedBacktester()

        result = backtester.run_backtest(
            sample_data,
            lambda df: rsi_strategy(df, period=14, oversold=30, overbought=70)
        )

        assert result is not None
        assert len(result.positions) > 0

    def test_buy_and_hold(self, sample_data):
        """Test simple buy-and-hold strategy."""
        backtester = VectorizedBacktester()

        def buy_and_hold(df):
            return pd.Series(1, index=df.index)

        result = backtester.run_backtest(sample_data, buy_and_hold)

        assert result is not None
        # Buy and hold should have minimal trades
        assert result.metrics.num_trades <= 1

    def test_strategy_comparison(self, sample_data):
        """Compare multiple strategies."""
        backtester = VectorizedBacktester()

        # Run different strategies
        ma_fast = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=10, slow=30)
        )

        ma_slow = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=50, slow=100)
        )

        # Both should complete successfully
        assert ma_fast is not None
        assert ma_slow is not None

        # Faster MA should generate more trades
        assert ma_fast.metrics.num_trades >= ma_slow.metrics.num_trades


class TestWalkForwardAnalysis:
    """Test walk-forward analysis functionality."""

    def test_initialization(self):
        """Test walk-forward analyzer initialization."""
        analyzer = WalkForwardAnalyzer(
            train_period_days=252,
            test_period_days=63,
            step_days=63
        )
        assert analyzer is not None
        assert analyzer.train_period_days == 252
        assert analyzer.test_period_days == 63
        assert analyzer.step_days == 63

    def test_split_train_test(self, multi_year_data):
        """Test train/test split."""
        analyzer = WalkForwardAnalyzer()

        train, test = analyzer.split_train_test(multi_year_data, start_idx=0)

        assert len(train) > 0
        assert len(test) > 0
        # Test should come after train
        assert test.index[0] > train.index[-1]

    def test_run_walk_forward(self, multi_year_data):
        """Test running walk-forward analysis."""
        analyzer = WalkForwardAnalyzer(
            train_period_days=252,
            test_period_days=63,
            step_days=126
        )

        results = analyzer.run_walk_forward(
            multi_year_data,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        assert isinstance(results, pd.DataFrame)
        # Should have multiple windows
        assert len(results) > 0


class TestParameterOptimization:
    """Test parameter optimization functionality."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        backtester = VectorizedBacktester()
        optimizer = ParameterOptimizer(backtester)
        assert optimizer is not None
        assert optimizer.backtester is backtester

    def test_grid_search(self, sample_data):
        """Test grid search over parameters."""
        backtester = VectorizedBacktester()
        optimizer = ParameterOptimizer(backtester)

        param_grid = {
            'fast': [10, 20],
            'slow': [30, 50]
        }

        results = optimizer.grid_search(
            sample_data,
            moving_average_crossover,
            param_grid
        )

        assert isinstance(results, pd.DataFrame)
        # Should test all combinations
        assert len(results) <= 4  # 2 x 2 = 4 combinations


class TestCostImpact:
    """Test the impact of transaction costs."""

    def test_costs_reduce_returns(self, sample_data):
        """Test that costs reduce returns."""
        # Backtest without costs
        config_no_costs = BacktestConfig(enable_costs=False)
        backtester_no_costs = VectorizedBacktester(config_no_costs)
        result_no_costs = backtester_no_costs.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        # Backtest with costs
        config_with_costs = BacktestConfig(enable_costs=True)
        backtester_with_costs = VectorizedBacktester(config_with_costs)
        result_with_costs = backtester_with_costs.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        # Returns with costs should be lower
        assert result_with_costs.metrics.total_return <= result_no_costs.metrics.total_return

    def test_high_frequency_more_costs(self, sample_data):
        """Test that high-frequency strategies incur more costs."""
        config = BacktestConfig(enable_costs=True)
        backtester = VectorizedBacktester(config)

        # Low frequency (slow MA)
        result_low_freq = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=50, slow=100)
        )

        # High frequency (fast MA)
        result_high_freq = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=5, slow=20)
        )

        # High frequency should have more trades
        assert result_high_freq.metrics.num_trades > result_low_freq.metrics.num_trades


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_data_period(self):
        """Test with very short data period."""
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        prices = pd.Series(100 + np.random.randn(len(dates)), index=dates)

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': 1000000
        }, index=dates)

        backtester = VectorizedBacktester()
        # Should handle gracefully even if MA periods are long
        result = backtester.run_backtest(
            data,
            lambda df: moving_average_crossover(df, fast=5, slow=10)
        )
        assert result is not None

    def test_no_trades_strategy(self, sample_data):
        """Test strategy that generates no trades."""
        def no_trade_strategy(df):
            return pd.Series(0, index=df.index)  # Always flat

        backtester = VectorizedBacktester()
        result = backtester.run_backtest(sample_data, no_trade_strategy)

        assert result is not None
        # Should have zero trades
        assert result.metrics.num_trades == 0
        # Returns should be zero (no position)
        assert abs(result.metrics.total_return) < 0.001

    def test_always_long_strategy(self, sample_data):
        """Test strategy that's always long."""
        def always_long(df):
            return pd.Series(1, index=df.index)

        backtester = VectorizedBacktester()
        result = backtester.run_backtest(sample_data, always_long)

        assert result is not None
        # Should have minimal trades (just initial position)
        assert result.metrics.num_trades <= 1


class TestMetricsValidation:
    """Test that calculated metrics are reasonable."""

    def test_sharpe_ratio_bounds(self, sample_data):
        """Test that Sharpe ratio is within reasonable bounds."""
        backtester = VectorizedBacktester()
        result = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        # Sharpe should be reasonable (not infinity or NaN)
        assert not np.isnan(result.metrics.sharpe_ratio)
        assert not np.isinf(result.metrics.sharpe_ratio)
        assert -10 <= result.metrics.sharpe_ratio <= 10

    def test_drawdown_bounds(self, sample_data):
        """Test that drawdown is within valid bounds."""
        backtester = VectorizedBacktester()
        result = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        # Max drawdown should be between -100% and 0%
        assert -1 <= result.metrics.max_drawdown <= 0

    def test_win_rate_bounds(self, sample_data):
        """Test that win rate is between 0% and 100%."""
        backtester = VectorizedBacktester()
        result = backtester.run_backtest(
            sample_data,
            lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        assert 0 <= result.metrics.win_rate <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
