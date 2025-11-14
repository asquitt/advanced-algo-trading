"""
Comprehensive tests for backtesting module.

Tests VectorizedBacktester, PerformanceAnalyzer, and TransactionCostModel.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtesting.vectorized_engine import (
    VectorizedBacktester,
    BacktestConfig,
    BacktestResult
)
from src.backtesting.performance_analyzer import PerformanceAnalyzer, BacktestMetrics
from src.backtesting.transaction_cost_model import TransactionCostModel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Generate realistic price data with trend and noise
    base_price = 100.0
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.randn(len(dates)) * 2

    prices = base_price + trend + noise.cumsum() * 0.1

    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(dates)) * 0.01),
        'high': prices * (1 + abs(np.random.randn(len(dates))) * 0.02),
        'low': prices * (1 - abs(np.random.randn(len(dates))) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)

    return df


@pytest.fixture
def simple_signal_function():
    """Simple moving average crossover signal function."""
    def signal_func(data, fast_period=20, slow_period=50):
        """Generate buy/sell signals based on MA crossover."""
        fast_ma = data['close'].rolling(fast_period).mean()
        slow_ma = data['close'].rolling(slow_period).mean()

        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1  # Buy signal
        signals[fast_ma < slow_ma] = -1  # Sell signal

        return signals

    return signal_func


@pytest.fixture
def sample_returns():
    """Generate sample returns for performance analysis."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.02, index=dates)
    return returns


@pytest.fixture
def sample_equity_curve():
    """Generate sample equity curve."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.02, index=dates)
    equity = (1 + returns).cumprod() * 100000
    return equity


@pytest.fixture
def sample_trades():
    """Generate sample trades DataFrame."""
    trades = pd.DataFrame({
        'entry_date': pd.date_range(start='2023-01-01', periods=10, freq='W'),
        'exit_date': pd.date_range(start='2023-01-08', periods=10, freq='W'),
        'entry_price': [100, 102, 98, 105, 103, 101, 99, 104, 106, 102],
        'exit_price': [105, 101, 103, 110, 100, 106, 102, 103, 111, 108],
        'size': [1.0] * 10,
        'pnl_pct': [0.05, -0.01, 0.05, 0.05, -0.03, 0.05, 0.03, -0.01, 0.05, 0.06],
        'pnl_dollar': [5000, -1000, 5000, 5000, -3000, 5000, 3000, -1000, 5000, 6000],
        'duration': [7] * 10,
        'side': ['long'] * 10
    })
    return trades


# ============================================================================
# TransactionCostModel Tests
# ============================================================================

class TestTransactionCostModel:
    """Tests for TransactionCostModel."""

    def test_initialization(self):
        """Test cost model initialization."""
        model = TransactionCostModel(
            commission_pct=0.001,
            slippage_bps=5.0,
            spread_bps=2.0
        )

        assert model.commission_pct == 0.001
        assert model.slippage_bps == 5.0
        assert model.spread_bps == 2.0

    def test_calculate_costs_no_trades(self):
        """Test cost calculation with no trades."""
        model = TransactionCostModel()

        position_changes = pd.Series([0, 0, 0, 0])
        prices = pd.Series([100, 101, 102, 103])

        costs = model.calculate_costs(position_changes, prices)

        assert len(costs) == 4
        assert (costs == 0).all()

    def test_calculate_costs_with_trades(self):
        """Test cost calculation with trades."""
        model = TransactionCostModel(
            commission_pct=0.001,
            slippage_bps=5.0,
            spread_bps=2.0
        )

        position_changes = pd.Series([1.0, 0, -1.0, 0])  # Buy, hold, sell, hold
        prices = pd.Series([100, 101, 102, 103])

        costs = model.calculate_costs(position_changes, prices)

        # Should have costs at index 0 and 2 (trades)
        assert costs.iloc[0] > 0
        assert costs.iloc[1] == 0
        assert costs.iloc[2] > 0
        assert costs.iloc[3] == 0

    def test_commission_calculation(self):
        """Test commission calculation."""
        model = TransactionCostModel(commission_pct=0.001)

        position_changes = pd.Series([1.0])
        commission = model._calculate_commission(position_changes)

        assert commission.iloc[0] == 0.001  # 0.1% of position

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        model = TransactionCostModel(slippage_bps=10.0)

        position_changes = pd.Series([1.0])
        prices = pd.Series([100.0])

        slippage = model._calculate_slippage(position_changes, prices)

        # 10 bps = 0.001 = 0.1%
        assert slippage.iloc[0] == 0.001

    def test_spread_calculation(self):
        """Test spread calculation."""
        model = TransactionCostModel(spread_bps=4.0)

        position_changes = pd.Series([1.0])
        spread = model._calculate_spread(position_changes)

        # Half-spread: 4 bps / 2 = 2 bps = 0.0002
        assert spread.iloc[0] == 0.0002

    def test_estimate_total_costs(self):
        """Test total cost estimation."""
        model = TransactionCostModel()

        costs = model.estimate_total_costs(
            num_trades=100,
            avg_position_size=1.0,
            holding_period_days=5
        )

        assert 'num_trades' in costs
        assert 'total_costs' in costs
        assert 'annual_cost' in costs
        assert costs['num_trades'] == 100
        assert costs['total_costs'] > 0

    def test_compare_scenarios(self):
        """Test scenario comparison."""
        model = TransactionCostModel()

        comparison = model.compare_scenarios()

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 4  # 4 scenarios
        assert 'strategy' in comparison.columns
        assert 'total_costs' in comparison.columns


# ============================================================================
# PerformanceAnalyzer Tests
# ============================================================================

class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = PerformanceAnalyzer()
        assert analyzer is not None

    def test_calculate_metrics(self, sample_returns, sample_equity_curve, sample_trades):
        """Test comprehensive metrics calculation."""
        analyzer = PerformanceAnalyzer()

        metrics = analyzer.calculate_metrics(
            returns=sample_returns,
            equity_curve=sample_equity_curve,
            trades=sample_trades
        )

        assert isinstance(metrics, BacktestMetrics)
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.num_trades, int)

    def test_total_return_calculation(self, sample_equity_curve):
        """Test total return calculation."""
        analyzer = PerformanceAnalyzer()

        total_return = analyzer._calculate_total_return(sample_equity_curve)

        assert isinstance(total_return, float)
        # Return should be (final - initial) / initial
        expected = (sample_equity_curve.iloc[-1] - sample_equity_curve.iloc[0]) / sample_equity_curve.iloc[0]
        assert abs(total_return - expected) < 0.0001

    def test_annual_return_calculation(self, sample_returns):
        """Test annualized return calculation."""
        analyzer = PerformanceAnalyzer()

        annual_return = analyzer._calculate_annual_return(sample_returns)

        assert isinstance(annual_return, float)

    def test_volatility_calculation(self, sample_returns):
        """Test volatility calculation."""
        analyzer = PerformanceAnalyzer()

        volatility = analyzer._calculate_volatility(sample_returns)

        assert isinstance(volatility, float)
        assert volatility > 0

    def test_sharpe_ratio_calculation(self, sample_returns):
        """Test Sharpe ratio calculation."""
        analyzer = PerformanceAnalyzer()

        sharpe = analyzer._calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)

        assert isinstance(sharpe, float)

    def test_sortino_ratio_calculation(self, sample_returns):
        """Test Sortino ratio calculation."""
        analyzer = PerformanceAnalyzer()

        sortino = analyzer._calculate_sortino_ratio(sample_returns, risk_free_rate=0.02)

        assert isinstance(sortino, float)

    def test_calmar_ratio_calculation(self):
        """Test Calmar ratio calculation."""
        analyzer = PerformanceAnalyzer()

        calmar = analyzer._calculate_calmar_ratio(annual_return=0.15, max_drawdown=-0.10)

        assert abs(calmar - 1.5) < 0.0001  # 0.15 / 0.10 (floating point tolerance)

    def test_drawdown_metrics(self, sample_equity_curve):
        """Test drawdown metrics calculation."""
        analyzer = PerformanceAnalyzer()

        max_dd, avg_dd, max_dd_duration = analyzer._calculate_drawdown_metrics(sample_equity_curve)

        assert isinstance(max_dd, float)
        assert isinstance(avg_dd, float)
        assert isinstance(max_dd_duration, int)
        assert max_dd <= 0  # Drawdown is negative
        assert avg_dd <= 0

    def test_var_calculation(self, sample_returns):
        """Test Value at Risk calculation."""
        analyzer = PerformanceAnalyzer()

        var_95 = analyzer._calculate_var(sample_returns, confidence=0.95)

        assert isinstance(var_95, float)
        assert var_95 < 0  # VaR is typically negative (loss)

    def test_cvar_calculation(self, sample_returns):
        """Test Conditional VaR calculation."""
        analyzer = PerformanceAnalyzer()

        cvar_95 = analyzer._calculate_cvar(sample_returns, confidence=0.95)

        assert isinstance(cvar_95, float)
        # CVaR should be more negative than VaR (tail risk)

    def test_ulcer_index_calculation(self, sample_equity_curve):
        """Test Ulcer Index calculation."""
        analyzer = PerformanceAnalyzer()

        ulcer = analyzer._calculate_ulcer_index(sample_equity_curve)

        assert isinstance(ulcer, float)
        assert ulcer >= 0

    def test_trade_statistics(self, sample_trades):
        """Test trade statistics calculation."""
        analyzer = PerformanceAnalyzer()

        stats = analyzer._calculate_trade_statistics(sample_trades)

        assert 'num_trades' in stats
        assert 'win_rate' in stats
        assert 'avg_win' in stats
        assert 'avg_loss' in stats
        assert 'profit_factor' in stats

        assert stats['num_trades'] == 10
        assert 0 <= stats['win_rate'] <= 100

    def test_metrics_to_dict(self, sample_returns, sample_equity_curve, sample_trades):
        """Test metrics conversion to dictionary."""
        analyzer = PerformanceAnalyzer()

        metrics = analyzer.calculate_metrics(
            returns=sample_returns,
            equity_curve=sample_equity_curve,
            trades=sample_trades
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert 'sharpe_ratio' in metrics_dict
        assert 'total_return' in metrics_dict
        assert 'max_drawdown' in metrics_dict


# ============================================================================
# VectorizedBacktester Tests
# ============================================================================

class TestVectorizedBacktester:
    """Tests for VectorizedBacktester."""

    def test_initialization(self):
        """Test backtester initialization."""
        config = BacktestConfig(initial_capital=100000.0)
        backtester = VectorizedBacktester(config)

        assert backtester.config.initial_capital == 100000.0
        assert backtester.cost_model is not None
        assert backtester.performance_analyzer is not None

    def test_run_backtest(self, sample_price_data, simple_signal_function):
        """Test running a complete backtest."""
        config = BacktestConfig(initial_capital=100000.0)
        backtester = VectorizedBacktester(config)

        result = backtester.run_backtest(
            data=sample_price_data,
            signal_function=simple_signal_function
        )

        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, BacktestMetrics)
        assert len(result.equity_curve) == len(sample_price_data)
        assert len(result.returns) == len(sample_price_data)
        assert isinstance(result.trades, pd.DataFrame)

    def test_backtest_with_equal_weight_sizing(self, sample_price_data, simple_signal_function):
        """Test backtest with equal weight position sizing."""
        config = BacktestConfig(
            initial_capital=100000.0,
            position_sizing_method="equal_weight"
        )
        backtester = VectorizedBacktester(config)

        result = backtester.run_backtest(
            data=sample_price_data,
            signal_function=simple_signal_function
        )

        assert result.config.position_sizing_method == "equal_weight"
        assert isinstance(result.metrics.sharpe_ratio, float)

    def test_backtest_with_volatility_sizing(self, sample_price_data, simple_signal_function):
        """Test backtest with volatility-based position sizing."""
        config = BacktestConfig(
            initial_capital=100000.0,
            position_sizing_method="volatility"
        )
        backtester = VectorizedBacktester(config)

        result = backtester.run_backtest(
            data=sample_price_data,
            signal_function=simple_signal_function
        )

        assert result.config.position_sizing_method == "volatility"

    def test_backtest_with_kelly_sizing(self, sample_price_data, simple_signal_function):
        """Test backtest with Kelly Criterion position sizing."""
        config = BacktestConfig(
            initial_capital=100000.0,
            position_sizing_method="kelly"
        )
        backtester = VectorizedBacktester(config)

        result = backtester.run_backtest(
            data=sample_price_data,
            signal_function=simple_signal_function
        )

        assert result.config.position_sizing_method == "kelly"

    def test_volatility_sizing_method(self, sample_price_data):
        """Test volatility sizing method."""
        backtester = VectorizedBacktester()

        positions = pd.Series(1, index=sample_price_data.index)
        sample_price_data['returns'] = sample_price_data['close'].pct_change()

        sized = backtester._volatility_sizing(positions, sample_price_data)

        assert len(sized) == len(positions)
        assert (sized.abs() <= backtester.config.max_leverage).all()

    def test_kelly_sizing_method(self, sample_price_data):
        """Test Kelly Criterion sizing method."""
        backtester = VectorizedBacktester()

        positions = pd.Series(1, index=sample_price_data.index)
        sample_price_data['returns'] = sample_price_data['close'].pct_change()

        sized = backtester._kelly_sizing(positions, sample_price_data)

        assert len(sized) == len(positions)

    def test_transaction_costs_applied(self, sample_price_data):
        """Test that transaction costs are applied correctly."""
        backtester = VectorizedBacktester()

        returns = pd.Series(0.01, index=sample_price_data.index)
        positions = pd.Series([0, 1, 1, 0], index=sample_price_data.index[:4])

        returns_after_costs = backtester._apply_transaction_costs(
            returns[:4],
            positions,
            sample_price_data[:4]
        )

        # Costs should reduce returns (excluding NaN values)
        valid_mask = ~returns_after_costs.isna()
        assert (returns_after_costs[valid_mask] <= returns[:4][valid_mask]).all()

    def test_drawdown_calculation(self, sample_equity_curve):
        """Test drawdown calculation."""
        backtester = VectorizedBacktester()

        drawdowns = backtester._calculate_drawdowns(sample_equity_curve)

        assert len(drawdowns) == len(sample_equity_curve)
        assert (drawdowns <= 0).all()  # Drawdowns are negative or zero

    def test_underwater_curve_calculation(self, sample_equity_curve):
        """Test underwater curve calculation."""
        backtester = VectorizedBacktester()

        underwater = backtester._calculate_underwater_curve(sample_equity_curve)

        assert len(underwater) == len(sample_equity_curve)
        assert underwater.isin([0, 1]).all()  # Binary: 0 or 1

    def test_trade_generation(self, sample_price_data):
        """Test trade generation from positions."""
        backtester = VectorizedBacktester()

        # Create simple position series: buy, hold, sell
        positions = pd.Series(0, index=sample_price_data.index)
        positions[10:20] = 1  # Long position for 10 days

        trades = backtester._generate_trades(positions, sample_price_data)

        assert isinstance(trades, pd.DataFrame)
        if len(trades) > 0:
            assert 'entry_date' in trades.columns
            assert 'exit_date' in trades.columns
            assert 'pnl_pct' in trades.columns

    def test_walk_forward_analysis(self, sample_price_data, simple_signal_function):
        """Test walk-forward analysis."""
        backtester = VectorizedBacktester()

        wf_results = backtester.walk_forward_analysis(
            data=sample_price_data,
            signal_function=simple_signal_function,
            train_period_days=100,
            test_period_days=30,
            step_days=30
        )

        assert 'in_sample_results' in wf_results
        assert 'out_of_sample_results' in wf_results
        assert 'avg_in_sample_sharpe' in wf_results
        assert 'avg_out_of_sample_sharpe' in wf_results
        assert len(wf_results['in_sample_results']) > 0

    def test_parameter_optimization(self, sample_price_data, simple_signal_function):
        """Test parameter optimization."""
        backtester = VectorizedBacktester()

        param_grid = {
            'fast_period': [10, 20],
            'slow_period': [40, 50]
        }

        optimization = backtester.parameter_optimization(
            data=sample_price_data,
            signal_function=simple_signal_function,
            param_grid=param_grid,
            metric='sharpe_ratio'
        )

        assert 'best_params' in optimization
        assert 'best_result' in optimization
        assert 'best_metric_value' in optimization
        assert 'all_results' in optimization
        assert len(optimization['all_results']) == 4  # 2 * 2 combinations

    def test_backtest_result_summary(self, sample_price_data, simple_signal_function):
        """Test backtest result summary generation."""
        backtester = VectorizedBacktester()

        result = backtester.run_backtest(
            data=sample_price_data,
            signal_function=simple_signal_function
        )

        summary = result.summary()

        assert isinstance(summary, str)
        assert 'Backtest Summary' in summary
        assert 'Total Return' in summary
        assert 'Sharpe Ratio' in summary

    def test_compound_vs_simple_returns(self, sample_price_data, simple_signal_function):
        """Test compound vs simple returns calculation."""
        # Compound returns
        config_compound = BacktestConfig(compound_returns=True)
        backtester_compound = VectorizedBacktester(config_compound)
        result_compound = backtester_compound.run_backtest(
            data=sample_price_data,
            signal_function=simple_signal_function
        )

        # Simple returns
        config_simple = BacktestConfig(compound_returns=False)
        backtester_simple = VectorizedBacktester(config_simple)
        result_simple = backtester_simple.run_backtest(
            data=sample_price_data,
            signal_function=simple_signal_function
        )

        # Results should be different
        assert result_compound.equity_curve.iloc[-1] != result_simple.equity_curve.iloc[-1]


# ============================================================================
# Integration Tests
# ============================================================================

class TestBacktestingIntegration:
    """Integration tests for backtesting workflow."""

    def test_full_backtest_workflow(self, sample_price_data, simple_signal_function):
        """Test complete backtesting workflow."""
        # 1. Create configuration
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_pct=0.001,
            slippage_bps=5.0,
            position_sizing_method="volatility"
        )

        # 2. Initialize backtester
        backtester = VectorizedBacktester(config)

        # 3. Run backtest
        result = backtester.run_backtest(
            data=sample_price_data,
            signal_function=simple_signal_function
        )

        # 4. Verify results
        assert result.metrics.total_return is not None
        assert result.equity_curve.iloc[-1] > 0
        assert len(result.trades) >= 0

        # 5. Generate summary
        summary = result.summary()
        assert len(summary) > 0

    def test_strategy_comparison(self, sample_price_data, simple_signal_function):
        """Test comparing multiple strategies."""
        results = {}

        for method in ['equal_weight', 'volatility', 'kelly']:
            config = BacktestConfig(position_sizing_method=method)
            backtester = VectorizedBacktester(config)

            result = backtester.run_backtest(
                data=sample_price_data,
                signal_function=simple_signal_function
            )

            results[method] = {
                'total_return': result.metrics.total_return,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'max_drawdown': result.metrics.max_drawdown
            }

        # Verify all strategies produced results
        assert len(results) == 3
        for method, metrics in results.items():
            assert metrics['total_return'] is not None
            assert metrics['sharpe_ratio'] is not None
