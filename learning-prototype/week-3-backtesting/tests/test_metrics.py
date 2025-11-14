"""
Unit tests for performance metrics module.

Run with: pytest tests/test_metrics.py -v
"""

import sys
sys.path.append('../starter-code')
sys.path.append('../solutions')

import pytest
import pandas as pd
import numpy as np
from performance_metrics_complete import PerformanceAnalyzer, BacktestMetrics


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.01 + 0.0005, index=dates)
    return returns


@pytest.fixture
def sample_equity_curve(sample_returns):
    """Generate sample equity curve from returns."""
    return (1 + sample_returns).cumprod() * 100000


@pytest.fixture
def sample_trades():
    """Generate sample trades for testing."""
    return pd.DataFrame({
        'pnl_pct': [0.02, -0.01, 0.03, -0.005, 0.015, 0.01, -0.02, 0.025]
    })


@pytest.fixture
def analyzer():
    """Create a PerformanceAnalyzer instance."""
    return PerformanceAnalyzer()


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer class."""

    def test_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None

    def test_calculate_total_return(self, analyzer, sample_equity_curve):
        """Test total return calculation."""
        total_return = analyzer.calculate_total_return(sample_equity_curve)

        assert isinstance(total_return, float)
        assert -1 <= total_return <= 10  # Reasonable bounds

        # Test with known values
        simple_equity = pd.Series([100, 110, 120])
        assert abs(analyzer.calculate_total_return(simple_equity) - 0.20) < 0.001

    def test_calculate_annual_return(self, analyzer, sample_returns):
        """Test annualized return calculation."""
        annual_return = analyzer.calculate_annual_return(sample_returns)

        assert isinstance(annual_return, float)
        assert -1 <= annual_return <= 5  # Reasonable bounds

    def test_calculate_volatility(self, analyzer, sample_returns):
        """Test volatility calculation."""
        volatility = analyzer.calculate_volatility(sample_returns)

        assert isinstance(volatility, float)
        assert volatility >= 0  # Volatility is always positive
        assert volatility <= 1.0  # Reasonable upper bound for daily data

    def test_calculate_downside_deviation(self, analyzer, sample_returns):
        """Test downside deviation calculation."""
        downside_dev = analyzer.calculate_downside_deviation(sample_returns)

        assert isinstance(downside_dev, float)
        assert downside_dev >= 0  # Always positive

        # Downside deviation should be less than or equal to total volatility
        volatility = analyzer.calculate_volatility(sample_returns)
        assert downside_dev <= volatility

    def test_calculate_sharpe_ratio(self, analyzer, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = analyzer.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)

        assert isinstance(sharpe, float)
        assert -5 <= sharpe <= 10  # Reasonable bounds

    def test_calculate_sortino_ratio(self, analyzer, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = analyzer.calculate_sortino_ratio(sample_returns, risk_free_rate=0.02)

        assert isinstance(sortino, float)
        assert -5 <= sortino <= 10  # Reasonable bounds

    def test_calculate_max_drawdown(self, analyzer, sample_equity_curve):
        """Test maximum drawdown calculation."""
        max_dd = analyzer.calculate_max_drawdown(sample_equity_curve)

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown is always negative or zero
        assert max_dd >= -1  # Can't lose more than 100%

        # Test with known drawdown
        equity_with_dd = pd.Series([100, 110, 90, 95, 100])
        dd = analyzer.calculate_max_drawdown(equity_with_dd)
        assert abs(dd - (-0.1818)) < 0.01  # -18.18% from 110 to 90

    def test_calculate_calmar_ratio(self, analyzer):
        """Test Calmar ratio calculation."""
        calmar = analyzer.calculate_calmar_ratio(
            annual_return=0.20,
            max_drawdown=-0.10
        )

        assert isinstance(calmar, float)
        assert abs(calmar - 2.0) < 0.001  # 0.20 / 0.10 = 2.0

        # Test with zero drawdown
        calmar_zero = analyzer.calculate_calmar_ratio(0.20, 0.0)
        assert calmar_zero == 0.0  # Handle division by zero

    def test_calculate_var(self, analyzer, sample_returns):
        """Test Value at Risk calculation."""
        var_95 = analyzer.calculate_var(sample_returns, confidence=0.95)

        assert isinstance(var_95, float)
        assert var_95 <= 0  # VaR is typically negative

        # Check that VaR is at the correct percentile
        percentile_5 = np.percentile(sample_returns, 5)
        assert abs(var_95 - percentile_5) < 0.0001

    def test_calculate_cvar(self, analyzer, sample_returns):
        """Test Conditional VaR calculation."""
        cvar_95 = analyzer.calculate_cvar(sample_returns, confidence=0.95)

        assert isinstance(cvar_95, float)
        assert cvar_95 <= 0  # CVaR is typically negative

        # CVaR should be worse (more negative) than VaR
        var_95 = analyzer.calculate_var(sample_returns, confidence=0.95)
        assert cvar_95 <= var_95

    def test_calculate_trade_statistics(self, analyzer, sample_trades):
        """Test trade statistics calculation."""
        stats = analyzer.calculate_trade_statistics(sample_trades)

        assert isinstance(stats, dict)
        assert 'num_trades' in stats
        assert 'win_rate' in stats
        assert 'avg_win' in stats
        assert 'avg_loss' in stats
        assert 'profit_factor' in stats

        # Check values
        assert stats['num_trades'] == 8
        assert 0 <= stats['win_rate'] <= 100
        assert stats['avg_win'] > 0
        assert stats['avg_loss'] < 0
        assert stats['profit_factor'] > 0

    def test_calculate_metrics(self, analyzer, sample_returns, sample_equity_curve, sample_trades):
        """Test complete metrics calculation."""
        metrics = analyzer.calculate_metrics(
            returns=sample_returns,
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            risk_free_rate=0.02
        )

        assert isinstance(metrics, BacktestMetrics)

        # Check that all metrics are calculated
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.annual_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.calmar_ratio, float)
        assert isinstance(metrics.volatility, float)
        assert isinstance(metrics.downside_deviation, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.value_at_risk_95, float)
        assert isinstance(metrics.cvar_95, float)
        assert isinstance(metrics.num_trades, int)
        assert isinstance(metrics.win_rate, float)

    def test_empty_data_handling(self, analyzer):
        """Test handling of empty data."""
        empty_returns = pd.Series([], dtype=float)
        empty_equity = pd.Series([], dtype=float)
        empty_trades = pd.DataFrame()

        # Should not crash
        total_return = analyzer.calculate_total_return(empty_equity)
        assert total_return == 0.0

        volatility = analyzer.calculate_volatility(empty_returns)
        assert volatility == 0.0

        stats = analyzer.calculate_trade_statistics(empty_trades)
        assert stats['num_trades'] == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_all_positive_returns(self, analyzer):
        """Test with all positive returns (no downside)."""
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.01])
        downside_dev = analyzer.calculate_downside_deviation(positive_returns)
        assert downside_dev >= 0  # Should handle gracefully

    def test_all_negative_returns(self, analyzer):
        """Test with all negative returns."""
        negative_returns = pd.Series([-0.01, -0.02, -0.015, -0.01])
        sharpe = analyzer.calculate_sharpe_ratio(negative_returns)
        assert sharpe < 0  # Should be negative

    def test_zero_volatility(self, analyzer):
        """Test with constant returns (zero volatility)."""
        constant_returns = pd.Series([0.01] * 100)
        sharpe = analyzer.calculate_sharpe_ratio(constant_returns)
        assert sharpe == 0.0  # Should handle division by zero

    def test_single_trade(self, analyzer):
        """Test with single trade."""
        single_trade = pd.DataFrame({'pnl_pct': [0.05]})
        stats = analyzer.calculate_trade_statistics(single_trade)
        assert stats['num_trades'] == 1
        assert stats['win_rate'] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
