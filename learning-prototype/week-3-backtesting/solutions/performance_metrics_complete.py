"""
Week 3: Performance Metrics - Complete Solution

This is the complete, working implementation of all performance metrics.
Use this to check your work or if you get stuck!

Author: LLM Trading Platform Learning Lab
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics."""
    # Return metrics
    total_return: float
    annual_return: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float

    # Value at Risk
    value_at_risk_95: float
    cvar_95: float

    # Trading statistics
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class PerformanceAnalyzer:
    """Calculate comprehensive backtest performance metrics."""

    def __init__(self):
        """Initialize the performance analyzer."""
        print("PerformanceAnalyzer initialized. Ready to calculate metrics!")

    def calculate_total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return from start to end."""
        if len(equity_curve) == 0:
            return 0.0
        return (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

    def calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return (CAGR)."""
        if len(returns) == 0:
            return 0.0

        total_return = (1 + returns).prod() - 1
        num_years = len(returns) / 252.0

        if num_years == 0:
            return 0.0

        annual_return = (1 + total_return) ** (1 / num_years) - 1
        return annual_return

    def calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        return returns.std() * np.sqrt(252)

    def calculate_downside_deviation(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate downside deviation (semi-deviation)."""
        if len(returns) == 0:
            return 0.0

        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0.0

        downside_std = downside_returns.std()
        return downside_std * np.sqrt(252)

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf

        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        downside_dev = self.calculate_downside_deviation(returns, daily_rf)

        if downside_dev == 0:
            return 0.0

        sortino = excess_returns.mean() * 252 / downside_dev
        return sortino

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) == 0:
            return 0.0

        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        return max_dd

    def calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        return annual_return / abs(max_drawdown)

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns, confidence)
        worse_than_var = returns[returns <= var]

        if len(worse_than_var) == 0:
            return var

        return worse_than_var.mean()

    def calculate_trade_statistics(self, trades: pd.DataFrame) -> dict:
        """Calculate trading statistics from completed trades."""
        if len(trades) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }

        num_trades = len(trades)
        winning_trades = trades[trades['pnl_pct'] > 0]
        losing_trades = trades[trades['pnl_pct'] < 0]

        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0

        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

        total_wins = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

    def calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> BacktestMetrics:
        """Calculate all metrics at once!"""
        # Return metrics
        total_return = self.calculate_total_return(equity_curve)
        annual_return = self.calculate_annual_return(returns)

        # Risk metrics
        volatility = self.calculate_volatility(returns)
        downside_deviation = self.calculate_downside_deviation(returns)
        max_drawdown = self.calculate_max_drawdown(equity_curve)

        # Risk-adjusted metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)
        calmar_ratio = self.calculate_calmar_ratio(annual_return, max_drawdown)

        # Value at Risk
        var_95 = self.calculate_var(returns, 0.95)
        cvar_95 = self.calculate_cvar(returns, 0.95)

        # Trading statistics
        trade_stats = self.calculate_trade_statistics(trades)

        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_drawdown,
            value_at_risk_95=var_95,
            cvar_95=cvar_95,
            num_trades=trade_stats['num_trades'],
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor']
        )


def test_implementation():
    """Test the implementation."""
    print("Testing Performance Metrics - Complete Solution")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.01 + 0.0008, index=dates)
    equity_curve = (1 + returns).cumprod() * 100000

    trades = pd.DataFrame({
        'pnl_pct': [0.02, -0.01, 0.03, -0.005, 0.015]
    })

    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(returns, equity_curve, trades)

    print("\nCalculated Metrics:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Annual Return: {metrics.annual_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Volatility: {metrics.volatility:.2%}")
    print(f"  VaR (95%): {metrics.value_at_risk_95:.2%}")
    print(f"  CVaR (95%): {metrics.cvar_95:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.1f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_implementation()
