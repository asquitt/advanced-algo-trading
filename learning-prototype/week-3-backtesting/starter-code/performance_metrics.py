"""
Week 3: Performance Metrics - Starter Code

Your mission: Fill in all the TODOs to implement a complete performance metrics calculator!

Total TODOs: 30
Estimated time: 4 hours

Hint levels:
🟢 Easy: Direct implementation
🟡 Medium: Requires understanding of concept
🔴 Hard: Complex calculation or logic

Author: LLM Trading Platform Learning Lab
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class BacktestMetrics:
    """
    Complete backtest performance metrics.

    This class will hold all the metrics you calculate.
    No TODOs here - just use it to return your results!
    """
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
    """
    Calculate comprehensive backtest performance metrics.

    This is the main class you'll implement. Each method calculates
    a specific metric. Follow the TODOs in order!
    """

    def __init__(self):
        """Initialize the performance analyzer."""
        print("PerformanceAnalyzer initialized. Ready to calculate metrics!")

    def calculate_total_return(self, equity_curve: pd.Series) -> float:
        """
        Calculate total return from start to end.

        Formula: (Final Value - Initial Value) / Initial Value

        Example:
        - Start: $100,000
        - End: $120,000
        - Total Return = (120,000 - 100,000) / 100,000 = 0.20 = 20%

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Total return as a decimal (0.20 for 20%)

        🟢 Easy TODO #1: Calculate total return
        """
        # TODO #1: Calculate total return
        # HINT: (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        # YOUR CODE HERE
        pass

    def calculate_annual_return(self, returns: pd.Series) -> float:
        """
        Calculate annualized return (CAGR - Compound Annual Growth Rate).

        Formula: (1 + total_return)^(1 / years) - 1

        Why annualize? To compare strategies with different time periods.
        - Strategy A: 30% in 2 years
        - Strategy B: 20% in 1 year
        - Which is better? Need to annualize to compare!

        Args:
            returns: Daily returns series

        Returns:
            Annualized return

        🟡 Medium TODO #2: Calculate annualized return
        """
        # TODO #2: Calculate annualized return
        # HINT: Assume 252 trading days per year
        # HINT: total_return = (1 + returns).prod() - 1
        # HINT: num_years = len(returns) / 252.0
        # HINT: annual = (1 + total_return) ** (1 / num_years) - 1
        # YOUR CODE HERE
        pass

    def calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).

        Formula: std(returns) * sqrt(252)

        Why sqrt(252)? Volatility scales with square root of time!
        - Daily vol: 1%
        - Annual vol: 1% * sqrt(252) = 15.87%

        Args:
            returns: Daily returns series

        Returns:
            Annualized volatility

        🟢 Easy TODO #3: Calculate volatility
        """
        # TODO #3: Calculate annualized volatility
        # HINT: returns.std() * np.sqrt(252)
        # YOUR CODE HERE
        pass

    def calculate_downside_deviation(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """
        Calculate downside deviation (semi-deviation).

        Only considers returns BELOW the target (negative returns).
        Used in Sortino ratio instead of total volatility.

        Why? We only care about downside risk, not upside volatility!

        Args:
            returns: Daily returns series
            target_return: Minimum acceptable return (default: 0)

        Returns:
            Annualized downside deviation

        🟡 Medium TODO #4: Calculate downside deviation
        """
        # TODO #4: Calculate downside deviation
        # HINT: Step 1: Filter returns below target
        # HINT: downside_returns = returns[returns < target_return]
        # HINT: Step 2: Calculate std of downside returns
        # HINT: Step 3: Annualize by multiplying by sqrt(252)
        # YOUR CODE HERE
        pass

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).

        Formula: (Return - Risk_Free_Rate) / Volatility

        Interpretation:
        - < 1.0: Not good
        - 1.0-2.0: Good
        - > 2.0: Excellent
        - > 3.0: Check for errors!

        Example:
        - Return: 20% (0.20)
        - Risk-free rate: 2% (0.02)
        - Volatility: 15% (0.15)
        - Sharpe = (0.20 - 0.02) / 0.15 = 1.20 (Good!)

        Args:
            returns: Daily returns series
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Sharpe ratio

        🟡 Medium TODO #5: Calculate Sharpe ratio
        """
        # TODO #5: Calculate Sharpe ratio
        # HINT: Convert annual risk-free rate to daily: risk_free_rate / 252
        # HINT: Calculate excess returns: returns - daily_risk_free_rate
        # HINT: Sharpe = mean(excess_returns) / std(excess_returns) * sqrt(252)
        # YOUR CODE HERE
        pass

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (risk-adjusted return using downside risk).

        Like Sharpe, but uses downside deviation instead of total volatility.
        Better for asymmetric strategies (different upside/downside).

        Formula: (Return - Risk_Free_Rate) / Downside_Deviation

        Args:
            returns: Daily returns series
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio

        🟡 Medium TODO #6: Calculate Sortino ratio
        """
        # TODO #6: Calculate Sortino ratio
        # HINT: Similar to Sharpe, but use downside_deviation instead
        # HINT: You already implemented calculate_downside_deviation!
        # YOUR CODE HERE
        pass

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).

        Drawdown = How much you lose from a peak before recovering.

        Example:
        - Peak: $110,000
        - Trough: $95,000
        - Drawdown: ($95,000 - $110,000) / $110,000 = -13.6%

        Why important? Shows worst-case scenario investors experienced.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Maximum drawdown (negative number, e.g., -0.136 for -13.6%)

        🔴 Hard TODO #7: Calculate maximum drawdown
        """
        # TODO #7: Calculate maximum drawdown
        # HINT: Step 1: Calculate running maximum
        # running_max = equity_curve.expanding().max()
        # HINT: Step 2: Calculate drawdown at each point
        # drawdown = (equity_curve - running_max) / running_max
        # HINT: Step 3: Find the minimum (most negative)
        # max_dd = drawdown.min()
        # YOUR CODE HERE
        pass

    def calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Measures return per unit of drawdown risk.
        Higher is better.

        Example:
        - Annual return: 20%
        - Max drawdown: -10%
        - Calmar = 0.20 / 0.10 = 2.0

        Args:
            annual_return: Annualized return
            max_drawdown: Maximum drawdown (negative number)

        Returns:
            Calmar ratio

        🟢 Easy TODO #8: Calculate Calmar ratio
        """
        # TODO #8: Calculate Calmar ratio
        # HINT: annual_return / abs(max_drawdown)
        # HINT: Handle division by zero if max_drawdown is 0
        # YOUR CODE HERE
        pass

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        VaR = "What's the maximum I could lose with X% confidence?"

        Example (95% VaR):
        - "I'm 95% confident I won't lose more than 2% in a day"
        - VaR_95 = -0.02

        Args:
            returns: Daily returns series
            confidence: Confidence level (default: 95%)

        Returns:
            VaR at given confidence (negative number)

        🟡 Medium TODO #9: Calculate VaR
        """
        # TODO #9: Calculate Value at Risk
        # HINT: VaR is the percentile of the returns distribution
        # HINT: Use np.percentile(returns, (1 - confidence) * 100)
        # HINT: Example: VaR_95 = np.percentile(returns, 5)
        # YOUR CODE HERE
        pass

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR = "If I lose more than VaR, how much do I lose on average?"

        More conservative than VaR - looks at tail risk.

        Example:
        - VaR_95: -2% (95% of the time, loss <= 2%)
        - CVaR_95: -3% (when loss > 2%, average loss is 3%)

        Args:
            returns: Daily returns series
            confidence: Confidence level

        Returns:
            CVaR at given confidence

        🔴 Hard TODO #10: Calculate CVaR
        """
        # TODO #10: Calculate Conditional VaR
        # HINT: Step 1: Calculate VaR
        # var = self.calculate_var(returns, confidence)
        # HINT: Step 2: Filter returns worse than VaR
        # worse_than_var = returns[returns <= var]
        # HINT: Step 3: Take the mean
        # cvar = worse_than_var.mean()
        # YOUR CODE HERE
        pass

    def calculate_trade_statistics(self, trades: pd.DataFrame) -> dict:
        """
        Calculate trading statistics from completed trades.

        Metrics:
        - Number of trades
        - Win rate (% of profitable trades)
        - Average win
        - Average loss
        - Profit factor (total wins / total losses)

        Args:
            trades: DataFrame with columns: entry_price, exit_price, pnl_pct

        Returns:
            Dictionary with trade statistics

        🟡 Medium TODO #11-15: Calculate trade stats
        """
        if len(trades) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }

        # TODO #11: Calculate number of trades
        # HINT: num_trades = len(trades)
        num_trades = None  # YOUR CODE HERE

        # TODO #12: Separate winning and losing trades
        # HINT: winning_trades = trades[trades['pnl_pct'] > 0]
        # HINT: losing_trades = trades[trades['pnl_pct'] < 0]
        winning_trades = None  # YOUR CODE HERE
        losing_trades = None  # YOUR CODE HERE

        # TODO #13: Calculate win rate
        # HINT: win_rate = len(winning_trades) / num_trades * 100
        win_rate = None  # YOUR CODE HERE

        # TODO #14: Calculate average win and loss
        # HINT: avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_win = None  # YOUR CODE HERE
        avg_loss = None  # YOUR CODE HERE

        # TODO #15: Calculate profit factor
        # HINT: total_wins = winning_trades['pnl_pct'].sum()
        # HINT: total_losses = abs(losing_trades['pnl_pct'].sum())
        # HINT: profit_factor = total_wins / total_losses if total_losses > 0 else 0
        profit_factor = None  # YOUR CODE HERE

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
        """
        Calculate all metrics at once!

        This is the main method that calls all your other methods.
        Once you've implemented everything above, this will work automatically!

        Args:
            returns: Daily returns series
            equity_curve: Equity curve series
            trades: DataFrame of trades
            risk_free_rate: Annual risk-free rate

        Returns:
            BacktestMetrics dataclass with all metrics

        🟢 Easy TODO #16: Wire up all your methods!
        """
        # TODO #16: Call all your implemented methods
        # HINT: Use the methods you implemented above!

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


# ============================================================================
# Testing Your Implementation
# ============================================================================

def test_your_implementation():
    """
    Quick test to see if your implementation works!

    Run this file to test: python performance_metrics.py
    """
    print("🧪 Testing your implementation...")
    print("-" * 50)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
    equity_curve = (1 + returns).cumprod() * 100000

    # Sample trades
    trades = pd.DataFrame({
        'pnl_pct': [0.02, -0.01, 0.03, -0.005, 0.015]
    })

    # Initialize analyzer
    analyzer = PerformanceAnalyzer()

    # Calculate metrics
    try:
        metrics = analyzer.calculate_metrics(returns, equity_curve, trades)

        print("✅ SUCCESS! Your implementation works!")
        print("\nCalculated Metrics:")
        print(f"  Total Return: {metrics.total_return:.2%}")
        print(f"  Annual Return: {metrics.annual_return:.2%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  Win Rate: {metrics.win_rate:.1f}%")
        print("\n🎉 Great job! Now run the full test suite:")
        print("   pytest tests/test_metrics.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Debug tips:")
        print("  1. Check that all TODOs are filled in")
        print("  2. Make sure you're returning the right type")
        print("  3. Handle edge cases (empty data, division by zero)")
        print("  4. Use print() to debug intermediate values")


if __name__ == "__main__":
    test_your_implementation()
