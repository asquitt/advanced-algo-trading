"""
Exercise 3: Backtesting Engine

In this exercise, you'll run your first complete backtest!

Learning goals:
- Use the VectorizedBacktester class
- Test different trading strategies
- Understand the importance of avoiding look-ahead bias
- Compare strategies with and without transaction costs

Estimated time: 45 minutes
"""

import sys
sys.path.append('../starter-code')

import pandas as pd
import numpy as np
from backtesting_engine import (
    VectorizedBacktester,
    BacktestConfig,
    moving_average_crossover,
    rsi_strategy
)

# ============================================================================
# Setup: Generate Sample Price Data
# ============================================================================

print("=" * 70)
print("EXERCISE 3: Backtesting Engine")
print("=" * 70)

# Generate realistic price data
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')

# Create a trending market with some volatility
trend = np.linspace(0, 0.3, len(dates))
noise = np.random.randn(len(dates)) * 0.015
returns = trend[1:] - trend[:-1] + noise[1:]
returns = np.insert(returns, 0, 0)
prices = 100 * np.exp(np.cumsum(returns))

# Create OHLCV data
data = pd.DataFrame({
    'open': prices * (1 + np.random.randn(len(dates)) * 0.002),
    'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.005)),
    'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.005)),
    'close': prices,
    'volume': np.random.randint(1000000, 10000000, len(dates))
}, index=dates)

print(f"\nGenerated price data:")
print(f"  Start date: {data.index[0].date()}")
print(f"  End date: {data.index[-1].date()}")
print(f"  Starting price: ${data['close'].iloc[0]:.2f}")
print(f"  Ending price: ${data['close'].iloc[-1]:.2f}")
print(f"  Total return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1):.2%}")

# ============================================================================
# TASK 1: Run Your First Backtest
# ============================================================================

print("\n" + "=" * 70)
print("TASK 1: Run a simple moving average crossover backtest")
print("=" * 70)

# TODO: Create a backtester with no transaction costs first
# YOUR CODE HERE:
config_no_costs = BacktestConfig(
    initial_capital=100000,
    enable_costs=False  # Start without costs to see gross returns
)
backtester_no_costs = None  # Create VectorizedBacktester

# TODO: Define a simple MA crossover strategy
# HINT: Use the moving_average_crossover function with fast=20, slow=50
# YOUR CODE HERE:
def my_strategy(df):
    # YOUR CODE HERE
    pass

# TODO: Run the backtest
# YOUR CODE HERE:
result_no_costs = None

# Print results
if result_no_costs:
    print("\nBacktest Results (WITHOUT transaction costs):")
    print(f"  Total Return: {result_no_costs.metrics.total_return:.2%}")
    print(f"  Annual Return: {result_no_costs.metrics.annual_return:.2%}")
    print(f"  Sharpe Ratio: {result_no_costs.metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result_no_costs.metrics.max_drawdown:.2%}")
    print(f"  Number of Trades: {result_no_costs.metrics.num_trades}")
    print(f"  Win Rate: {result_no_costs.metrics.win_rate:.1f}%")

# ============================================================================
# TASK 2: Add Transaction Costs
# ============================================================================

print("\n" + "=" * 70)
print("TASK 2: Run the same backtest WITH transaction costs")
print("=" * 70)

# TODO: Create a backtester with transaction costs enabled
# YOUR CODE HERE:
config_with_costs = BacktestConfig(
    initial_capital=100000,
    enable_costs=True  # Enable costs to see realistic returns
)
backtester_with_costs = None  # Create VectorizedBacktester

# TODO: Run the backtest with costs
# YOUR CODE HERE:
result_with_costs = None

# Print results
if result_with_costs:
    print("\nBacktest Results (WITH transaction costs):")
    print(f"  Total Return: {result_with_costs.metrics.total_return:.2%}")
    print(f"  Annual Return: {result_with_costs.metrics.annual_return:.2%}")
    print(f"  Sharpe Ratio: {result_with_costs.metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result_with_costs.metrics.max_drawdown:.2%}")
    print(f"  Number of Trades: {result_with_costs.metrics.num_trades}")
    print(f"  Win Rate: {result_with_costs.metrics.win_rate:.1f}%")

    # Compare with and without costs
    if result_no_costs:
        cost_impact = result_no_costs.metrics.total_return - result_with_costs.metrics.total_return
        print(f"\n  Cost Impact: {cost_impact:.2%}")
        print(f"  Costs reduced returns by {cost_impact/result_no_costs.metrics.total_return:.1%}!")

# ============================================================================
# TASK 3: Test Different MA Parameters
# ============================================================================

print("\n" + "=" * 70)
print("TASK 3: Test different moving average parameters")
print("=" * 70)

# TODO: Test these parameter combinations:
parameter_sets = [
    {'fast': 10, 'slow': 30},
    {'fast': 20, 'slow': 50},
    {'fast': 50, 'slow': 100},
    {'fast': 5, 'slow': 20},
]

print("\nComparing different MA parameters:")
print(f"{'Fast/Slow':<12} {'Return':<10} {'Sharpe':<10} {'Trades':<10} {'Win Rate':<10}")
print("-" * 52)

results_comparison = []

for params in parameter_sets:
    # TODO: Create strategy with these parameters
    # YOUR CODE HERE:
    def test_strategy(df):
        return moving_average_crossover(df, fast=params['fast'], slow=params['slow'])

    # TODO: Run backtest
    # YOUR CODE HERE:
    result = None

    if result:
        print(f"{params['fast']}/{params['slow']:<8} {result.metrics.total_return:<10.2%} {result.metrics.sharpe_ratio:<10.2f} {result.metrics.num_trades:<10} {result.metrics.win_rate:<10.1f}%")
        results_comparison.append({
            'params': f"{params['fast']}/{params['slow']}",
            'return': result.metrics.total_return,
            'sharpe': result.metrics.sharpe_ratio,
            'trades': result.metrics.num_trades
        })

print("\nKey Insight: Notice how parameters affect results!")

# ============================================================================
# TASK 4: Test RSI Strategy
# ============================================================================

print("\n" + "=" * 70)
print("TASK 4: Backtest an RSI mean reversion strategy")
print("=" * 70)

# TODO: Test RSI strategy with different parameters
# YOUR CODE HERE:
def rsi_test_strategy(df):
    return rsi_strategy(df, period=14, oversold=30, overbought=70)

# TODO: Run backtest
# YOUR CODE HERE:
result_rsi = None

if result_rsi:
    print("\nRSI Strategy Results:")
    print(f"  Total Return: {result_rsi.metrics.total_return:.2%}")
    print(f"  Annual Return: {result_rsi.metrics.annual_return:.2%}")
    print(f"  Sharpe Ratio: {result_rsi.metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result_rsi.metrics.max_drawdown:.2%}")
    print(f"  Number of Trades: {result_rsi.metrics.num_trades}")
    print(f"  Win Rate: {result_rsi.metrics.win_rate:.1f}%")

# ============================================================================
# TASK 5: Compare Multiple Strategies
# ============================================================================

print("\n" + "=" * 70)
print("TASK 5: Compare MA Crossover vs RSI vs Buy-and-Hold")
print("=" * 70)

# TODO: Define buy-and-hold strategy
# YOUR CODE HERE:
def buy_and_hold(df):
    # Simply return 1 (always long)
    return pd.Series(1, index=df.index)

# TODO: Run backtests for all three strategies
# YOUR CODE HERE:
result_ma = None  # MA crossover
result_rsi = None  # RSI
result_bh = None  # Buy and hold

# Create comparison
if result_ma and result_rsi and result_bh:
    comparison = pd.DataFrame({
        'Strategy': ['MA Crossover', 'RSI', 'Buy & Hold'],
        'Total Return': [
            result_ma.metrics.total_return,
            result_rsi.metrics.total_return,
            result_bh.metrics.total_return
        ],
        'Sharpe Ratio': [
            result_ma.metrics.sharpe_ratio,
            result_rsi.metrics.sharpe_ratio,
            result_bh.metrics.sharpe_ratio
        ],
        'Max Drawdown': [
            result_ma.metrics.max_drawdown,
            result_rsi.metrics.max_drawdown,
            result_bh.metrics.max_drawdown
        ],
        'Num Trades': [
            result_ma.metrics.num_trades,
            result_rsi.metrics.num_trades,
            result_bh.metrics.num_trades
        ]
    })

    print("\nStrategy Comparison:")
    print(comparison.to_string(index=False))

# ============================================================================
# TASK 6: Understand Look-Ahead Bias
# ============================================================================

print("\n" + "=" * 70)
print("TASK 6: Demonstrate look-ahead bias (WRONG way)")
print("=" * 70)

print("""
Let's intentionally create look-ahead bias to see why it's wrong!

WRONG: Using today's signal with today's return
RIGHT: Using yesterday's signal with today's return (shift by 1)
""")

# TODO: Create a strategy with look-ahead bias (DON'T shift signals)
# YOUR CODE HERE:
def biased_strategy(df):
    # Calculate MA crossover
    fast_ma = df['close'].rolling(20).mean()
    slow_ma = df['close'].rolling(50).mean()
    signals = pd.Series(0, index=df.index)
    signals[fast_ma > slow_ma] = 1
    # NOTE: Not shifting! This is WRONG but we're doing it to demonstrate
    return signals

# TODO: Calculate returns WITHOUT shifting (biased)
# YOUR CODE HERE:
# biased_returns = ...

# TODO: Calculate returns WITH shifting (correct)
# YOUR CODE HERE:
# correct_returns = ...

print("\nCompare biased vs correct:")
# YOUR CODE HERE - print the difference

# ============================================================================
# TASK 7: Analyze Equity Curves
# ============================================================================

print("\n" + "=" * 70)
print("TASK 7: Analyze equity curve characteristics")
print("=" * 70)

# TODO: Get equity curve from one of your backtests
# YOUR CODE HERE:
equity_curve = None

if equity_curve is not None:
    # Calculate drawdown series
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    # Find worst drawdown period
    worst_dd_idx = drawdown.idxmin()
    worst_dd_value = drawdown.min()

    print(f"\nEquity Curve Analysis:")
    print(f"  Starting equity: ${equity_curve.iloc[0]:,.2f}")
    print(f"  Ending equity: ${equity_curve.iloc[-1]:,.2f}")
    print(f"  Peak equity: ${equity_curve.max():,.2f}")
    print(f"  Worst drawdown: {worst_dd_value:.2%}")
    print(f"  Worst drawdown date: {worst_dd_idx.date()}")

    # Calculate recovery time
    # TODO: Calculate how long it took to recover from worst drawdown
    # YOUR CODE HERE:

# ============================================================================
# TASK 8: Answer These Questions
# ============================================================================

print("\n" + "=" * 70)
print("TASK 8: Analysis Questions")
print("=" * 70)

print("""
Answer these questions based on your backtests:

1. How much did transaction costs reduce your returns?
   YOUR ANSWER:

2. Which strategy had the best Sharpe ratio?
   YOUR ANSWER:

3. Which strategy had the smallest maximum drawdown?
   YOUR ANSWER:

4. How many trades did the MA crossover strategy generate?
   Is this too many or too few?
   YOUR ANSWER:

5. Did any strategy beat buy-and-hold on a risk-adjusted basis?
   YOUR ANSWER:

6. What happens when you use very short MA periods (e.g., 5/20)?
   Why?
   YOUR ANSWER:

7. Why is it important to shift signals by 1 day?
   YOUR ANSWER:

8. If you had to trade this strategy with real money, which one
   would you choose? Why?
   YOUR ANSWER:
""")

# ============================================================================
# Bonus Challenge
# ============================================================================

print("\n" + "=" * 70)
print("BONUS CHALLENGE")
print("=" * 70)

print("""
Create a combined strategy that uses both MA crossover AND RSI:

Rules:
- Only go long when BOTH:
  * Fast MA > Slow MA (trend is up)
  * RSI < 30 (price is oversold)
- Exit when RSI > 70 (price is overbought)

Compare this combined strategy to the individual strategies.
Does combining them improve performance?

YOUR CODE HERE:
""")

# TODO: Implement combined strategy
# YOUR CODE HERE:
def combined_strategy(df):
    # YOUR CODE HERE
    pass

print("\n" + "=" * 70)
print("Exercise 3 Complete!")
print("=" * 70)
print("""
Key Takeaways:
✅ Backtesting lets you test strategies before risking real money
✅ Transaction costs significantly impact profitability
✅ Different parameters produce very different results
✅ Always shift signals to avoid look-ahead bias
✅ Compare strategies on risk-adjusted metrics, not just returns
✅ Buy-and-hold is often a tough benchmark to beat!
✅ Equity curve analysis reveals strategy behavior

Next: Exercise 4 - Walk-Forward Analysis
""")
