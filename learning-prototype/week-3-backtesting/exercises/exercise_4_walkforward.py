"""
Exercise 4: Walk-Forward Analysis

In this exercise, you'll implement walk-forward analysis to avoid overfitting!

Learning goals:
- Understand the danger of overfitting
- Implement train/test splits for time series
- Use walk-forward analysis for out-of-sample validation
- Optimize parameters without curve-fitting

Estimated time: 45 minutes
"""

import sys
sys.path.append('../starter-code')

import pandas as pd
import numpy as np
from backtesting_engine import (
    VectorizedBacktester,
    WalkForwardAnalyzer,
    BacktestConfig,
    moving_average_crossover
)

# ============================================================================
# Setup: Generate Multi-Year Price Data
# ============================================================================

print("=" * 70)
print("EXERCISE 4: Walk-Forward Analysis")
print("=" * 70)

# Generate 3 years of price data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

# Create realistic price data with regime changes
returns = []
for i in range(len(dates)):
    # Different market regimes
    if i < len(dates) // 3:  # First year: Bull market
        mean_return = 0.0008
        volatility = 0.01
    elif i < 2 * len(dates) // 3:  # Second year: Sideways
        mean_return = 0.0
        volatility = 0.015
    else:  # Third year: Volatile
        mean_return = 0.0003
        volatility = 0.025
    returns.append(np.random.randn() * volatility + mean_return)

prices = 100 * np.exp(np.cumsum(returns))

data = pd.DataFrame({
    'open': prices * (1 + np.random.randn(len(dates)) * 0.002),
    'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.005)),
    'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.005)),
    'close': prices,
    'volume': np.random.randint(1000000, 10000000, len(dates))
}, index=dates)

print(f"\nGenerated 3 years of price data:")
print(f"  Period: {data.index[0].date()} to {data.index[-1].date()}")
print(f"  Year 1 (Bull): {data.index[0].date()} - {dates[len(dates)//3].date()}")
print(f"  Year 2 (Sideways): {dates[len(dates)//3].date()} - {dates[2*len(dates)//3].date()}")
print(f"  Year 3 (Volatile): {dates[2*len(dates)//3].date()} - {data.index[-1].date()}")

# ============================================================================
# TASK 1: Understand Train/Test Split for Time Series
# ============================================================================

print("\n" + "=" * 70)
print("TASK 1: Split data into train and test sets")
print("=" * 70)

print("""
For time series:
- NEVER use random splits (would use future data!)
- ALWAYS use sequential splits (train on past, test on future)
""")

# TODO: Split data into train (first 2 years) and test (last year)
# YOUR CODE HERE:
train_data = None
test_data = None

if train_data is not None and test_data is not None:
    print(f"\nTrain set: {train_data.index[0].date()} to {train_data.index[-1].date()}")
    print(f"  Length: {len(train_data)} days")

    print(f"\nTest set: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print(f"  Length: {len(test_data)} days")

# ============================================================================
# TASK 2: Optimize on Training Data
# ============================================================================

print("\n" + "=" * 70)
print("TASK 2: Find best parameters on training data")
print("=" * 70)

# TODO: Test different MA parameters on training data
parameter_grid = [
    {'fast': 10, 'slow': 30},
    {'fast': 20, 'slow': 50},
    {'fast': 30, 'slow': 70},
    {'fast': 50, 'slow': 100},
]

print("\nOptimizing on training data...")
print(f"{'Fast/Slow':<12} {'Return':<10} {'Sharpe':<10} {'Drawdown':<10}")
print("-" * 42)

best_params = None
best_sharpe = -np.inf
train_results = []

for params in parameter_grid:
    # TODO: Run backtest on training data
    # YOUR CODE HERE:
    config = BacktestConfig(initial_capital=100000, enable_costs=True)
    backtester = None  # Create backtester

    def strategy(df):
        return moving_average_crossover(df, fast=params['fast'], slow=params['slow'])

    result = None  # Run backtest on train_data

    if result:
        print(f"{params['fast']}/{params['slow']:<8} {result.metrics.total_return:<10.2%} {result.metrics.sharpe_ratio:<10.2f} {result.metrics.max_drawdown:<10.2%}")

        train_results.append({
            'params': params,
            'return': result.metrics.total_return,
            'sharpe': result.metrics.sharpe_ratio
        })

        # TODO: Track best parameters
        # YOUR CODE HERE:
        if result.metrics.sharpe_ratio > best_sharpe:
            best_sharpe = result.metrics.sharpe_ratio
            best_params = params

if best_params:
    print(f"\nBest parameters on training data: {best_params}")
    print(f"  Sharpe ratio: {best_sharpe:.2f}")

# ============================================================================
# TASK 3: Validate on Test Data
# ============================================================================

print("\n" + "=" * 70)
print("TASK 3: Validate best parameters on test data (out-of-sample)")
print("=" * 70)

if best_params:
    # TODO: Run backtest on test data with best parameters
    # YOUR CODE HERE:
    def best_strategy(df):
        return moving_average_crossover(df, fast=best_params['fast'], slow=best_params['slow'])

    backtester_test = None  # Create backtester
    result_test = None  # Run backtest on test_data

    if result_test:
        print(f"\nOut-of-sample performance (test data):")
        print(f"  Total Return: {result_test.metrics.total_return:.2%}")
        print(f"  Sharpe Ratio: {result_test.metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result_test.metrics.max_drawdown:.2%}")

        print(f"\nComparison:")
        print(f"  In-sample Sharpe: {best_sharpe:.2f}")
        print(f"  Out-of-sample Sharpe: {result_test.metrics.sharpe_ratio:.2f}")
        print(f"  Degradation: {best_sharpe - result_test.metrics.sharpe_ratio:.2f}")

        if result_test.metrics.sharpe_ratio < best_sharpe * 0.5:
            print("\n⚠️  WARNING: Significant performance degradation!")
            print("   This suggests overfitting to training data.")
        else:
            print("\n✅ Performance held up reasonably well out-of-sample.")

# ============================================================================
# TASK 4: Implement Rolling Window Walk-Forward
# ============================================================================

print("\n" + "=" * 70)
print("TASK 4: Implement rolling walk-forward analysis")
print("=" * 70)

print("""
Walk-forward process:
1. Train on months 1-12, test on month 13
2. Train on months 2-13, test on month 14
3. Train on months 3-14, test on month 15
... and so on

This gives you multiple out-of-sample periods!
""")

# TODO: Implement simple walk-forward analysis
# YOUR CODE HERE:

train_window = 252  # 1 year training
test_window = 63   # 1 quarter testing
step = 63          # Move forward by 1 quarter

walk_forward_results = []

# TODO: Loop through windows
# YOUR CODE HERE:
num_windows = (len(data) - train_window) // step

print(f"\nRunning {num_windows} walk-forward iterations...")

for i in range(num_windows):
    start_idx = i * step
    train_end = start_idx + train_window
    test_end = train_end + test_window

    if test_end > len(data):
        break

    window_train = data.iloc[start_idx:train_end]
    window_test = data.iloc[train_end:test_end]

    # TODO: Optimize on train window (simplified: just use best_params)
    # In real implementation, you'd re-optimize for each window
    # YOUR CODE HERE:

    # TODO: Test on test window
    # YOUR CODE HERE:

    # TODO: Store results
    # YOUR CODE HERE:

# TODO: Analyze walk-forward results
# YOUR CODE HERE:
if walk_forward_results:
    wf_df = pd.DataFrame(walk_forward_results)
    print(f"\nWalk-Forward Results Summary:")
    print(f"  Number of windows: {len(wf_df)}")
    print(f"  Average return: {wf_df['return'].mean():.2%}")
    print(f"  Average Sharpe: {wf_df['sharpe'].mean():.2f}")
    print(f"  Win rate (profitable windows): {(wf_df['return'] > 0).sum() / len(wf_df) * 100:.1f}%")

# ============================================================================
# TASK 5: Compare to Full-Period Backtest
# ============================================================================

print("\n" + "=" * 70)
print("TASK 5: Compare walk-forward to full-period backtest")
print("=" * 70)

# TODO: Run backtest on full data with best_params
# YOUR CODE HERE:
full_result = None

if full_result:
    print(f"\nFull-period backtest (all data):")
    print(f"  Total Return: {full_result.metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {full_result.metrics.sharpe_ratio:.2f}")

if walk_forward_results and full_result:
    wf_avg_sharpe = pd.DataFrame(walk_forward_results)['sharpe'].mean()
    print(f"\nWalk-forward average Sharpe: {wf_avg_sharpe:.2f}")
    print(f"Full-period Sharpe: {full_result.metrics.sharpe_ratio:.2f}")

    if wf_avg_sharpe < full_result.metrics.sharpe_ratio * 0.7:
        print("\n⚠️  Walk-forward performs much worse than full-period backtest!")
        print("   This suggests the strategy is overfit to historical data.")
    else:
        print("\n✅ Walk-forward results are consistent with full-period backtest.")

# ============================================================================
# TASK 6: Analyze Performance Across Different Market Regimes
# ============================================================================

print("\n" + "=" * 70)
print("TASK 6: How does strategy perform in different market conditions?")
print("=" * 70)

# TODO: Split data by regime and test separately
# YOUR CODE HERE:
regime_1 = data.iloc[:len(data)//3]  # Bull market
regime_2 = data.iloc[len(data)//3:2*len(data)//3]  # Sideways
regime_3 = data.iloc[2*len(data)//3:]  # Volatile

print("\nPerformance by market regime:")

# TODO: Test strategy in each regime
# YOUR CODE HERE:
for regime_name, regime_data in [
    ('Bull Market', regime_1),
    ('Sideways', regime_2),
    ('Volatile', regime_3)
]:
    # Run backtest on this regime
    # YOUR CODE HERE:
    pass

print("\nKey Insight: Strategy performance varies by market regime!")

# ============================================================================
# TASK 7: Implement Parameter Reoptimization
# ============================================================================

print("\n" + "=" * 70)
print("TASK 7: Walk-forward with parameter reoptimization")
print("=" * 70)

print("""
Advanced walk-forward:
- Reoptimize parameters for each train window
- Test with those parameters on the test window
- More realistic than using same parameters throughout

This is computationally expensive but more robust!
""")

# TODO: Implement walk-forward with reoptimization
# YOUR CODE HERE:

print("\nThis is left as an advanced exercise!")
print("Hint: For each train window, run grid search to find best params,")
print("      then test those params on the corresponding test window.")

# ============================================================================
# TASK 8: Answer These Questions
# ============================================================================

print("\n" + "=" * 70)
print("TASK 8: Analysis Questions")
print("=" * 70)

print("""
Answer these questions based on your analysis:

1. How did out-of-sample performance compare to in-sample?
   YOUR ANSWER:

2. Why is walk-forward analysis better than a single train/test split?
   YOUR ANSWER:

3. What percentage of walk-forward windows were profitable?
   Is this good enough?
   YOUR ANSWER:

4. Did the strategy perform equally well in all market regimes?
   Why or why not?
   YOUR ANSWER:

5. What does it mean if out-of-sample performance is much worse
   than in-sample?
   YOUR ANSWER:

6. How would you decide if a strategy is overfit?
   YOUR ANSWER:

7. What's the trade-off between longer and shorter training windows?
   YOUR ANSWER:

8. Would you trade this strategy with real money based on the
   walk-forward results? Why or why not?
   YOUR ANSWER:
""")

# ============================================================================
# Bonus Challenge
# ============================================================================

print("\n" + "=" * 70)
print("BONUS CHALLENGE")
print("=" * 70)

print("""
Implement adaptive walk-forward analysis:

Instead of using fixed window sizes, use expanding windows:
- Window 1: Train on year 1, test on quarter 1 of year 2
- Window 2: Train on year 1 + Q1, test on Q2 of year 2
- Window 3: Train on year 1 + Q1-Q2, test on Q3 of year 2
... and so on

This uses all available history for each test period.
Compare this to the rolling window approach.

YOUR CODE HERE:
""")

# TODO: Implement expanding window walk-forward
# YOUR CODE HERE:

print("\n" + "=" * 70)
print("Exercise 4 Complete!")
print("=" * 70)
print("""
Key Takeaways:
✅ Overfitting is a major risk in backtesting
✅ Always validate strategies out-of-sample
✅ Walk-forward analysis prevents curve-fitting
✅ Parameters optimized on one period may not work in another
✅ Test strategies across different market regimes
✅ Be skeptical of perfect backtests!
✅ If it looks too good to be true, it probably is

Congratulations! You've completed Week 3: Backtesting Engine!

You now know how to:
- Calculate comprehensive performance metrics
- Model realistic transaction costs
- Build vectorized backtesting engines
- Avoid common backtesting pitfalls
- Validate strategies properly with walk-forward analysis

Next steps:
- Apply these techniques to your own trading strategies
- Read "Advances in Financial Machine Learning" by Lopez de Prado
- Start building real strategies in Week 4!
""")
