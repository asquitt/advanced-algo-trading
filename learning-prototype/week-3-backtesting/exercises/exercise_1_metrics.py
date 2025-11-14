"""
Exercise 1: Calculate Performance Metrics

In this exercise, you'll practice calculating performance metrics for a sample strategy.

Learning goals:
- Understand how to use the PerformanceAnalyzer class
- Interpret different performance metrics
- Compare strategies using metrics

Estimated time: 30 minutes
"""

import sys
sys.path.append('../starter-code')

import pandas as pd
import numpy as np
from performance_metrics import PerformanceAnalyzer, BacktestMetrics

# ============================================================================
# Part 1: Sample Strategy Returns
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Performance Metrics")
print("=" * 70)

# Generate sample strategy returns
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

# Strategy A: High return, high volatility
returns_a = pd.Series(np.random.randn(len(dates)) * 0.015 + 0.0008, index=dates)
equity_a = (1 + returns_a).cumprod() * 100000

# Strategy B: Lower return, lower volatility
returns_b = pd.Series(np.random.randn(len(dates)) * 0.008 + 0.0005, index=dates)
equity_b = (1 + returns_b).cumprod() * 100000

# Sample trades for both strategies
trades_a = pd.DataFrame({
    'pnl_pct': [0.03, -0.01, 0.025, -0.015, 0.02, 0.01, -0.02, 0.035]
})

trades_b = pd.DataFrame({
    'pnl_pct': [0.015, -0.005, 0.01, -0.008, 0.012, 0.008, -0.003, 0.011]
})

# ============================================================================
# TASK 1: Calculate Metrics for Strategy A
# ============================================================================

print("\n" + "=" * 70)
print("TASK 1: Calculate metrics for Strategy A (High Risk)")
print("=" * 70)

# TODO: Create a PerformanceAnalyzer instance
# YOUR CODE HERE:
analyzer = None

# TODO: Calculate metrics for Strategy A
# YOUR CODE HERE:
metrics_a = None

# Print results
if metrics_a:
    print(f"\nStrategy A Results:")
    print(f"  Total Return: {metrics_a.total_return:.2%}")
    print(f"  Annual Return: {metrics_a.annual_return:.2%}")
    print(f"  Sharpe Ratio: {metrics_a.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics_a.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics_a.calmar_ratio:.2f}")
    print(f"  Max Drawdown: {metrics_a.max_drawdown:.2%}")
    print(f"  Volatility: {metrics_a.volatility:.2%}")
    print(f"  VaR (95%): {metrics_a.value_at_risk_95:.2%}")
    print(f"  CVaR (95%): {metrics_a.cvar_95:.2%}")
    print(f"  Win Rate: {metrics_a.win_rate:.1f}%")
    print(f"  Profit Factor: {metrics_a.profit_factor:.2f}")

# ============================================================================
# TASK 2: Calculate Metrics for Strategy B
# ============================================================================

print("\n" + "=" * 70)
print("TASK 2: Calculate metrics for Strategy B (Low Risk)")
print("=" * 70)

# TODO: Calculate metrics for Strategy B
# YOUR CODE HERE:
metrics_b = None

# Print results
if metrics_b:
    print(f"\nStrategy B Results:")
    print(f"  Total Return: {metrics_b.total_return:.2%}")
    print(f"  Annual Return: {metrics_b.annual_return:.2%}")
    print(f"  Sharpe Ratio: {metrics_b.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics_b.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics_b.calmar_ratio:.2f}")
    print(f"  Max Drawdown: {metrics_b.max_drawdown:.2%}")
    print(f"  Volatility: {metrics_b.volatility:.2%}")
    print(f"  VaR (95%): {metrics_b.value_at_risk_95:.2%}")
    print(f"  CVaR (95%): {metrics_b.cvar_95:.2%}")
    print(f"  Win Rate: {metrics_b.win_rate:.1f}%")
    print(f"  Profit Factor: {metrics_b.profit_factor:.2f}")

# ============================================================================
# TASK 3: Compare Strategies
# ============================================================================

print("\n" + "=" * 70)
print("TASK 3: Compare strategies side-by-side")
print("=" * 70)

# TODO: Create a comparison DataFrame
# HINT: Create a DataFrame with both strategies' metrics
# YOUR CODE HERE:
comparison = None

if comparison is not None:
    print("\nStrategy Comparison:")
    print(comparison.to_string())

# ============================================================================
# TASK 4: Answer These Questions
# ============================================================================

print("\n" + "=" * 70)
print("TASK 4: Analysis Questions")
print("=" * 70)

print("""
Answer these questions based on the metrics:

1. Which strategy has higher absolute returns?
   YOUR ANSWER:

2. Which strategy has better risk-adjusted returns (Sharpe)?
   YOUR ANSWER:

3. Which strategy has smaller maximum drawdown?
   YOUR ANSWER:

4. If you could only choose one strategy, which would you pick and why?
   YOUR ANSWER:

5. What is the main trade-off between Strategy A and Strategy B?
   YOUR ANSWER:

6. What does the Sortino ratio tell you that Sharpe doesn't?
   YOUR ANSWER:

7. Why might CVaR be more important than VaR for risk management?
   YOUR ANSWER:

8. Based on the profit factor, which strategy has better trade quality?
   YOUR ANSWER:
""")

# ============================================================================
# TASK 5: Test Individual Metrics
# ============================================================================

print("\n" + "=" * 70)
print("TASK 5: Test individual metric calculations")
print("=" * 70)

# TODO: Test calculating individual metrics
# HINT: Use analyzer.calculate_sharpe_ratio(returns_a), etc.

# Test Sharpe ratio
# YOUR CODE HERE:
sharpe_a = None

print(f"\nManual Sharpe calculation for Strategy A: {sharpe_a}")

# Test max drawdown
# YOUR CODE HERE:
max_dd_a = None

print(f"Manual max drawdown calculation for Strategy A: {max_dd_a:.2%}")

# Test VaR
# YOUR CODE HERE:
var_a = None

print(f"Manual VaR(95%) calculation for Strategy A: {var_a:.2%}")

# ============================================================================
# Bonus Challenge
# ============================================================================

print("\n" + "=" * 70)
print("BONUS CHALLENGE")
print("=" * 70)

print("""
Create a third strategy (Strategy C) with these characteristics:
- Total return between A and B
- Better Sharpe ratio than both A and B
- Maximum drawdown < 10%

HINT: You need to engineer the returns carefully. Think about:
- Mean return (controls total return)
- Standard deviation (controls volatility)
- How to reduce drawdowns (reduce negative streaks)

Try to create this strategy by modifying the random seed and parameters!
""")

# TODO: Create Strategy C
# YOUR CODE HERE:
# returns_c = ...
# equity_c = ...
# trades_c = ...
# metrics_c = analyzer.calculate_metrics(returns_c, equity_c, trades_c)

print("\n" + "=" * 70)
print("Exercise 1 Complete!")
print("=" * 70)
print("""
Key Takeaways:
✅ Performance metrics help you understand strategy characteristics
✅ High returns don't always mean better risk-adjusted performance
✅ Multiple metrics give you a complete picture
✅ Sharpe ratio is the most common risk-adjusted metric
✅ Drawdown shows worst-case scenarios
✅ Trade statistics reveal strategy behavior

Next: Exercise 2 - Transaction Costs
""")
