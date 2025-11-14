"""
Exercise 2: Transaction Cost Analysis

In this exercise, you'll learn how transaction costs impact strategy profitability.

Learning goals:
- Calculate realistic transaction costs
- Compare high-frequency vs low-frequency strategies
- Understand cost breakdown (commission, spread, slippage, impact)
- See how costs can turn profitable strategies into losers

Estimated time: 30 minutes
"""

import sys
sys.path.append('../starter-code')

import pandas as pd
import numpy as np
from transaction_costs import (
    TransactionCostModel,
    TransactionCostConfig,
    CostBreakdown
)

# ============================================================================
# Setup: Different Cost Scenarios
# ============================================================================

print("=" * 70)
print("EXERCISE 2: Transaction Cost Analysis")
print("=" * 70)

# Scenario 1: Retail trader (high costs)
retail_config = TransactionCostConfig(
    commission_fixed=1.0,
    commission_pct=0.001,  # 0.1%
    spread_bps=5.0,
    slippage_bps=10.0
)

# Scenario 2: Professional trader (low costs)
professional_config = TransactionCostConfig(
    commission_fixed=0.0,
    commission_pct=0.0002,  # 0.02%
    spread_bps=1.0,
    slippage_bps=2.0
)

# Scenario 3: High-frequency trader (very low costs, but many trades)
hft_config = TransactionCostConfig(
    commission_fixed=0.0,
    commission_pct=0.00005,  # 0.005%
    spread_bps=0.5,
    slippage_bps=0.5
)

# ============================================================================
# TASK 1: Calculate Costs for Single Trade
# ============================================================================

print("\n" + "=" * 70)
print("TASK 1: Calculate costs for a single $10,000 trade")
print("=" * 70)

trade_value = 10000.0

# TODO: Create cost models for each scenario
# YOUR CODE HERE:
retail_model = None
professional_model = None
hft_model = None

# TODO: Calculate costs for each scenario
# YOUR CODE HERE:
retail_costs = None
professional_costs = None
hft_costs = None

# Print results
if retail_costs:
    print(f"\nRetail Trader (high costs):")
    print(f"  Commission: ${retail_costs.commission:.2f}")
    print(f"  Spread: ${retail_costs.spread:.2f}")
    print(f"  Slippage: ${retail_costs.slippage:.2f}")
    print(f"  Market Impact: ${retail_costs.market_impact:.2f}")
    print(f"  Total: ${retail_costs.total_cost:.2f} ({retail_costs.total_cost_pct:.3%})")

if professional_costs:
    print(f"\nProfessional Trader (low costs):")
    print(f"  Commission: ${professional_costs.commission:.2f}")
    print(f"  Spread: ${professional_costs.spread:.2f}")
    print(f"  Slippage: ${professional_costs.slippage:.2f}")
    print(f"  Market Impact: ${professional_costs.market_impact:.2f}")
    print(f"  Total: ${professional_costs.total_cost:.2f} ({professional_costs.total_cost_pct:.3%})")

if hft_costs:
    print(f"\nHFT Trader (very low costs):")
    print(f"  Commission: ${hft_costs.commission:.2f}")
    print(f"  Spread: ${hft_costs.spread:.2f}")
    print(f"  Slippage: ${hft_costs.slippage:.2f}")
    print(f"  Market Impact: ${hft_costs.market_impact:.2f}")
    print(f"  Total: ${hft_costs.total_cost:.2f} ({hft_costs.total_cost_pct:.3%})")

# ============================================================================
# TASK 2: Compare Strategy Types
# ============================================================================

print("\n" + "=" * 70)
print("TASK 2: Compare different trading strategies")
print("=" * 70)

# Define three strategies with different trading frequencies
strategies = {
    'Buy & Hold': {
        'avg_trade_value': 100000,
        'trades_per_year': 2  # Buy once, sell once
    },
    'Swing Trading': {
        'avg_trade_value': 50000,
        'trades_per_year': 50  # Weekly trades
    },
    'Day Trading': {
        'avg_trade_value': 10000,
        'trades_per_year': 500  # 2 trades per day
    }
}

# TODO: Compare costs for retail trader
# HINT: Use retail_model.compare_strategies(strategies)
# YOUR CODE HERE:
retail_comparison = None

if retail_comparison is not None:
    print("\nRetail Trader - Strategy Comparison:")
    print(retail_comparison.to_string())
    print("\nKey Insight: Notice how day trading costs add up!")

# TODO: Compare costs for professional trader
# YOUR CODE HERE:
professional_comparison = None

if professional_comparison is not None:
    print("\nProfessional Trader - Strategy Comparison:")
    print(professional_comparison.to_string())
    print("\nKey Insight: Lower costs make higher frequency more viable")

# ============================================================================
# TASK 3: Calculate Break-Even Returns
# ============================================================================

print("\n" + "=" * 70)
print("TASK 3: Calculate break-even returns needed")
print("=" * 70)

print("""
Break-even return = The return you need just to cover transaction costs!

If your strategy doesn't beat this, you're losing money.
""")

# TODO: Calculate break-even for each strategy as retail trader
# HINT: Use estimate_annual_costs() for each strategy
# YOUR CODE HERE:

for strategy_name, config in strategies.items():
    # costs = retail_model.estimate_annual_costs(...)
    # YOUR CODE HERE:
    costs = None

    if costs:
        print(f"\n{strategy_name}:")
        print(f"  Annual cost: ${costs['annual_cost']:,.2f}")
        print(f"  Break-even return: {costs['breakeven_return']:.2%}")
        print(f"  --> You need {costs['breakeven_return']:.2%} return just to break even!")

# ============================================================================
# TASK 4: Impact of Volatility on Slippage
# ============================================================================

print("\n" + "=" * 70)
print("TASK 4: How volatility affects slippage")
print("=" * 70)

volatilities = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%

print("\nSlippage costs for $10,000 trade at different volatilities:")
print(f"{'Volatility':<12} {'Retail':<12} {'Professional':<15} {'HFT':<12}")
print("-" * 51)

for vol in volatilities:
    # TODO: Calculate costs at different volatilities
    # YOUR CODE HERE:
    retail_cost = None
    pro_cost = None
    hft_cost = None

    if retail_cost:
        print(f"{vol:.1%}{'':<8} ${retail_cost.total_cost:<11.2f} ${pro_cost.total_cost:<14.2f} ${hft_cost.total_cost:<11.2f}")

print("\nKey Insight: Slippage increases with volatility!")

# ============================================================================
# TASK 5: Market Impact Analysis
# ============================================================================

print("\n" + "=" * 70)
print("TASK 5: Market impact for different order sizes")
print("=" * 70)

order_sizes = [0.001, 0.01, 0.05, 0.10]  # 0.1%, 1%, 5%, 10% of volume

print("\nMarket impact for $100,000 trade:")
print(f"{'% of Volume':<15} {'Impact $':<12} {'Impact %':<12}")
print("-" * 39)

for size in order_sizes:
    # TODO: Calculate market impact
    # YOUR CODE HERE:
    costs = None

    if costs:
        print(f"{size:.1%}{'':<12} ${costs.market_impact:<11.2f} {costs.market_impact/100000:.3%}")

print("\nKey Insight: Large orders have bigger market impact!")

# ============================================================================
# TASK 6: Before and After Costs
# ============================================================================

print("\n" + "=" * 70)
print("TASK 6: Strategy performance before and after costs")
print("=" * 70)

# Simulate a day trading strategy
np.random.seed(42)
num_trades = 250  # 1 trade per day for a year
avg_return_per_trade = 0.002  # 0.2% average per trade
trade_values = np.full(num_trades, 10000.0)

# Calculate gross returns (before costs)
gross_return = num_trades * avg_return_per_trade
gross_profit = gross_return * 10000  # On $10k per trade

print(f"\nDay Trading Strategy (250 trades/year):")
print(f"  Average return per trade: {avg_return_per_trade:.2%}")
print(f"  Gross return: {gross_return:.2%}")
print(f"  Gross profit: ${gross_profit:,.2f}")

# TODO: Calculate total costs for retail trader
# HINT: Calculate cost per trade and multiply by num_trades
# YOUR CODE HERE:
total_retail_costs = None

# TODO: Calculate net returns
# YOUR CODE HERE:
net_return_retail = None
net_profit_retail = None

if net_return_retail is not None:
    print(f"\nRetail Trader (after costs):")
    print(f"  Total costs: ${total_retail_costs:,.2f}")
    print(f"  Net return: {net_return_retail:.2%}")
    print(f"  Net profit: ${net_profit_retail:,.2f}")
    print(f"  Costs consumed {(total_retail_costs/gross_profit)*100:.1f}% of profit!")

# TODO: Calculate for professional trader
# YOUR CODE HERE:
total_pro_costs = None
net_return_pro = None
net_profit_pro = None

if net_return_pro is not None:
    print(f"\nProfessional Trader (after costs):")
    print(f"  Total costs: ${total_pro_costs:,.2f}")
    print(f"  Net return: {net_return_pro:.2%}")
    print(f"  Net profit: ${net_profit_pro:,.2f}")
    print(f"  Costs consumed {(total_pro_costs/gross_profit)*100:.1f}% of profit!")

# ============================================================================
# TASK 7: Answer These Questions
# ============================================================================

print("\n" + "=" * 70)
print("TASK 7: Analysis Questions")
print("=" * 70)

print("""
Answer these questions based on your analysis:

1. Which cost component is usually the largest for retail traders?
   YOUR ANSWER:

2. Why can professional traders profit from high-frequency strategies
   while retail traders cannot?
   YOUR ANSWER:

3. How does volatility affect your transaction costs?
   YOUR ANSWER:

4. What is the break-even return for a day trading strategy with
   retail costs?
   YOUR ANSWER:

5. If you had $100,000 and could only use retail broker costs,
   which strategy would you choose? Why?
   YOUR ANSWER:

6. How much do transaction costs matter for a buy-and-hold strategy?
   YOUR ANSWER:

7. What can you do to reduce transaction costs?
   YOUR ANSWER:

8. Why is it important to include costs in backtesting?
   YOUR ANSWER:
""")

# ============================================================================
# Bonus Challenge
# ============================================================================

print("\n" + "=" * 70)
print("BONUS CHALLENGE")
print("=" * 70)

print("""
Calculate the maximum trading frequency for a retail trader:

Given:
- Annual target return: 15%
- Average return per trade: 0.3%
- Capital: $100,000
- Retail costs (use the config above)

Question: What's the maximum number of trades per year where you
          can still achieve your target return after costs?

HINT: Work backwards from the target return. Calculate how much
      gross return you need to cover costs and hit your target.

YOUR CODE HERE:
""")

# TODO: Solve the bonus challenge
# YOUR CODE HERE:

print("\n" + "=" * 70)
print("Exercise 2 Complete!")
print("=" * 70)
print("""
Key Takeaways:
✅ Transaction costs can destroy strategy profitability
✅ High-frequency strategies require very low costs to be viable
✅ Retail traders should focus on lower-frequency strategies
✅ Always include realistic costs in backtesting
✅ Volatility increases slippage costs
✅ Market impact matters for large orders
✅ Know your break-even return!

Next: Exercise 3 - Backtesting Engine
""")
