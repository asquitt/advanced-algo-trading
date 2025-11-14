"""
Week 3: Transaction Costs - Starter Code

Your mission: Fill in all the TODOs to implement realistic transaction cost modeling!

Total TODOs: 20
Estimated time: 3 hours

Hint levels:
🟢 Easy: Direct implementation
🟡 Medium: Requires understanding of concept
🔴 Hard: Complex calculation or logic

Author: LLM Trading Platform Learning Lab
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class TransactionCostConfig:
    """
    Configuration for transaction cost modeling.

    No TODOs here - just configure your costs!
    """
    # Commission per trade (fixed + percentage)
    commission_fixed: float = 1.0  # $1 per trade
    commission_pct: float = 0.001  # 0.1% of trade value

    # Bid-ask spread
    spread_bps: float = 2.0  # 2 basis points (0.02%)

    # Slippage model
    slippage_bps: float = 5.0  # 5 basis points (0.05%)
    slippage_volatility_factor: float = 0.1  # Extra slippage during volatility

    # Market impact (for large orders)
    market_impact_factor: float = 0.01  # 1% impact per 1% of volume


@dataclass
class CostBreakdown:
    """
    Breakdown of all transaction costs for a trade.

    This helps you understand where your costs come from!
    """
    commission: float
    spread: float
    slippage: float
    market_impact: float
    total_cost: float
    total_cost_pct: float  # As percentage of trade value


class TransactionCostModel:
    """
    Model realistic transaction costs for backtesting.

    Why this matters:
    - Ignoring costs = unrealistic backtest results
    - High-frequency strategies especially sensitive to costs
    - Can turn profitable strategy into losing one!

    Costs to model:
    1. Commission - Broker fees
    2. Spread - Bid-ask spread
    3. Slippage - Price moves against you
    4. Market Impact - Your order moves the market
    """

    def __init__(self, config: Optional[TransactionCostConfig] = None):
        """
        Initialize transaction cost model.

        Args:
            config: Cost configuration (uses defaults if None)

        🟢 Easy TODO #1: Initialize the cost model
        """
        # TODO #1: Store the config (or use default)
        # HINT: self.config = config if config else TransactionCostConfig()
        # YOUR CODE HERE
        pass

        print(f"TransactionCostModel initialized")
        print(f"  Commission: ${self.config.commission_fixed} + {self.config.commission_pct*100:.2f}%")
        print(f"  Spread: {self.config.spread_bps} bps")
        print(f"  Slippage: {self.config.slippage_bps} bps")

    def calculate_commission(self, trade_value: float) -> float:
        """
        Calculate broker commission.

        Formula: Fixed + (Trade_Value * Percentage)

        Example:
        - Trade value: $10,000
        - Fixed: $1
        - Percentage: 0.1%
        - Commission = $1 + ($10,000 * 0.001) = $1 + $10 = $11

        Args:
            trade_value: Dollar value of the trade

        Returns:
            Commission in dollars

        🟢 Easy TODO #2: Calculate commission
        """
        # TODO #2: Calculate commission
        # HINT: fixed + (trade_value * percentage)
        # YOUR CODE HERE
        pass

    def calculate_spread_cost(self, trade_value: float) -> float:
        """
        Calculate bid-ask spread cost.

        Spread = Difference between buy and sell price
        - Buy at ask (higher)
        - Sell at bid (lower)
        - You lose the spread!

        Example:
        - Bid: $99.99
        - Ask: $100.01
        - Spread: $0.02 = 2 cents = 0.02%

        Args:
            trade_value: Dollar value of the trade

        Returns:
            Spread cost in dollars

        🟢 Easy TODO #3: Calculate spread cost
        """
        # TODO #3: Calculate spread cost
        # HINT: trade_value * (spread_bps / 10000)
        # HINT: Divide by 10000 to convert basis points to decimal
        # YOUR CODE HERE
        pass

    def calculate_slippage(
        self,
        trade_value: float,
        volatility: float = 0.02
    ) -> float:
        """
        Calculate slippage cost.

        Slippage = Price moves against you between signal and execution
        - Want to buy? Price goes up!
        - Want to sell? Price goes down!
        - Worse during high volatility

        Formula: Base_Slippage + Volatility_Factor * Volatility

        Example (high volatility day):
        - Base slippage: 5 bps
        - Volatility: 3% (0.03)
        - Vol factor: 0.1
        - Total slippage = 5 bps + (0.1 * 300 bps) = 35 bps

        Args:
            trade_value: Dollar value of the trade
            volatility: Current volatility (default: 2%)

        Returns:
            Slippage cost in dollars

        🟡 Medium TODO #4: Calculate slippage
        """
        # TODO #4: Calculate slippage
        # HINT: base_slippage_bps = self.config.slippage_bps
        # HINT: volatility_bps = volatility * 10000  # Convert to bps
        # HINT: extra_slippage = self.config.slippage_volatility_factor * volatility_bps
        # HINT: total_slippage_bps = base_slippage_bps + extra_slippage
        # HINT: cost = trade_value * (total_slippage_bps / 10000)
        # YOUR CODE HERE
        pass

    def calculate_market_impact(
        self,
        trade_value: float,
        volume_participation: float = 0.01
    ) -> float:
        """
        Calculate market impact cost.

        Market Impact = Your order moves the market price
        - Large orders have more impact
        - Illiquid markets have more impact

        Formula: Trade_Value * Impact_Factor * Volume_Participation

        Example:
        - Trade: $100,000
        - Daily volume: $10,000,000
        - Your trade = 1% of volume
        - Impact factor: 0.01
        - Impact = $100,000 * 0.01 * 0.01 = $10

        Args:
            trade_value: Dollar value of the trade
            volume_participation: Trade size as % of daily volume

        Returns:
            Market impact cost in dollars

        🟡 Medium TODO #5: Calculate market impact
        """
        # TODO #5: Calculate market impact
        # HINT: trade_value * impact_factor * volume_participation
        # YOUR CODE HERE
        pass

    def calculate_total_cost(
        self,
        trade_value: float,
        volatility: float = 0.02,
        volume_participation: float = 0.01
    ) -> CostBreakdown:
        """
        Calculate total transaction costs with breakdown.

        This is the main method that combines all cost components!

        Args:
            trade_value: Dollar value of the trade
            volatility: Current volatility
            volume_participation: Trade size as % of volume

        Returns:
            CostBreakdown with all components

        🟡 Medium TODO #6: Calculate total cost
        """
        # TODO #6: Calculate all cost components
        # HINT: Call the methods you implemented above!

        commission = None  # YOUR CODE HERE
        spread = None  # YOUR CODE HERE
        slippage = None  # YOUR CODE HERE
        market_impact = None  # YOUR CODE HERE

        total = None  # YOUR CODE HERE (sum all components)
        total_pct = None  # YOUR CODE HERE (total / trade_value)

        return CostBreakdown(
            commission=commission,
            spread=spread,
            slippage=slippage,
            market_impact=market_impact,
            total_cost=total,
            total_cost_pct=total_pct
        )

    def apply_costs_to_returns(
        self,
        returns: pd.Series,
        positions: pd.Series,
        prices: pd.Series,
        initial_capital: float = 100000.0
    ) -> pd.Series:
        """
        Apply transaction costs to a returns series.

        This is how you integrate costs into your backtest!

        Process:
        1. Detect when position changes (trades happen)
        2. Calculate trade value
        3. Calculate transaction costs
        4. Subtract costs from returns

        Args:
            returns: Daily returns series
            positions: Position sizes (-1, 0, 1 for short, flat, long)
            prices: Price series
            initial_capital: Starting capital

        Returns:
            Returns series with costs applied

        🔴 Hard TODO #7-11: Apply costs to returns
        """
        # TODO #7: Create a copy of returns
        # HINT: adjusted_returns = returns.copy()
        adjusted_returns = None  # YOUR CODE HERE

        # TODO #8: Detect position changes (trades)
        # HINT: position_changes = positions.diff()
        # HINT: Fill first value with positions.iloc[0] to capture initial position
        position_changes = None  # YOUR CODE HERE

        # TODO #9: Calculate running equity
        # HINT: equity = initial_capital * (1 + returns).cumprod()
        equity = None  # YOUR CODE HERE

        # TODO #10: Loop through and apply costs when positions change
        # HINT: Use a for loop or vectorized operations
        # HINT: For each trade: calculate trade_value, calculate costs, subtract from returns
        # YOUR CODE HERE
        for i in range(len(position_changes)):
            if position_changes.iloc[i] != 0:  # Trade happened
                # Calculate trade value
                # TODO #11: Calculate trade value for this position change
                # HINT: trade_value = abs(position_changes.iloc[i]) * equity.iloc[i]
                trade_value = None  # YOUR CODE HERE

                if trade_value > 0:
                    # Calculate costs
                    costs = self.calculate_total_cost(trade_value)

                    # Convert cost to return impact
                    cost_as_return = costs.total_cost / equity.iloc[i]

                    # Subtract cost from return
                    adjusted_returns.iloc[i] -= cost_as_return

        return adjusted_returns

    def estimate_annual_costs(
        self,
        avg_trade_value: float,
        trades_per_year: int
    ) -> dict:
        """
        Estimate annual transaction costs for a strategy.

        Useful for comparing strategies!

        Example:
        - Strategy A: 100 trades/year, $50 avg cost = $5,000/year
        - Strategy B: 10 trades/year, $100 avg cost = $1,000/year
        - Strategy B is cheaper!

        Args:
            avg_trade_value: Average trade size
            trades_per_year: Number of trades per year

        Returns:
            Dictionary with cost estimates

        🟡 Medium TODO #12-14: Estimate annual costs
        """
        # TODO #12: Calculate cost per trade
        # HINT: Use calculate_total_cost()
        cost_per_trade = None  # YOUR CODE HERE

        # TODO #13: Calculate annual cost
        # HINT: annual_cost = cost_per_trade.total_cost * trades_per_year
        annual_cost = None  # YOUR CODE HERE

        # TODO #14: Calculate as percentage of capital
        # HINT: Assume you trade your full capital
        # HINT: cost_pct = annual_cost / avg_trade_value
        cost_pct = None  # YOUR CODE HERE

        return {
            'cost_per_trade': cost_per_trade.total_cost,
            'trades_per_year': trades_per_year,
            'annual_cost': annual_cost,
            'cost_percentage': cost_pct,
            'breakeven_return': cost_pct  # Need this return just to break even!
        }

    def compare_strategies(
        self,
        strategies: dict
    ) -> pd.DataFrame:
        """
        Compare transaction costs across multiple strategies.

        Args:
            strategies: Dict with strategy configs
                       {name: {'avg_trade_value': X, 'trades_per_year': Y}}

        Returns:
            DataFrame comparing costs

        🟡 Medium TODO #15-16: Compare strategies
        """
        # TODO #15: Calculate costs for each strategy
        # HINT: Use estimate_annual_costs() for each strategy
        results = []
        for name, config in strategies.items():
            costs = None  # YOUR CODE HERE - call estimate_annual_costs()
            results.append({
                'strategy': name,
                'trades_per_year': config['trades_per_year'],
                'cost_per_trade': costs['cost_per_trade'],
                'annual_cost': costs['annual_cost'],
                'breakeven_return_pct': costs['breakeven_return'] * 100
            })

        # TODO #16: Create and return DataFrame
        # HINT: pd.DataFrame(results)
        # YOUR CODE HERE
        pass


# ============================================================================
# Advanced: Realistic Slippage Model
# ============================================================================

class AdvancedSlippageModel:
    """
    More realistic slippage model based on order book dynamics.

    Optional advanced section - implement if you want extra challenge!
    """

    def __init__(self, depth_factor: float = 0.01):
        """
        Initialize advanced slippage model.

        Args:
            depth_factor: Order book depth factor

        🔴 Hard TODO #17: Initialize advanced slippage model
        """
        # TODO #17: Store depth factor
        # YOUR CODE HERE
        pass

    def calculate_slippage_from_order_size(
        self,
        order_size: float,
        average_order_size: float,
        volatility: float
    ) -> float:
        """
        Calculate slippage based on order size relative to market.

        Larger orders relative to normal size = more slippage

        Formula: base_rate * (order_size / avg_size)^power * volatility

        Args:
            order_size: Size of your order
            average_order_size: Average market order size
            volatility: Current market volatility

        Returns:
            Slippage as percentage

        🔴 Hard TODO #18: Calculate realistic slippage
        """
        # TODO #18: Implement advanced slippage
        # HINT: size_ratio = order_size / average_order_size
        # HINT: slippage = self.depth_factor * (size_ratio ** 1.5) * volatility
        # YOUR CODE HERE
        pass


# ============================================================================
# Utility Functions
# ============================================================================

def bps_to_percentage(bps: float) -> float:
    """
    Convert basis points to percentage.

    Args:
        bps: Basis points

    Returns:
        Percentage (decimal)

    🟢 Easy TODO #19: Convert basis points
    """
    # TODO #19: Convert basis points to percentage
    # HINT: bps / 10000
    # YOUR CODE HERE
    pass


def estimate_bid_ask_spread(
    prices: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Estimate bid-ask spread from price data.

    Uses high-low range as proxy for spread.
    Real data would have actual bid/ask quotes.

    Args:
        prices: Price series
        window: Rolling window for estimation

    Returns:
        Estimated spread series

    🔴 Hard TODO #20: Estimate spread from prices
    """
    # TODO #20: Estimate bid-ask spread
    # HINT: Calculate rolling high-low range
    # HINT: spread = (high - low) / ((high + low) / 2)
    # HINT: In reality, use: prices.rolling(window).std() / prices * 0.01
    # YOUR CODE HERE
    pass


# ============================================================================
# Testing Your Implementation
# ============================================================================

def test_your_implementation():
    """
    Quick test to see if your implementation works!

    Run this file to test: python transaction_costs.py
    """
    print("🧪 Testing your implementation...")
    print("-" * 50)

    # Create cost model
    config = TransactionCostConfig(
        commission_fixed=1.0,
        commission_pct=0.001,
        spread_bps=2.0,
        slippage_bps=5.0
    )
    model = TransactionCostModel(config)

    # Test single trade
    trade_value = 10000.0

    try:
        costs = model.calculate_total_cost(
            trade_value=trade_value,
            volatility=0.02,
            volume_participation=0.01
        )

        print("✅ SUCCESS! Your implementation works!")
        print(f"\nCost breakdown for ${trade_value:,.0f} trade:")
        print(f"  Commission: ${costs.commission:.2f}")
        print(f"  Spread: ${costs.spread:.2f}")
        print(f"  Slippage: ${costs.slippage:.2f}")
        print(f"  Market Impact: ${costs.market_impact:.2f}")
        print(f"  Total: ${costs.total_cost:.2f} ({costs.total_cost_pct:.3%})")
        print("\n🎉 Great job! Now run the full test suite:")
        print("   pytest tests/test_costs.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Debug tips:")
        print("  1. Check that all TODOs are filled in")
        print("  2. Make sure you're returning the right type")
        print("  3. Handle edge cases (division by zero)")
        print("  4. Use print() to debug intermediate values")


if __name__ == "__main__":
    test_your_implementation()
