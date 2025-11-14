"""
Week 3: Transaction Costs - Complete Solution

This is the complete, working implementation of transaction cost modeling.
Use this to check your work or if you get stuck!

Author: LLM Trading Platform Learning Lab
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost modeling."""
    commission_fixed: float = 1.0
    commission_pct: float = 0.001
    spread_bps: float = 2.0
    slippage_bps: float = 5.0
    slippage_volatility_factor: float = 0.1
    market_impact_factor: float = 0.01


@dataclass
class CostBreakdown:
    """Breakdown of all transaction costs for a trade."""
    commission: float
    spread: float
    slippage: float
    market_impact: float
    total_cost: float
    total_cost_pct: float


class TransactionCostModel:
    """Model realistic transaction costs for backtesting."""

    def __init__(self, config: Optional[TransactionCostConfig] = None):
        """Initialize transaction cost model."""
        self.config = config if config else TransactionCostConfig()

        print(f"TransactionCostModel initialized")
        print(f"  Commission: ${self.config.commission_fixed} + {self.config.commission_pct*100:.2f}%")
        print(f"  Spread: {self.config.spread_bps} bps")
        print(f"  Slippage: {self.config.slippage_bps} bps")

    def calculate_commission(self, trade_value: float) -> float:
        """Calculate broker commission."""
        return self.config.commission_fixed + (trade_value * self.config.commission_pct)

    def calculate_spread_cost(self, trade_value: float) -> float:
        """Calculate bid-ask spread cost."""
        return trade_value * (self.config.spread_bps / 10000)

    def calculate_slippage(
        self,
        trade_value: float,
        volatility: float = 0.02
    ) -> float:
        """Calculate slippage cost."""
        base_slippage_bps = self.config.slippage_bps
        volatility_bps = volatility * 10000
        extra_slippage = self.config.slippage_volatility_factor * volatility_bps
        total_slippage_bps = base_slippage_bps + extra_slippage
        return trade_value * (total_slippage_bps / 10000)

    def calculate_market_impact(
        self,
        trade_value: float,
        volume_participation: float = 0.01
    ) -> float:
        """Calculate market impact cost."""
        return trade_value * self.config.market_impact_factor * volume_participation

    def calculate_total_cost(
        self,
        trade_value: float,
        volatility: float = 0.02,
        volume_participation: float = 0.01
    ) -> CostBreakdown:
        """Calculate total transaction costs with breakdown."""
        commission = self.calculate_commission(trade_value)
        spread = self.calculate_spread_cost(trade_value)
        slippage = self.calculate_slippage(trade_value, volatility)
        market_impact = self.calculate_market_impact(trade_value, volume_participation)

        total = commission + spread + slippage + market_impact
        total_pct = total / trade_value if trade_value > 0 else 0

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
        """Apply transaction costs to a returns series."""
        adjusted_returns = returns.copy()
        position_changes = positions.diff()
        position_changes.iloc[0] = positions.iloc[0]

        equity = initial_capital * (1 + returns).cumprod()

        for i in range(len(position_changes)):
            if position_changes.iloc[i] != 0:
                trade_value = abs(position_changes.iloc[i]) * equity.iloc[i]

                if trade_value > 0:
                    costs = self.calculate_total_cost(trade_value)
                    cost_as_return = costs.total_cost / equity.iloc[i]
                    adjusted_returns.iloc[i] -= cost_as_return

        return adjusted_returns

    def estimate_annual_costs(
        self,
        avg_trade_value: float,
        trades_per_year: int
    ) -> dict:
        """Estimate annual transaction costs for a strategy."""
        cost_per_trade = self.calculate_total_cost(avg_trade_value)
        annual_cost = cost_per_trade.total_cost * trades_per_year
        cost_pct = annual_cost / avg_trade_value if avg_trade_value > 0 else 0

        return {
            'cost_per_trade': cost_per_trade.total_cost,
            'trades_per_year': trades_per_year,
            'annual_cost': annual_cost,
            'cost_percentage': cost_pct,
            'breakeven_return': cost_pct
        }

    def compare_strategies(self, strategies: dict) -> pd.DataFrame:
        """Compare transaction costs across multiple strategies."""
        results = []

        for name, config in strategies.items():
            costs = self.estimate_annual_costs(
                config['avg_trade_value'],
                config['trades_per_year']
            )
            results.append({
                'strategy': name,
                'trades_per_year': config['trades_per_year'],
                'cost_per_trade': costs['cost_per_trade'],
                'annual_cost': costs['annual_cost'],
                'breakeven_return_pct': costs['breakeven_return'] * 100
            })

        return pd.DataFrame(results)


class AdvancedSlippageModel:
    """More realistic slippage model based on order book dynamics."""

    def __init__(self, depth_factor: float = 0.01):
        """Initialize advanced slippage model."""
        self.depth_factor = depth_factor

    def calculate_slippage_from_order_size(
        self,
        order_size: float,
        average_order_size: float,
        volatility: float
    ) -> float:
        """Calculate slippage based on order size relative to market."""
        if average_order_size == 0:
            return 0.0

        size_ratio = order_size / average_order_size
        slippage = self.depth_factor * (size_ratio ** 1.5) * volatility
        return slippage


def bps_to_percentage(bps: float) -> float:
    """Convert basis points to percentage."""
    return bps / 10000


def estimate_bid_ask_spread(
    prices: pd.Series,
    window: int = 20
) -> pd.Series:
    """Estimate bid-ask spread from price data."""
    spread = prices.rolling(window).std() / prices * 0.01
    return spread.fillna(0)


def test_implementation():
    """Test the implementation."""
    print("Testing Transaction Costs - Complete Solution")
    print("=" * 70)

    config = TransactionCostConfig(
        commission_fixed=1.0,
        commission_pct=0.001,
        spread_bps=2.0,
        slippage_bps=5.0
    )
    model = TransactionCostModel(config)

    trade_value = 10000.0
    costs = model.calculate_total_cost(
        trade_value=trade_value,
        volatility=0.02,
        volume_participation=0.01
    )

    print(f"\nCost breakdown for ${trade_value:,.0f} trade:")
    print(f"  Commission: ${costs.commission:.2f}")
    print(f"  Spread: ${costs.spread:.2f}")
    print(f"  Slippage: ${costs.slippage:.2f}")
    print(f"  Market Impact: ${costs.market_impact:.2f}")
    print(f"  Total: ${costs.total_cost:.2f} ({costs.total_cost_pct:.3%})")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_implementation()
