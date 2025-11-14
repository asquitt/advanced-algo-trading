"""
Transaction Cost Model

Models realistic trading costs including commissions, slippage, and spreads.
Essential for accurate backtesting performance estimation.

Author: LLM Trading Platform
"""

from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


class TransactionCostModel:
    """
    Models realistic transaction costs for backtesting.

    Includes:
    - Fixed and percentage-based commissions
    - Market impact slippage (based on volume)
    - Bid-ask spread costs
    - Variable slippage based on volatility
    """

    def __init__(
        self,
        commission_pct: float = 0.001,  # 0.1% commission
        commission_fixed: float = 0.0,  # Fixed commission per trade
        slippage_bps: float = 5.0,  # 5 basis points slippage
        spread_bps: float = 2.0,  # 2 basis points spread
        use_volume_slippage: bool = True,  # Adjust slippage based on volume
        use_volatility_slippage: bool = True  # Adjust slippage based on volatility
    ):
        """
        Initialize transaction cost model.

        Args:
            commission_pct: Percentage commission (e.g., 0.001 = 0.1%)
            commission_fixed: Fixed commission per trade (in dollars)
            slippage_bps: Base slippage in basis points (1 bp = 0.01%)
            spread_bps: Bid-ask spread in basis points
            use_volume_slippage: Apply volume-based slippage adjustment
            use_volatility_slippage: Apply volatility-based slippage adjustment
        """
        self.commission_pct = commission_pct
        self.commission_fixed = commission_fixed
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.use_volume_slippage = use_volume_slippage
        self.use_volatility_slippage = use_volatility_slippage

        logger.info(
            f"TransactionCostModel initialized: "
            f"commission={commission_pct:.3%}, "
            f"slippage={slippage_bps}bps, "
            f"spread={spread_bps}bps"
        )

    def calculate_costs(
        self,
        position_changes: pd.Series,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate total transaction costs for position changes.

        Args:
            position_changes: Series of position size changes (trades)
            prices: Series of prices at trade execution
            volumes: Optional series of volumes for slippage calculation

        Returns:
            Series of costs as percentage of position value
        """
        # Initialize cost series
        costs = pd.Series(0.0, index=position_changes.index)

        # Only calculate costs when there's a position change (trade)
        trades = position_changes != 0

        if not trades.any():
            return costs

        # Commission costs
        commission_costs = self._calculate_commission(position_changes[trades])

        # Slippage costs
        slippage_costs = self._calculate_slippage(
            position_changes[trades],
            prices[trades],
            volumes[trades] if volumes is not None else None
        )

        # Spread costs
        spread_costs = self._calculate_spread(position_changes[trades])

        # Total costs
        costs[trades] = commission_costs + slippage_costs + spread_costs

        return costs

    def _calculate_commission(self, position_changes: pd.Series) -> pd.Series:
        """
        Calculate commission costs.

        Commission is charged on the absolute value of position changes.
        """
        # Percentage commission on trade value
        commission = position_changes.abs() * self.commission_pct

        # Add fixed commission if applicable
        if self.commission_fixed > 0:
            # Fixed commission as percentage of trade value
            # (needs to be converted to percentage based on position size)
            # For simplicity, we add it to each trade
            commission = commission + (self.commission_fixed / 100000)  # Normalize

        return commission

    def _calculate_slippage(
        self,
        position_changes: pd.Series,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate slippage costs.

        Slippage increases with:
        - Trade size (larger trades = more slippage)
        - Lower volume (less liquidity = more slippage)
        - Higher volatility (more uncertain execution)
        """
        # Base slippage in decimal (5 bps = 0.0005)
        base_slippage = self.slippage_bps / 10000.0

        # Start with base slippage
        slippage = pd.Series(base_slippage, index=position_changes.index)

        # Volume-based adjustment
        if self.use_volume_slippage and volumes is not None:
            volume_factor = self._calculate_volume_impact(volumes)
            slippage = slippage * volume_factor

        # Volatility-based adjustment
        if self.use_volatility_slippage:
            volatility_factor = self._calculate_volatility_impact(prices)
            slippage = slippage * volatility_factor

        # Apply slippage to absolute position change
        slippage_cost = position_changes.abs() * slippage

        return slippage_cost

    def _calculate_spread(self, position_changes: pd.Series) -> pd.Series:
        """
        Calculate bid-ask spread costs.

        Spread is paid on every trade (buy at ask, sell at bid).
        """
        spread_decimal = self.spread_bps / 10000.0

        # Half-spread cost on absolute position change
        # (we pay half the spread when crossing the spread)
        spread_cost = position_changes.abs() * (spread_decimal / 2)

        return spread_cost

    def _calculate_volume_impact(self, volumes: pd.Series) -> pd.Series:
        """
        Calculate volume impact factor for slippage.

        Lower volume = higher slippage.
        """
        # Normalize volumes to mean volume
        avg_volume = volumes.rolling(20, min_periods=1).mean()

        # Impact factor: higher when volume is below average
        # Formula: 1.0 + (1 - current_vol/avg_vol) * 0.5
        # - If volume = average: factor = 1.0
        # - If volume = 0.5 * average: factor = 1.25
        # - If volume = 2 * average: factor = 0.75
        volume_ratio = volumes / avg_volume.replace(0, 1)  # Avoid division by zero
        impact_factor = 1.0 + (1.0 - volume_ratio) * 0.5

        # Clip to reasonable range
        impact_factor = impact_factor.clip(0.5, 3.0)

        return impact_factor

    def _calculate_volatility_impact(self, prices: pd.Series) -> pd.Series:
        """
        Calculate volatility impact factor for slippage.

        Higher volatility = higher slippage (less certain execution price).
        """
        # Calculate rolling volatility
        returns = prices.pct_change()
        volatility = returns.rolling(20, min_periods=1).std()

        # Normalize to mean volatility
        avg_volatility = volatility.mean()

        if avg_volatility == 0:
            return pd.Series(1.0, index=prices.index)

        volatility_ratio = volatility / avg_volatility

        # Impact factor: higher when volatility is above average
        # Formula: 1.0 + (vol_ratio - 1.0) * 0.3
        # - If vol = average: factor = 1.0
        # - If vol = 2 * average: factor = 1.3
        # - If vol = 0.5 * average: factor = 0.85
        impact_factor = 1.0 + (volatility_ratio - 1.0) * 0.3

        # Clip to reasonable range
        impact_factor = impact_factor.clip(0.5, 2.5).fillna(1.0)

        return impact_factor

    def estimate_total_costs(
        self,
        num_trades: int,
        avg_position_size: float = 1.0,
        holding_period_days: int = 1
    ) -> dict:
        """
        Estimate total trading costs for a strategy.

        Args:
            num_trades: Expected number of trades
            avg_position_size: Average position size (as fraction of portfolio)
            holding_period_days: Average holding period

        Returns:
            Dictionary with cost breakdown
        """
        # Calculate per-trade costs
        commission_per_trade = self.commission_pct * avg_position_size * 2  # Entry + exit
        slippage_per_trade = (self.slippage_bps / 10000.0) * avg_position_size * 2
        spread_per_trade = (self.spread_bps / 10000.0) * avg_position_size * 2

        total_per_trade = commission_per_trade + slippage_per_trade + spread_per_trade

        # Total costs for all trades
        total_commission = commission_per_trade * num_trades
        total_slippage = slippage_per_trade * num_trades
        total_spread = spread_per_trade * num_trades
        total_costs = total_per_trade * num_trades

        # Annualized cost (assuming 252 trading days)
        trades_per_year = num_trades * (252 / (holding_period_days * num_trades))
        annual_cost = total_per_trade * trades_per_year

        logger.info(
            f"Cost estimate: {num_trades} trades, "
            f"avg size={avg_position_size:.1%}, "
            f"total cost={total_costs:.2%}"
        )

        return {
            'num_trades': num_trades,
            'commission_per_trade': commission_per_trade,
            'slippage_per_trade': slippage_per_trade,
            'spread_per_trade': spread_per_trade,
            'total_per_trade': total_per_trade,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_spread': total_spread,
            'total_costs': total_costs,
            'annual_cost': annual_cost
        }

    def compare_scenarios(self) -> pd.DataFrame:
        """
        Compare costs across different trading frequency scenarios.

        Returns:
            DataFrame with cost comparisons
        """
        scenarios = [
            {'name': 'High Frequency', 'trades': 1000, 'size': 0.1, 'days': 1},
            {'name': 'Day Trading', 'trades': 250, 'size': 0.25, 'days': 1},
            {'name': 'Swing Trading', 'trades': 100, 'size': 0.5, 'days': 5},
            {'name': 'Position Trading', 'trades': 20, 'size': 1.0, 'days': 30},
        ]

        results = []
        for scenario in scenarios:
            costs = self.estimate_total_costs(
                num_trades=scenario['trades'],
                avg_position_size=scenario['size'],
                holding_period_days=scenario['days']
            )
            costs['strategy'] = scenario['name']
            results.append(costs)

        return pd.DataFrame(results)
