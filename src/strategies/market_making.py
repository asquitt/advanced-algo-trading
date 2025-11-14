"""
Market Making Strategy with Inventory Management

Provides liquidity by placing simultaneous buy and sell orders around the mid-price,
capturing the bid-ask spread while managing inventory risk.

Key Concepts:
- Bid-Ask Spread: Profit from the difference between buy and sell prices
- Inventory Risk: Managing position size to avoid directional exposure
- Order Book Imbalance: Adjusting quotes based on supply/demand
- Adverse Selection: Avoiding informed traders

Author: LLM Trading Platform
"""

from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class MarketMakerConfig:
    """Configuration for market making strategy."""
    spread_bps: float = 10.0  # Target spread in basis points
    max_inventory: int = 1000  # Maximum inventory (shares)
    target_inventory: int = 0  # Target inventory (neutral)
    inventory_skew_factor: float = 0.5  # How much to skew quotes based on inventory
    order_size: int = 100  # Size of each quote
    min_edge_bps: float = 5.0  # Minimum edge to quote
    quote_ttl_seconds: int = 5  # Time to live for quotes
    cancel_on_fill: bool = True  # Cancel opposite side on fill
    use_orderbook_imbalance: bool = True  # Adjust based on order book


class MarketMakingStrategy:
    """
    High-frequency market making strategy.

    Strategy:
    1. Calculate fair price (mid-price or model-based)
    2. Place buy order at (fair_price - spread/2)
    3. Place sell order at (fair_price + spread/2)
    4. Adjust quotes based on inventory
    5. Manage risk through position limits
    """

    def __init__(self, config: Optional[MarketMakerConfig] = None):
        """
        Initialize market making strategy.

        Args:
            config: Market maker configuration
        """
        self.config = config or MarketMakerConfig()
        self.current_inventory = 0
        self.active_orders = {'bid': None, 'ask': None}

        logger.info(
            f"MarketMakingStrategy initialized: "
            f"spread={self.config.spread_bps}bps, "
            f"max_inventory={self.config.max_inventory}"
        )

    def calculate_fair_price(
        self,
        data: pd.DataFrame,
        method: str = 'mid'
    ) -> float:
        """
        Calculate fair price for market making.

        Args:
            data: Market data with OHLC
            method: 'mid' (mid-price), 'vwap', or 'micro'

        Returns:
            Fair price estimate
        """
        if method == 'mid':
            # Simple mid-price
            if 'bid' in data.columns and 'ask' in data.columns:
                bid = data['bid'].iloc[-1]
                ask = data['ask'].iloc[-1]
                return (bid + ask) / 2
            else:
                # Fallback to close price
                return data['close'].iloc[-1]

        elif method == 'vwap':
            # Volume-weighted average price
            if 'vwap' in data.columns:
                return data['vwap'].iloc[-1]
            else:
                # Calculate VWAP
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                vwap = (typical_price * data['volume']).sum() / data['volume'].sum()
                return vwap

        elif method == 'micro':
            # Microprice (order book weighted mid)
            if 'bid' in data.columns and 'ask' in data.columns:
                bid = data['bid'].iloc[-1]
                ask = data['ask'].iloc[-1]
                bid_size = data.get('bid_size', pd.Series([1])).iloc[-1]
                ask_size = data.get('ask_size', pd.Series([1])).iloc[-1]

                # Weighted by inverse of size (more weight to smaller size side)
                total_size = bid_size + ask_size
                if total_size > 0:
                    microprice = (ask * bid_size + bid * ask_size) / total_size
                    return microprice
                else:
                    return (bid + ask) / 2
            else:
                return data['close'].iloc[-1]

        return data['close'].iloc[-1]

    def calculate_orderbook_imbalance(
        self,
        data: pd.DataFrame,
        levels: int = 5
    ) -> float:
        """
        Calculate order book imbalance.

        Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Args:
            data: Market data with order book
            levels: Number of levels to consider

        Returns:
            Imbalance (-1 to 1, negative = more asks, positive = more bids)
        """
        if 'bid_levels' not in data.columns or 'ask_levels' not in data.columns:
            return 0.0

        bid_levels = data['bid_levels'].iloc[-1]
        ask_levels = data['ask_levels'].iloc[-1]

        if not bid_levels or not ask_levels:
            return 0.0

        # Sum volume at top N levels
        bid_volume = sum(size for price, size in bid_levels[:levels])
        ask_volume = sum(size for price, size in ask_levels[:levels])

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        imbalance = (bid_volume - ask_volume) / total_volume

        return imbalance

    def calculate_inventory_skew(self) -> float:
        """
        Calculate inventory skew adjustment.

        Positive inventory (long) -> widen ask, tighten bid (encourage sells)
        Negative inventory (short) -> widen bid, tighten ask (encourage buys)

        Returns:
            Skew factor in basis points
        """
        # Normalize inventory to -1 to 1
        if self.config.max_inventory == 0:
            return 0.0

        inventory_pct = self.current_inventory / self.config.max_inventory

        # Skew spreads based on inventory
        # If long, make it easier to sell (tighter ask spread)
        # If short, make it easier to buy (tighter bid spread)
        skew_bps = inventory_pct * self.config.inventory_skew_factor * self.config.spread_bps

        return skew_bps

    def calculate_quote_prices(
        self,
        fair_price: float,
        volatility: Optional[float] = None,
        imbalance: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate bid and ask quote prices.

        Args:
            fair_price: Fair/mid price
            volatility: Current volatility (optional, for dynamic spreads)
            imbalance: Order book imbalance

        Returns:
            (bid_price, ask_price) tuple
        """
        # Base spread
        spread_decimal = self.config.spread_bps / 10000.0

        # Adjust spread based on volatility
        if volatility is not None and volatility > 0:
            # Widen spread in high volatility
            vol_adjustment = min(volatility / 0.02, 2.0)  # Cap at 2x
            spread_decimal *= vol_adjustment

        # Inventory skew
        inventory_skew_bps = self.calculate_inventory_skew()
        inventory_skew_decimal = inventory_skew_bps / 10000.0

        # Order book imbalance adjustment
        # If more bids (positive imbalance), widen bid spread (less aggressive)
        # If more asks (negative imbalance), widen ask spread (less aggressive)
        imbalance_adjustment = 0.0
        if self.config.use_orderbook_imbalance:
            imbalance_adjustment = imbalance * spread_decimal * 0.5

        # Calculate bid and ask spreads from mid
        bid_spread = spread_decimal / 2 + inventory_skew_decimal + imbalance_adjustment
        ask_spread = spread_decimal / 2 - inventory_skew_decimal - imbalance_adjustment

        # Ensure minimum edge
        min_edge = self.config.min_edge_bps / 10000.0
        bid_spread = max(bid_spread, min_edge)
        ask_spread = max(ask_spread, min_edge)

        bid_price = fair_price * (1 - bid_spread)
        ask_price = fair_price * (1 + ask_spread)

        return bid_price, ask_price

    def check_inventory_limits(self, side: str, quantity: int) -> bool:
        """
        Check if trade would exceed inventory limits.

        Args:
            side: 'buy' or 'sell'
            quantity: Trade quantity

        Returns:
            True if trade is allowed
        """
        if side == 'buy':
            new_inventory = self.current_inventory + quantity
        else:
            new_inventory = self.current_inventory - quantity

        return abs(new_inventory) <= self.config.max_inventory

    def update_inventory(self, side: str, quantity: int):
        """
        Update inventory after trade execution.

        Args:
            side: 'buy' or 'sell'
            quantity: Trade quantity
        """
        if side == 'buy':
            self.current_inventory += quantity
        else:
            self.current_inventory -= quantity

        logger.debug(f"Inventory updated: {self.current_inventory}")

    def generate_quotes(
        self,
        data: pd.DataFrame,
        fair_price_method: str = 'mid'
    ) -> Dict[str, Dict]:
        """
        Generate bid and ask quotes.

        Args:
            data: Market data
            fair_price_method: Method to calculate fair price

        Returns:
            Dictionary with 'bid' and 'ask' quote details
        """
        # Calculate fair price
        fair_price = self.calculate_fair_price(data, method=fair_price_method)

        # Calculate volatility (for dynamic spreads)
        if len(data) >= 20:
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
        else:
            volatility = None

        # Calculate order book imbalance
        imbalance = self.calculate_orderbook_imbalance(data)

        # Calculate quote prices
        bid_price, ask_price = self.calculate_quote_prices(
            fair_price,
            volatility=volatility,
            imbalance=imbalance
        )

        # Check inventory limits
        can_buy = self.check_inventory_limits('buy', self.config.order_size)
        can_sell = self.check_inventory_limits('sell', self.config.order_size)

        quotes = {}

        if can_buy:
            quotes['bid'] = {
                'price': bid_price,
                'size': self.config.order_size,
                'side': 'buy',
                'ttl': self.config.quote_ttl_seconds
            }

        if can_sell:
            quotes['ask'] = {
                'price': ask_price,
                'size': self.config.order_size,
                'side': 'sell',
                'ttl': self.config.quote_ttl_seconds
            }

        logger.debug(
            f"Quotes generated: fair={fair_price:.2f}, "
            f"bid={bid_price:.2f}, ask={ask_price:.2f}, "
            f"inventory={self.current_inventory}"
        )

        return quotes

    def backtest_signal_function(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Signal function for backtesting.

        Market making doesn't have traditional buy/sell signals.
        Instead, we simulate capturing the spread when both sides execute.

        Args:
            data: Market data
            **kwargs: Additional parameters

        Returns:
            Position series (mostly neutral, occasional small positions)
        """
        signals = pd.Series(0, index=data.index)

        # Market making is market-neutral, so we don't generate directional signals
        # In a real backtest, you'd simulate both bid and ask executions
        # For simplicity, we'll generate small mean-reverting signals

        if len(data) < 20:
            return signals

        # Calculate fair price deviation
        fair_price = self.calculate_fair_price(data)
        current_price = data['close'].iloc[-1]

        deviation = (current_price - fair_price) / fair_price

        # Small mean-reversion signals (market making profits from mean reversion)
        if deviation > 0.001:  # Price above fair -> sell signal
            signals.iloc[-1] = -0.1  # Small short position
        elif deviation < -0.001:  # Price below fair -> buy signal
            signals.iloc[-1] = 0.1  # Small long position

        return signals

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        side: str
    ) -> float:
        """
        Calculate P&L from a trade.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            side: 'buy' or 'sell'

        Returns:
            P&L in dollars
        """
        if side == 'buy':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        return pnl

    def calculate_expected_spread_capture(
        self,
        fair_price: float,
        bid_price: float,
        ask_price: float,
        execution_probability: float = 0.5
    ) -> float:
        """
        Calculate expected profit from spread capture.

        Args:
            fair_price: Fair/mid price
            bid_price: Bid quote price
            ask_price: Ask quote price
            execution_probability: Probability of both sides executing

        Returns:
            Expected spread capture in dollars
        """
        bid_edge = fair_price - bid_price
        ask_edge = ask_price - fair_price

        # Expected profit if both sides execute
        spread_profit = (bid_edge + ask_edge) * self.config.order_size

        # Adjust for execution probability
        expected_profit = spread_profit * execution_probability

        return expected_profit
