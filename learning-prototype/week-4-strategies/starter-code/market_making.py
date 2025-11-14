"""
Market Making Strategy - Starter Code with TODOs

Your mission: Implement a market making strategy with inventory management!

This strategy:
1. Quotes both bid and ask prices
2. Captures the spread
3. Manages inventory risk
4. Adjusts quotes based on order book imbalance
5. Handles adverse selection

Difficulty levels:
🟢 Easy: Basic implementation
🟡 Medium: Requires some thinking
🔴 Hard: Advanced concepts

Author: Learning Lab Week 4
"""

from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class OrderSide(Enum):
    """Order side."""
    BID = "bid"
    ASK = "ask"


@dataclass
class MarketMakingConfig:
    """Configuration for market making strategy."""
    # Spread parameters
    base_spread_bps: float = 10.0  # 10 bps base spread
    min_spread_bps: float = 5.0    # Minimum spread
    max_spread_bps: float = 50.0   # Maximum spread

    # Quote size
    base_quote_size: int = 100     # Base size per quote

    # Inventory management
    max_inventory: int = 1000      # Maximum inventory position
    target_inventory: int = 0      # Target inventory (usually 0)
    inventory_skew_factor: float = 0.5  # How much to skew for inventory

    # Risk management
    adverse_selection_lookback: int = 20
    adverse_selection_threshold: float = 0.6  # Stop if >60% adverse fills

    # Order book parameters
    use_order_book_imbalance: bool = True
    imbalance_adjustment_factor: float = 0.3

    # Volatility adjustment
    vol_lookback: int = 20
    vol_spread_multiplier: float = 2.0  # Widen spread in high vol


class MarketMakingStrategy:
    """
    Market making strategy with inventory management.

    Core concepts:
    1. Quote bid and ask prices with spread
    2. Manage inventory to stay market-neutral
    3. Adjust quotes based on market conditions
    4. Avoid adverse selection
    """

    def __init__(self, config: Optional[MarketMakingConfig] = None):
        """Initialize market making strategy."""
        self.config = config or MarketMakingConfig()
        self.current_inventory = 0
        self.pnl = 0.0

    # ========================================
    # Part 1: Fair Price Estimation
    # ========================================

    def estimate_fair_price(
        self,
        bid_price: float,
        ask_price: float,
        bid_size: int,
        ask_size: int
    ) -> float:
        """
        Estimate fair price from bid-ask quotes.

        🟢 TODO #1: Estimate fair price

        Common approaches:
        - Mid price: (bid + ask) / 2
        - Size-weighted: (bid*ask_size + ask*bid_size) / (bid_size + ask_size)

        Use size-weighted for better estimate.

        HINT: Weight by opposite side size

        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            bid_size: Bid size
            ask_size: Ask size

        Returns:
            Estimated fair price
        """
        # YOUR CODE HERE
        pass

    def calculate_microprice(
        self,
        bid_price: float,
        ask_price: float,
        bid_size: int,
        ask_size: int
    ) -> float:
        """
        Calculate microprice (more accurate fair price).

        🟡 TODO #2: Calculate microprice

        Formula: microprice = (bid*ask_size + ask*bid_size) / (bid_size + ask_size)

        This weights prices by opposite side liquidity.

        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            bid_size: Bid size
            ask_size: Ask size

        Returns:
            Microprice
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 2: Spread Calculation
    # ========================================

    def calculate_base_spread(
        self,
        fair_price: float,
        spread_bps: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate base bid-ask spread.

        🟢 TODO #3: Calculate base spread

        Formula:
        - half_spread = fair_price * (spread_bps / 10000) / 2
        - bid = fair_price - half_spread
        - ask = fair_price + half_spread

        HINT: Use self.config.base_spread_bps if spread_bps is None
        HINT: BPS = basis points (1 bps = 0.01%)

        Args:
            fair_price: Fair price estimate
            spread_bps: Spread in basis points

        Returns:
            (bid_price, ask_price) tuple
        """
        # YOUR CODE HERE
        pass

    def adjust_spread_for_volatility(
        self,
        base_spread_bps: float,
        volatility: float,
        avg_volatility: float
    ) -> float:
        """
        Adjust spread based on volatility regime.

        🟡 TODO #4: Adjust for volatility

        Logic:
        - High volatility → wider spread (more risk)
        - Low volatility → tighter spread (less risk)

        Formula: adjusted = base * (1 + multiplier * (vol - avg_vol) / avg_vol)

        HINT: Use self.config.vol_spread_multiplier
        HINT: Clip to min_spread_bps and max_spread_bps

        Args:
            base_spread_bps: Base spread in bps
            volatility: Current volatility
            avg_volatility: Average volatility

        Returns:
            Adjusted spread in bps
        """
        # YOUR CODE HERE
        pass

    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        lookback: Optional[int] = None
    ) -> float:
        """
        Calculate realized volatility.

        🟢 TODO #5: Calculate volatility

        Formula: vol = returns.rolling(lookback).std() * sqrt(periods_per_day)

        HINT: Use self.config.vol_lookback if lookback is None
        HINT: For minute data, periods_per_day = sqrt(390)
        HINT: For daily data, periods_per_day = sqrt(252)

        Args:
            returns: Returns series
            lookback: Lookback periods

        Returns:
            Current volatility (last value)
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 3: Inventory Management
    # ========================================

    def calculate_inventory_skew(
        self,
        current_inventory: int,
        target_inventory: Optional[int] = None,
        max_inventory: Optional[int] = None
    ) -> float:
        """
        Calculate inventory skew factor.

        🟡 TODO #6: Calculate inventory skew

        Logic:
        - If inventory > target: skew positive (make bid less attractive)
        - If inventory < target: skew negative (make ask less attractive)

        Formula: skew = (current - target) / max_inventory

        This returns a value in [-1, 1].

        HINT: Use self.config values if parameters are None

        Args:
            current_inventory: Current inventory position
            target_inventory: Target inventory (default 0)
            max_inventory: Maximum allowed inventory

        Returns:
            Skew factor in [-1, 1]
        """
        # YOUR CODE HERE
        pass

    def apply_inventory_skew_to_quotes(
        self,
        bid_price: float,
        ask_price: float,
        fair_price: float,
        skew: float
    ) -> Tuple[float, float]:
        """
        Adjust bid/ask quotes based on inventory skew.

        🟡 TODO #7: Apply inventory skew

        Logic:
        - Positive skew (long inventory): Widen bid, tighten ask (encourage selling)
        - Negative skew (short inventory): Tighten bid, widen ask (encourage buying)

        Formula (example):
        - spread = ask - bid
        - bid_adjustment = -skew * spread * skew_factor
        - ask_adjustment = +skew * spread * skew_factor

        HINT: Use self.config.inventory_skew_factor
        HINT: Ensure bid < fair < ask after adjustment

        Args:
            bid_price: Base bid price
            ask_price: Base ask price
            fair_price: Fair price
            skew: Inventory skew factor

        Returns:
            (adjusted_bid, adjusted_ask) tuple
        """
        # YOUR CODE HERE
        pass

    def calculate_inventory_risk(
        self,
        inventory: int,
        position_value: float,
        volatility: float
    ) -> float:
        """
        Calculate inventory risk (potential loss from adverse move).

        🟡 TODO #8: Calculate inventory risk

        Formula: risk = abs(inventory * position_value * volatility * sqrt(time))

        This is essentially Value at Risk (VaR) for inventory.

        HINT: Use 1-day time horizon (time = 1/252)
        HINT: Return absolute value

        Args:
            inventory: Current inventory position
            position_value: Value per unit
            volatility: Annualized volatility

        Returns:
            Estimated inventory risk (dollar amount)
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 4: Order Book Analysis
    # ========================================

    def calculate_order_book_imbalance(
        self,
        bid_size: int,
        ask_size: int,
        levels: int = 1
    ) -> float:
        """
        Calculate order book imbalance.

        🟢 TODO #9: Calculate order book imbalance

        Formula: imbalance = (bid_size - ask_size) / (bid_size + ask_size)

        Returns value in [-1, 1]:
        - Positive: More buy pressure (bid heavy)
        - Negative: More sell pressure (ask heavy)

        HINT: Handle zero division case

        Args:
            bid_size: Total bid size
            ask_size: Total ask size
            levels: Number of price levels (for future extension)

        Returns:
            Imbalance in [-1, 1]
        """
        # YOUR CODE HERE
        pass

    def adjust_quotes_for_imbalance(
        self,
        bid_price: float,
        ask_price: float,
        imbalance: float
    ) -> Tuple[float, float]:
        """
        Adjust quotes based on order book imbalance.

        🟡 TODO #10: Adjust for imbalance

        Logic:
        - Positive imbalance (bid heavy): Market likely to move up
          → Make ask more aggressive (lower), bid less aggressive (lower)
        - Negative imbalance (ask heavy): Market likely to move down
          → Make bid more aggressive (higher), ask less aggressive (higher)

        Formula (example):
        - spread = ask - bid
        - adjustment = imbalance * spread * adjustment_factor
        - bid = bid + adjustment
        - ask = ask + adjustment

        HINT: Use self.config.imbalance_adjustment_factor

        Args:
            bid_price: Base bid price
            ask_price: Base ask price
            imbalance: Order book imbalance

        Returns:
            (adjusted_bid, adjusted_ask) tuple
        """
        # YOUR CODE HERE
        pass

    def calculate_book_depth(
        self,
        order_book: Dict[float, int],
        side: str,
        num_levels: int = 5
    ) -> float:
        """
        Calculate total depth in order book.

        🟢 TODO #11: Calculate book depth

        Sum up the sizes of top N levels.

        HINT: order_book is dict of {price: size}
        HINT: For bid side, take highest N prices
        HINT: For ask side, take lowest N prices

        Args:
            order_book: Dict mapping price to size
            side: 'bid' or 'ask'
            num_levels: Number of levels to sum

        Returns:
            Total depth (sum of sizes)
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 5: Adverse Selection Detection
    # ========================================

    def calculate_effective_spread(
        self,
        execution_price: float,
        mid_price: float,
        side: str
    ) -> float:
        """
        Calculate effective spread (measure of transaction cost).

        🟢 TODO #12: Calculate effective spread

        Formula:
        - For buy: effective_spread = 2 * (execution_price - mid_price)
        - For sell: effective_spread = 2 * (mid_price - execution_price)

        HINT: side is 'buy' or 'sell'

        Args:
            execution_price: Actual execution price
            mid_price: Mid price at execution time
            side: 'buy' or 'sell'

        Returns:
            Effective spread (positive value)
        """
        # YOUR CODE HERE
        pass

    def calculate_realized_spread(
        self,
        execution_price: float,
        mid_price_after: float,
        side: str,
        time_horizon: int = 5
    ) -> float:
        """
        Calculate realized spread (profit after adverse selection).

        🟡 TODO #13: Calculate realized spread

        Formula:
        - For buy (we sold): realized = 2 * (execution_price - mid_price_after)
        - For sell (we bought): realized = 2 * (mid_price_after - execution_price)

        If realized < effective: We suffered adverse selection.

        Args:
            execution_price: Our execution price
            mid_price_after: Mid price after time_horizon
            side: 'buy' or 'sell' (customer side, opposite of our side)
            time_horizon: Minutes after execution

        Returns:
            Realized spread
        """
        # YOUR CODE HERE
        pass

    def detect_adverse_selection(
        self,
        execution_prices: pd.Series,
        mid_prices_before: pd.Series,
        mid_prices_after: pd.Series,
        sides: pd.Series,
        lookback: Optional[int] = None
    ) -> bool:
        """
        Detect if we're experiencing adverse selection.

        🔴 TODO #14: Detect adverse selection

        Adverse selection occurs when:
        - We buy, then price drops (we overpaid)
        - We sell, then price rises (we undersold)

        Calculate ratio of adverse fills to total fills.

        HINT: Calculate realized spread for each fill
        HINT: Adverse if realized < 0
        HINT: If ratio > threshold, return True

        Args:
            execution_prices: Our execution prices
            mid_prices_before: Mid prices before fills
            mid_prices_after: Mid prices after fills
            sides: Customer sides ('buy' or 'sell')
            lookback: Number of recent fills to check

        Returns:
            True if adverse selection detected
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 6: Quote Sizing
    # ========================================

    def calculate_optimal_quote_size(
        self,
        base_size: int,
        inventory: int,
        max_inventory: int,
        volatility: float
    ) -> Tuple[int, int]:
        """
        Calculate optimal bid and ask sizes.

        🟡 TODO #15: Calculate quote sizes

        Logic:
        - If near max inventory: Reduce bid size, increase ask size
        - If near min inventory: Increase bid size, reduce ask size
        - Reduce sizes in high volatility

        HINT: Use inventory skew to adjust
        HINT: Ensure sizes are positive

        Args:
            base_size: Base quote size
            inventory: Current inventory
            max_inventory: Maximum inventory
            volatility: Current volatility

        Returns:
            (bid_size, ask_size) tuple
        """
        # YOUR CODE HERE
        pass

    def apply_size_limits(
        self,
        bid_size: int,
        ask_size: int,
        current_inventory: int,
        max_inventory: int
    ) -> Tuple[int, int]:
        """
        Apply inventory limits to quote sizes.

        🟡 TODO #16: Apply size limits

        Logic:
        - If inventory + bid_size > max_inventory: Reduce bid_size
        - If inventory - ask_size < -max_inventory: Reduce ask_size
        - Set to 0 if at limit

        Args:
            bid_size: Desired bid size
            ask_size: Desired ask size
            current_inventory: Current inventory
            max_inventory: Max allowed inventory

        Returns:
            (limited_bid_size, limited_ask_size) tuple
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 7: PnL Tracking
    # ========================================

    def calculate_trade_pnl(
        self,
        execution_price: float,
        size: int,
        side: str,
        fees: float = 0.0
    ) -> float:
        """
        Calculate PnL from a single trade.

        🟢 TODO #17: Calculate trade PnL

        For market maker:
        - We buy (provide ask liquidity): -execution_price * size - fees
        - We sell (provide bid liquidity): +execution_price * size - fees

        HINT: side is 'buy' or 'sell' (customer side, opposite of ours)

        Args:
            execution_price: Trade price
            size: Trade size
            side: Customer side ('buy' or 'sell')
            fees: Exchange fees

        Returns:
            PnL (positive = profit)
        """
        # YOUR CODE HERE
        pass

    def calculate_inventory_pnl(
        self,
        inventory: int,
        entry_price: float,
        current_price: float
    ) -> float:
        """
        Calculate unrealized PnL from inventory.

        🟢 TODO #18: Calculate inventory PnL

        Formula: pnl = inventory * (current_price - entry_price)

        Positive inventory (long): Profit if price increases
        Negative inventory (short): Profit if price decreases

        Args:
            inventory: Current inventory (positive or negative)
            entry_price: Average entry price
            current_price: Current market price

        Returns:
            Unrealized PnL
        """
        # YOUR CODE HERE
        pass

    def calculate_total_pnl(
        self,
        realized_pnl: float,
        unrealized_pnl: float
    ) -> float:
        """
        Calculate total PnL.

        🟢 TODO #19: Calculate total PnL

        HINT: Simple addition!

        Args:
            realized_pnl: Realized PnL from closed trades
            unrealized_pnl: Unrealized PnL from open inventory

        Returns:
            Total PnL
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 8: Quote Generation
    # ========================================

    def generate_quotes(
        self,
        fair_price: float,
        volatility: float,
        avg_volatility: float,
        current_inventory: int,
        bid_size_book: int,
        ask_size_book: int
    ) -> Dict:
        """
        Generate bid and ask quotes.

        🔴 TODO #20: Generate quotes

        Steps:
        1. Calculate base spread
        2. Adjust spread for volatility
        3. Calculate base bid/ask from fair price and spread
        4. Calculate inventory skew
        5. Apply inventory skew to quotes
        6. Calculate order book imbalance
        7. Adjust quotes for imbalance
        8. Calculate quote sizes
        9. Apply size limits
        10. Return quote dict

        Args:
            fair_price: Estimated fair price
            volatility: Current volatility
            avg_volatility: Average volatility
            current_inventory: Current inventory
            bid_size_book: Bid size in order book
            ask_size_book: Ask size in order book

        Returns:
            Dict with:
            - 'bid_price': Bid quote price
            - 'ask_price': Ask quote price
            - 'bid_size': Bid quote size
            - 'ask_size': Ask quote size
            - 'spread_bps': Spread in bps
            - 'inventory_skew': Inventory skew factor
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 9: Main Strategy Logic
    # ========================================

    def run_strategy(
        self,
        prices: pd.Series,
        bid_prices: pd.Series,
        ask_prices: pd.Series,
        bid_sizes: pd.Series,
        ask_sizes: pd.Series
    ) -> Dict:
        """
        Run complete market making strategy.

        🔴 TODO #21: Implement main strategy

        Steps:
        1. Calculate returns and volatility
        2. For each timestamp:
           a. Estimate fair price
           b. Calculate microprice
           c. Generate quotes
           d. Simulate fills (simplified)
           e. Update inventory
           f. Calculate PnL
        3. Return results

        HINT: This is complex - focus on quote generation first
        HINT: For now, skip actual fill simulation
        HINT: Track quotes over time

        Args:
            prices: Price series (trades)
            bid_prices: Best bid prices
            ask_prices: Best ask prices
            bid_sizes: Bid sizes
            ask_sizes: Ask sizes

        Returns:
            Dict with:
            - 'quotes': DataFrame of generated quotes
            - 'inventory': Inventory series
            - 'pnl': PnL series
            - 'fair_prices': Fair price series
        """
        # YOUR CODE HERE
        pass

    def calculate_strategy_metrics(
        self,
        quotes: pd.DataFrame,
        inventory: pd.Series,
        pnl: pd.Series
    ) -> Dict:
        """
        Calculate strategy performance metrics.

        🟡 TODO #22: Calculate metrics

        Metrics to calculate:
        - Total PnL
        - Average spread captured
        - Fill rate (assumed)
        - Average inventory
        - Max inventory
        - Sharpe ratio
        - PnL per trade

        Args:
            quotes: DataFrame with quote history
            inventory: Inventory series
            pnl: PnL series

        Returns:
            Dict with metrics
        """
        # YOUR CODE HERE
        pass


# ========================================
# Self-Test Function
# ========================================

def test_implementation():
    """
    Test your market making implementation!

    🎯 Run this function to check if your code works:
        python market_making.py
    """
    print("🧪 Testing Market Making Implementation...\n")

    # Generate synthetic market data
    np.random.seed(42)
    n = 390  # One trading day

    timestamps = pd.date_range('2024-01-15 09:30', periods=n, freq='1min')

    # Simulate bid-ask quotes
    mid = 100.0
    spread = 0.02
    bid_prices = pd.Series(mid - spread/2 + np.random.normal(0, 0.001, n), index=timestamps)
    ask_prices = pd.Series(mid + spread/2 + np.random.normal(0, 0.001, n), index=timestamps)
    prices = (bid_prices + ask_prices) / 2

    bid_sizes = pd.Series(np.random.randint(100, 500, n), index=timestamps)
    ask_sizes = pd.Series(np.random.randint(100, 500, n), index=timestamps)

    # Create strategy
    strategy = MarketMakingStrategy()

    # Test 1: Fair price
    print("Test 1: Fair Price Estimation")
    try:
        fair = strategy.estimate_fair_price(
            bid_prices.iloc[0],
            ask_prices.iloc[0],
            bid_sizes.iloc[0],
            ask_sizes.iloc[0]
        )
        microprice = strategy.calculate_microprice(
            bid_prices.iloc[0],
            ask_prices.iloc[0],
            bid_sizes.iloc[0],
            ask_sizes.iloc[0]
        )
        print(f"✅ Fair price: ${fair:.4f}")
        print(f"   Microprice: ${microprice:.4f}")
    except:
        print("❌ Fair price functions not implemented")

    # Test 2: Spread calculation
    print("\nTest 2: Spread Calculation")
    try:
        bid, ask = strategy.calculate_base_spread(100.0, spread_bps=10)
        spread_bps = (ask - bid) / 100.0 * 10000
        print(f"✅ Base spread calculated:")
        print(f"   Bid: ${bid:.4f}")
        print(f"   Ask: ${ask:.4f}")
        print(f"   Spread: {spread_bps:.2f} bps")
    except:
        print("❌ calculate_base_spread() not implemented")

    # Test 3: Inventory skew
    print("\nTest 3: Inventory Management")
    try:
        skew = strategy.calculate_inventory_skew(500, 0, 1000)
        print(f"✅ Inventory skew calculated: {skew:.3f}")

        adj_bid, adj_ask = strategy.apply_inventory_skew_to_quotes(
            99.99, 100.01, 100.0, skew
        )
        print(f"   Adjusted bid: ${adj_bid:.4f}")
        print(f"   Adjusted ask: ${adj_ask:.4f}")
    except:
        print("❌ Inventory management functions not implemented")

    # Test 4: Order book imbalance
    print("\nTest 4: Order Book Imbalance")
    try:
        imbalance = strategy.calculate_order_book_imbalance(300, 200)
        print(f"✅ Order book imbalance: {imbalance:.3f}")

        adj_bid, adj_ask = strategy.adjust_quotes_for_imbalance(
            99.99, 100.01, imbalance
        )
        print(f"   Imbalance-adjusted bid: ${adj_bid:.4f}")
        print(f"   Imbalance-adjusted ask: ${adj_ask:.4f}")
    except:
        print("❌ Order book functions not implemented")

    # Test 5: PnL calculation
    print("\nTest 5: PnL Calculation")
    try:
        trade_pnl = strategy.calculate_trade_pnl(100.01, 100, 'buy', fees=0.1)
        inv_pnl = strategy.calculate_inventory_pnl(100, 99.95, 100.05)
        total_pnl = strategy.calculate_total_pnl(trade_pnl, inv_pnl)
        print(f"✅ PnL calculated:")
        print(f"   Trade PnL: ${trade_pnl:.2f}")
        print(f"   Inventory PnL: ${inv_pnl:.2f}")
        print(f"   Total PnL: ${total_pnl:.2f}")
    except:
        print("❌ PnL functions not implemented")

    # Test 6: Quote sizing
    print("\nTest 6: Quote Sizing")
    try:
        bid_size, ask_size = strategy.calculate_optimal_quote_size(
            100, 200, 1000, 0.02
        )
        print(f"✅ Quote sizes calculated:")
        print(f"   Bid size: {bid_size}")
        print(f"   Ask size: {ask_size}")
    except:
        print("❌ calculate_optimal_quote_size() not implemented")

    # Test 7: Volatility
    print("\nTest 7: Volatility Calculation")
    try:
        returns = prices.pct_change()
        vol = strategy.calculate_realized_volatility(returns, lookback=20)
        print(f"✅ Volatility: {vol:.2%}")
    except:
        print("❌ calculate_realized_volatility() not implemented")

    # Test 8: Quote generation
    print("\nTest 8: Quote Generation")
    try:
        quotes = strategy.generate_quotes(
            fair_price=100.0,
            volatility=0.02,
            avg_volatility=0.02,
            current_inventory=100,
            bid_size_book=300,
            ask_size_book=250
        )
        print(f"✅ Quotes generated:")
        print(f"   Bid: ${quotes['bid_price']:.4f} x {quotes['bid_size']}")
        print(f"   Ask: ${quotes['ask_price']:.4f} x {quotes['ask_size']}")
        print(f"   Spread: {quotes['spread_bps']:.2f} bps")
    except:
        print("❌ generate_quotes() not implemented")

    # Test 9: Full strategy
    print("\nTest 9: Full Strategy")
    try:
        result = strategy.run_strategy(
            prices, bid_prices, ask_prices, bid_sizes, ask_sizes
        )
        print(f"✅ Strategy run successfully!")
        if 'quotes' in result:
            print(f"   Quotes generated: {len(result['quotes'])}")
    except:
        print("❌ run_strategy() not implemented")

    print("\n" + "="*50)
    print("🎉 Testing complete!")
    print("="*50)
    print("\n💡 Market making is complex! Focus on:")
    print("   1. Quote generation with inventory management")
    print("   2. Spread adjustment for volatility")
    print("   3. Order book imbalance detection")
    print("   4. Adverse selection avoidance")


if __name__ == "__main__":
    test_implementation()
