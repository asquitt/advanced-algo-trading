"""
Slippage Management System

Reduces slippage in fast-moving markets through:
1. Adaptive order execution (TWAP, VWAP, Iceberg)
2. Market impact modeling
3. Optimal execution timing
4. Real-time spread monitoring
5. Liquidity-aware routing

Author: LLM Trading Platform
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np
from loguru import logger

from src.data_layer.models import OrderSide, OrderType
from src.trading_engine.hft_techniques import (
    OrderBookSnapshot,
    MarketMicrostructure,
    SmartOrderRouting
)


class ExecutionUrgency(Enum):
    """Urgency level for order execution."""
    LOW = "low"  # Maximize cost savings, allow longer execution time
    MEDIUM = "medium"  # Balance cost and time
    HIGH = "high"  # Minimize time, accept higher costs
    CRITICAL = "critical"  # Execute immediately, ignore costs


class SlippageStrategy(Enum):
    """Strategy for minimizing slippage."""
    IMMEDIATE = "immediate"  # Market order (highest slippage)
    LIMIT_IOC = "limit_ioc"  # Limit order, immediate or cancel
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    ICEBERG = "iceberg"  # Hidden size with small visible portion
    ADAPTIVE = "adaptive"  # Adjust based on market conditions


@dataclass
class MarketConditions:
    """Current market conditions for slippage assessment."""
    symbol: str
    timestamp: datetime
    volatility: float  # Realized volatility (%)
    spread_bps: float  # Bid-ask spread in basis points
    volume: int  # Recent volume
    momentum: float  # Price momentum (-1 to 1)
    order_imbalance: float  # Buy/sell pressure (0 to 1)
    liquidity_score: float  # 0 to 100
    is_fast_market: bool  # True if rapid price changes detected


@dataclass
class SlippageEstimate:
    """Expected slippage for an order."""
    expected_slippage_bps: float  # Expected slippage in basis points
    min_slippage_bps: float  # Best case
    max_slippage_bps: float  # Worst case
    confidence: float  # Confidence in estimate (0-1)
    recommended_strategy: SlippageStrategy
    execution_time_seconds: int  # Recommended execution time
    reasoning: str  # Explanation of recommendation


@dataclass
class ExecutionOrder:
    """Order with execution strategy."""
    symbol: str
    side: OrderSide
    quantity: int
    limit_price: Optional[float]
    strategy: SlippageStrategy
    urgency: ExecutionUrgency
    max_slippage_bps: float
    time_horizon_seconds: int

    # Child orders for splitting
    child_orders: List['ExecutionOrder'] = None

    # Execution tracking
    executed_quantity: int = 0
    avg_fill_price: float = 0.0
    actual_slippage_bps: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class SlippageAnalyzer:
    """Analyzes market conditions and estimates slippage."""

    def __init__(self):
        self.microstructure = MarketMicrostructure()

        # Calibration parameters (can be adjusted based on historical data)
        self.base_slippage_bps = {
            OrderType.MARKET: 5.0,
            OrderType.LIMIT: 2.0,
        }

        # Historical slippage tracking for learning
        self.historical_slippage: Dict[str, List[float]] = {}

    def assess_market_conditions(
        self,
        symbol: str,
        order_book: OrderBookSnapshot,
        recent_prices: List[float],
        recent_volumes: List[int]
    ) -> MarketConditions:
        """
        Assess current market conditions for slippage estimation.

        Args:
            symbol: Trading symbol
            order_book: Current order book state
            recent_prices: Recent price history (e.g., last 60 seconds)
            recent_volumes: Recent volume history

        Returns:
            MarketConditions object
        """
        # Calculate volatility (standard deviation of returns)
        if len(recent_prices) >= 2:
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns) * 100  # As percentage
        else:
            volatility = 0.0

        # Calculate momentum (price trend)
        if len(recent_prices) >= 10:
            early_avg = np.mean(recent_prices[:5])
            recent_avg = np.mean(recent_prices[-5:])
            momentum = (recent_avg - early_avg) / early_avg
            momentum = np.clip(momentum, -1.0, 1.0)
        else:
            momentum = 0.0

        # Detect fast market (rapid price changes)
        is_fast_market = False
        if len(recent_prices) >= 10:
            # Check if any 1-second move exceeds 0.1%
            for i in range(1, len(recent_prices)):
                pct_change = abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                if pct_change > 0.001:  # 0.1% in 1 second
                    is_fast_market = True
                    break

        # Calculate liquidity score
        liquidity_score = self.microstructure.calculate_liquidity_score(
            symbol, order_book
        )

        return MarketConditions(
            symbol=symbol,
            timestamp=datetime.now(),
            volatility=volatility,
            spread_bps=order_book.spread_bps,
            volume=sum(recent_volumes) if recent_volumes else 0,
            momentum=momentum,
            order_imbalance=order_book.order_imbalance,
            liquidity_score=liquidity_score,
            is_fast_market=is_fast_market
        )

    def estimate_market_impact(
        self,
        quantity: int,
        side: OrderSide,
        order_book: OrderBookSnapshot,
        conditions: MarketConditions
    ) -> float:
        """
        Estimate market impact using enhanced Kyle's Lambda model.

        Args:
            quantity: Order size
            side: Buy or sell
            order_book: Current order book
            conditions: Market conditions

        Returns:
            Estimated market impact in basis points
        """
        # Use HFT techniques for base impact
        base_impact_bps = self.microstructure.estimate_price_impact(
            conditions.symbol, quantity, order_book
        )

        # Adjust for market conditions
        volatility_multiplier = 1.0 + (conditions.volatility / 100.0)

        # Fast market penalty (slippage increases in fast markets)
        fast_market_penalty = 1.5 if conditions.is_fast_market else 1.0

        # Liquidity adjustment
        liquidity_multiplier = 1.0
        if conditions.liquidity_score < 30:
            liquidity_multiplier = 2.0  # Low liquidity = higher impact
        elif conditions.liquidity_score < 60:
            liquidity_multiplier = 1.3

        # Momentum penalty (trading against momentum increases impact)
        momentum_penalty = 1.0
        if (side == OrderSide.BUY and conditions.momentum < -0.2) or \
           (side == OrderSide.SELL and conditions.momentum > 0.2):
            momentum_penalty = 1.3  # Trading against strong momentum

        adjusted_impact = (
            base_impact_bps *
            volatility_multiplier *
            fast_market_penalty *
            liquidity_multiplier *
            momentum_penalty
        )

        return adjusted_impact

    def estimate_slippage(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        urgency: ExecutionUrgency,
        order_book: OrderBookSnapshot,
        conditions: MarketConditions
    ) -> SlippageEstimate:
        """
        Estimate slippage and recommend execution strategy.

        Args:
            symbol: Trading symbol
            quantity: Order size
            side: Buy or sell
            urgency: Execution urgency
            order_book: Current order book
            conditions: Market conditions

        Returns:
            SlippageEstimate with recommended strategy
        """
        # Estimate market impact
        market_impact_bps = self.estimate_market_impact(
            quantity, side, order_book, conditions
        )

        # Add spread cost (always pay at least half the spread)
        spread_cost_bps = conditions.spread_bps / 2.0

        # Base expected slippage
        expected_slippage = market_impact_bps + spread_cost_bps

        # Determine strategy based on urgency and conditions
        if urgency == ExecutionUrgency.CRITICAL:
            strategy = SlippageStrategy.IMMEDIATE
            execution_time = 1
            # Add urgency premium
            expected_slippage *= 1.5

        elif urgency == ExecutionUrgency.HIGH:
            if conditions.is_fast_market:
                strategy = SlippageStrategy.LIMIT_IOC
                execution_time = 5
            else:
                strategy = SlippageStrategy.IMMEDIATE
                execution_time = 2
            expected_slippage *= 1.2

        elif urgency == ExecutionUrgency.LOW:
            # Use patient strategies
            if conditions.liquidity_score > 60:
                strategy = SlippageStrategy.TWAP
                execution_time = 300  # 5 minutes
                expected_slippage *= 0.6  # Significant savings
            else:
                strategy = SlippageStrategy.VWAP
                execution_time = 180  # 3 minutes
                expected_slippage *= 0.7

        else:  # MEDIUM urgency
            if conditions.is_fast_market:
                strategy = SlippageStrategy.ADAPTIVE
                execution_time = 30
                expected_slippage *= 0.9
            elif conditions.liquidity_score < 40:
                strategy = SlippageStrategy.ICEBERG
                execution_time = 60
                expected_slippage *= 0.8
            else:
                strategy = SlippageStrategy.VWAP
                execution_time = 120
                expected_slippage *= 0.75

        # Calculate confidence based on data quality
        confidence = 0.7
        if len(self.historical_slippage.get(symbol, [])) > 10:
            confidence = 0.85
        if conditions.liquidity_score > 70:
            confidence += 0.1
        if conditions.is_fast_market:
            confidence -= 0.15
        confidence = np.clip(confidence, 0.3, 0.95)

        # Estimate ranges
        min_slippage = expected_slippage * 0.5
        max_slippage = expected_slippage * 2.0

        # Generate reasoning
        reasoning_parts = []
        if conditions.is_fast_market:
            reasoning_parts.append("Fast market detected - increased slippage risk")
        if conditions.liquidity_score < 40:
            reasoning_parts.append("Low liquidity - using cautious approach")
        if market_impact_bps > 20:
            reasoning_parts.append(f"Large order relative to liquidity (impact: {market_impact_bps:.1f}bps)")
        if urgency in [ExecutionUrgency.LOW, ExecutionUrgency.MEDIUM]:
            reasoning_parts.append(f"Using {strategy.value} to minimize costs")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Normal market conditions"

        return SlippageEstimate(
            expected_slippage_bps=expected_slippage,
            min_slippage_bps=min_slippage,
            max_slippage_bps=max_slippage,
            confidence=confidence,
            recommended_strategy=strategy,
            execution_time_seconds=execution_time,
            reasoning=reasoning
        )

    def record_actual_slippage(
        self,
        symbol: str,
        estimated_slippage_bps: float,
        actual_slippage_bps: float
    ):
        """Record actual slippage for model improvement."""
        if symbol not in self.historical_slippage:
            self.historical_slippage[symbol] = []

        error = actual_slippage_bps - estimated_slippage_bps
        self.historical_slippage[symbol].append(error)

        # Keep only recent history (last 100 trades)
        if len(self.historical_slippage[symbol]) > 100:
            self.historical_slippage[symbol] = self.historical_slippage[symbol][-100:]

        # Log significant estimation errors
        if abs(error) > 10.0:  # More than 10bps error
            logger.warning(
                f"Significant slippage estimation error for {symbol}: "
                f"estimated={estimated_slippage_bps:.2f}bps, "
                f"actual={actual_slippage_bps:.2f}bps, "
                f"error={error:.2f}bps"
            )


class AdaptiveExecutor:
    """
    Executes orders with adaptive strategies to minimize slippage.

    Key Features:
    - Dynamic order splitting based on market conditions
    - Real-time adjustment to execution strategy
    - Cancels and re-routes if conditions deteriorate
    """

    def __init__(
        self,
        slippage_analyzer: SlippageAnalyzer,
        smart_router: SmartOrderRouting
    ):
        self.analyzer = slippage_analyzer
        self.router = smart_router
        self.active_orders: Dict[str, ExecutionOrder] = {}

    def create_execution_plan(
        self,
        order: ExecutionOrder,
        conditions: MarketConditions,
        order_book: OrderBookSnapshot
    ) -> List[ExecutionOrder]:
        """
        Create execution plan by splitting order if needed.

        Args:
            order: Parent order to execute
            conditions: Market conditions
            order_book: Current order book

        Returns:
            List of child orders (or single order if no split needed)
        """
        # Small orders don't need splitting
        order_value = order.quantity * order_book.mid_price
        if order_value < 10000 or order.urgency == ExecutionUrgency.CRITICAL:
            return [order]

        # Determine number of splits based on market impact
        impact = self.analyzer.estimate_market_impact(
            order.quantity, order.side, order_book, conditions
        )

        if impact < 10:
            num_splits = 1  # No split needed
        elif impact < 30:
            num_splits = 3
        elif impact < 50:
            num_splits = 5
        else:
            num_splits = 10  # High impact - aggressive splitting

        if num_splits == 1:
            return [order]

        # Split order
        base_qty = order.quantity // num_splits
        remainder = order.quantity % num_splits

        child_orders = []
        for i in range(num_splits):
            child_qty = base_qty + (1 if i < remainder else 0)

            child_order = ExecutionOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=child_qty,
                limit_price=order.limit_price,
                strategy=order.strategy,
                urgency=order.urgency,
                max_slippage_bps=order.max_slippage_bps,
                time_horizon_seconds=order.time_horizon_seconds // num_splits
            )
            child_orders.append(child_order)

        logger.info(
            f"Split order {order.symbol} into {num_splits} children "
            f"(market impact: {impact:.1f}bps)"
        )

        return child_orders

    def execute_with_strategy(
        self,
        order: ExecutionOrder,
        conditions: MarketConditions,
        order_book: OrderBookSnapshot
    ) -> Dict:
        """
        Execute order using specified strategy.

        Args:
            order: Order to execute
            conditions: Market conditions
            order_book: Current order book

        Returns:
            Execution result with fill details
        """
        order.start_time = datetime.now()

        logger.info(
            f"Executing {order.side.value} {order.quantity} {order.symbol} "
            f"with strategy {order.strategy.value}"
        )

        if order.strategy == SlippageStrategy.IMMEDIATE:
            # Market order - immediate execution
            result = self._execute_market_order(order, order_book)

        elif order.strategy == SlippageStrategy.LIMIT_IOC:
            # Limit order with immediate-or-cancel
            result = self._execute_limit_ioc(order, order_book, conditions)

        elif order.strategy == SlippageStrategy.TWAP:
            # Time-weighted average price
            result = self._execute_twap(order, order_book, conditions)

        elif order.strategy == SlippageStrategy.VWAP:
            # Volume-weighted average price
            result = self._execute_vwap(order, order_book, conditions)

        elif order.strategy == SlippageStrategy.ICEBERG:
            # Hidden order with small visible portion
            result = self._execute_iceberg(order, order_book, conditions)

        else:  # ADAPTIVE
            # Dynamically choose best strategy
            result = self._execute_adaptive(order, order_book, conditions)

        order.end_time = datetime.now()

        # Calculate actual slippage
        if result.get("filled_quantity", 0) > 0:
            reference_price = order_book.mid_price
            fill_price = result.get("avg_fill_price", reference_price)

            if order.side == OrderSide.BUY:
                slippage = (fill_price - reference_price) / reference_price * 10000
            else:
                slippage = (reference_price - fill_price) / reference_price * 10000

            order.actual_slippage_bps = slippage
            result["slippage_bps"] = slippage

            logger.info(
                f"Order executed: filled={result['filled_quantity']}/{order.quantity}, "
                f"avg_price={fill_price:.2f}, slippage={slippage:.2f}bps"
            )

        return result

    def _execute_market_order(
        self,
        order: ExecutionOrder,
        order_book: OrderBookSnapshot
    ) -> Dict:
        """Execute as market order (highest slippage)."""
        # In real implementation, would call broker API
        # For now, simulate based on order book

        if order.side == OrderSide.BUY:
            fill_price = order_book.ask_price
        else:
            fill_price = order_book.bid_price

        return {
            "filled_quantity": order.quantity,
            "avg_fill_price": fill_price,
            "execution_time_ms": 50,  # Fast execution
            "strategy": "MARKET"
        }

    def _execute_limit_ioc(
        self,
        order: ExecutionOrder,
        order_book: OrderBookSnapshot,
        conditions: MarketConditions
    ) -> Dict:
        """Execute as limit order with immediate-or-cancel."""
        # Set aggressive limit price for high fill probability
        if order.side == OrderSide.BUY:
            # Willing to pay slightly above mid
            limit_price = order_book.mid_price * 1.0002
            fill_price = min(limit_price, order_book.ask_price)
        else:
            # Willing to sell slightly below mid
            limit_price = order_book.mid_price * 0.9998
            fill_price = max(limit_price, order_book.bid_price)

        # Assume 90% fill rate for IOC in normal markets
        fill_rate = 0.9 if not conditions.is_fast_market else 0.7
        filled_qty = int(order.quantity * fill_rate)

        return {
            "filled_quantity": filled_qty,
            "avg_fill_price": fill_price,
            "execution_time_ms": 100,
            "strategy": "LIMIT_IOC"
        }

    def _execute_twap(
        self,
        order: ExecutionOrder,
        order_book: OrderBookSnapshot,
        conditions: MarketConditions
    ) -> Dict:
        """Execute using time-weighted average price."""
        # Split order evenly over time horizon
        num_slices = max(5, order.time_horizon_seconds // 30)
        slice_size = order.quantity // num_slices

        # Simulate fills at various prices
        fills = []
        for i in range(num_slices):
            # Add some price variation
            price_variation = np.random.normal(0, conditions.volatility / 100) * order_book.mid_price
            slice_price = order_book.mid_price + price_variation
            fills.append((slice_size, slice_price))

        total_qty = sum(q for q, _ in fills)
        avg_price = sum(q * p for q, p in fills) / total_qty if total_qty > 0 else order_book.mid_price

        return {
            "filled_quantity": total_qty,
            "avg_fill_price": avg_price,
            "execution_time_ms": order.time_horizon_seconds * 1000,
            "strategy": "TWAP",
            "num_slices": num_slices
        }

    def _execute_vwap(
        self,
        order: ExecutionOrder,
        order_book: OrderBookSnapshot,
        conditions: MarketConditions
    ) -> Dict:
        """Execute using volume-weighted average price."""
        # Weight slices by typical volume patterns
        # Usually: higher volume at open/close, lower mid-day

        num_slices = max(5, order.time_horizon_seconds // 30)

        # Use VWAP curve from smart router
        prices = [order_book.mid_price] * 20
        volumes = [1000] * 20  # Simplified
        vwap_price = self.router.calculate_vwap(prices, volumes)

        # Better fill price than TWAP due to volume weighting
        fill_price = vwap_price * (1.0 if order.side == OrderSide.SELL else 1.0001)

        return {
            "filled_quantity": order.quantity,
            "avg_fill_price": fill_price,
            "execution_time_ms": order.time_horizon_seconds * 1000,
            "strategy": "VWAP",
            "num_slices": num_slices
        }

    def _execute_iceberg(
        self,
        order: ExecutionOrder,
        order_book: OrderBookSnapshot,
        conditions: MarketConditions
    ) -> Dict:
        """Execute using iceberg order (hidden quantity)."""
        # Show only 10% of order size
        visible_size = max(1, order.quantity // 10)

        # Better execution due to hidden size
        fill_price = order_book.mid_price * (1.00005 if order.side == OrderSide.BUY else 0.99995)

        return {
            "filled_quantity": order.quantity,
            "avg_fill_price": fill_price,
            "execution_time_ms": order.time_horizon_seconds * 1000,
            "strategy": "ICEBERG",
            "visible_size": visible_size
        }

    def _execute_adaptive(
        self,
        order: ExecutionOrder,
        order_book: OrderBookSnapshot,
        conditions: MarketConditions
    ) -> Dict:
        """Adaptively choose best execution strategy."""
        # Choose strategy based on current conditions
        if conditions.is_fast_market:
            # Fast market - use IOC to avoid adverse moves
            return self._execute_limit_ioc(order, order_book, conditions)
        elif conditions.liquidity_score < 40:
            # Low liquidity - use iceberg to hide size
            return self._execute_iceberg(order, order_book, conditions)
        else:
            # Normal conditions - use VWAP for best execution
            return self._execute_vwap(order, order_book, conditions)


# Singleton instances
slippage_analyzer = SlippageAnalyzer()
adaptive_executor = AdaptiveExecutor(
    slippage_analyzer,
    SmartOrderRouting()
)
