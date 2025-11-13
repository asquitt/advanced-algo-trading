"""
High-Frequency Trading Techniques Implementation.

This module implements techniques inspired by HFT firms:
1. Order Book Analysis - Bid-ask spread, market depth
2. Market Microstructure - Price impact, liquidity analysis
3. Latency Optimization - Fast signal generation
4. Statistical Arbitrage - Mean reversion, pairs trading
5. Smart Order Routing - VWAP, TWAP execution
6. Tick Data Processing - High-resolution price data

References:
- Algorithmic Trading: Winning Strategies and Their Rationale (Chan, 2013)
- High-Frequency Trading: A Practical Guide (Kissell, 2013)
- Market Microstructure in Practice (Lehalle & Laruelle, 2018)
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from src.utils.logger import app_logger
from src.data_layer.market_data import market_data


@dataclass
class OrderBookSnapshot:
    """
    Order book snapshot with bid/ask levels.

    HFT firms use order book imbalance to predict short-term price movements.
    """
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    bid_levels: List[Tuple[float, int]]  # [(price, size), ...]
    ask_levels: List[Tuple[float, int]]

    @property
    def spread(self) -> float:
        """Bid-ask spread in absolute terms."""
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Spread in basis points (HFT profitability metric)."""
        mid_price = (self.bid_price + self.ask_price) / 2
        return (self.spread / mid_price) * 10000

    @property
    def order_imbalance(self) -> float:
        """
        Order book imbalance ratio.

        Imbalance > 0.5: Buy pressure (bullish)
        Imbalance < 0.5: Sell pressure (bearish)

        HFT firms use this for sub-second predictions.
        """
        total_bid_volume = self.bid_size
        total_ask_volume = self.ask_size
        total_volume = total_bid_volume + total_ask_volume

        if total_volume == 0:
            return 0.5

        return total_bid_volume / total_volume

    @property
    def microprice(self) -> float:
        """
        Volume-weighted microprice.

        Better estimate of "true" price than mid-price.
        Used by HFT for more accurate valuations.

        Formula: (bid_size * ask + ask_size * bid) / (bid_size + ask_size)
        """
        if self.bid_size + self.ask_size == 0:
            return (self.bid_price + self.ask_price) / 2

        return (
            self.bid_size * self.ask_price +
            self.ask_size * self.bid_price
        ) / (self.bid_size + self.ask_size)


class MarketMicrostructure:
    """
    Market microstructure analysis used by HFT firms.

    Analyzes:
    - Liquidity dynamics
    - Price impact
    - Execution costs
    - Market quality metrics
    """

    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[int]] = {}

    def calculate_liquidity_score(
        self,
        symbol: str,
        order_book: OrderBookSnapshot
    ) -> float:
        """
        Calculate liquidity score (0-1).

        Higher score = more liquid market = better for HFT.

        Considers:
        - Bid-ask spread (tighter is better)
        - Order book depth (more depth is better)
        - Volume (higher is better)
        """
        # Spread component (inverse - tighter is better)
        # Typical spread: 0.01% = 1bp (good), 0.1% = 10bp (poor)
        spread_score = max(0, 1 - (order_book.spread_bps / 20))

        # Depth component (more size is better)
        depth = order_book.bid_size + order_book.ask_size
        depth_score = min(1, depth / 10000)  # Normalize to 10k shares

        # Combine with weights
        liquidity_score = 0.6 * spread_score + 0.4 * depth_score

        return liquidity_score

    def estimate_price_impact(
        self,
        symbol: str,
        order_size: int,
        order_book: OrderBookSnapshot
    ) -> float:
        """
        Estimate price impact of an order (in basis points).

        HFT firms minimize price impact through:
        - Order splitting
        - Smart order routing
        - Timing optimization

        Uses Kyle's Lambda model (simplified):
        Impact ∝ sqrt(order_size / average_daily_volume)
        """
        # Get average daily volume (simplified)
        avg_volume = 1_000_000  # Would query from historical data

        # Kyle's lambda (price impact coefficient)
        # Typical values: 0.1-1.0 for liquid stocks
        kyle_lambda = 0.3

        # Price impact in basis points
        impact_bps = kyle_lambda * np.sqrt(order_size / avg_volume) * 10000

        return impact_bps

    def detect_quote_stuffing(
        self,
        symbol: str,
        quote_updates_per_second: float
    ) -> bool:
        """
        Detect quote stuffing (HFT manipulation tactic to avoid).

        Quote stuffing: Flooding order book with fake orders to:
        - Slow down competitors
        - Create false signals

        Threshold: >100 updates/second is suspicious.
        """
        return quote_updates_per_second > 100

    def calculate_effective_spread(
        self,
        execution_price: float,
        mid_price: float,
        side: str
    ) -> float:
        """
        Calculate effective spread (actual execution cost).

        Effective spread = 2 * |execution_price - mid_price|

        HFT firms track this to measure execution quality.
        """
        return 2 * abs(execution_price - mid_price)


class StatisticalArbitrage:
    """
    Statistical arbitrage strategies used by HFT firms.

    Techniques:
    1. Mean Reversion - Price tends to revert to mean
    2. Pairs Trading - Correlated assets diverge/converge
    3. Cointegration - Long-term equilibrium relationships
    """

    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window

    def calculate_zscore(
        self,
        price_history: List[float]
    ) -> float:
        """
        Calculate z-score for mean reversion.

        Z-score = (current_price - mean) / std_dev

        Trading rules:
        - Z > +2: Overbought, consider selling
        - Z < -2: Oversold, consider buying
        - |Z| < 0.5: Mean reversion complete

        HFT firms use much shorter windows (seconds to minutes).
        """
        if len(price_history) < 2:
            return 0.0

        recent_prices = price_history[-self.lookback_window:]
        mean = np.mean(recent_prices)
        std = np.std(recent_prices)

        if std == 0:
            return 0.0

        current_price = recent_prices[-1]
        zscore = (current_price - mean) / std

        return zscore

    def detect_mean_reversion_signal(
        self,
        symbol: str,
        price_history: List[float],
        zscore_threshold: float = 2.0
    ) -> Optional[str]:
        """
        Detect mean reversion trading signal.

        Returns:
        - "BUY": Price below mean, expect reversion up
        - "SELL": Price above mean, expect reversion down
        - None: No signal
        """
        zscore = self.calculate_zscore(price_history)

        if zscore < -zscore_threshold:
            return "BUY"
        elif zscore > zscore_threshold:
            return "SELL"
        else:
            return None

    def calculate_correlation(
        self,
        prices_a: List[float],
        prices_b: List[float]
    ) -> float:
        """
        Calculate correlation for pairs trading.

        HFT firms trade pairs of highly correlated stocks:
        - Buy underperformer, sell outperformer
        - Profit from convergence

        Typical pairs: SPY/IWM, XLE/OIH, etc.
        """
        if len(prices_a) != len(prices_b) or len(prices_a) < 2:
            return 0.0

        return np.corrcoef(prices_a, prices_b)[0, 1]

    def calculate_half_life(
        self,
        price_history: List[float]
    ) -> float:
        """
        Calculate mean reversion half-life.

        Half-life: Time for price to revert halfway to mean.

        Shorter half-life = faster mean reversion = better for HFT
        Typical HFT half-life: seconds to minutes
        """
        if len(price_history) < 2:
            return float('inf')

        # Calculate log returns
        returns = np.diff(np.log(price_history))

        # AR(1) regression: returns[t] = rho * returns[t-1] + epsilon
        if len(returns) < 2:
            return float('inf')

        y = returns[1:]
        x = returns[:-1]

        # Simple linear regression
        rho = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0

        # Half-life calculation
        if rho >= 1 or rho <= -1:
            return float('inf')

        half_life = -np.log(2) / np.log(abs(rho))

        return half_life


class SmartOrderRouting:
    """
    Smart order routing (SOR) used by HFT firms.

    Techniques:
    1. VWAP - Volume-Weighted Average Price
    2. TWAP - Time-Weighted Average Price
    3. Implementation Shortfall
    4. Adaptive execution
    """

    def calculate_vwap(
        self,
        prices: List[float],
        volumes: List[int]
    ) -> float:
        """
        Calculate Volume-Weighted Average Price.

        VWAP = Σ(price * volume) / Σ(volume)

        HFT firms use VWAP to:
        - Benchmark execution quality
        - Execute large orders without moving market

        Goal: Execute close to VWAP to minimize market impact.
        """
        if len(prices) != len(volumes) or not prices:
            return 0.0

        total_value = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)

        if total_volume == 0:
            return 0.0

        return total_value / total_volume

    def calculate_twap(
        self,
        prices: List[float]
    ) -> float:
        """
        Calculate Time-Weighted Average Price.

        TWAP = Σ(price) / n

        Simpler than VWAP, used for:
        - Illiquid stocks
        - Consistent execution pace
        """
        if not prices:
            return 0.0

        return sum(prices) / len(prices)

    def split_order(
        self,
        total_quantity: int,
        num_slices: int,
        strategy: str = "TWAP"
    ) -> List[int]:
        """
        Split large order into smaller slices.

        Prevents:
        - Market impact
        - Information leakage
        - Adverse selection

        Strategies:
        - TWAP: Equal slices
        - VWAP: Volume-weighted slices
        - Adaptive: Respond to market conditions
        """
        if strategy == "TWAP":
            # Equal slices
            base_size = total_quantity // num_slices
            remainder = total_quantity % num_slices

            slices = [base_size] * num_slices
            for i in range(remainder):
                slices[i] += 1

            return slices

        # Add more strategies (VWAP, Adaptive) as needed
        return [total_quantity // num_slices] * num_slices

    def calculate_implementation_shortfall(
        self,
        decision_price: float,
        execution_prices: List[float],
        quantities: List[int]
    ) -> float:
        """
        Calculate implementation shortfall (cost of execution).

        IS = Σ((execution_price - decision_price) * quantity)

        HFT firms minimize this through:
        - Fast execution
        - Smart routing
        - Low latency infrastructure

        Returns cost in dollars.
        """
        if len(execution_prices) != len(quantities):
            return 0.0

        shortfall = sum(
            (exec_price - decision_price) * qty
            for exec_price, qty in zip(execution_prices, quantities)
        )

        return shortfall


class LatencyOptimization:
    """
    Latency optimization techniques from HFT.

    For HFT, microseconds matter:
    - Co-location: Servers near exchange
    - Fast networks: Fiber optic, microwave
    - Optimized code: C++, FPGA
    - Algorithm efficiency: O(1) operations

    Our platform focuses on:
    - Efficient data structures
    - Caching
    - Async operations
    - Batch processing
    """

    @staticmethod
    def estimate_latency_budget(
        signal_generation_ms: float,
        order_routing_ms: float,
        exchange_latency_ms: float
    ) -> Dict[str, float]:
        """
        Estimate total latency budget.

        HFT target: <1ms total
        Our platform target: <100ms (still very fast for LLM-based)

        Breakdown:
        - Signal generation: 50-100ms (optimized LLM)
        - Order routing: 10-20ms
        - Exchange: 5-10ms
        """
        total_latency = (
            signal_generation_ms +
            order_routing_ms +
            exchange_latency_ms
        )

        return {
            "signal_generation_ms": signal_generation_ms,
            "order_routing_ms": order_routing_ms,
            "exchange_latency_ms": exchange_latency_ms,
            "total_latency_ms": total_latency,
            "can_compete_hft": total_latency < 1,  # Realistic HFT
            "is_fast": total_latency < 100,  # Fast for LLM-based
        }

    @staticmethod
    def optimize_cache_hit_rate(
        cache_ttl_seconds: int,
        update_frequency_seconds: int
    ) -> float:
        """
        Calculate optimal cache TTL for hit rate.

        HFT firms balance:
        - Freshness: Shorter TTL
        - Performance: Longer TTL

        Our recommendation:
        - Market data: 1-15 seconds
        - LLM analysis: 3600 seconds (1 hour)
        """
        if update_frequency_seconds == 0:
            return 0.0

        hit_rate = min(1.0, cache_ttl_seconds / update_frequency_seconds)
        return hit_rate


# Export classes
__all__ = [
    "OrderBookSnapshot",
    "MarketMicrostructure",
    "StatisticalArbitrage",
    "SmartOrderRouting",
    "LatencyOptimization",
]
