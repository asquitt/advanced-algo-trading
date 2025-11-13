"""
Unit tests for HFT techniques.

Tests:
- Order book analysis
- Market microstructure
- Statistical arbitrage
- Smart order routing
- Latency optimization
"""

import pytest
import numpy as np
from datetime import datetime
from src.trading_engine.hft_techniques import (
    OrderBookSnapshot,
    MarketMicrostructure,
    StatisticalArbitrage,
    SmartOrderRouting,
    LatencyOptimization,
)


class TestOrderBookSnapshot:
    """Test order book snapshot calculations."""

    def test_order_book_creation(self, mock_order_book):
        """Test creating order book snapshot."""
        assert mock_order_book.symbol == "AAPL"
        assert mock_order_book.bid_price == 150.45
        assert mock_order_book.ask_price == 150.55

    def test_spread_calculation(self, mock_order_book):
        """Test bid-ask spread calculation."""
        expected_spread = 150.55 - 150.45
        assert abs(mock_order_book.spread - expected_spread) < 0.001

    def test_spread_bps_calculation(self, mock_order_book):
        """Test spread in basis points."""
        mid_price = (150.45 + 150.55) / 2
        expected_bps = ((0.10 / mid_price) * 10000)
        assert abs(mock_order_book.spread_bps - expected_bps) < 0.1

    def test_order_imbalance_balanced(self):
        """Test order imbalance when balanced."""
        ob = OrderBookSnapshot(
            symbol="TEST",
            timestamp=datetime.utcnow(),
            bid_price=100.0,
            bid_size=100,  # Equal size
            ask_price=100.1,
            ask_size=100,  # Equal size
            bid_levels=[],
            ask_levels=[],
        )

        # Balanced order book should have imbalance of 0.5
        assert abs(ob.order_imbalance - 0.5) < 0.001

    def test_order_imbalance_buy_pressure(self):
        """Test order imbalance with buy pressure."""
        ob = OrderBookSnapshot(
            symbol="TEST",
            timestamp=datetime.utcnow(),
            bid_price=100.0,
            bid_size=200,  # More bids
            ask_price=100.1,
            ask_size=100,  # Fewer asks
            bid_levels=[],
            ask_levels=[],
        )

        # More buy pressure, imbalance > 0.5
        assert ob.order_imbalance > 0.5

    def test_order_imbalance_sell_pressure(self):
        """Test order imbalance with sell pressure."""
        ob = OrderBookSnapshot(
            symbol="TEST",
            timestamp=datetime.utcnow(),
            bid_price=100.0,
            bid_size=100,  # Fewer bids
            ask_price=100.1,
            ask_size=200,  # More asks
            bid_levels=[],
            ask_levels=[],
        )

        # More sell pressure, imbalance < 0.5
        assert ob.order_imbalance < 0.5

    def test_microprice_calculation(self):
        """Test volume-weighted microprice."""
        ob = OrderBookSnapshot(
            symbol="TEST",
            timestamp=datetime.utcnow(),
            bid_price=100.0,
            bid_size=100,
            ask_price=100.2,
            ask_size=100,
            bid_levels=[],
            ask_levels=[],
        )

        # Microprice should be between bid and ask
        assert ob.bid_price < ob.microprice < ob.ask_price

        # With equal volumes, should be close to mid-price
        mid_price = (ob.bid_price + ob.ask_price) / 2
        assert abs(ob.microprice - mid_price) < 0.01


class TestMarketMicrostructure:
    """Test market microstructure analysis."""

    def test_liquidity_score_calculation(self, mock_order_book):
        """Test liquidity score calculation."""
        mm = MarketMicrostructure()
        score = mm.calculate_liquidity_score("AAPL", mock_order_book)

        # Score should be between 0 and 1
        assert 0 <= score <= 1

    def test_liquidity_score_tight_spread(self):
        """Test that tight spread increases liquidity score."""
        mm = MarketMicrostructure()

        # Tight spread
        ob_tight = OrderBookSnapshot(
            symbol="TEST",
            timestamp=datetime.utcnow(),
            bid_price=100.0,
            bid_size=1000,
            ask_price=100.01,  # 1 cent spread
            ask_size=1000,
            bid_levels=[],
            ask_levels=[],
        )

        # Wide spread
        ob_wide = OrderBookSnapshot(
            symbol="TEST",
            timestamp=datetime.utcnow(),
            bid_price=100.0,
            bid_size=1000,
            ask_price=100.50,  # 50 cent spread
            ask_size=1000,
            bid_levels=[],
            ask_levels=[],
        )

        score_tight = mm.calculate_liquidity_score("TEST", ob_tight)
        score_wide = mm.calculate_liquidity_score("TEST", ob_wide)

        assert score_tight > score_wide

    def test_price_impact_estimation(self, mock_order_book):
        """Test price impact estimation."""
        mm = MarketMicrostructure()

        # Small order
        impact_small = mm.estimate_price_impact("AAPL", 100, mock_order_book)

        # Large order
        impact_large = mm.estimate_price_impact("AAPL", 10000, mock_order_book)

        # Large order should have higher impact
        assert impact_large > impact_small
        assert impact_large > 0

    def test_quote_stuffing_detection(self):
        """Test quote stuffing detection."""
        mm = MarketMicrostructure()

        # Normal rate
        assert mm.detect_quote_stuffing("TEST", 50) is False

        # Suspicious rate
        assert mm.detect_quote_stuffing("TEST", 150) is True

    def test_effective_spread_calculation(self):
        """Test effective spread calculation."""
        mm = MarketMicrostructure()

        mid_price = 100.0
        execution_price = 100.05

        spread = mm.calculate_effective_spread(execution_price, mid_price, "buy")

        # Effective spread = 2 * |100.05 - 100.0| = 0.10
        assert abs(spread - 0.10) < 0.001


class TestStatisticalArbitrage:
    """Test statistical arbitrage strategies."""

    def test_zscore_calculation(self):
        """Test z-score calculation for mean reversion."""
        stat_arb = StatisticalArbitrage(lookback_window=20)

        # Create price series with clear mean
        prices = [100.0] * 18 + [110.0, 115.0]  # Last two are outliers

        zscore = stat_arb.calculate_zscore(prices)

        # Z-score should be positive (price above mean)
        assert zscore > 0

    def test_zscore_mean_reversion(self):
        """Test z-score returns to 0 at mean."""
        stat_arb = StatisticalArbitrage(lookback_window=10)

        # Prices at mean
        prices = [100.0] * 10

        zscore = stat_arb.calculate_zscore(prices)

        # Z-score should be ~0 when at mean
        assert abs(zscore) < 0.1

    def test_mean_reversion_buy_signal(self):
        """Test mean reversion buy signal."""
        stat_arb = StatisticalArbitrage(lookback_window=20)

        # Prices significantly below mean
        prices = [100.0] * 19 + [85.0]

        signal = stat_arb.detect_mean_reversion_signal("TEST", prices, zscore_threshold=2.0)

        # Should generate BUY signal
        assert signal == "BUY"

    def test_mean_reversion_sell_signal(self):
        """Test mean reversion sell signal."""
        stat_arb = StatisticalArbitrage(lookback_window=20)

        # Prices significantly above mean
        prices = [100.0] * 19 + [115.0]

        signal = stat_arb.detect_mean_reversion_signal("TEST", prices, zscore_threshold=2.0)

        # Should generate SELL signal
        assert signal == "SELL"

    def test_mean_reversion_no_signal(self):
        """Test no signal when near mean."""
        stat_arb = StatisticalArbitrage(lookback_window=20)

        # Prices at mean
        prices = [100.0] * 20

        signal = stat_arb.detect_mean_reversion_signal("TEST", prices)

        # Should not generate signal
        assert signal is None

    def test_correlation_calculation(self):
        """Test correlation calculation for pairs trading."""
        stat_arb = StatisticalArbitrage()

        # Perfectly correlated
        prices_a = [1, 2, 3, 4, 5]
        prices_b = [2, 4, 6, 8, 10]  # 2x prices_a

        correlation = stat_arb.calculate_correlation(prices_a, prices_b)

        # Should be perfectly correlated
        assert abs(correlation - 1.0) < 0.001

    def test_half_life_calculation(self):
        """Test mean reversion half-life."""
        stat_arb = StatisticalArbitrage()

        # Mean-reverting series
        prices = [100, 105, 102, 104, 101, 103, 100, 102, 101, 100]

        half_life = stat_arb.calculate_half_life(prices)

        # Should have finite half-life
        assert half_life < float('inf')
        assert half_life > 0


class TestSmartOrderRouting:
    """Test smart order routing algorithms."""

    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        sor = SmartOrderRouting()

        prices = [100.0, 100.5, 101.0]
        volumes = [1000, 2000, 1000]

        vwap = sor.calculate_vwap(prices, volumes)

        # VWAP should be weighted by volume
        expected_vwap = (100.0*1000 + 100.5*2000 + 101.0*1000) / 4000
        assert abs(vwap - expected_vwap) < 0.001

    def test_vwap_empty_data(self):
        """Test VWAP with empty data."""
        sor = SmartOrderRouting()

        vwap = sor.calculate_vwap([], [])
        assert vwap == 0.0

    def test_twap_calculation(self):
        """Test TWAP calculation."""
        sor = SmartOrderRouting()

        prices = [100.0, 100.5, 101.0, 100.5]

        twap = sor.calculate_twap(prices)

        # TWAP is simple average
        expected_twap = sum(prices) / len(prices)
        assert abs(twap - expected_twap) < 0.001

    def test_order_splitting_twap(self):
        """Test order splitting with TWAP strategy."""
        sor = SmartOrderRouting()

        total_qty = 1000
        num_slices = 10

        slices = sor.split_order(total_qty, num_slices, strategy="TWAP")

        # Should have correct number of slices
        assert len(slices) == num_slices

        # Total should equal original quantity
        assert sum(slices) == total_qty

        # TWAP slices should be roughly equal
        avg_slice = total_qty / num_slices
        for slice_qty in slices:
            assert abs(slice_qty - avg_slice) <= 1  # Allow ±1 for rounding

    def test_implementation_shortfall_calculation(self):
        """Test implementation shortfall calculation."""
        sor = SmartOrderRouting()

        decision_price = 100.0
        execution_prices = [100.05, 100.10, 100.08]
        quantities = [100, 100, 100]

        shortfall = sor.calculate_implementation_shortfall(
            decision_price,
            execution_prices,
            quantities
        )

        # Shortfall = (100.05-100)*100 + (100.10-100)*100 + (100.08-100)*100
        expected = (0.05 + 0.10 + 0.08) * 100
        assert abs(shortfall - expected) < 0.01


class TestLatencyOptimization:
    """Test latency optimization utilities."""

    def test_latency_budget_estimation(self):
        """Test total latency budget calculation."""
        result = LatencyOptimization.estimate_latency_budget(
            signal_generation_ms=50,
            order_routing_ms=10,
            exchange_latency_ms=5
        )

        assert result["total_latency_ms"] == 65
        assert result["is_fast"] is True
        assert result["can_compete_hft"] is False  # 65ms is not HFT level

    def test_hft_level_latency(self):
        """Test HFT-level latency detection."""
        result = LatencyOptimization.estimate_latency_budget(
            signal_generation_ms=0.3,
            order_routing_ms=0.2,
            exchange_latency_ms=0.1
        )

        assert result["total_latency_ms"] < 1
        assert result["can_compete_hft"] is True

    def test_cache_hit_rate_optimization(self):
        """Test optimal cache hit rate calculation."""
        # Cache TTL longer than update frequency
        hit_rate = LatencyOptimization.optimize_cache_hit_rate(
            cache_ttl_seconds=3600,
            update_frequency_seconds=60
        )

        assert hit_rate == 1.0  # 100% hit rate

        # Cache TTL shorter than update frequency
        hit_rate = LatencyOptimization.optimize_cache_hit_rate(
            cache_ttl_seconds=30,
            update_frequency_seconds=60
        )

        assert hit_rate == 0.5  # 50% hit rate
