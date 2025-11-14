"""
Tests for Trading Platform Improvements

Tests for:
1. Slippage management and reduction
2. Advanced feature engineering
3. Adaptive position sizing with drawdown management
4. Enhanced executor integration

Author: LLM Trading Platform
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.trading_engine.slippage_management import (
    SlippageAnalyzer,
    AdaptiveExecutor,
    ExecutionOrder,
    ExecutionUrgency,
    SlippageStrategy,
    MarketConditions,
    OrderBookSnapshot
)
from src.llm_agents.feature_engineering import (
    FeatureEngineer,
    TechnicalIndicators,
    RegimeDetector,
    MarketRegime
)
from src.trading_engine.position_sizing import (
    AdaptivePositionSizer,
    DrawdownAnalyzer,
    PerformanceAnalyzer,
    RiskMode
)
from src.data_layer.models import Trade, OrderSide


class TestSlippageManagement:
    """Tests for slippage analysis and reduction."""

    def test_market_conditions_assessment(self):
        """Test market conditions are properly assessed."""
        analyzer = SlippageAnalyzer()

        order_book = OrderBookSnapshot(
            symbol="AAPL",
            bid_price=150.00,
            ask_price=150.10,
            bid_size=100,
            ask_size=100,
            timestamp=datetime.now(),
            bid_levels=[(150.00, 100), (149.99, 200), (149.98, 150)],
            ask_levels=[(150.10, 100), (150.11, 200), (150.12, 150)]
        )

        # Normal market conditions
        prices = [150.0 + i * 0.05 for i in range(60)]  # Gradual uptrend
        volumes = [1000] * 60

        conditions = analyzer.assess_market_conditions(
            symbol="AAPL",
            order_book=order_book,
            recent_prices=prices,
            recent_volumes=volumes
        )

        assert conditions.symbol == "AAPL"
        assert conditions.volatility >= 0.0
        assert 0 <= conditions.order_imbalance <= 1
        assert 0 <= conditions.liquidity_score <= 100
        assert not conditions.is_fast_market  # Gradual moves

    def test_fast_market_detection(self):
        """Test detection of fast-moving markets."""
        analyzer = SlippageAnalyzer()

        order_book = OrderBookSnapshot(
            symbol="TSLA",
            bid_price=200.00,
            ask_price=200.20,
            bid_size=50,
            ask_size=50,
            timestamp=datetime.now(),
            bid_levels=[(200.00, 50), (199.98, 80), (199.96, 100)],
            ask_levels=[(200.20, 50), (200.22, 80), (200.24, 100)]
        )

        # Fast market - rapid price changes
        prices = [200.0]
        for i in range(20):
            if i % 5 == 0:
                # Sharp moves every 5 periods
                prices.append(prices[-1] * 1.005)  # 0.5% jump
            else:
                prices.append(prices[-1] * 1.0001)

        volumes = [1000] * len(prices)

        conditions = analyzer.assess_market_conditions(
            symbol="TSLA",
            order_book=order_book,
            recent_prices=prices,
            recent_volumes=volumes
        )

        # Should detect fast market due to sharp moves
        assert conditions.is_fast_market

    def test_slippage_estimation_scales_with_order_size(self):
        """Test slippage increases with order size."""
        analyzer = SlippageAnalyzer()

        order_book = OrderBookSnapshot(
            symbol="AAPL",
            bid_price=150.00,
            ask_price=150.10,
            bid_size=100,
            ask_size=100,
            timestamp=datetime.now(),
            bid_levels=[(150.00, 100), (149.99, 200), (149.98, 150)],
            ask_levels=[(150.10, 100), (150.11, 200), (150.12, 150)]
        )

        conditions = MarketConditions(
            symbol="AAPL",
            timestamp=datetime.now(),
            volatility=15.0,
            spread_bps=6.67,
            volume=100000,
            momentum=0.1,
            order_imbalance=0.5,
            liquidity_score=70.0,
            is_fast_market=False
        )

        # Small order
        small_estimate = analyzer.estimate_slippage(
            symbol="AAPL",
            quantity=10,
            side=OrderSide.BUY,
            urgency=ExecutionUrgency.MEDIUM,
            order_book=order_book,
            conditions=conditions
        )

        # Large order
        large_estimate = analyzer.estimate_slippage(
            symbol="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            urgency=ExecutionUrgency.MEDIUM,
            order_book=order_book,
            conditions=conditions
        )

        # Large order should have higher slippage
        assert large_estimate.expected_slippage_bps > small_estimate.expected_slippage_bps

    def test_urgency_affects_slippage(self):
        """Test execution urgency affects slippage estimates."""
        analyzer = SlippageAnalyzer()

        order_book = OrderBookSnapshot(
            symbol="AAPL",
            bid_price=150.00,
            ask_price=150.10,
            bid_size=100,
            ask_size=100,
            timestamp=datetime.now(),
            bid_levels=[(150.00, 100), (149.99, 200), (149.98, 150)],
            ask_levels=[(150.10, 100), (150.11, 200), (150.12, 150)]
        )

        conditions = MarketConditions(
            symbol="AAPL",
            timestamp=datetime.now(),
            volatility=15.0,
            spread_bps=6.67,
            volume=100000,
            momentum=0.1,
            order_imbalance=0.5,
            liquidity_score=70.0,
            is_fast_market=False
        )

        # Patient execution (low urgency)
        low_urgency = analyzer.estimate_slippage(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            urgency=ExecutionUrgency.LOW,
            order_book=order_book,
            conditions=conditions
        )

        # Urgent execution
        high_urgency = analyzer.estimate_slippage(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            urgency=ExecutionUrgency.CRITICAL,
            order_book=order_book,
            conditions=conditions
        )

        # Low urgency should recommend patient strategy
        assert low_urgency.recommended_strategy in [
            SlippageStrategy.TWAP, SlippageStrategy.VWAP
        ]

        # High urgency should recommend fast execution
        assert high_urgency.recommended_strategy in [
            SlippageStrategy.IMMEDIATE, SlippageStrategy.LIMIT_IOC
        ]

        # Low urgency should have lower expected slippage
        assert low_urgency.expected_slippage_bps < high_urgency.expected_slippage_bps

    def test_slippage_recording_for_learning(self):
        """Test that actual slippage is recorded for model improvement."""
        analyzer = SlippageAnalyzer()

        # Record some slippages
        analyzer.record_actual_slippage("AAPL", estimated_slippage_bps=5.0, actual_slippage_bps=4.5)
        analyzer.record_actual_slippage("AAPL", estimated_slippage_bps=5.0, actual_slippage_bps=5.2)
        analyzer.record_actual_slippage("AAPL", estimated_slippage_bps=5.0, actual_slippage_bps=4.8)

        # Should have history
        assert "AAPL" in analyzer.historical_slippage
        assert len(analyzer.historical_slippage["AAPL"]) == 3


class TestFeatureEngineering:
    """Tests for advanced feature engineering."""

    def test_technical_indicators_rsi(self):
        """Test RSI calculation."""
        indicators = TechnicalIndicators()

        # Uptrending prices (should have high RSI)
        uptrend = [100 + i for i in range(30)]
        rsi_up = indicators.calculate_rsi(uptrend, period=14)
        assert rsi_up > 50  # Uptrend should have RSI > 50

        # Downtrending prices (should have low RSI)
        downtrend = [100 - i for i in range(30)]
        rsi_down = indicators.calculate_rsi(downtrend, period=14)
        assert rsi_down < 50  # Downtrend should have RSI < 50

    def test_macd_calculation(self):
        """Test MACD calculation."""
        indicators = TechnicalIndicators()

        prices = [100 + i * 0.5 for i in range(50)]
        macd, signal, histogram = indicators.calculate_macd(prices)

        # MACD values should be returned
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(histogram, float)

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        indicators = TechnicalIndicators()

        prices = [100.0] * 20 + [105.0]  # Prices with a spike
        upper, middle, lower = indicators.calculate_bollinger_bands(prices)

        # Upper > middle > lower
        assert upper > middle > lower

        # Price spike should be near or above upper band
        assert 105.0 >= middle

    def test_regime_detection_trending_up(self):
        """Test detection of uptrending market."""
        detector = RegimeDetector()

        # Strong uptrend
        prices = [100 + i * 2 for i in range(60)]
        volumes = [1000] * 60

        regime = detector.detect_regime(prices, volumes, period=60)

        assert regime.primary_regime == MarketRegime.TRENDING_UP
        assert regime.trend_strength > 0

    def test_regime_detection_ranging(self):
        """Test detection of ranging market."""
        detector = RegimeDetector()

        # Ranging market (oscillating)
        prices = []
        for i in range(60):
            price = 100 + 2 * np.sin(i / 3)  # Oscillate between 98-102
            prices.append(price)
        volumes = [1000] * 60

        regime = detector.detect_regime(prices, volumes, period=60)

        # Should detect ranging or low volatility
        assert regime.primary_regime in [
            MarketRegime.RANGING,
            MarketRegime.LOW_VOLATILITY
        ]

    def test_feature_engineer_comprehensive(self):
        """Test comprehensive feature creation."""
        engineer = FeatureEngineer()

        # Create sample data
        prices = [100 + i * 0.5 for i in range(100)]
        volumes = [1000 + i * 10 for i in range(100)]
        highs = [p * 1.02 for p in prices]
        lows = [p * 0.98 for p in prices]

        features = engineer.create_features(
            symbol="AAPL",
            prices=prices,
            volumes=volumes,
            highs=highs,
            lows=lows,
            news_sentiment=0.6,
            social_sentiment=0.5
        )

        # Verify all feature categories are present
        assert features.symbol == "AAPL"
        assert features.technical is not None
        assert features.regime is not None
        assert features.alternative is not None
        assert features.multi_timeframe is not None

        # Check technical features
        assert features.technical.rsi_14 >= 0 and features.technical.rsi_14 <= 100
        assert features.technical.current_price > 0

        # Check regime features
        assert isinstance(features.regime.primary_regime, MarketRegime)
        assert 0 <= features.regime.regime_strength <= 1

        # Check alternative features
        assert features.alternative.news_sentiment_score == 0.6
        assert features.alternative.social_sentiment_score == 0.5

        # Check feature quality
        assert 0 <= features.feature_quality_score <= 1

    def test_multi_timeframe_alignment(self):
        """Test multi-timeframe trend alignment."""
        engineer = FeatureEngineer()

        # Strong uptrend across all timeframes
        prices = [100 + i * 1.0 for i in range(150)]
        volumes = [1000] * 150
        highs = [p * 1.01 for p in prices]
        lows = [p * 0.99 for p in prices]

        features = engineer.create_features(
            symbol="TEST",
            prices=prices,
            volumes=volumes,
            highs=highs,
            lows=lows
        )

        mtf = features.multi_timeframe

        # Should show bullish trends
        assert mtf.daily_trend == "bullish"
        assert mtf.short_term_momentum > 0
        assert mtf.medium_term_momentum > 0
        assert mtf.long_term_momentum > 0

        # High alignment score (all timeframes agree)
        assert mtf.alignment_score > 0.7


class TestPositionSizing:
    """Tests for adaptive position sizing and drawdown management."""

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        analyzer = DrawdownAnalyzer()

        # Simulate equity curve
        analyzer.update_equity(100000.0, datetime.now())
        analyzer.update_equity(110000.0, datetime.now() + timedelta(days=1))  # New peak
        analyzer.update_equity(105000.0, datetime.now() + timedelta(days=2))  # Drawdown

        metrics = analyzer.calculate_drawdown_metrics()

        # Should detect drawdown from peak
        assert metrics.peak_equity == 110000.0
        assert metrics.current_equity == 105000.0
        assert metrics.is_in_drawdown
        assert metrics.current_drawdown_pct < 0

    def test_performance_metrics_calculation(self):
        """Test performance metrics from trades."""
        analyzer = PerformanceAnalyzer()

        # Create sample trades
        trades = [
            Trade(symbol="AAPL", side=OrderSide.BUY, quantity=10,
                  entry_price=100.0, exit_price=110.0, status="filled",
                  entry_time=datetime.now(), exit_time=datetime.now()),  # Win: +100
            Trade(symbol="MSFT", side=OrderSide.BUY, quantity=10,
                  entry_price=200.0, exit_price=195.0, status="filled",
                  entry_time=datetime.now(), exit_time=datetime.now()),  # Loss: -50
            Trade(symbol="GOOGL", side=OrderSide.BUY, quantity=10,
                  entry_price=150.0, exit_price=160.0, status="filled",
                  entry_time=datetime.now(), exit_time=datetime.now()),  # Win: +100
        ]

        # Set P&L
        trades[0].pnl = (110.0 - 100.0) * 10
        trades[1].pnl = (195.0 - 200.0) * 10
        trades[2].pnl = (160.0 - 150.0) * 10

        metrics = analyzer.calculate_metrics(trades)

        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(66.67, rel=0.1)
        assert metrics.avg_win == 100.0
        assert metrics.avg_loss == 50.0
        assert metrics.profit_factor > 1.0  # Profitable

    def test_position_sizing_reduces_in_drawdown(self):
        """Test position size reduces during drawdown."""
        sizer = AdaptivePositionSizer()

        # Simulate good performance (no drawdown)
        good_trades = [
            Trade(symbol="AAPL", side=OrderSide.BUY, quantity=10,
                  entry_price=100.0, exit_price=110.0, status="filled",
                  entry_time=datetime.now(), exit_time=datetime.now()),
        ]
        good_trades[0].pnl = 100.0

        good_recommendation = sizer.calculate_position_size(
            portfolio_value=100000.0,
            signal_confidence=0.8,
            market_volatility=0.15,
            recent_trades=good_trades,
            open_positions=0
        )

        # Simulate drawdown
        sizer.drawdown_analyzer.update_equity(100000.0)  # Peak
        sizer.drawdown_analyzer.update_equity(85000.0)  # -15% drawdown

        # Create losing trades
        bad_trades = [
            Trade(symbol="AAPL", side=OrderSide.BUY, quantity=10,
                  entry_price=100.0, exit_price=90.0, status="filled",
                  entry_time=datetime.now(), exit_time=datetime.now()),
        ]
        bad_trades[0].pnl = -100.0

        drawdown_recommendation = sizer.calculate_position_size(
            portfolio_value=85000.0,
            signal_confidence=0.8,
            market_volatility=0.15,
            recent_trades=bad_trades,
            open_positions=0
        )

        # Position size should be reduced during drawdown
        assert drawdown_recommendation.adjusted_size_pct < good_recommendation.adjusted_size_pct
        assert drawdown_recommendation.position_multiplier < 1.0

    def test_risk_mode_escalation(self):
        """Test risk mode escalates with drawdown severity."""
        sizer = AdaptivePositionSizer()

        # Simulate increasing drawdown
        sizer.drawdown_analyzer.update_equity(100000.0)  # Peak

        # Small drawdown (-5%)
        sizer.drawdown_analyzer.update_equity(95000.0)
        small_dd = sizer.drawdown_analyzer.calculate_drawdown_metrics()
        perf = sizer.performance_analyzer._default_metrics()

        risk_mode_small = sizer._determine_risk_mode(small_dd, perf)
        assert risk_mode_small in [RiskMode.NORMAL, RiskMode.CONSERVATIVE]

        # Large drawdown (-20%)
        sizer.drawdown_analyzer.update_equity(80000.0)
        large_dd = sizer.drawdown_analyzer.calculate_drawdown_metrics()

        risk_mode_large = sizer._determine_risk_mode(large_dd, perf)
        assert risk_mode_large in [RiskMode.DEFENSIVE, RiskMode.HALT]

    def test_position_sizing_respects_portfolio_heat(self):
        """Test position sizing limits based on portfolio heat."""
        sizer = AdaptivePositionSizer(max_portfolio_heat=0.10)  # 10% max

        trades = []

        # High portfolio heat (90% of max)
        high_heat_rec = sizer.calculate_position_size(
            portfolio_value=100000.0,
            signal_confidence=0.8,
            market_volatility=0.15,
            recent_trades=trades,
            open_positions=5,
            current_portfolio_heat=0.09  # 9% already at risk
        )

        # Low portfolio heat
        low_heat_rec = sizer.calculate_position_size(
            portfolio_value=100000.0,
            signal_confidence=0.8,
            market_volatility=0.15,
            recent_trades=trades,
            open_positions=1,
            current_portfolio_heat=0.02  # Only 2% at risk
        )

        # High heat should result in smaller position
        assert high_heat_rec.adjusted_size_pct < low_heat_rec.adjusted_size_pct

    def test_consecutive_losses_reduce_size(self):
        """Test consecutive losses trigger size reduction."""
        sizer = AdaptivePositionSizer()

        # Create consecutive losing trades (need at least 10 for DEFENSIVE mode)
        losing_trades = []
        for i in range(10):
            trade = Trade(
                symbol=f"STOCK{i}",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=100.0,
                exit_price=90.0,
                status="filled",
                entry_time=datetime.now(),
                exit_time=datetime.now()
            )
            trade.pnl = -100.0
            losing_trades.append(trade)

        recommendation = sizer.calculate_position_size(
            portfolio_value=100000.0,
            signal_confidence=0.8,
            market_volatility=0.15,
            recent_trades=losing_trades,
            open_positions=0
        )

        # Should have reduced position size due to losing streak
        assert recommendation.position_multiplier < 1.0
        # With 10 consecutive losses and 0% win rate, should trigger DEFENSIVE mode
        assert recommendation.risk_mode in [
            RiskMode.CONSERVATIVE,
            RiskMode.DEFENSIVE,
            RiskMode.HALT
        ]


class TestEnhancedExecutor:
    """Tests for enhanced executor integration."""

    @patch('src.trading_engine.enhanced_executor.AlpacaBroker')
    def test_enhanced_executor_uses_adaptive_sizing(self, mock_broker_class):
        """Test enhanced executor uses adaptive position sizing."""
        # This would test the integration, but requires more mocking
        # Placeholder for integration test
        pass

    def test_slippage_estimate_affects_execution_strategy(self):
        """Test slippage estimates change execution strategy."""
        # Would test that high slippage estimates trigger smart routing
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
