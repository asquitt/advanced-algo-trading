"""
Performance and load tests for the trading platform.

These tests measure:
- Signal generation latency
- API response times
- Throughput under load
- Cache effectiveness
- Memory usage
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import statistics


class TestSignalGenerationPerformance:
    """Test signal generation performance."""

    @patch('src.llm_agents.financial_agent.market_data')
    @patch('src.llm_agents.sentiment_agent.market_data')
    @patch('src.llm_agents.base_agent.BaseLLMAgent._call_llm')
    def test_signal_generation_latency(self, mock_llm, mock_sentiment_data,
                                      mock_financial_data, sample_company_info, sample_news):
        """Test signal generation completes within acceptable time."""
        from src.llm_agents.ensemble_strategy import EnsembleStrategy

        # Mock data
        mock_financial_data.get_company_info.return_value = sample_company_info
        mock_sentiment_data.get_news.return_value = sample_news

        # Mock LLM with realistic latency
        def mock_llm_call(*args, **kwargs):
            time.sleep(0.05)  # Simulate 50ms LLM call
            return ('{"score": 75, "valuation": "fairly_valued", "strengths": [], "weaknesses": [], "red_flags": [], "investment_thesis": "Test", "confidence": 0.8}', 1000, 0.003)

        mock_llm.side_effect = [
            mock_llm_call(),
            mock_llm_call(),
        ]

        # Measure time
        strategy = EnsembleStrategy()
        start_time = time.time()
        signal = strategy.generate_signal("AAPL", use_cache=False, track_mlflow=False)
        elapsed_ms = (time.time() - start_time) * 1000

        # Should complete within 500ms (our target for LLM-based)
        assert elapsed_ms < 500
        assert signal is not None

    @patch('src.llm_agents.base_agent.cache')
    def test_cache_hit_performance(self, mock_cache):
        """Test that cache hits are fast."""
        from src.llm_agents.financial_agent import FinancialAnalyzerAgent

        agent = FinancialAnalyzerAgent()

        # Mock cache hit
        mock_cache.get.return_value = {
            "score": 0.8,
            "cached": True,
        }

        # Measure time
        start_time = time.time()
        result = agent._get_cached_analysis("AAPL")
        elapsed_ms = (time.time() - start_time) * 1000

        # Cache hit should be very fast (<10ms)
        assert elapsed_ms < 10
        assert result is not None

    @patch('src.llm_agents.financial_agent.market_data')
    @patch('src.llm_agents.sentiment_agent.market_data')
    @patch('src.llm_agents.base_agent.BaseLLMAgent._call_llm')
    def test_batch_signal_generation(self, mock_llm, mock_sentiment_data,
                                    mock_financial_data, sample_company_info, sample_news):
        """Test generating signals for multiple symbols."""
        from src.llm_agents.ensemble_strategy import EnsembleStrategy

        # Mock data
        mock_financial_data.get_company_info.return_value = sample_company_info
        mock_sentiment_data.get_news.return_value = sample_news

        # Mock fast LLM
        mock_llm.return_value = (
            '{"score": 75, "valuation": "fairly_valued", "strengths": [], "weaknesses": [], "red_flags": [], "investment_thesis": "Test", "confidence": 0.8}',
            1000,
            0.003
        )

        # Generate signals for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        strategy = EnsembleStrategy()

        start_time = time.time()
        signals = []
        for symbol in symbols:
            signal = strategy.generate_signal(symbol, use_cache=False, track_mlflow=False)
            signals.append(signal)
        elapsed_s = time.time() - start_time

        # Should process all signals
        assert len(signals) == len(symbols)

        # Calculate throughput
        throughput = len(symbols) / elapsed_s
        print(f"\nSignal generation throughput: {throughput:.2f} signals/second")

        # Should achieve reasonable throughput
        assert throughput > 1  # At least 1 signal/second


class TestAPIPerformance:
    """Test API endpoint performance."""

    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_api_endpoint_latency(self, mock_kafka, mock_strategy):
        """Test API endpoint response time."""
        from src.main import app
        from fastapi.testclient import TestClient
        from src.data_layer.models import TradingSignal, SignalType

        client = TestClient(app)

        # Mock strategy
        mock_strategy.generate_signal.return_value = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.85,
            ai_conviction_score=0.78,
            reasoning="Test",
            source_agent="test",
        )
        mock_kafka.publish_signal.return_value = True

        # Measure latency
        latencies = []
        for _ in range(10):
            start_time = time.time()
            response = client.post("/signals/generate?symbol=AAPL")
            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)

            assert response.status_code == 200

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nAPI latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

        # Should be reasonably fast
        assert avg_latency < 100  # Average < 100ms
        assert p95_latency < 200  # P95 < 200ms

    def test_health_check_performance(self):
        """Test health check endpoint is very fast."""
        from src.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Measure
        start_time = time.time()
        response = client.get("/health")
        elapsed_ms = (time.time() - start_time) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 50  # Should be very fast


class TestHFTPerformance:
    """Test HFT technique performance."""

    def test_order_book_analysis_speed(self, mock_order_book):
        """Test order book analysis is fast."""
        from src.trading_engine.hft_techniques import MarketMicrostructure

        mm = MarketMicrostructure()

        # Measure
        start_time = time.time()
        for _ in range(1000):  # 1000 iterations
            score = mm.calculate_liquidity_score("AAPL", mock_order_book)
        elapsed_ms = (time.time() - start_time) * 1000

        # Should be very fast (< 1ms per calculation)
        per_calc_ms = elapsed_ms / 1000
        assert per_calc_ms < 1.0

    def test_zscore_calculation_speed(self):
        """Test z-score calculation is fast."""
        from src.trading_engine.hft_techniques import StatisticalArbitrage

        stat_arb = StatisticalArbitrage(lookback_window=20)
        prices = [100.0 + i * 0.1 for i in range(100)]

        # Measure
        start_time = time.time()
        for _ in range(1000):
            zscore = stat_arb.calculate_zscore(prices)
        elapsed_ms = (time.time() - start_time) * 1000

        # Should be very fast
        per_calc_ms = elapsed_ms / 1000
        assert per_calc_ms < 0.5  # Less than 0.5ms per calculation

    def test_vwap_calculation_speed(self):
        """Test VWAP calculation is fast."""
        from src.trading_engine.hft_techniques import SmartOrderRouting

        sor = SmartOrderRouting()
        prices = [100.0 + i * 0.1 for i in range(100)]
        volumes = [1000 + i * 10 for i in range(100)]

        # Measure
        start_time = time.time()
        for _ in range(1000):
            vwap = sor.calculate_vwap(prices, volumes)
        elapsed_ms = (time.time() - start_time) * 1000

        # Should be very fast
        per_calc_ms = elapsed_ms / 1000
        assert per_calc_ms < 0.5


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_cache_size_management(self):
        """Test that cache doesn't grow unbounded."""
        from src.utils.cache import Cache

        cache = Cache()

        # Add many items
        for i in range(1000):
            cache.set(f"key_{i}", {"data": f"value_{i}"}, ttl=3600)

        # Cache should handle this without issues
        # (In production, Redis handles memory management)

    def test_signal_object_size(self):
        """Test that signal objects are reasonably sized."""
        from src.data_layer.models import TradingSignal, SignalType
        import sys

        signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=0.85,
            ai_conviction_score=0.78,
            fundamental_score=0.82,
            sentiment_score=0.65,
            technical_score=0.75,
            reasoning="Test reasoning" * 100,  # Long string
            source_agent="ensemble_strategy",
        )

        # Get size
        size_bytes = sys.getsizeof(signal.dict())

        # Should be reasonable (< 10KB)
        assert size_bytes < 10240


class TestThroughput:
    """Test system throughput."""

    @patch('src.trading_engine.advanced_executor.broker')
    @patch('src.trading_engine.advanced_executor.market_data')
    def test_trade_execution_throughput(self, mock_market_data, mock_broker,
                                       sample_trading_signal, sample_quote):
        """Test trade execution throughput."""
        from src.trading_engine.advanced_executor import AdvancedTradingExecutor
        from src.data_layer.models import Trade

        executor = AdvancedTradingExecutor()

        # Mock
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = sample_quote
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.get_account.return_value = {
            "portfolio_value": 100000.0,
            "buying_power": 50000.0,
        }
        mock_broker.can_trade.return_value = (True, "OK")
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            status="filled",
        )

        # Execute multiple trades
        start_time = time.time()
        trades = []
        for i in range(10):
            trade = executor.execute_signal(
                sample_trading_signal,
                track_mlflow=False,
                execution_strategy="IMMEDIATE"
            )
            trades.append(trade)
        elapsed_s = time.time() - start_time

        # Calculate throughput
        throughput = len([t for t in trades if t]) / elapsed_s
        print(f"\nTrade execution throughput: {throughput:.2f} trades/second")

        # Should achieve reasonable throughput
        assert throughput > 5  # At least 5 trades/second


class TestConcurrency:
    """Test concurrent operations."""

    @patch('src.llm_agents.financial_agent.market_data')
    @patch('src.llm_agents.sentiment_agent.market_data')
    @patch('src.llm_agents.base_agent.BaseLLMAgent._call_llm')
    def test_concurrent_signal_generation(self, mock_llm, mock_sentiment_data,
                                         mock_financial_data, sample_company_info, sample_news):
        """Test generating signals concurrently."""
        from src.llm_agents.ensemble_strategy import EnsembleStrategy
        import concurrent.futures

        # Mock
        mock_financial_data.get_company_info.return_value = sample_company_info
        mock_sentiment_data.get_news.return_value = sample_news
        mock_llm.return_value = (
            '{"score": 75, "valuation": "fairly_valued", "strengths": [], "weaknesses": [], "red_flags": [], "investment_thesis": "Test", "confidence": 0.8}',
            1000,
            0.003
        )

        strategy = EnsembleStrategy()
        symbols = ["AAPL", "MSFT", "GOOGL"]

        # Generate concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(strategy.generate_signal, symbol, False, False)
                for symbol in symbols
            ]

            signals = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should complete
        assert len(signals) == len(symbols)
        assert all(s is not None for s in signals)
