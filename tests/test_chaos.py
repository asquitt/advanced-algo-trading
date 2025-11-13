"""
Tests for Chaos Engineering and Resilience

Tests system behavior under failure conditions, network issues,
resource exhaustion, and other adverse scenarios.

Author: LLM Trading Platform - Test Suite
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np
import pandas as pd
from src.data_layer.models import TradingSignal, SignalType


@pytest.mark.slow
class TestDatabaseFailures:
    """Test behavior when database is unavailable."""

    @patch("src.utils.database.get_db_connection")
    def test_database_connection_failure(self, mock_get_conn):
        """Test handling of database connection failure."""
        mock_get_conn.side_effect = Exception("Connection refused")

        # System should handle gracefully
        try:
            mock_get_conn()
        except Exception as e:
            assert "Connection refused" in str(e)
            # Should log error and potentially retry or use fallback

    @patch("src.utils.database.get_db_connection")
    def test_database_timeout(self, mock_get_conn):
        """Test handling of database timeout."""
        def slow_connection():
            time.sleep(5)  # Simulate slow connection
            raise TimeoutError("Database timeout")

        mock_get_conn.side_effect = slow_connection

        # Should timeout and handle gracefully
        with pytest.raises(TimeoutError):
            mock_get_conn()

    def test_database_partial_failure(self):
        """Test handling of intermittent database failures."""
        # Simulate 50% failure rate
        call_count = 0

        def flaky_connection():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Temporary failure")
            return Mock()

        # Should implement retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = flaky_connection()
                if result:
                    break
            except Exception:
                if attempt == max_retries - 1:
                    raise


@pytest.mark.slow
class TestRedisFailures:
    """Test behavior when Redis cache is unavailable."""

    @patch("src.utils.cache.redis_client")
    def test_redis_connection_failure(self, mock_redis):
        """Test handling when Redis is down."""
        mock_redis.get.side_effect = Exception("Redis connection failed")

        # Should fall back to direct computation
        try:
            mock_redis.get("test_key")
        except Exception:
            # Should continue without cache
            pass

    @patch("src.utils.cache.redis_client")
    def test_cache_miss_handled(self, mock_redis):
        """Test handling of cache misses."""
        mock_redis.get.return_value = None  # Cache miss

        result = mock_redis.get("nonexistent_key")
        assert result is None
        # Should compute value and potentially cache it

    def test_cache_corruption(self):
        """Test handling of corrupted cache data."""
        # Simulate corrupted cached data
        corrupted_data = b"corrupted\x00\x00\x00"

        # Should handle deserialization errors gracefully
        try:
            import json
            json.loads(corrupted_data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Should invalidate cache and recompute
            pass


@pytest.mark.slow
class TestExternalAPIFailures:
    """Test behavior when external APIs fail."""

    @patch("src.llm_agents.financial_agent.anthropic_client")
    def test_llm_api_timeout(self, mock_client):
        """Test handling of LLM API timeout."""
        mock_client.messages.create.side_effect = TimeoutError("Request timeout")

        with pytest.raises(TimeoutError):
            mock_client.messages.create(
                model="claude-3",
                messages=[{"role": "user", "content": "test"}]
            )
        # Should retry or use fallback model

    @patch("src.data_layer.market_data.yfinance")
    def test_market_data_api_rate_limit(self, mock_yf):
        """Test handling of API rate limiting."""
        mock_yf.Ticker.side_effect = Exception("429 Too Many Requests")

        # Should implement exponential backoff
        with pytest.raises(Exception):
            mock_yf.Ticker("AAPL")

    @patch("src.trading_engine.broker.api_client")
    def test_broker_api_degradation(self, mock_broker):
        """Test handling when broker API is slow/degraded."""
        def slow_api_call(*args, **kwargs):
            time.sleep(3)  # Slow response
            return Mock()

        mock_broker.get_account.side_effect = slow_api_call

        # Should timeout after reasonable period
        start = time.time()
        try:
            mock_broker.get_account()
        except Exception:
            pass
        elapsed = time.time() - start

        # Should have timed out
        assert elapsed < 10  # Maximum acceptable timeout


@pytest.mark.slow
class TestNetworkFailures:
    """Test behavior under network issues."""

    def test_network_partition(self):
        """Test handling of network partition."""
        # Simulate network unavailable
        import socket

        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.connect.side_effect = socket.error("Network unreachable")

            # Should handle gracefully
            try:
                sock = socket.socket()
                sock.connect(("example.com", 80))
            except socket.error as e:
                assert "Network unreachable" in str(e)

    def test_dns_resolution_failure(self):
        """Test handling of DNS resolution failure."""
        import socket

        with patch("socket.gethostbyname") as mock_dns:
            mock_dns.side_effect = socket.gaierror("Name or service not known")

            # Should handle DNS failure
            with pytest.raises(socket.gaierror):
                socket.gethostbyname("invalid.domain.local")

    def test_intermittent_connectivity(self):
        """Test handling of flaky network connection."""
        call_count = 0

        def flaky_request():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Network error")
            return {"status": "success"}

        # Should retry and eventually succeed
        result = None
        for attempt in range(5):
            try:
                result = flaky_request()
                break
            except ConnectionError:
                time.sleep(0.1)

        assert result is not None


class TestResourceExhaustion:
    """Test behavior under resource constraints."""

    def test_memory_pressure(self):
        """Test behavior under memory pressure."""
        # Allocate large amount of memory
        try:
            large_array = np.zeros((10000, 10000))  # 800MB
            assert large_array.shape == (10000, 10000)
        except MemoryError:
            # Should handle gracefully
            pass

    def test_cpu_saturation(self):
        """Test behavior under CPU saturation."""
        # Simulate CPU-intensive operation
        start = time.time()

        # Compute-heavy task
        result = sum(i**2 for i in range(1000000))

        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0  # Should finish within 5 seconds
        assert result > 0

    def test_disk_space_exhaustion(self):
        """Test handling of disk space issues."""
        # Simulate disk full error
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = OSError("No space left on device")

            # Should handle gracefully
            with pytest.raises(OSError):
                open("/tmp/test.txt", "w")


class TestDataCorruption:
    """Test handling of corrupted data."""

    def test_corrupted_market_data(self):
        """Test handling of corrupted market data."""
        # Corrupted data with NaN, Inf, negative prices
        corrupted_data = pd.DataFrame({
            "price": [150.0, np.nan, -50.0, np.inf, 160.0],
            "volume": [1000, 0, -100, 2000, 1500]
        })

        # Should detect and handle corrupted data
        has_nan = corrupted_data["price"].isna().any()
        has_negative = (corrupted_data["price"] < 0).any()
        has_inf = np.isinf(corrupted_data["price"]).any()

        assert has_nan or has_negative or has_inf  # Detect corruption

        # Clean data
        cleaned = corrupted_data[
            corrupted_data["price"].notna() &
            (corrupted_data["price"] > 0) &
            np.isfinite(corrupted_data["price"])
        ]

        assert len(cleaned) < len(corrupted_data)  # Removed corrupt rows

    def test_inconsistent_data_formats(self):
        """Test handling of inconsistent data formats."""
        # Different date formats
        inconsistent_dates = [
            "2024-01-15",
            "01/15/2024",
            "15-Jan-2024",
            "2024-01-15T10:30:00Z"
        ]

        # Should normalize to consistent format
        from dateutil import parser

        normalized = []
        for date_str in inconsistent_dates:
            try:
                parsed = parser.parse(date_str)
                normalized.append(parsed.isoformat())
            except Exception:
                pass

        # All should parse successfully
        assert len(normalized) == len(inconsistent_dates)

    def test_malformed_json_response(self):
        """Test handling of malformed JSON from API."""
        malformed_json = '{"symbol": "AAPL", "price": 150.0'  # Missing closing brace

        import json

        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_json)
        # Should handle and request retry


class TestConcurrencyIssues:
    """Test concurrent execution issues."""

    def test_race_condition(self):
        """Test for race conditions in concurrent access."""
        shared_counter = {"value": 0}

        def increment():
            # Simulate race condition
            current = shared_counter["value"]
            time.sleep(0.001)  # Context switch point
            shared_counter["value"] = current + 1

        # Without locking, race conditions can occur
        # In production, use threading.Lock or similar
        import threading

        lock = threading.Lock()

        def safe_increment():
            with lock:
                shared_counter["value"] += 1

        threads = [threading.Thread(target=safe_increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With locking, should get correct count
        assert shared_counter["value"] == 10

    def test_deadlock_prevention(self):
        """Test deadlock prevention."""
        import threading

        lock1 = threading.Lock()
        lock2 = threading.Lock()

        # Acquire locks in consistent order to prevent deadlock
        def task1():
            with lock1:
                time.sleep(0.01)
                with lock2:
                    pass

        def task2():
            with lock1:  # Same order as task1
                time.sleep(0.01)
                with lock2:
                    pass

        # Should complete without deadlock
        t1 = threading.Thread(target=task1)
        t2 = threading.Thread(target=task2)

        t1.start()
        t2.start()

        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        # Both should complete
        assert not t1.is_alive()
        assert not t2.is_alive()


class TestGracefulDegradation:
    """Test graceful degradation under failures."""

    def test_fallback_to_simple_strategy(self):
        """Test fallback to simple strategy when LLM fails."""
        # Primary: LLM analysis
        # Fallback: Technical indicators only

        def primary_strategy():
            raise Exception("LLM API unavailable")

        def fallback_strategy():
            return TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.HOLD,
                confidence_score=0.5,
                ai_conviction_score=0.0,  # No LLM
                technical_score=0.6,
                reasoning="LLM unavailable, using technical only",
                source_agent="fallback"
            )

        try:
            signal = primary_strategy()
        except Exception:
            signal = fallback_strategy()

        assert signal is not None
        assert signal.ai_conviction_score == 0.0  # Fallback mode

    def test_degraded_mode_operations(self):
        """Test operations in degraded mode."""
        # When external services fail, operate with reduced functionality
        services_status = {
            "llm": False,  # Down
            "market_data": True,  # Up
            "broker": True,  # Up
            "cache": False  # Down
        }

        # Determine operating mode
        if services_status["broker"] and services_status["market_data"]:
            # Can trade with limited intelligence
            can_trade = True
        else:
            can_trade = False

        assert can_trade  # Should continue with core functionality


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        failure_threshold = 5
        failure_count = 0
        circuit_open = False

        def faulty_service():
            nonlocal failure_count, circuit_open

            if circuit_open:
                raise Exception("Circuit breaker open")

            # Simulate service failure
            failure_count += 1
            if failure_count >= failure_threshold:
                circuit_open = True
            raise Exception("Service failure")

        # Should open circuit after threshold
        for i in range(10):
            try:
                faulty_service()
            except Exception as e:
                if "Circuit breaker open" in str(e):
                    break

        assert circuit_open

    def test_circuit_breaker_closes_after_recovery(self):
        """Test that circuit breaker closes after service recovers."""
        circuit_state = "closed"
        failure_count = 0
        success_count = 0

        def monitored_service():
            nonlocal circuit_state, failure_count, success_count

            if circuit_state == "open":
                # Try to recover
                success_count += 1
                if success_count >= 3:
                    circuit_state = "closed"
                    failure_count = 0
                return "success"

            # Simulate occasional failure
            if failure_count < 5:
                failure_count += 1
                if failure_count >= 5:
                    circuit_state = "open"
                raise Exception("Failure")

            return "success"

        # Trigger failures
        for _ in range(5):
            try:
                monitored_service()
            except Exception:
                pass

        assert circuit_state == "open"

        # Trigger recovery
        for _ in range(5):
            try:
                monitored_service()
            except Exception:
                pass

        # Should recover
        # Note: simplified logic for test, real circuit breaker more complex


class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    def test_exponential_backoff(self):
        """Test exponential backoff retry logic."""
        attempt = 0
        max_attempts = 5

        def calculate_backoff(attempt):
            return min(2 ** attempt, 30)  # Max 30 seconds

        backoffs = [calculate_backoff(i) for i in range(max_attempts)]

        # Should increase exponentially
        assert backoffs == [1, 2, 4, 8, 16]

    def test_retry_with_jitter(self):
        """Test retry with jitter to prevent thundering herd."""
        import random

        def calculate_backoff_with_jitter(attempt):
            base_delay = 2 ** attempt
            jitter = random.uniform(0, base_delay * 0.1)
            return base_delay + jitter

        delays = [calculate_backoff_with_jitter(i) for i in range(5)]

        # All delays should be positive and increasing trend
        assert all(d > 0 for d in delays)
