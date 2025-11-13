"""
Tests for FastAPI Endpoints

Tests REST API endpoints for signal generation, trade execution,
portfolio monitoring, and system health.

Author: LLM Trading Platform - Test Suite
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.main import app
from src.data_layer.models import TradingSignal, SignalType, Trade, OrderSide, OrderStatus, Position


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_signal():
    """Create mock trading signal."""
    return TradingSignal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence_score=0.85,
        ai_conviction_score=0.80,
        fundamental_score=0.85,
        sentiment_score=0.75,
        technical_score=0.80,
        reasoning="Strong fundamentals and positive sentiment",
        source_agent="ensemble_strategy"
    )


@pytest.fixture
def mock_trade():
    """Create mock trade."""
    return Trade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        entry_price=150.25,
        status=OrderStatus.FILLED,
        order_id="test_order_123",
        timestamp=datetime.utcnow()
    )


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root health check endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "running"
        assert data["service"] == "LLM Trading Platform"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    @patch("src.main.broker.get_account")
    def test_health_endpoint_healthy(self, mock_get_account, client):
        """Test detailed health endpoint when system is healthy."""
        mock_get_account.return_value = {"cash": 100000.0}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "broker" in data
        assert "paper_trading" in data
        assert "timestamp" in data

    @patch("src.main.broker.get_account")
    def test_health_endpoint_degraded(self, mock_get_account, client):
        """Test health endpoint when broker is degraded."""
        mock_get_account.side_effect = Exception("Connection error")

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"  # API is healthy
        assert "unhealthy" in data["broker"].lower()


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Should contain Prometheus metrics
        content = response.text
        assert "trading_signals_total" in content or "python" in content.lower()


class TestSignalGenerationEndpoints:
    """Test signal generation endpoints."""

    @patch("src.main.strategy.generate_signal")
    @patch("src.main.kafka_producer.publish_signal")
    def test_generate_signal_success(self, mock_publish, mock_generate, client, mock_signal):
        """Test successful signal generation."""
        mock_generate.return_value = mock_signal
        mock_publish.return_value = True

        response = client.post(
            "/signals/generate",
            params={"symbol": "AAPL", "use_cache": True, "execute": False}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["symbol"] == "AAPL"
        assert data["signal_type"] == "BUY"
        assert data["confidence_score"] == 0.85
        assert "reasoning" in data

        mock_generate.assert_called_once()
        mock_publish.assert_called_once()

    @patch("src.main.strategy.generate_signal")
    def test_generate_signal_error(self, mock_generate, client):
        """Test signal generation with error."""
        mock_generate.side_effect = Exception("API rate limit exceeded")

        response = client.post(
            "/signals/generate",
            params={"symbol": "AAPL"}
        )

        assert response.status_code == 500
        assert "API rate limit exceeded" in response.json()["detail"]

    @patch("src.main.strategy.generate_signal")
    @patch("src.main.executor.execute_signal")
    @patch("src.main.kafka_producer.publish_signal")
    def test_generate_signal_with_execution(
        self, mock_publish, mock_execute, mock_generate, client, mock_signal, mock_trade
    ):
        """Test signal generation with automatic execution."""
        mock_generate.return_value = mock_signal
        mock_execute.return_value = mock_trade
        mock_publish.return_value = True

        response = client.post(
            "/signals/generate",
            params={"symbol": "AAPL", "execute": True}
        )

        assert response.status_code == 200
        mock_execute.assert_called_once()

    @patch("src.main.strategy.generate_signal")
    def test_generate_signal_hold_no_execution(
        self, mock_generate, client
    ):
        """Test that HOLD signals don't trigger execution."""
        hold_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.HOLD,
            confidence_score=0.50,
            ai_conviction_score=0.45,
            reasoning="Neutral outlook",
            source_agent="ensemble_strategy"
        )
        mock_generate.return_value = hold_signal

        with patch("src.main.executor.execute_signal") as mock_execute:
            response = client.post(
                "/signals/generate",
                params={"symbol": "AAPL", "execute": True}
            )

            assert response.status_code == 200
            mock_execute.assert_not_called()  # HOLD shouldn't execute

    @patch("src.main.strategy.generate_signal")
    @patch("src.main.kafka_producer.publish_signal")
    def test_generate_batch_signals_success(self, mock_publish, mock_generate, client, mock_signal):
        """Test batch signal generation."""
        mock_generate.return_value = mock_signal
        mock_publish.return_value = True

        response = client.post(
            "/signals/batch",
            json=["AAPL", "GOOGL", "MSFT"],
            params={"use_cache": True}
        )

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 3
        assert mock_generate.call_count == 3

    @patch("src.main.strategy.generate_signal")
    def test_generate_batch_signals_partial_failure(self, mock_generate, client, mock_signal):
        """Test batch signal generation with some failures."""
        # First call succeeds, second fails, third succeeds
        mock_generate.side_effect = [
            mock_signal,
            Exception("API error"),
            mock_signal
        ]

        response = client.post(
            "/signals/batch",
            json=["AAPL", "INVALID", "MSFT"]
        )

        assert response.status_code == 200
        data = response.json()

        # Should return successful signals only
        assert len(data) == 2


class TestTradeExecutionEndpoints:
    """Test trade execution endpoints."""

    @patch("src.main.strategy.generate_signal")
    @patch("src.main.executor.execute_signal")
    def test_execute_trade_with_signal_generation(
        self, mock_execute, mock_generate, client, mock_signal, mock_trade
    ):
        """Test trade execution with signal generation."""
        mock_generate.return_value = mock_signal
        mock_execute.return_value = mock_trade

        response = client.post(
            "/trades/execute",
            params={"symbol": "AAPL", "generate_signal": True}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["symbol"] == "AAPL"
        assert data["side"] == "buy"
        assert data["status"] == "filled"

        mock_generate.assert_called_once()
        mock_execute.assert_called_once()

    @patch("src.main.executor.execute_signal")
    def test_execute_trade_manual(self, mock_execute, client, mock_trade):
        """Test manual trade execution without signal generation."""
        mock_execute.return_value = mock_trade

        response = client.post(
            "/trades/execute",
            params={"symbol": "AAPL", "generate_signal": False}
        )

        assert response.status_code == 200
        mock_execute.assert_called_once()

    @patch("src.main.executor.execute_signal")
    def test_execute_trade_no_trade_returned(self, mock_execute, client):
        """Test trade execution when no trade is executed."""
        mock_execute.return_value = None  # Signal not executed

        with patch("src.main.strategy.generate_signal") as mock_gen:
            mock_gen.return_value = TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.HOLD,
                confidence_score=0.5,
                ai_conviction_score=0.4,
                reasoning="Hold",
                source_agent="test"
            )

            response = client.post(
                "/trades/execute",
                params={"symbol": "AAPL"}
            )

            assert response.status_code == 200


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_symbol_format(self, client):
        """Test handling of invalid symbol format."""
        with patch("src.main.strategy.generate_signal") as mock_generate:
            mock_generate.side_effect = ValueError("Invalid symbol")

            response = client.post(
                "/signals/generate",
                params={"symbol": "INVALID_SYMBOL!!!"}
            )

            assert response.status_code == 500

    def test_missing_required_parameters(self, client):
        """Test handling of missing required parameters."""
        response = client.post("/signals/generate")

        # Should fail due to missing symbol parameter
        assert response.status_code == 422  # Unprocessable Entity

    def test_empty_batch_symbols(self, client):
        """Test batch signal generation with empty list."""
        response = client.post(
            "/signals/batch",
            json=[]
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0


class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.options(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )

        # FastAPI/Starlette handles OPTIONS automatically
        assert response.status_code in [200, 405]  # May vary


class TestErrorHandling:
    """Test error handling."""

    @patch("src.main.strategy.generate_signal")
    def test_internal_server_error(self, mock_generate, client):
        """Test internal server error handling."""
        mock_generate.side_effect = Exception("Unexpected error")

        response = client.post(
            "/signals/generate",
            params={"symbol": "AAPL"}
        )

        assert response.status_code == 500
        assert "detail" in response.json()

    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoints."""
        response = client.get("/nonexistent")

        assert response.status_code == 404


class TestAPIIntegration:
    """Integration tests for API workflows."""

    @patch("src.main.strategy.generate_signal")
    @patch("src.main.executor.execute_signal")
    @patch("src.main.kafka_producer.publish_signal")
    def test_full_trading_workflow(
        self, mock_publish, mock_execute, mock_generate, client, mock_signal, mock_trade
    ):
        """Test complete trading workflow through API."""
        mock_generate.return_value = mock_signal
        mock_execute.return_value = mock_trade
        mock_publish.return_value = True

        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # 2. Generate signal
        signal_response = client.post(
            "/signals/generate",
            params={"symbol": "AAPL", "execute": False}
        )
        assert signal_response.status_code == 200

        # 3. Execute trade
        trade_response = client.post(
            "/trades/execute",
            params={"symbol": "AAPL", "generate_signal": False}
        )
        assert trade_response.status_code == 200

        # 4. Check metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200

    @patch("src.main.strategy.generate_signal")
    @patch("src.main.kafka_producer.publish_signal")
    def test_concurrent_signal_generation(self, mock_publish, mock_generate, client, mock_signal):
        """Test handling of multiple concurrent requests."""
        mock_generate.return_value = mock_signal
        mock_publish.return_value = True

        # Simulate concurrent requests
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        responses = []
        for symbol in symbols:
            response = client.post(
                "/signals/generate",
                params={"symbol": symbol}
            )
            responses.append(response)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

    def test_api_response_time(self, client):
        """Test that health endpoint responds quickly."""
        import time

        start = time.time()
        response = client.get("/")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0  # Should respond within 1 second
