"""
Pytest configuration and fixtures.

This file provides shared test fixtures and configuration for all tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
import os
import sys

# Set test environment variables BEFORE any imports
os.environ["GROQ_API_KEY"] = "test_groq_key"
os.environ["ANTHROPIC_API_KEY"] = "test_anthropic_key"
os.environ["ALPACA_API_KEY"] = "test_alpaca_key"
os.environ["ALPACA_SECRET_KEY"] = "test_alpaca_secret"

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from src.utils.config import Settings

    return Settings(
        groq_api_key="test_groq_key",
        anthropic_api_key="test_anthropic_key",
        alpaca_api_key="test_alpaca_key",
        alpaca_secret_key="test_alpaca_secret",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="test_db",
        postgres_user="test_user",
        postgres_password="test_pass",
        redis_host="localhost",
        redis_port=6379,
        kafka_bootstrap_servers="localhost:9092",
        paper_trading=True,
        max_position_size=10000.0,
        risk_per_trade=0.02,
        max_tokens_per_analysis=2000,
        cache_analysis_hours=24,
    )


@pytest.fixture
def mock_cache():
    """Mock Redis cache."""
    cache_data = {}

    class MockCache:
        def get(self, key):
            return cache_data.get(key)

        def set(self, key, value, ttl=None):
            cache_data[key] = value
            return True

        def delete(self, key):
            if key in cache_data:
                del cache_data[key]
            return True

        def clear_pattern(self, pattern):
            keys_to_delete = [k for k in cache_data.keys() if pattern.replace("*", "") in k]
            for key in keys_to_delete:
                del cache_data[key]
            return len(keys_to_delete)

    return MockCache()


@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for testing."""
    from src.data_layer.models import TradingSignal, SignalType

    return TradingSignal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence_score=0.85,
        ai_conviction_score=0.78,
        fundamental_score=0.82,
        sentiment_score=0.65,
        technical_score=0.75,
        reasoning="Strong fundamentals and positive sentiment",
        source_agent="ensemble_strategy",
    )


@pytest.fixture
def sample_trade():
    """Sample trade for testing."""
    from src.data_layer.models import Trade, OrderSide, OrderStatus

    return Trade(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        entry_price=150.25,
        status=OrderStatus.FILLED,
        order_id="test_order_123",
    )


@pytest.fixture
def sample_quote():
    """Sample market quote."""
    return {
        "symbol": "AAPL",
        "price": 150.50,
        "bid": 150.45,
        "ask": 150.55,
        "bid_size": 100,
        "ask_size": 100,
        "volume": 1000000,
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_news():
    """Sample news articles."""
    from src.data_layer.models import MarketNews

    return [
        MarketNews(
            symbol="AAPL",
            headline="Apple announces record earnings",
            summary="Strong iPhone sales drive revenue growth",
            source="Reuters",
            published_at=datetime.utcnow(),
        ),
        MarketNews(
            symbol="AAPL",
            headline="Apple stock upgraded by analysts",
            summary="Price target raised to $200",
            source="Bloomberg",
            published_at=datetime.utcnow(),
        ),
    ]


@pytest.fixture
def sample_company_info():
    """Sample company information."""
    return {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 2500000000000,
        "pe_ratio": 25.5,
        "forward_pe": 22.3,
        "peg_ratio": 1.8,
        "price_to_book": 35.2,
        "dividend_yield": 0.005,
        "52_week_high": 180.0,
        "52_week_low": 125.0,
        "avg_volume": 50000000,
    }


@pytest.fixture
def mock_broker():
    """Mock broker for testing."""
    broker_mock = Mock()

    # Mock account
    broker_mock.get_account.return_value = {
        "cash": 100000.0,
        "portfolio_value": 120000.0,
        "buying_power": 200000.0,
        "equity": 120000.0,
        "last_equity": 115000.0,
        "pattern_day_trader": False,
        "trading_blocked": False,
        "transfers_blocked": False,
    }

    # Mock positions
    broker_mock.get_positions.return_value = []
    broker_mock.get_position.return_value = None

    # Mock can_trade
    broker_mock.can_trade.return_value = (True, "OK")

    return broker_mock


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""
    return {
        "score": 75,
        "valuation": "fairly_valued",
        "strengths": ["Strong revenue growth", "High profit margins"],
        "weaknesses": ["High valuation", "Intense competition"],
        "red_flags": [],
        "investment_thesis": "Strong fundamentals support current valuation",
        "confidence": 0.8,
    }


@pytest.fixture
def mock_order_book():
    """Mock order book snapshot."""
    from src.trading_engine.hft_techniques import OrderBookSnapshot

    return OrderBookSnapshot(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        bid_price=150.45,
        bid_size=100,
        ask_price=150.55,
        ask_size=100,
        bid_levels=[(150.45, 100), (150.40, 200), (150.35, 300)],
        ask_levels=[(150.55, 100), (150.60, 200), (150.65, 300)],
    )


@pytest.fixture(autouse=True)
def reset_mlflow():
    """Reset MLflow between tests."""
    # This prevents MLflow from interfering with tests
    os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/test_mlflow"
    yield
    # Cleanup after test
    if "MLFLOW_TRACKING_URI" in os.environ:
        del os.environ["MLFLOW_TRACKING_URI"]


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer."""
    producer_mock = Mock()
    producer_mock.publish_news.return_value = True
    producer_mock.publish_filing.return_value = True
    producer_mock.publish_signal.return_value = True
    return producer_mock
