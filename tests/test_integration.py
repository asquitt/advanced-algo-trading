"""
Integration tests for the full trading platform.

These tests verify that components work together correctly:
- End-to-end signal generation
- Complete trade execution flow
- Data layer integration
- API endpoints
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


class TestEndToEndSignalGeneration:
    """Test complete signal generation pipeline."""

    @patch('src.llm_agents.financial_agent.market_data')
    @patch('src.llm_agents.sentiment_agent.market_data')
    @patch.object('src.llm_agents.base_agent.BaseLLMAgent', '_call_llm')
    def test_full_signal_pipeline(self, mock_llm, mock_sentiment_data, mock_financial_data,
                                 sample_company_info, sample_news):
        """Test generating a signal through the full pipeline."""
        from src.llm_agents.ensemble_strategy import EnsembleStrategy

        # Mock data sources
        mock_financial_data.get_company_info.return_value = sample_company_info
        mock_sentiment_data.get_news.return_value = sample_news

        # Mock LLM responses
        mock_llm.side_effect = [
            # Financial analysis
            ('{"score": 80, "valuation": "fairly_valued", "strengths": ["Growth"], "weaknesses": [], "red_flags": [], "investment_thesis": "Good", "confidence": 0.85}', 1500, 0.0045),
            # Sentiment analysis
            ('{"sentiment_score": 0.7, "sentiment_label": "positive", "positive_themes": ["earnings"], "negative_themes": [], "market_impact": "high", "key_catalysts": ["product"], "summary": "Positive", "confidence": 0.8}', 800, 0.0001),
        ]

        # Generate signal
        strategy = EnsembleStrategy()
        signal = strategy.generate_signal("AAPL", use_cache=False, track_mlflow=False)

        # Verify signal was generated
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.signal_type in ["BUY", "SELL", "HOLD"]
        assert 0 <= signal.ai_conviction_score <= 1

    @patch('src.llm_agents.financial_agent.market_data')
    @patch('src.llm_agents.sentiment_agent.market_data')
    @patch('src.llm_agents.base_agent.BaseLLMAgent._call_llm')
    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_signal_to_trade_execution(self, mock_exec_data, mock_broker,
                                       mock_llm, mock_sentiment_data, mock_financial_data,
                                       sample_company_info, sample_news, sample_quote):
        """Test complete flow from signal generation to trade execution."""
        from src.llm_agents.ensemble_strategy import EnsembleStrategy
        from src.trading_engine.executor import TradingExecutor

        # Mock data for signal generation
        mock_financial_data.get_company_info.return_value = sample_company_info
        mock_sentiment_data.get_news.return_value = sample_news

        # Mock strong LLM responses for BUY signal
        mock_llm.side_effect = [
            ('{"score": 90, "valuation": "undervalued", "strengths": ["Growth"], "weaknesses": [], "red_flags": [], "investment_thesis": "Strong buy", "confidence": 0.9}', 1500, 0.0045),
            ('{"sentiment_score": 0.8, "sentiment_label": "very_positive", "positive_themes": ["earnings"], "negative_themes": [], "market_impact": "high", "key_catalysts": ["product"], "summary": "Very positive", "confidence": 0.85}', 800, 0.0001),
        ]

        # Mock execution
        mock_exec_data.is_market_open.return_value = True
        mock_exec_data.get_quote.return_value = sample_quote

        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.get_account.return_value = {
            "portfolio_value": 100000.0,
            "buying_power": 50000.0,
        }
        mock_broker.can_trade.return_value = (True, "OK")

        from src.data_layer.models import Trade
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            entry_price=150.50,
            status="filled",
        )

        # Generate signal
        strategy = EnsembleStrategy()
        signal = strategy.generate_signal("AAPL", use_cache=False, track_mlflow=False)

        # Execute signal
        executor = TradingExecutor()
        trade = executor.execute_signal(signal, track_mlflow=False)

        # Verify execution
        assert trade is not None
        assert trade.symbol == "AAPL"


class TestAPIIntegration:
    """Test FastAPI endpoints integration."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        from src.main import app
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "paper_trading" in data

    def test_root_endpoint(self):
        """Test root endpoint."""
        from src.main import app
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "running"

    @patch('src.main.market_data')
    @patch('src.main.strategy')
    @patch('src.main.kafka_producer')
    def test_generate_signal_endpoint(self, mock_kafka, mock_strategy, mock_market_data):
        """Test signal generation endpoint."""
        from src.main import app
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

        # Mock Kafka
        mock_kafka.publish_signal.return_value = True

        # Call endpoint
        response = client.post("/signals/generate?symbol=AAPL")

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert "signal_type" in data

    @patch('src.main.broker')
    def test_portfolio_endpoint(self, mock_broker):
        """Test portfolio endpoint."""
        from src.main import app
        from src.data_layer.models import PortfolioState

        client = TestClient(app)

        # Mock broker
        mock_broker.get_portfolio_state.return_value = PortfolioState(
            cash_balance=50000.0,
            portfolio_value=100000.0,
            total_pnl=5000.0,
            total_pnl_percent=0.05,
            active_positions=3,
        )

        # Call endpoint
        response = client.get("/portfolio")

        assert response.status_code == 200
        data = response.json()
        assert data["cash_balance"] == 50000.0


class TestKafkaIntegration:
    """Test Kafka streaming integration."""

    def test_kafka_producer_creation(self):
        """Test creating Kafka producer."""
        from src.data_layer.kafka_stream import KafkaStreamProducer

        # Should create without error
        # (actual Kafka connection not tested in unit tests)
        producer = KafkaStreamProducer()
        assert producer is not None

    @patch('src.data_layer.kafka_stream.KafkaProducer')
    def test_publish_signal(self, mock_kafka_producer):
        """Test publishing signal to Kafka."""
        from src.data_layer.kafka_stream import KafkaStreamProducer

        # Mock Kafka
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        future = MagicMock()
        future.get.return_value = MagicMock(partition=0, offset=123)
        mock_producer_instance.send.return_value = future

        # Create producer and publish
        producer = KafkaStreamProducer()
        result = producer.publish_signal({
            "symbol": "AAPL",
            "signal_type": "BUY",
        })

        # Verify
        assert result is True


class TestCacheIntegration:
    """Test caching integration."""

    @patch('src.llm_agents.financial_agent.market_data')
    @patch('src.llm_agents.base_agent.cache')
    def test_cache_usage_in_agents(self, mock_cache, mock_market_data, sample_company_info):
        """Test that agents use cache correctly."""
        from src.llm_agents.financial_agent import FinancialAnalyzerAgent

        agent = FinancialAnalyzerAgent()

        # Mock cache hit
        cached_analysis = {
            "score": 0.8,
            "valuation": "fairly_valued",
            "strengths": [],
            "weaknesses": [],
            "red_flags": [],
            "investment_thesis": "Cached",
            "confidence": 0.8,
        }
        mock_cache.get.return_value = cached_analysis

        # Analyze
        result = agent.analyze("AAPL", use_cache=True)

        # Should use cache
        assert result.cache_hit is True
        assert result.tokens_used == 0
        assert result.api_cost == 0.0


class TestDatabaseIntegration:
    """Test database integration."""

    @patch('src.utils.database.engine')
    def test_database_connection(self, mock_engine):
        """Test database connection setup."""
        from src.utils.database import init_db

        # Should not raise exception
        init_db()

    @patch('src.utils.database.SessionLocal')
    def test_session_context_manager(self, mock_session_local):
        """Test database session context manager."""
        from src.utils.database import get_db

        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Use context manager
        with get_db() as db:
            pass

        # Verify commit and close
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
