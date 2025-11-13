"""
Unit tests for data models.

Tests Pydantic models for:
- Validation
- Serialization
- Business logic
"""

import pytest
from datetime import datetime
from src.data_layer.models import (
    TradingSignal,
    Trade,
    Position,
    PortfolioState,
    SignalType,
    OrderSide,
    OrderStatus,
    MarketNews,
)


class TestTradingSignal:
    """Test TradingSignal model."""

    def test_trading_signal_creation(self, sample_trading_signal):
        """Test creating a valid trading signal."""
        assert sample_trading_signal.symbol == "AAPL"
        assert sample_trading_signal.signal_type == SignalType.BUY
        assert 0 <= sample_trading_signal.confidence_score <= 1
        assert 0 <= sample_trading_signal.ai_conviction_score <= 1

    def test_signal_type_validation(self):
        """Test signal type must be valid enum."""
        with pytest.raises(ValueError):
            TradingSignal(
                symbol="AAPL",
                signal_type="INVALID",  # Invalid signal type
                confidence_score=0.8,
                ai_conviction_score=0.75,
                reasoning="Test",
                source_agent="test"
            )

    def test_confidence_score_validation(self):
        """Test confidence score must be between 0 and 1."""
        with pytest.raises(ValueError):
            TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence_score=1.5,  # Invalid - too high
                ai_conviction_score=0.75,
                reasoning="Test",
                source_agent="test"
            )

    def test_sentiment_score_range(self):
        """Test sentiment score can be negative."""
        signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.SELL,
            confidence_score=0.8,
            ai_conviction_score=0.6,
            sentiment_score=-0.5,  # Negative sentiment is valid
            reasoning="Negative news",
            source_agent="test"
        )
        assert signal.sentiment_score == -0.5

    def test_signal_serialization(self, sample_trading_signal):
        """Test signal can be serialized to dict."""
        signal_dict = sample_trading_signal.dict()

        assert signal_dict["symbol"] == "AAPL"
        assert signal_dict["signal_type"] == "BUY"
        assert "created_at" in signal_dict


class TestTrade:
    """Test Trade model."""

    def test_trade_creation(self, sample_trade):
        """Test creating a valid trade."""
        assert sample_trade.symbol == "AAPL"
        assert sample_trade.side == OrderSide.BUY
        assert sample_trade.quantity == 10
        assert sample_trade.entry_price == 150.25

    def test_pnl_calculation_buy(self):
        """Test P&L calculation for buy trade."""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            entry_price=100.0,
            exit_price=110.0,
        )

        # For buy: (exit - entry) * quantity
        expected_pnl = (110.0 - 100.0) * 10
        assert trade.pnl == expected_pnl

    def test_pnl_calculation_sell(self):
        """Test P&L calculation for sell trade."""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            entry_price=110.0,
            exit_price=100.0,
        )

        # For sell: (entry - exit) * quantity
        expected_pnl = (110.0 - 100.0) * 10
        assert trade.pnl == expected_pnl

    def test_quantity_must_be_positive(self):
        """Test quantity must be greater than 0."""
        with pytest.raises(ValueError):
            Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=0,  # Invalid
            )

    def test_price_must_be_positive(self):
        """Test price must be greater than 0."""
        with pytest.raises(ValueError):
            Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=-100.0,  # Invalid
            )


class TestPosition:
    """Test Position model."""

    def test_position_creation(self):
        """Test creating a valid position."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.0,
            current_price=155.0,
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_entry_price == 150.0

    def test_position_serialization(self):
        """Test position can be serialized."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            unrealized_pnl_percent=0.0333,
        )

        pos_dict = position.dict()
        assert pos_dict["symbol"] == "AAPL"
        assert pos_dict["unrealized_pnl"] == 500.0


class TestPortfolioState:
    """Test PortfolioState model."""

    def test_portfolio_state_creation(self):
        """Test creating portfolio state."""
        state = PortfolioState(
            cash_balance=50000.0,
            portfolio_value=100000.0,
            total_pnl=5000.0,
            total_pnl_percent=0.05,
            active_positions=5,
        )

        assert state.cash_balance == 50000.0
        assert state.portfolio_value == 100000.0
        assert state.active_positions == 5

    def test_win_rate_validation(self):
        """Test win rate must be between 0 and 1."""
        with pytest.raises(ValueError):
            PortfolioState(
                cash_balance=50000.0,
                portfolio_value=100000.0,
                win_rate=1.5,  # Invalid - too high
            )


class TestMarketNews:
    """Test MarketNews model."""

    def test_market_news_creation(self):
        """Test creating market news."""
        news = MarketNews(
            symbol="AAPL",
            headline="Apple announces new product",
            source="Reuters",
            published_at=datetime.utcnow(),
        )

        assert news.symbol == "AAPL"
        assert news.headline == "Apple announces new product"
        assert news.source == "Reuters"

    def test_sentiment_score_range(self):
        """Test sentiment score validation."""
        # Valid sentiment scores
        news1 = MarketNews(
            symbol="AAPL",
            headline="Test",
            source="Test",
            published_at=datetime.utcnow(),
            sentiment_score=0.5,
        )
        assert news1.sentiment_score == 0.5

        # Invalid sentiment score
        with pytest.raises(ValueError):
            MarketNews(
                symbol="AAPL",
                headline="Test",
                source="Test",
                published_at=datetime.utcnow(),
                sentiment_score=2.0,  # Too high
            )
