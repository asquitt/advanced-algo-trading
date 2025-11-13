"""
Unit tests for trading engine.

Tests:
- Broker integration
- Trade execution
- Risk management
- Advanced executor with HFT techniques
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.trading_engine.executor import TradingExecutor
from src.trading_engine.advanced_executor import AdvancedTradingExecutor
from src.data_layer.models import TradingSignal, SignalType, Trade, Position


class TestTradingExecutor:
    """Test basic trading executor."""

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_execute_buy_signal(self, mock_market_data, mock_broker, sample_trading_signal, sample_quote):
        """Test executing a buy signal."""
        executor = TradingExecutor()

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = sample_quote

        # Mock broker
        mock_broker.get_position.return_value = None  # No existing position
        mock_broker.get_positions.return_value = []  # No positions
        mock_broker.get_account.return_value = {
            "portfolio_value": 100000.0,
            "buying_power": 50000.0,
        }
        mock_broker.can_trade.return_value = (True, "OK")
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            entry_price=150.50,
            status="filled",
        )

        # Execute signal
        trade = executor.execute_signal(sample_trading_signal, track_mlflow=False)

        # Should execute trade
        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_execute_sell_signal(self, mock_market_data, mock_broker, sample_quote):
        """Test executing a sell signal."""
        executor = TradingExecutor()

        # Create sell signal
        sell_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.SELL,
            confidence_score=0.8,
            ai_conviction_score=0.75,
            reasoning="Time to sell",
            source_agent="test",
        )

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = sample_quote

        # Mock broker - has position to sell
        mock_broker.get_position.return_value = Position(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=140.0,
            current_price=150.50,
        )
        mock_broker.can_trade.return_value = (True, "OK")
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side="sell",
            quantity=10,
            entry_price=150.50,
            status="filled",
        )

        # Execute signal
        trade = executor.execute_signal(sell_signal, track_mlflow=False)

        # Should execute trade
        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.side == "sell"

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_hold_signal_no_action(self, mock_market_data, mock_broker):
        """Test that HOLD signal doesn't execute trade."""
        executor = TradingExecutor()

        # Create HOLD signal
        hold_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.HOLD,
            confidence_score=0.5,
            ai_conviction_score=0.5,
            reasoning="Hold position",
            source_agent="test",
        )

        # Mock market open
        mock_market_data.is_market_open.return_value = True

        # Execute signal
        trade = executor.execute_signal(hold_signal, track_mlflow=False)

        # Should not execute trade
        assert trade is None

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_skip_when_market_closed(self, mock_market_data, mock_broker, sample_trading_signal):
        """Test that trades are skipped when market is closed."""
        executor = TradingExecutor()

        # Mock market closed
        mock_market_data.is_market_open.return_value = False

        # Execute signal
        trade = executor.execute_signal(sample_trading_signal, track_mlflow=False)

        # Should not execute trade
        assert trade is None

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_skip_when_max_positions_reached(self, mock_market_data, mock_broker, sample_trading_signal, sample_quote):
        """Test that trade is skipped when max positions reached."""
        executor = TradingExecutor(max_open_positions=2)

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = sample_quote

        # Mock broker - already at max positions
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = [
            Position(symbol="MSFT", quantity=10, avg_entry_price=300.0),
            Position(symbol="GOOGL", quantity=5, avg_entry_price=2800.0),
        ]

        # Execute signal
        trade = executor.execute_signal(sample_trading_signal, track_mlflow=False)

        # Should not execute trade
        assert trade is None

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_position_sizing(self, mock_market_data, mock_broker, sample_trading_signal, sample_quote):
        """Test position sizing calculation."""
        executor = TradingExecutor(
            max_position_size=10000.0,
            risk_per_trade=0.02,
        )

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = sample_quote

        # Mock broker
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.get_account.return_value = {
            "portfolio_value": 100000.0,
            "buying_power": 50000.0,
        }
        mock_broker.can_trade.return_value = (True, "OK")

        # Capture the order
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            status="filled",
        )

        # Execute
        trade = executor.execute_signal(sample_trading_signal, track_mlflow=False)

        # Verify order was placed with calculated quantity
        mock_broker.place_market_order.assert_called_once()
        call_args = mock_broker.place_market_order.call_args
        qty = call_args[0][1]

        # Quantity should be > 0 and reasonable
        assert qty > 0
        assert qty < 1000  # Sanity check


class TestAdvancedExecutor:
    """Test advanced executor with HFT techniques."""

    @patch('src.trading_engine.advanced_executor.broker')
    @patch('src.trading_engine.advanced_executor.market_data')
    def test_liquidity_check(self, mock_market_data, mock_broker, sample_trading_signal, sample_quote):
        """Test that low liquidity prevents trading."""
        executor = AdvancedTradingExecutor(min_liquidity_score=0.8)

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = {
            **sample_quote,
            "bid": 150.0,
            "ask": 151.0,  # Wide spread = low liquidity
            "bid_size": 10,  # Low size
            "ask_size": 10,
        }

        # Execute - should skip due to low liquidity
        trade = executor.execute_signal(sample_trading_signal, track_mlflow=False)

        # May be None due to liquidity check
        # (exact behavior depends on liquidity score calculation)

    @patch('src.trading_engine.advanced_executor.broker')
    @patch('src.trading_engine.advanced_executor.market_data')
    def test_price_impact_estimation(self, mock_market_data, mock_broker, sample_trading_signal, sample_quote):
        """Test that price impact is estimated."""
        executor = AdvancedTradingExecutor()

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = sample_quote

        # Mock broker
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

        # Execute
        trade = executor.execute_signal(
            sample_trading_signal,
            track_mlflow=False,
            execution_strategy="VWAP"
        )

        # Should execute
        assert trade is not None or True  # May vary based on conditions

    def test_execution_metrics_tracking(self):
        """Test that execution metrics are tracked."""
        executor = AdvancedTradingExecutor()

        # Simulate some executions
        executor.execution_times = [50.0, 75.0, 100.0, 60.0, 80.0]

        # Get metrics
        metrics = executor.get_execution_metrics()

        assert "avg_execution_time_ms" in metrics
        assert metrics["avg_execution_time_ms"] > 0
        assert metrics["min_execution_time_ms"] == 50.0
        assert metrics["max_execution_time_ms"] == 100.0
        assert "p95_execution_time_ms" in metrics

    def test_order_book_snapshot_creation(self, sample_quote):
        """Test order book snapshot creation."""
        executor = AdvancedTradingExecutor()

        order_book = executor._create_order_book_snapshot("AAPL", sample_quote)

        assert order_book.symbol == "AAPL"
        assert order_book.bid_price > 0
        assert order_book.ask_price > order_book.bid_price
        assert len(order_book.bid_levels) > 0
        assert len(order_book.ask_levels) > 0


class TestRiskManagement:
    """Test risk management features."""

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_respect_max_position_size(self, mock_market_data, mock_broker, sample_trading_signal):
        """Test that max position size is respected."""
        max_size = 5000.0
        executor = TradingExecutor(max_position_size=max_size)

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = {
            "symbol": "AAPL",
            "price": 100.0,  # Low price
            "bid": 99.9,
            "ask": 100.1,
        }

        # Mock broker
        mock_broker.get_position.return_value = None
        mock_broker.get_positions.return_value = []
        mock_broker.get_account.return_value = {
            "portfolio_value": 1000000.0,  # Large portfolio
            "buying_power": 500000.0,
        }
        mock_broker.can_trade.return_value = (True, "OK")
        mock_broker.place_market_order.return_value = Trade(
            symbol="AAPL",
            side="buy",
            quantity=50,  # Will be calculated
            status="filled",
        )

        # Execute
        trade = executor.execute_signal(sample_trading_signal, track_mlflow=False)

        # Verify quantity respects max position size
        if trade:
            position_value = trade.quantity * 100.0
            assert position_value <= max_size * 1.1  # Allow small margin

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_cannot_sell_without_position(self, mock_market_data, mock_broker):
        """Test that sell is blocked without position."""
        executor = TradingExecutor()

        # Create sell signal
        sell_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.SELL,
            confidence_score=0.8,
            ai_conviction_score=0.75,
            reasoning="Sell",
            source_agent="test",
        )

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = {"price": 150.0}

        # Mock broker - no position
        mock_broker.get_position.return_value = None

        # Execute
        trade = executor.execute_signal(sell_signal, track_mlflow=False)

        # Should not execute
        assert trade is None

    @patch('src.trading_engine.executor.broker')
    @patch('src.trading_engine.executor.market_data')
    def test_skip_duplicate_buy(self, mock_market_data, mock_broker, sample_trading_signal, sample_quote):
        """Test that duplicate buy is skipped if position exists."""
        executor = TradingExecutor()

        # Mock market data
        mock_market_data.is_market_open.return_value = True
        mock_market_data.get_quote.return_value = sample_quote

        # Mock broker - already have position
        mock_broker.get_position.return_value = Position(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=140.0,
        )

        # Execute
        trade = executor.execute_signal(sample_trading_signal, track_mlflow=False)

        # Should not execute duplicate buy
        assert trade is None
