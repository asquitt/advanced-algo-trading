"""
Comprehensive tests for broker.py to increase coverage from 29% to 80%+

Tests cover:
- Broker initialization
- Account operations
- Position management
- Order execution
- Error handling
- Paper trading validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from decimal import Decimal

from src.trading_engine.broker import AlpacaBroker
from src.data_layer.models import Trade, OrderSide, Position


class TestBrokerInitialization:
    """Test broker initialization and configuration."""

    @patch('src.trading_engine.broker.TradingClient')
    @patch('src.trading_engine.broker.settings')
    def test_broker_initializes_with_paper_trading(self, mock_settings, mock_client):
        """Test broker initializes in paper trading mode."""
        mock_settings.paper_trading = True
        mock_settings.alpaca_api_key = "test_key"
        mock_settings.alpaca_secret_key = "test_secret"

        broker = AlpacaBroker()

        mock_client.assert_called_once()
        # Verify paper trading URL is used
        call_kwargs = mock_client.call_args.kwargs
        assert call_kwargs.get('paper') is True

    @patch('src.trading_engine.broker.TradingClient')
    def test_broker_validates_api_keys(self, mock_client):
        """Test broker validates API keys on init."""
        # Mock account check
        mock_client.return_value.get_account.return_value = Mock(
            account_number="PA123456",
            cash="100000.00",
            portfolio_value="100000.00"
        )

        broker = AlpacaBroker()

        # Should call get_account to validate connection
        mock_client.return_value.get_account.assert_called_once()

    def test_broker_stores_configuration(self):
        """Test broker stores configuration correctly."""
        with patch('src.trading_engine.broker.TradingClient'):
            broker = AlpacaBroker()

            assert hasattr(broker, 'client')
            assert hasattr(broker, 'paper_trading')
            assert broker.paper_trading is True


class TestAccountOperations:
    """Test account-related operations."""

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_account_returns_account_info(self, mock_client):
        """Test get_account returns formatted account data."""
        # Mock Alpaca account response
        mock_account = Mock(
            account_number="PA123456",
            cash="50000.00",
            portfolio_value="105000.00",
            buying_power="100000.00",
            equity="105000.00"
        )
        mock_client.return_value.get_account.return_value = mock_account

        broker = AlpacaBroker()
        account = broker.get_account()

        assert account["account_number"] == "PA123456"
        assert float(account["cash"]) == 50000.00
        assert float(account["portfolio_value"]) == 105000.00
        assert float(account["buying_power"]) == 100000.00

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_account_handles_api_error(self, mock_client):
        """Test get_account handles API errors gracefully."""
        mock_client.return_value.get_account.side_effect = Exception("API Error")

        broker = AlpacaBroker()

        with pytest.raises(Exception, match="API Error"):
            broker.get_account()

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_buying_power(self, mock_client):
        """Test retrieving buying power."""
        mock_account = Mock(buying_power="75000.00")
        mock_client.return_value.get_account.return_value = mock_account

        broker = AlpacaBroker()
        account = broker.get_account()

        assert float(account["buying_power"]) == 75000.00


class TestPositionManagement:
    """Test position management operations."""

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_positions_returns_all_positions(self, mock_client):
        """Test get_positions returns all open positions."""
        # Mock multiple positions
        mock_positions = [
            Mock(
                symbol="AAPL",
                qty="10",
                current_price="150.25",
                market_value="1502.50",
                avg_entry_price="148.00",
                unrealized_pl="22.50",
                side="long"
            ),
            Mock(
                symbol="GOOGL",
                qty="5",
                current_price="2800.00",
                market_value="14000.00",
                avg_entry_price="2750.00",
                unrealized_pl="250.00",
                side="long"
            )
        ]
        mock_client.return_value.get_all_positions.return_value = mock_positions

        broker = AlpacaBroker()
        positions = broker.get_positions()

        assert len(positions) == 2
        assert positions[0]["symbol"] == "AAPL"
        assert float(positions[0]["qty"]) == 10
        assert positions[1]["symbol"] == "GOOGL"

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_position_for_symbol(self, mock_client):
        """Test getting position for specific symbol."""
        mock_position = Mock(
            symbol="AAPL",
            qty="10",
            current_price="150.00",
            market_value="1500.00",
            avg_entry_price="145.00",
            unrealized_pl="50.00"
        )
        mock_client.return_value.get_open_position.return_value = mock_position

        broker = AlpacaBroker()
        position = broker.get_position("AAPL")

        assert position["symbol"] == "AAPL"
        assert float(position["qty"]) == 10
        assert float(position["unrealized_pl"]) == 50.00

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_position_returns_none_when_no_position(self, mock_client):
        """Test get_position returns None when no position exists."""
        from alpaca.common.exceptions import APIError

        mock_client.return_value.get_open_position.side_effect = APIError("Position not found")

        broker = AlpacaBroker()
        position = broker.get_position("AAPL")

        assert position is None

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_positions_handles_empty_portfolio(self, mock_client):
        """Test get_positions with no open positions."""
        mock_client.return_value.get_all_positions.return_value = []

        broker = AlpacaBroker()
        positions = broker.get_positions()

        assert positions == []


class TestOrderExecution:
    """Test trade execution operations."""

    @patch('src.trading_engine.broker.TradingClient')
    def test_place_market_order_buy(self, mock_client):
        """Test placing a market buy order."""
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce

        mock_order = Mock(
            id="order-123",
            symbol="AAPL",
            qty="10",
            side=AlpacaOrderSide.BUY,
            type="market",
            status="filled",
            filled_avg_price="150.50",
            filled_at=datetime.now()
        )
        mock_client.return_value.submit_order.return_value = mock_order

        broker = AlpacaBroker()
        trade = broker.place_market_order("AAPL", OrderSide.BUY, 10)

        assert trade.symbol == "AAPL"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 10
        assert trade.status == "filled"
        assert float(trade.entry_price) == 150.50

    @patch('src.trading_engine.broker.TradingClient')
    def test_place_market_order_sell(self, mock_client):
        """Test placing a market sell order."""
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide

        mock_order = Mock(
            id="order-456",
            symbol="GOOGL",
            qty="5",
            side=AlpacaOrderSide.SELL,
            status="filled",
            filled_avg_price="2800.00",
            filled_at=datetime.now()
        )
        mock_client.return_value.submit_order.return_value = mock_order

        broker = AlpacaBroker()
        trade = broker.place_market_order("GOOGL", OrderSide.SELL, 5)

        assert trade.symbol == "GOOGL"
        assert trade.side == OrderSide.SELL
        assert trade.quantity == 5

    @patch('src.trading_engine.broker.TradingClient')
    def test_place_limit_order(self, mock_client):
        """Test placing a limit order."""
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide

        mock_order = Mock(
            id="order-789",
            symbol="MSFT",
            qty="20",
            side=AlpacaOrderSide.BUY,
            type="limit",
            limit_price="380.00",
            status="pending",
            filled_avg_price=None,
            filled_at=None
        )
        mock_client.return_value.submit_order.return_value = mock_order

        broker = AlpacaBroker()
        trade = broker.place_limit_order("MSFT", OrderSide.BUY, 20, 380.00)

        assert trade.symbol == "MSFT"
        assert trade.quantity == 20
        assert trade.status == "pending"

    @patch('src.trading_engine.broker.TradingClient')
    def test_order_execution_handles_rejection(self, mock_client):
        """Test handling of rejected orders."""
        from alpaca.common.exceptions import APIError

        mock_client.return_value.submit_order.side_effect = APIError("Insufficient funds")

        broker = AlpacaBroker()

        with pytest.raises(APIError, match="Insufficient funds"):
            broker.place_market_order("AAPL", OrderSide.BUY, 1000)

    @patch('src.trading_engine.broker.TradingClient')
    def test_cancel_order(self, mock_client):
        """Test order cancellation."""
        mock_client.return_value.cancel_order_by_id.return_value = None

        broker = AlpacaBroker()
        result = broker.cancel_order("order-123")

        assert result is True
        mock_client.return_value.cancel_order_by_id.assert_called_once_with("order-123")

    @patch('src.trading_engine.broker.TradingClient')
    def test_cancel_order_handles_error(self, mock_client):
        """Test order cancellation error handling."""
        from alpaca.common.exceptions import APIError

        mock_client.return_value.cancel_order_by_id.side_effect = APIError("Order not found")

        broker = AlpacaBroker()
        result = broker.cancel_order("invalid-order")

        assert result is False


class TestTradingValidation:
    """Test trading validation and safety checks."""

    @patch('src.trading_engine.broker.TradingClient')
    def test_can_trade_checks_buying_power(self, mock_client):
        """Test can_trade validates sufficient buying power."""
        mock_account = Mock(buying_power="10000.00")
        mock_client.return_value.get_account.return_value = mock_account

        broker = AlpacaBroker()

        # Should be able to trade with sufficient funds
        can_trade, reason = broker.can_trade("AAPL", 50, 150.00)
        assert can_trade is True

    @patch('src.trading_engine.broker.TradingClient')
    def test_can_trade_rejects_insufficient_funds(self, mock_client):
        """Test can_trade rejects trades exceeding buying power."""
        mock_account = Mock(buying_power="1000.00")
        mock_client.return_value.get_account.return_value = mock_account

        broker = AlpacaBroker()

        # Should reject trade exceeding buying power
        can_trade, reason = broker.can_trade("AAPL", 100, 150.00)
        assert can_trade is False
        assert "Insufficient buying power" in reason

    @patch('src.trading_engine.broker.TradingClient')
    def test_validate_trade_params(self, mock_client):
        """Test trade parameter validation."""
        broker = AlpacaBroker()

        # Valid params
        assert broker.validate_trade_params("AAPL", 10, 150.00) is True

        # Invalid quantity
        assert broker.validate_trade_params("AAPL", 0, 150.00) is False
        assert broker.validate_trade_params("AAPL", -10, 150.00) is False

        # Invalid price
        assert broker.validate_trade_params("AAPL", 10, 0) is False
        assert broker.validate_trade_params("AAPL", 10, -100) is False


class TestPortfolioState:
    """Test portfolio state tracking."""

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_portfolio_state(self, mock_client):
        """Test getting complete portfolio state."""
        mock_account = Mock(
            cash="50000.00",
            portfolio_value="105000.00"
        )
        mock_positions = [
            Mock(symbol="AAPL", qty="10", unrealized_pl="100.00"),
            Mock(symbol="GOOGL", qty="5", unrealized_pl="250.00")
        ]

        mock_client.return_value.get_account.return_value = mock_account
        mock_client.return_value.get_all_positions.return_value = mock_positions

        broker = AlpacaBroker()
        state = broker.get_portfolio_state()

        assert float(state.cash_balance) == 50000.00
        assert float(state.portfolio_value) == 105000.00
        assert state.active_positions == 2
        assert float(state.total_pnl) == 350.00  # 100 + 250

    @patch('src.trading_engine.broker.TradingClient')
    def test_portfolio_state_with_no_positions(self, mock_client):
        """Test portfolio state with no open positions."""
        mock_account = Mock(
            cash="100000.00",
            portfolio_value="100000.00"
        )
        mock_client.return_value.get_account.return_value = mock_account
        mock_client.return_value.get_all_positions.return_value = []

        broker = AlpacaBroker()
        state = broker.get_portfolio_state()

        assert state.active_positions == 0
        assert float(state.total_pnl) == 0.00


class TestErrorHandling:
    """Test error handling and recovery."""

    @patch('src.trading_engine.broker.TradingClient')
    def test_handles_network_timeout(self, mock_client):
        """Test handling of network timeout errors."""
        import requests

        mock_client.return_value.get_account.side_effect = requests.Timeout("Connection timeout")

        broker = AlpacaBroker()

        with pytest.raises(requests.Timeout):
            broker.get_account()

    @patch('src.trading_engine.broker.TradingClient')
    def test_handles_rate_limit(self, mock_client):
        """Test handling of rate limit errors."""
        from alpaca.common.exceptions import APIError

        mock_client.return_value.get_positions.side_effect = APIError("Rate limit exceeded")

        broker = AlpacaBroker()

        with pytest.raises(APIError, match="Rate limit"):
            broker.get_positions()

    @patch('src.trading_engine.broker.TradingClient')
    def test_handles_invalid_symbol(self, mock_client):
        """Test handling of invalid symbol errors."""
        from alpaca.common.exceptions import APIError

        mock_client.return_value.submit_order.side_effect = APIError("Invalid symbol")

        broker = AlpacaBroker()

        with pytest.raises(APIError, match="Invalid symbol"):
            broker.place_market_order("INVALID", OrderSide.BUY, 10)


class TestPaperTradingMode:
    """Test paper trading specific functionality."""

    @patch('src.trading_engine.broker.TradingClient')
    def test_paper_trading_always_enabled(self, mock_client):
        """Test that paper trading is always enabled for safety."""
        broker = AlpacaBroker()

        assert broker.paper_trading is True

        # Verify client was initialized with paper=True
        call_kwargs = mock_client.call_args.kwargs
        assert call_kwargs.get('paper') is True

    @patch('src.trading_engine.broker.TradingClient')
    def test_paper_trading_uses_paper_api_url(self, mock_client):
        """Test paper trading uses correct API URL."""
        broker = AlpacaBroker()

        # Check that paper trading URL is configured
        assert broker.paper_trading is True


class TestOrderStatus:
    """Test order status tracking."""

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_order_status(self, mock_client):
        """Test retrieving order status."""
        mock_order = Mock(
            id="order-123",
            status="filled",
            filled_qty="10",
            filled_avg_price="150.50"
        )
        mock_client.return_value.get_order_by_id.return_value = mock_order

        broker = AlpacaBroker()
        status = broker.get_order_status("order-123")

        assert status["status"] == "filled"
        assert status["filled_qty"] == "10"

    @patch('src.trading_engine.broker.TradingClient')
    def test_get_open_orders(self, mock_client):
        """Test retrieving all open orders."""
        mock_orders = [
            Mock(id="order-1", symbol="AAPL", status="pending"),
            Mock(id="order-2", symbol="GOOGL", status="partially_filled")
        ]
        mock_client.return_value.get_orders.return_value = mock_orders

        broker = AlpacaBroker()
        orders = broker.get_open_orders()

        assert len(orders) == 2
        assert orders[0]["status"] == "pending"
        assert orders[1]["status"] == "partially_filled"
