"""
Broker interface for executing trades.

This module integrates with Alpaca for paper trading.
All trades are executed in paper trading mode by default for safety.
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.common.exceptions import APIError
from typing import Optional, Dict, Any, List
from src.utils.config import settings
from src.utils.logger import app_logger
from src.data_layer.models import Trade, Position, PortfolioState, OrderStatus
from datetime import datetime


class AlpacaBroker:
    """
    Alpaca broker integration for paper trading.

    IMPORTANT: This is configured for PAPER TRADING ONLY by default.
    Real money trading requires changing the base URL and using live API keys.
    """

    def __init__(self):
        """Initialize Alpaca trading client."""
        self.client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.paper_trading,  # ALWAYS True by default!
        )

        if not settings.paper_trading:
            app_logger.warning("🚨 LIVE TRADING MODE ENABLED! Use with extreme caution!")
        else:
            app_logger.info("✅ Paper trading mode enabled (safe)")

    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dict with account details (cash, buying power, portfolio value)
        """
        try:
            account = self.client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
            }
        except APIError as e:
            app_logger.error(f"Failed to get account info: {e}")
            raise

    def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        try:
            positions = self.client.get_all_positions()
            return [
                Position(
                    symbol=pos.symbol,
                    quantity=int(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_percent=float(pos.unrealized_plpc),
                )
                for pos in positions
            ]
        except APIError as e:
            app_logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Position object or None if no position
        """
        try:
            pos = self.client.get_open_position(symbol)
            return Position(
                symbol=pos.symbol,
                quantity=int(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_percent=float(pos.unrealized_plpc),
            )
        except APIError:
            return None

    def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,  # 'buy' or 'sell'
    ) -> Optional[Trade]:
        """
        Place a market order.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'

        Returns:
            Trade object or None if failed
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            order = self.client.submit_order(order_request)

            app_logger.info(
                f"Market order placed: {side.upper()} {qty} {symbol} "
                f"(Order ID: {order.id})"
            )

            return Trade(
                symbol=symbol,
                side=side.lower(),
                quantity=qty,
                status=self._map_order_status(order.status),
                order_id=str(order.id),
                filled_at=order.filled_at,
                entry_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )

        except APIError as e:
            app_logger.error(f"Failed to place market order for {symbol}: {e}")
            return None

    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
    ) -> Optional[Trade]:
        """
        Place a limit order.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'
            limit_price: Limit price

        Returns:
            Trade object or None if failed
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )

            order = self.client.submit_order(order_request)

            app_logger.info(
                f"Limit order placed: {side.upper()} {qty} {symbol} @ ${limit_price} "
                f"(Order ID: {order.id})"
            )

            return Trade(
                symbol=symbol,
                side=side.lower(),
                quantity=qty,
                entry_price=limit_price,
                status=self._map_order_status(order.status),
                order_id=str(order.id),
            )

        except APIError as e:
            app_logger.error(f"Failed to place limit order for {symbol}: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            self.client.cancel_order_by_id(order_id)
            app_logger.info(f"Order {order_id} cancelled")
            return True
        except APIError as e:
            app_logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """
        Close an entire position.

        Args:
            symbol: Stock ticker

        Returns:
            True if closed successfully
        """
        try:
            self.client.close_position(symbol)
            app_logger.info(f"Position closed for {symbol}")
            return True
        except APIError as e:
            app_logger.error(f"Failed to close position for {symbol}: {e}")
            return False

    def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state.

        Returns:
            PortfolioState object
        """
        try:
            account = self.client.get_account()
            positions = self.get_positions()

            # Calculate metrics
            total_pnl = float(account.equity) - float(account.last_equity)
            total_pnl_percent = (total_pnl / float(account.last_equity)) if float(account.last_equity) > 0 else 0

            return PortfolioState(
                cash_balance=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                active_positions=len(positions),
            )
        except APIError as e:
            app_logger.error(f"Failed to get portfolio state: {e}")
            raise

    def _map_order_status(self, alpaca_status: str) -> OrderStatus:
        """
        Map Alpaca order status to our OrderStatus enum.

        Args:
            alpaca_status: Alpaca order status

        Returns:
            OrderStatus enum value
        """
        status_map = {
            "new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return status_map.get(alpaca_status.lower(), OrderStatus.PENDING)

    def can_trade(self, symbol: str, qty: int, side: str) -> tuple[bool, str]:
        """
        Check if we can execute a trade.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        try:
            account = self.get_account()

            # Check if trading is blocked
            if account.get("trading_blocked"):
                return False, "Trading is blocked for this account"

            # For buy orders, check buying power
            if side.lower() == "buy":
                # Rough estimate (would need current price for exact check)
                # This is simplified - in production, get real-time quote
                buying_power = account.get("buying_power", 0)
                if buying_power < 100:  # Arbitrary minimum
                    return False, f"Insufficient buying power: ${buying_power}"

            # For sell orders, check if we have the position
            if side.lower() == "sell":
                position = self.get_position(symbol)
                if not position or position.quantity < qty:
                    return False, f"Insufficient shares to sell (have {position.quantity if position else 0}, need {qty})"

            return True, "OK"

        except Exception as e:
            app_logger.error(f"Error checking trade eligibility: {e}")
            return False, f"Error: {str(e)}"


# Global broker instance
broker = AlpacaBroker()
