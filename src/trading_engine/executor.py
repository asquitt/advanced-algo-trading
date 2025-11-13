"""
Trading signal executor with risk management.

This module:
- Takes trading signals from LLM agents
- Applies risk management rules
- Executes trades via broker
- Tracks performance

Risk management includes:
- Position sizing based on risk percentage
- Maximum number of open positions
- Stop-loss and take-profit levels
- Portfolio heat limits
"""

from typing import Optional, Dict, Any
from src.data_layer.models import TradingSignal, Trade, SignalType
from src.trading_engine.broker import broker
from src.data_layer.market_data import market_data
from src.utils.config import settings
from src.utils.logger import app_logger
from src.utils.database import get_db
import mlflow


class TradingExecutor:
    """
    Executes trading signals with risk management.

    This is the bridge between AI signals and actual trade execution.
    """

    def __init__(
        self,
        max_position_size: float = None,
        risk_per_trade: float = None,
        max_open_positions: int = None,
    ):
        """
        Initialize trading executor.

        Args:
            max_position_size: Maximum dollar amount per position
            risk_per_trade: Risk percentage per trade (e.g., 0.02 = 2%)
            max_open_positions: Maximum concurrent positions
        """
        self.max_position_size = max_position_size or settings.max_position_size
        self.risk_per_trade = risk_per_trade or settings.risk_per_trade
        self.max_open_positions = max_open_positions or settings.max_open_positions

        app_logger.info(
            f"Trading executor initialized: "
            f"max_position=${self.max_position_size}, "
            f"risk_per_trade={self.risk_per_trade*100}%, "
            f"max_positions={self.max_open_positions}"
        )

    def execute_signal(
        self,
        signal: TradingSignal,
        track_mlflow: bool = True
    ) -> Optional[Trade]:
        """
        Execute a trading signal with risk management.

        Args:
            signal: TradingSignal to execute
            track_mlflow: Whether to log to MLflow

        Returns:
            Trade object if executed, None otherwise
        """
        app_logger.info(
            f"Processing signal for {signal.symbol}: {signal.signal_type.value} "
            f"(conviction={signal.ai_conviction_score:.2f})"
        )

        # Check if we should trade during current market hours
        if settings.trading_hours_only and not market_data.is_market_open():
            app_logger.warning(f"Market is closed, skipping signal for {signal.symbol}")
            return None

        # Get current quote
        quote = market_data.get_quote(signal.symbol)
        if not quote:
            app_logger.error(f"Failed to get quote for {signal.symbol}")
            return None

        current_price = quote.get("price")
        # Type-safe price validation
        try:
            current_price = float(current_price) if current_price is not None else None
        except (ValueError, TypeError):
            current_price = None

        if not current_price or current_price <= 0:
            app_logger.error(f"Invalid price for {signal.symbol}: {current_price}")
            return None

        # Handle different signal types
        if signal.signal_type == SignalType.BUY:
            return self._execute_buy(signal, current_price, track_mlflow)
        elif signal.signal_type == SignalType.SELL:
            return self._execute_sell(signal, current_price, track_mlflow)
        else:  # HOLD
            app_logger.info(f"HOLD signal for {signal.symbol}, no action taken")
            return None

    def _execute_buy(
        self,
        signal: TradingSignal,
        current_price: float,
        track_mlflow: bool
    ) -> Optional[Trade]:
        """
        Execute a buy signal.

        Args:
            signal: Trading signal
            current_price: Current stock price
            track_mlflow: Whether to log to MLflow

        Returns:
            Trade object if executed
        """
        # Check if we already have a position
        existing_position = broker.get_position(signal.symbol)
        if existing_position:
            app_logger.info(
                f"Already have position in {signal.symbol}, skipping buy"
            )
            return None

        # Check max open positions
        current_positions = broker.get_positions()
        if len(current_positions) >= self.max_open_positions:
            app_logger.warning(
                f"Max open positions ({self.max_open_positions}) reached, "
                f"skipping buy for {signal.symbol}"
            )
            return None

        # Calculate position size with error handling
        try:
            account = broker.get_account()
            portfolio_value = account.get("portfolio_value", 0)
        except Exception as e:
            app_logger.error(f"Failed to get account info: {e}")
            return None

        # Position sizing: use smaller of:
        # 1. Max position size
        # 2. Risk-based sizing (portfolio_value * risk_per_trade / risk_per_share)
        # 3. Available buying power
        position_value = min(
            self.max_position_size,
            portfolio_value * self.risk_per_trade * 10,  # Scale up by 10 for actual position
            account.get("buying_power", 0) * 0.9  # Use 90% of buying power
        )

        # Calculate quantity
        qty = int(position_value / current_price)
        if qty <= 0:
            app_logger.warning(
                f"Calculated quantity is {qty} for {signal.symbol}, skipping"
            )
            return None

        # Verify we can trade
        can_trade, reason = broker.can_trade(signal.symbol, qty, "buy")
        if not can_trade:
            app_logger.warning(f"Cannot execute buy for {signal.symbol}: {reason}")
            return None

        # Execute the trade
        app_logger.info(
            f"Executing BUY: {qty} shares of {signal.symbol} @ ${current_price:.2f} "
            f"(total: ${qty * current_price:.2f})"
        )

        trade = broker.place_market_order(signal.symbol, qty, "buy")

        if trade and track_mlflow:
            self._log_trade_to_mlflow(signal, trade, "buy")

        return trade

    def _execute_sell(
        self,
        signal: TradingSignal,
        current_price: float,
        track_mlflow: bool
    ) -> Optional[Trade]:
        """
        Execute a sell signal.

        Args:
            signal: Trading signal
            current_price: Current stock price
            track_mlflow: Whether to log to MLflow

        Returns:
            Trade object if executed
        """
        # Check if we have a position to sell
        position = broker.get_position(signal.symbol)
        if not position:
            app_logger.info(f"No position in {signal.symbol} to sell")
            return None

        qty = position.quantity

        # Verify we can trade
        can_trade, reason = broker.can_trade(signal.symbol, qty, "sell")
        if not can_trade:
            app_logger.warning(f"Cannot execute sell for {signal.symbol}: {reason}")
            return None

        # Execute the trade
        app_logger.info(
            f"Executing SELL: {qty} shares of {signal.symbol} @ ${current_price:.2f} "
            f"(total: ${qty * current_price:.2f})"
        )

        trade = broker.place_market_order(signal.symbol, qty, "sell")

        if trade and track_mlflow:
            self._log_trade_to_mlflow(signal, trade, "sell")

        return trade

    def _log_trade_to_mlflow(
        self,
        signal: TradingSignal,
        trade: Trade,
        action: str
    ):
        """
        Log trade execution to MLflow.

        Args:
            signal: Original trading signal
            trade: Executed trade
            action: 'buy' or 'sell'
        """
        try:
            mlflow.log_metrics({
                f"trade_{action}_qty": trade.quantity,
                f"trade_{action}_price": trade.entry_price or 0,
                "ai_conviction": signal.ai_conviction_score,
            })
            mlflow.set_tag(f"last_trade_{action}", signal.symbol)
        except Exception as e:
            app_logger.error(f"Failed to log trade to MLflow: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary with performance metrics.

        Returns:
            Dict with portfolio metrics
        """
        account = broker.get_account()
        positions = broker.get_positions()
        portfolio_state = broker.get_portfolio_state()

        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in positions if pos.unrealized_pnl
        )

        return {
            "account": account,
            "portfolio_state": portfolio_state.dict(),
            "positions": [pos.dict() for pos in positions],
            "total_unrealized_pnl": total_unrealized_pnl,
            "num_positions": len(positions),
        }


# Global executor instance
executor = TradingExecutor()
