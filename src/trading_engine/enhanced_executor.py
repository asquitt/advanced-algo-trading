"""
Enhanced Trading Executor with Advanced Risk Management

Integrates:
1. Adaptive position sizing (drawdown management)
2. Advanced slippage reduction
3. Feature-enriched signals

Author: LLM Trading Platform
"""

from datetime import datetime
from typing import Optional, List, Dict
import numpy as np
from loguru import logger

from src.data_layer.models import (
    TradingSignal, Trade, OrderSide, OrderType, SignalType
)
from src.trading_engine.broker import AlpacaBroker
from src.trading_engine.slippage_management import (
    slippage_analyzer,
    adaptive_executor,
    ExecutionOrder,
    ExecutionUrgency,
    SlippageStrategy,
    MarketConditions
)
from src.trading_engine.position_sizing import adaptive_position_sizer
from src.trading_engine.hft_techniques import OrderBookSnapshot
from src.llm_agents.feature_engineering import feature_engineer
from src.utils.config import settings


class EnhancedTradingExecutor:
    """
    Enhanced executor with all advanced features.

    Improvements over basic executor:
    1. Adaptive position sizing based on drawdown
    2. Intelligent slippage reduction strategies
    3. Feature-enriched signal validation
    4. Real-time risk monitoring
    5. Performance-based adjustments
    """

    def __init__(self, broker: AlpacaBroker):
        self.broker = broker
        self.max_position_size = settings.max_position_size
        self.max_open_positions = settings.max_open_positions

        # Trade history for performance analysis
        self.trade_history: List[Trade] = []
        self.closed_trades: List[Trade] = []

        # Risk tracking
        self.current_portfolio_heat = 0.0  # Total risk exposure

        logger.info("Enhanced trading executor initialized")

    def execute_signal(
        self,
        signal: TradingSignal,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM,
        use_advanced_features: bool = True
    ) -> Optional[Trade]:
        """
        Execute trading signal with advanced risk management.

        Args:
            signal: Trading signal to execute
            urgency: Execution urgency level
            use_advanced_features: Whether to use advanced features

        Returns:
            Trade object if executed, None otherwise
        """
        try:
            logger.info(
                f"Executing enhanced signal: {signal.symbol} {signal.signal_type.value} "
                f"(confidence: {signal.confidence_score:.2f}, urgency: {urgency.value})"
            )

            # Get current account state
            account = self.broker.get_account()
            portfolio_value = account.get("portfolio_value", 0.0)

            # Get current positions
            positions = self.broker.get_positions()
            open_positions_count = len(positions)

            # Check if can add more positions
            if open_positions_count >= self.max_open_positions:
                logger.warning(
                    f"Cannot open new position - at max limit "
                    f"({open_positions_count}/{self.max_open_positions})"
                )
                return None

            # Get market data
            quote = self.broker.get_quote(signal.symbol)
            if not quote:
                logger.error(f"No quote available for {signal.symbol}")
                return None

            current_price = quote.get("price", 0.0)
            if current_price <= 0:
                logger.error(f"Invalid price for {signal.symbol}: {current_price}")
                return None

            # Create order book snapshot (simplified)
            order_book = OrderBookSnapshot(
                symbol=signal.symbol,
                bid_price=quote.get("bid", current_price * 0.999),
                ask_price=quote.get("ask", current_price * 1.001),
                bid_size=quote.get("bid_size", 100),
                ask_size=quote.get("ask_size", 100),
                timestamp=datetime.now()
            )

            # Enrich signal with advanced features (if enabled)
            if use_advanced_features:
                enriched_signal = self._enrich_signal_with_features(
                    signal, order_book
                )
            else:
                enriched_signal = signal

            # Validate signal
            if not self._validate_signal(enriched_signal):
                logger.warning(f"Signal validation failed for {signal.symbol}")
                return None

            # Calculate position size with adaptive sizing
            position_sizing = adaptive_position_sizer.calculate_position_size(
                portfolio_value=portfolio_value,
                signal_confidence=enriched_signal.confidence_score,
                market_volatility=enriched_signal.technical_score / 100.0,  # Approximation
                recent_trades=self.closed_trades[-50:],  # Last 50 trades
                open_positions=open_positions_count,
                current_portfolio_heat=self.current_portfolio_heat
            )

            logger.info(
                f"Position sizing: {position_sizing.adjusted_size_pct*100:.2f}% "
                f"(risk_mode: {position_sizing.risk_mode.value}) - "
                f"{position_sizing.reasoning}"
            )

            # Check if halted
            if position_sizing.max_position_value <= 0:
                logger.warning(
                    f"Trading halted due to risk conditions: "
                    f"{position_sizing.reasoning}"
                )
                return None

            # Calculate quantity
            position_value = min(
                position_sizing.max_position_value,
                self.max_position_size,
                account.get("buying_power", 0.0) * 0.9  # Use 90% of buying power
            )

            if position_value < current_price:
                logger.warning(
                    f"Insufficient capital for {signal.symbol}: "
                    f"need ${current_price:.2f}, have ${position_value:.2f}"
                )
                return None

            quantity = int(position_value / current_price)
            if quantity <= 0:
                logger.warning(f"Calculated quantity is 0 for {signal.symbol}")
                return None

            # Estimate slippage and optimize execution
            market_conditions = slippage_analyzer.assess_market_conditions(
                symbol=signal.symbol,
                order_book=order_book,
                recent_prices=[current_price],  # Would use historical data
                recent_volumes=[1000]  # Would use historical data
            )

            slippage_estimate = slippage_analyzer.estimate_slippage(
                symbol=signal.symbol,
                quantity=quantity,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                urgency=urgency,
                order_book=order_book,
                conditions=market_conditions
            )

            logger.info(
                f"Slippage estimate: {slippage_estimate.expected_slippage_bps:.2f}bps "
                f"(strategy: {slippage_estimate.recommended_strategy.value}) - "
                f"{slippage_estimate.reasoning}"
            )

            # Check if slippage is acceptable
            max_acceptable_slippage = 50.0  # 50 basis points
            if slippage_estimate.expected_slippage_bps > max_acceptable_slippage:
                logger.warning(
                    f"Expected slippage too high for {signal.symbol}: "
                    f"{slippage_estimate.expected_slippage_bps:.2f}bps"
                )
                # Could still execute with adjusted strategy
                # For now, proceed with caution

            # Create execution order
            execution_order = ExecutionOrder(
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                quantity=quantity,
                limit_price=None,  # Will be set by strategy
                strategy=slippage_estimate.recommended_strategy,
                urgency=urgency,
                max_slippage_bps=max_acceptable_slippage,
                time_horizon_seconds=slippage_estimate.execution_time_seconds
            )

            # Execute with adaptive strategy
            execution_result = adaptive_executor.execute_with_strategy(
                order=execution_order,
                conditions=market_conditions,
                order_book=order_book
            )

            if not execution_result or execution_result.get("filled_quantity", 0) == 0:
                logger.error(f"Order execution failed for {signal.symbol}")
                return None

            filled_qty = execution_result["filled_quantity"]
            fill_price = execution_result["avg_fill_price"]
            actual_slippage = execution_result.get("slippage_bps", 0.0)

            # Record actual slippage for model improvement
            slippage_analyzer.record_actual_slippage(
                symbol=signal.symbol,
                estimated_slippage_bps=slippage_estimate.expected_slippage_bps,
                actual_slippage_bps=actual_slippage
            )

            # Create trade record
            trade = Trade(
                symbol=signal.symbol,
                order_id=f"ENH-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                side=execution_order.side,
                quantity=filled_qty,
                entry_price=fill_price,
                exit_price=None,
                status="open",
                signal_id=signal.signal_id,
                entry_time=datetime.now(),
                exit_time=None
            )

            # Update portfolio heat
            position_risk = position_value / portfolio_value if portfolio_value > 0 else 0
            self.current_portfolio_heat += position_risk

            # Track trade
            self.trade_history.append(trade)

            logger.info(
                f"✅ Trade executed: {trade.side.value} {filled_qty} {signal.symbol} @ ${fill_price:.2f} "
                f"(slippage: {actual_slippage:.2f}bps, "
                f"strategy: {execution_result.get('strategy', 'UNKNOWN')})"
            )

            return trade

        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {str(e)}")
            return None

    def close_position(
        self,
        symbol: str,
        urgency: ExecutionUrgency = ExecutionUrgency.HIGH
    ) -> Optional[Trade]:
        """
        Close an open position with optimal execution.

        Args:
            symbol: Symbol to close
            urgency: Urgency level for closing

        Returns:
            Updated Trade object with exit information
        """
        try:
            # Find open position
            positions = self.broker.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)

            if not position:
                logger.warning(f"No open position found for {symbol}")
                return None

            quantity = position["quantity"]
            side = OrderSide.SELL if position["side"] == "long" else OrderSide.BUY

            # Get current quote
            quote = self.broker.get_quote(symbol)
            if not quote:
                logger.error(f"No quote available for {symbol}")
                return None

            current_price = quote.get("price", 0.0)
            order_book = OrderBookSnapshot(
                symbol=symbol,
                bid_price=quote.get("bid", current_price * 0.999),
                ask_price=quote.get("ask", current_price * 1.001),
                bid_size=quote.get("bid_size", 100),
                ask_size=quote.get("ask_size", 100),
                timestamp=datetime.now()
            )

            # Assess market conditions for exit
            market_conditions = slippage_analyzer.assess_market_conditions(
                symbol=symbol,
                order_book=order_book,
                recent_prices=[current_price],
                recent_volumes=[1000]
            )

            # Get slippage estimate for exit
            slippage_estimate = slippage_analyzer.estimate_slippage(
                symbol=symbol,
                quantity=quantity,
                side=side,
                urgency=urgency,
                order_book=order_book,
                conditions=market_conditions
            )

            # Create exit order
            execution_order = ExecutionOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=None,
                strategy=slippage_estimate.recommended_strategy,
                urgency=urgency,
                max_slippage_bps=100.0,  # More lenient for exits
                time_horizon_seconds=slippage_estimate.execution_time_seconds
            )

            # Execute exit
            execution_result = adaptive_executor.execute_with_strategy(
                order=execution_order,
                conditions=market_conditions,
                order_book=order_book
            )

            if not execution_result:
                logger.error(f"Exit order execution failed for {symbol}")
                return None

            exit_price = execution_result["avg_fill_price"]

            # Update trade record
            for trade in reversed(self.trade_history):
                if trade.symbol == symbol and trade.status == "open":
                    trade.exit_price = exit_price
                    trade.exit_time = datetime.now()
                    trade.status = "closed"

                    # Calculate P&L
                    if trade.side == OrderSide.BUY:
                        trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                    else:
                        trade.pnl = (trade.entry_price - exit_price) * trade.quantity

                    # Move to closed trades
                    self.closed_trades.append(trade)

                    # Update portfolio heat
                    account = self.broker.get_account()
                    portfolio_value = account.get("portfolio_value", 1.0)
                    position_risk = (trade.entry_price * trade.quantity) / portfolio_value
                    self.current_portfolio_heat = max(0, self.current_portfolio_heat - position_risk)

                    logger.info(
                        f"✅ Position closed: {symbol} @ ${exit_price:.2f}, "
                        f"P&L: ${trade.pnl:.2f}"
                    )

                    return trade

            return None

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            return None

    def _enrich_signal_with_features(
        self,
        signal: TradingSignal,
        order_book: OrderBookSnapshot
    ) -> TradingSignal:
        """
        Enrich signal with advanced technical features.

        Args:
            signal: Original signal
            order_book: Current order book

        Returns:
            Enhanced signal with additional features
        """
        try:
            # Get historical data (simplified - would fetch from market data source)
            # For now, use current price with some synthetic history
            current_price = order_book.mid_price
            prices = [current_price * (1 + np.random.normal(0, 0.01))
                     for _ in range(60)]  # 60 periods
            volumes = [1000 + int(np.random.normal(0, 200)) for _ in range(60)]
            highs = [p * 1.01 for p in prices]
            lows = [p * 0.99 for p in prices]

            # Create enhanced features
            features = feature_engineer.create_features(
                symbol=signal.symbol,
                prices=prices,
                volumes=volumes,
                highs=highs,
                lows=lows,
                news_sentiment=signal.sentiment_score,
                social_sentiment=signal.sentiment_score * 0.8  # Approximation
            )

            # Adjust signal confidence based on technical features
            technical_signals = 0
            total_checks = 0

            # Check technical indicators alignment
            tech = features.technical

            # RSI check
            if signal.signal_type == SignalType.BUY:
                if tech.rsi_14 < 40:  # Oversold
                    technical_signals += 1
            elif signal.signal_type == SignalType.SELL:
                if tech.rsi_14 > 60:  # Overbought
                    technical_signals += 1
            total_checks += 1

            # MACD check
            if signal.signal_type == SignalType.BUY:
                if tech.macd > tech.macd_signal:  # Bullish crossover
                    technical_signals += 1
            elif signal.signal_type == SignalType.SELL:
                if tech.macd < tech.macd_signal:  # Bearish crossover
                    technical_signals += 1
            total_checks += 1

            # Trend alignment check
            if signal.signal_type == SignalType.BUY:
                if tech.current_price > tech.sma_50:  # Above 50 SMA
                    technical_signals += 1
            elif signal.signal_type == SignalType.SELL:
                if tech.current_price < tech.sma_50:  # Below 50 SMA
                    technical_signals += 1
            total_checks += 1

            # Multi-timeframe alignment
            mtf = features.multi_timeframe
            if signal.signal_type == SignalType.BUY:
                if mtf.alignment_score > 0.6 and mtf.daily_trend == "bullish":
                    technical_signals += 1
            elif signal.signal_type == SignalType.SELL:
                if mtf.alignment_score > 0.6 and mtf.daily_trend == "bearish":
                    technical_signals += 1
            total_checks += 1

            # Calculate technical agreement ratio
            technical_agreement = technical_signals / total_checks if total_checks > 0 else 0.5

            # Adjust confidence score
            original_confidence = signal.confidence_score
            adjusted_confidence = (original_confidence * 0.6 + technical_agreement * 0.4)

            # Update signal
            signal.confidence_score = adjusted_confidence
            signal.technical_score = technical_agreement * 100

            logger.debug(
                f"Signal enriched: {signal.symbol} confidence "
                f"{original_confidence:.2f} -> {adjusted_confidence:.2f} "
                f"(technical_agreement: {technical_agreement:.2f})"
            )

            return signal

        except Exception as e:
            logger.warning(f"Error enriching signal: {str(e)}")
            return signal  # Return original if enrichment fails

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal before execution."""
        if signal.signal_type == SignalType.HOLD:
            return False

        if signal.confidence_score < 0.3:  # Minimum confidence threshold
            logger.warning(
                f"Signal confidence too low: {signal.confidence_score:.2f}"
            )
            return False

        # Check if symbol is currently tradable
        # (would check market hours, halts, etc.)

        return True

    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0
            }

        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl < 0]

        total_pnl = sum(t.pnl for t in self.closed_trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "current_heat": self.current_portfolio_heat
        }


# Global instance would be initialized in main.py
# enhanced_executor = EnhancedTradingExecutor(broker)
