"""
Prometheus Metrics for Trading Platform

Comprehensive metrics collection for monitoring trading system performance,
risk, execution quality, and system health.

Metric types:
- Counter: Monotonically increasing value (e.g., total trades)
- Gauge: Value that can go up or down (e.g., portfolio value)
- Histogram: Distribution of values (e.g., order latency)
- Summary: Similar to histogram but calculates quantiles

Author: LLM Trading Platform
"""

from typing import Optional, Dict, List
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from loguru import logger


# Create custom registry for trading metrics
metrics_registry = CollectorRegistry()


class TradingMetrics:
    """
    Centralized metrics collection for the trading platform.

    Organizes metrics into categories:
    - Portfolio & P&L
    - Trading activity
    - Risk management
    - Execution quality
    - System performance
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize trading metrics.

        Args:
            registry: Prometheus registry (default: global registry)
        """
        self.registry = registry or metrics_registry

        logger.info("Initializing Prometheus trading metrics...")

        # ====================================================================
        # Portfolio & P&L Metrics
        # ====================================================================

        self.portfolio_value = Gauge(
            'portfolio_value_dollars',
            'Current portfolio value in dollars',
            registry=self.registry
        )

        self.portfolio_pnl = Gauge(
            'portfolio_pnl_dollars',
            'Portfolio profit and loss in dollars',
            registry=self.registry
        )

        self.account_buying_power = Gauge(
            'account_buying_power_dollars',
            'Available buying power in dollars',
            registry=self.registry
        )

        self.portfolio_return = Gauge(
            'portfolio_return_percent',
            'Portfolio return percentage',
            registry=self.registry
        )

        # ====================================================================
        # Trading Activity Metrics
        # ====================================================================

        self.trades_total = Counter(
            'trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'order_type'],
            registry=self.registry
        )

        self.winning_trades_total = Counter(
            'winning_trades_total',
            'Total number of winning trades',
            ['symbol'],
            registry=self.registry
        )

        self.losing_trades_total = Counter(
            'losing_trades_total',
            'Total number of losing trades',
            ['symbol'],
            registry=self.registry
        )

        self.open_positions_count = Gauge(
            'open_positions_count',
            'Number of currently open positions',
            registry=self.registry
        )

        self.position_value = Gauge(
            'position_value_dollars',
            'Value of position in dollars',
            ['symbol'],
            registry=self.registry
        )

        self.trade_volume = Counter(
            'trade_volume_shares',
            'Total trade volume in shares',
            ['symbol'],
            registry=self.registry
        )

        # ====================================================================
        # Risk Management Metrics
        # ====================================================================

        self.portfolio_drawdown = Gauge(
            'portfolio_drawdown_percent',
            'Current portfolio drawdown percentage',
            registry=self.registry
        )

        self.max_drawdown = Gauge(
            'max_drawdown_percent',
            'Maximum historical drawdown percentage',
            registry=self.registry
        )

        self.portfolio_var_95 = Gauge(
            'portfolio_var_95_percent',
            'Value at Risk at 95% confidence level',
            registry=self.registry
        )

        self.portfolio_cvar_95 = Gauge(
            'portfolio_cvar_95_percent',
            'Conditional Value at Risk at 95% confidence',
            registry=self.registry
        )

        self.sharpe_ratio = Gauge(
            'sharpe_ratio_30d',
            'Sharpe ratio calculated over 30 days',
            registry=self.registry
        )

        self.sortino_ratio = Gauge(
            'sortino_ratio_30d',
            'Sortino ratio calculated over 30 days',
            registry=self.registry
        )

        self.portfolio_volatility = Gauge(
            'portfolio_volatility_percent',
            'Portfolio volatility (annualized)',
            registry=self.registry
        )

        self.risk_score = Gauge(
            'risk_score',
            'Overall risk score (0-100)',
            registry=self.registry
        )

        self.risk_mode = Gauge(
            'risk_mode',
            'Current risk mode (0=NORMAL, 1=CONSERVATIVE, 2=DEFENSIVE, 3=HALT)',
            registry=self.registry
        )

        self.consecutive_losses = Gauge(
            'consecutive_losses',
            'Number of consecutive losing trades',
            registry=self.registry
        )

        self.kelly_fraction = Gauge(
            'kelly_fraction',
            'Kelly criterion fraction for position sizing',
            registry=self.registry
        )

        # ====================================================================
        # Execution Quality Metrics
        # ====================================================================

        self.orders_submitted_total = Counter(
            'orders_submitted_total',
            'Total orders submitted',
            ['symbol', 'order_type'],
            registry=self.registry
        )

        self.orders_filled_total = Counter(
            'orders_filled_total',
            'Total orders filled',
            ['symbol', 'order_type'],
            registry=self.registry
        )

        self.orders_rejected_total = Counter(
            'orders_rejected_total',
            'Total orders rejected',
            ['symbol', 'reason'],
            registry=self.registry
        )

        self.orders_cancelled_total = Counter(
            'orders_cancelled_total',
            'Total orders cancelled',
            ['symbol'],
            registry=self.registry
        )

        self.order_latency = Histogram(
            'order_latency_seconds',
            'Order execution latency in seconds',
            ['order_type'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )

        self.slippage_bps = Histogram(
            'slippage_bps',
            'Execution slippage in basis points',
            ['symbol'],
            buckets=(0, 1, 2, 5, 10, 20, 50, 100),
            registry=self.registry
        )

        self.market_impact_cost_bps = Gauge(
            'market_impact_cost_bps',
            'Market impact cost in basis points',
            ['symbol'],
            registry=self.registry
        )

        self.execution_quality_score = Gauge(
            'execution_quality_score',
            'Execution quality score (0-100)',
            ['algorithm'],
            registry=self.registry
        )

        # ====================================================================
        # Signal & Model Metrics
        # ====================================================================

        self.signals_generated_total = Counter(
            'signals_generated_total',
            'Total trading signals generated',
            ['symbol', 'signal_type', 'strategy'],
            registry=self.registry
        )

        self.signal_confidence = Histogram(
            'signal_confidence',
            'Signal confidence score distribution',
            ['strategy'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )

        self.model_prediction_errors_total = Counter(
            'model_prediction_errors_total',
            'Total model prediction errors',
            ['model_name'],
            registry=self.registry
        )

        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model prediction accuracy',
            ['model_name'],
            registry=self.registry
        )

        # ====================================================================
        # System Performance Metrics
        # ====================================================================

        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry
        )

        self.kafka_messages_produced_total = Counter(
            'kafka_messages_produced_total',
            'Total Kafka messages produced',
            ['topic'],
            registry=self.registry
        )

        self.kafka_messages_consumed_total = Counter(
            'kafka_messages_consumed_total',
            'Total Kafka messages consumed',
            ['topic'],
            registry=self.registry
        )

        self.kafka_consumer_lag = Gauge(
            'kafka_consumer_lag',
            'Kafka consumer lag',
            ['topic', 'partition'],
            registry=self.registry
        )

        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_name'],
            registry=self.registry
        )

        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_name'],
            registry=self.registry
        )

        logger.info("Prometheus trading metrics initialized successfully")

    # ========================================================================
    # Helper Methods for Recording Metrics
    # ========================================================================

    def record_trade(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None
    ):
        """
        Record a completed trade.

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', etc.
            quantity: Number of shares
            price: Execution price
            pnl: Profit/loss (if available)
        """
        self.trades_total.labels(
            symbol=symbol,
            side=side,
            order_type=order_type
        ).inc()

        self.trade_volume.labels(symbol=symbol).inc(quantity)

        if pnl is not None:
            if pnl > 0:
                self.winning_trades_total.labels(symbol=symbol).inc()
            elif pnl < 0:
                self.losing_trades_total.labels(symbol=symbol).inc()

    def record_order(
        self,
        symbol: str,
        order_type: str,
        status: str,
        latency: Optional[float] = None,
        rejection_reason: Optional[str] = None
    ):
        """
        Record order submission and outcome.

        Args:
            symbol: Stock symbol
            order_type: Order type
            status: 'submitted', 'filled', 'rejected', 'cancelled'
            latency: Order latency in seconds
            rejection_reason: Reason for rejection (if applicable)
        """
        if status == 'submitted':
            self.orders_submitted_total.labels(
                symbol=symbol,
                order_type=order_type
            ).inc()

        elif status == 'filled':
            self.orders_filled_total.labels(
                symbol=symbol,
                order_type=order_type
            ).inc()

            if latency is not None:
                self.order_latency.labels(order_type=order_type).observe(latency)

        elif status == 'rejected':
            self.orders_rejected_total.labels(
                symbol=symbol,
                reason=rejection_reason or 'unknown'
            ).inc()

        elif status == 'cancelled':
            self.orders_cancelled_total.labels(symbol=symbol).inc()

    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        strategy: str,
        confidence: float
    ):
        """
        Record a trading signal.

        Args:
            symbol: Stock symbol
            signal_type: 'buy', 'sell', 'hold'
            strategy: Strategy name
            confidence: Signal confidence (0-1)
        """
        self.signals_generated_total.labels(
            symbol=symbol,
            signal_type=signal_type,
            strategy=strategy
        ).inc()

        self.signal_confidence.labels(strategy=strategy).observe(confidence)

    def update_portfolio_metrics(
        self,
        value: float,
        pnl: float,
        buying_power: float,
        return_pct: float
    ):
        """
        Update portfolio-level metrics.

        Args:
            value: Portfolio value
            pnl: Profit/loss
            buying_power: Available buying power
            return_pct: Return percentage
        """
        self.portfolio_value.set(value)
        self.portfolio_pnl.set(pnl)
        self.account_buying_power.set(buying_power)
        self.portfolio_return.set(return_pct)

    def update_risk_metrics(
        self,
        drawdown: float,
        max_dd: float,
        var_95: float,
        cvar_95: float,
        sharpe: float,
        sortino: float,
        volatility: float
    ):
        """
        Update risk management metrics.

        Args:
            drawdown: Current drawdown
            max_dd: Maximum drawdown
            var_95: VaR at 95%
            cvar_95: CVaR at 95%
            sharpe: Sharpe ratio
            sortino: Sortino ratio
            volatility: Portfolio volatility
        """
        self.portfolio_drawdown.set(drawdown)
        self.max_drawdown.set(max_dd)
        self.portfolio_var_95.set(var_95)
        self.portfolio_cvar_95.set(cvar_95)
        self.sharpe_ratio.set(sharpe)
        self.sortino_ratio.set(sortino)
        self.portfolio_volatility.set(volatility)

    def get_metrics(self) -> bytes:
        """
        Get all metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)


# Global metrics instance
trading_metrics = TradingMetrics()
