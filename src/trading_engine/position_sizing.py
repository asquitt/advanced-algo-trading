"""
Adaptive Position Sizing with Drawdown Management

Dynamically adjusts position sizes based on:
1. Current drawdown level
2. Win rate and profitability
3. Market volatility
4. Portfolio heat
5. Risk-adjusted returns (Sharpe, Sortino)

Author: LLM Trading Platform
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict
import numpy as np
from loguru import logger

from src.data_layer.models import Trade, OrderSide


class RiskMode(Enum):
    """Risk management mode."""
    AGGRESSIVE = "aggressive"  # Full position sizing
    NORMAL = "normal"  # Standard position sizing
    CONSERVATIVE = "conservative"  # Reduced positions
    DEFENSIVE = "defensive"  # Minimal positions
    HALT = "halt"  # Stop trading temporarily


@dataclass
class DrawdownMetrics:
    """Drawdown analysis metrics."""
    current_drawdown_pct: float  # Current drawdown from peak
    max_drawdown_pct: float  # Maximum historical drawdown
    drawdown_duration_days: int  # Days in current drawdown
    peak_equity: float  # Equity at peak
    current_equity: float  # Current equity
    recovery_ratio: float  # How much recovered from max DD
    is_in_drawdown: bool  # Currently experiencing drawdown


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # Percentage
    avg_win: float
    avg_loss: float
    profit_factor: float  # Total wins / total losses
    sharpe_ratio: float  # Risk-adjusted returns
    sortino_ratio: float  # Downside risk-adjusted returns
    expectancy: float  # Expected value per trade
    consecutive_losses: int  # Current losing streak
    max_consecutive_losses: int  # Worst losing streak


@dataclass
class PositionSizingRecommendation:
    """Position sizing recommendation."""
    base_size_pct: float  # Base position as % of portfolio
    adjusted_size_pct: float  # After risk adjustments
    max_position_value: float  # Maximum dollar value
    risk_mode: RiskMode
    position_multiplier: float  # Adjustment factor
    reasoning: str  # Explanation of sizing decision
    confidence: float  # 0-1, confidence in recommendation


class DrawdownAnalyzer:
    """Analyzes drawdown and portfolio performance."""

    def __init__(self):
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.peak_equity = 0.0
        self.peak_date = datetime.now()

    def update_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """Update equity curve and track peak."""
        if timestamp is None:
            timestamp = datetime.now()

        self.equity_curve.append((timestamp, equity))

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_date = timestamp

        # Keep last 500 points
        if len(self.equity_curve) > 500:
            self.equity_curve = self.equity_curve[-500:]

    def calculate_drawdown_metrics(self) -> DrawdownMetrics:
        """Calculate current drawdown metrics."""
        if not self.equity_curve:
            return DrawdownMetrics(
                current_drawdown_pct=0.0,
                max_drawdown_pct=0.0,
                drawdown_duration_days=0,
                peak_equity=0.0,
                current_equity=0.0,
                recovery_ratio=1.0,
                is_in_drawdown=False
            )

        current_equity = self.equity_curve[-1][1]
        current_date = self.equity_curve[-1][0]

        # Current drawdown
        if self.peak_equity > 0:
            current_dd_pct = ((current_equity - self.peak_equity) / self.peak_equity) * 100
        else:
            current_dd_pct = 0.0

        # Calculate maximum drawdown
        max_dd_pct = 0.0
        peak = 0.0

        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = ((equity - peak) / peak * 100) if peak > 0 else 0.0
            if dd < max_dd_pct:
                max_dd_pct = dd

        # Drawdown duration
        if current_dd_pct < -1.0:  # More than 1% drawdown
            is_in_drawdown = True
            drawdown_duration = (current_date - self.peak_date).days
        else:
            is_in_drawdown = False
            drawdown_duration = 0

        # Recovery ratio
        if max_dd_pct < -1.0:
            recovery_ratio = current_dd_pct / max_dd_pct  # 0 to 1
        else:
            recovery_ratio = 1.0

        return DrawdownMetrics(
            current_drawdown_pct=current_dd_pct,
            max_drawdown_pct=max_dd_pct,
            drawdown_duration_days=drawdown_duration,
            peak_equity=self.peak_equity,
            current_equity=current_equity,
            recovery_ratio=recovery_ratio,
            is_in_drawdown=is_in_drawdown
        )


class PerformanceAnalyzer:
    """Analyzes trading performance."""

    def calculate_metrics(self, trades: List[Trade]) -> PerformanceMetrics:
        """
        Calculate performance metrics from trade history.

        Args:
            trades: List of completed trades

        Returns:
            PerformanceMetrics with analysis
        """
        if not trades:
            return self._default_metrics()

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Average win/loss
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Profit factor
        total_wins = sum(wins) if wins else 0.0
        total_losses = sum(losses) if losses else 0.0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

        # Expectancy
        win_rate_decimal = win_rate / 100.0
        expectancy = (win_rate_decimal * avg_win) - ((1 - win_rate_decimal) * avg_loss)

        # Calculate Sharpe and Sortino ratios
        returns = [t.pnl for t in trades]
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)

        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        current_streak = 0

        for trade in reversed(trades):
            if trade.pnl < 0:
                current_streak += 1
                consecutive_losses = current_streak
            else:
                if current_streak > max_consecutive_losses:
                    max_consecutive_losses = current_streak
                current_streak = 0

        if current_streak > max_consecutive_losses:
            max_consecutive_losses = current_streak

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            expectancy=expectancy,
            consecutive_losses=consecutive_losses,
            max_consecutive_losses=max_consecutive_losses
        )

    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize (assuming daily returns)
        sharpe = (avg_return - risk_free_rate / 252) / std_return * np.sqrt(252)

        return float(sharpe)

    def _calculate_sortino(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside risk)."""
        if not returns or len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)

        # Calculate downside deviation (only negative returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return float('inf')  # No downside risk

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        # Annualize
        sortino = (avg_return - risk_free_rate / 252) / downside_std * np.sqrt(252)

        return float(sortino)

    def _default_metrics(self) -> PerformanceMetrics:
        """Default metrics when no trades."""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            expectancy=0.0,
            consecutive_losses=0,
            max_consecutive_losses=0
        )


class AdaptivePositionSizer:
    """
    Adaptive position sizing with drawdown management.

    Key Features:
    - Reduces size during drawdowns
    - Increases size when performing well
    - Considers win rate and profit factor
    - Accounts for market volatility
    - Manages portfolio heat (total risk)
    """

    def __init__(
        self,
        base_risk_per_trade: float = 0.02,  # 2% of portfolio per trade
        max_portfolio_heat: float = 0.10,  # 10% total risk
        max_drawdown_threshold: float = -15.0,  # -15% triggers defensive mode
        min_win_rate: float = 45.0,  # Below 45% triggers caution
        min_profit_factor: float = 1.2  # Below 1.2 triggers caution
    ):
        self.base_risk_per_trade = base_risk_per_trade
        self.max_portfolio_heat = max_portfolio_heat
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor

        self.drawdown_analyzer = DrawdownAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()

        logger.info(
            f"Adaptive position sizer initialized: "
            f"base_risk={base_risk_per_trade*100}%, "
            f"max_heat={max_portfolio_heat*100}%, "
            f"dd_threshold={max_drawdown_threshold}%"
        )

    def calculate_position_size(
        self,
        portfolio_value: float,
        signal_confidence: float,
        market_volatility: float,
        recent_trades: List[Trade],
        open_positions: int,
        current_portfolio_heat: float = 0.0
    ) -> PositionSizingRecommendation:
        """
        Calculate recommended position size with risk adjustments.

        Args:
            portfolio_value: Current portfolio value
            signal_confidence: Signal confidence (0-1)
            market_volatility: Current market volatility
            recent_trades: Recent trade history
            open_positions: Number of open positions
            current_portfolio_heat: Current total risk exposure

        Returns:
            PositionSizingRecommendation
        """
        # Update equity curve
        self.drawdown_analyzer.update_equity(portfolio_value)

        # Calculate metrics
        dd_metrics = self.drawdown_analyzer.calculate_drawdown_metrics()
        perf_metrics = self.performance_analyzer.calculate_metrics(recent_trades)

        # Determine risk mode
        risk_mode = self._determine_risk_mode(dd_metrics, perf_metrics)

        # Base position size (as percentage of portfolio)
        base_size_pct = self.base_risk_per_trade

        # Apply adjustments
        multiplier = 1.0

        # 1. Drawdown adjustment (most important)
        dd_multiplier = self._calculate_drawdown_multiplier(dd_metrics, risk_mode)
        multiplier *= dd_multiplier

        # 2. Performance adjustment
        perf_multiplier = self._calculate_performance_multiplier(perf_metrics)
        multiplier *= perf_multiplier

        # 3. Signal confidence adjustment
        confidence_multiplier = 0.5 + (signal_confidence * 0.5)  # 0.5 to 1.0
        multiplier *= confidence_multiplier

        # 4. Volatility adjustment
        vol_multiplier = self._calculate_volatility_multiplier(market_volatility)
        multiplier *= vol_multiplier

        # 5. Portfolio heat adjustment
        heat_multiplier = self._calculate_heat_multiplier(
            current_portfolio_heat, open_positions
        )
        multiplier *= heat_multiplier

        # Calculate adjusted size
        adjusted_size_pct = base_size_pct * multiplier

        # Apply limits
        adjusted_size_pct = max(0.001, min(adjusted_size_pct, 0.05))  # 0.1% to 5%

        max_position_value = portfolio_value * adjusted_size_pct

        # Generate reasoning
        reasoning = self._generate_reasoning(
            risk_mode, dd_metrics, perf_metrics,
            dd_multiplier, perf_multiplier, confidence_multiplier,
            vol_multiplier, heat_multiplier
        )

        # Calculate confidence in recommendation
        confidence = self._calculate_recommendation_confidence(
            perf_metrics.total_trades,
            dd_metrics.is_in_drawdown,
            market_volatility
        )

        logger.info(
            f"Position sizing: {adjusted_size_pct*100:.2f}% "
            f"(multiplier: {multiplier:.2f}x, risk_mode: {risk_mode.value})"
        )

        return PositionSizingRecommendation(
            base_size_pct=base_size_pct,
            adjusted_size_pct=adjusted_size_pct,
            max_position_value=max_position_value,
            risk_mode=risk_mode,
            position_multiplier=multiplier,
            reasoning=reasoning,
            confidence=confidence
        )

    def _determine_risk_mode(
        self,
        dd_metrics: DrawdownMetrics,
        perf_metrics: PerformanceMetrics
    ) -> RiskMode:
        """Determine current risk management mode."""
        # Check for halt conditions (severe drawdown or poor performance)
        if dd_metrics.current_drawdown_pct < -25.0:  # -25% drawdown
            return RiskMode.HALT

        if (perf_metrics.total_trades >= 20 and
            perf_metrics.win_rate < 30.0 and
            perf_metrics.consecutive_losses >= 5):
            return RiskMode.HALT

        # Defensive mode
        if dd_metrics.current_drawdown_pct < self.max_drawdown_threshold:
            return RiskMode.DEFENSIVE

        if (perf_metrics.total_trades >= 10 and
            perf_metrics.win_rate < self.min_win_rate):
            return RiskMode.DEFENSIVE

        # Conservative mode
        if dd_metrics.current_drawdown_pct < -5.0:
            return RiskMode.CONSERVATIVE

        if (perf_metrics.total_trades >= 10 and
            perf_metrics.profit_factor < self.min_profit_factor):
            return RiskMode.CONSERVATIVE

        # Aggressive mode (performing well)
        if (perf_metrics.total_trades >= 20 and
            perf_metrics.win_rate > 60.0 and
            perf_metrics.profit_factor > 2.0 and
            dd_metrics.current_drawdown_pct > -2.0):
            return RiskMode.AGGRESSIVE

        # Normal mode
        return RiskMode.NORMAL

    def _calculate_drawdown_multiplier(
        self,
        dd_metrics: DrawdownMetrics,
        risk_mode: RiskMode
    ) -> float:
        """Calculate position size multiplier based on drawdown."""
        if risk_mode == RiskMode.HALT:
            return 0.0

        if not dd_metrics.is_in_drawdown:
            return 1.0

        dd_pct = abs(dd_metrics.current_drawdown_pct)

        if dd_pct < 5.0:
            return 1.0  # Minimal drawdown
        elif dd_pct < 10.0:
            return 0.8  # Small drawdown
        elif dd_pct < 15.0:
            return 0.5  # Moderate drawdown
        elif dd_pct < 20.0:
            return 0.3  # Significant drawdown
        else:
            return 0.1  # Severe drawdown

    def _calculate_performance_multiplier(
        self,
        perf_metrics: PerformanceMetrics
    ) -> float:
        """Calculate multiplier based on recent performance."""
        if perf_metrics.total_trades < 5:
            return 0.7  # Insufficient data, be cautious

        multiplier = 1.0

        # Win rate adjustment
        if perf_metrics.win_rate > 60.0:
            multiplier *= 1.2
        elif perf_metrics.win_rate < 40.0:
            multiplier *= 0.7

        # Profit factor adjustment
        if perf_metrics.profit_factor > 2.0:
            multiplier *= 1.2
        elif perf_metrics.profit_factor < 1.0:
            multiplier *= 0.5

        # Consecutive losses penalty
        if perf_metrics.consecutive_losses >= 3:
            multiplier *= 0.7
        if perf_metrics.consecutive_losses >= 5:
            multiplier *= 0.5

        # Sharpe ratio adjustment
        if perf_metrics.sharpe_ratio > 1.5:
            multiplier *= 1.1
        elif perf_metrics.sharpe_ratio < 0.5:
            multiplier *= 0.8

        return multiplier

    def _calculate_volatility_multiplier(self, market_volatility: float) -> float:
        """Adjust position size based on market volatility."""
        # market_volatility is expected to be annualized (e.g., 0.20 for 20%)

        if market_volatility < 0.15:  # Low volatility
            return 1.1
        elif market_volatility < 0.25:  # Normal volatility
            return 1.0
        elif market_volatility < 0.40:  # High volatility
            return 0.8
        else:  # Extreme volatility
            return 0.5

    def _calculate_heat_multiplier(
        self,
        current_heat: float,
        open_positions: int
    ) -> float:
        """Adjust based on current portfolio heat."""
        # Reduce size if approaching max heat
        heat_ratio = current_heat / self.max_portfolio_heat

        if heat_ratio > 0.9:
            return 0.3  # Near limit
        elif heat_ratio > 0.7:
            return 0.6
        elif heat_ratio > 0.5:
            return 0.8
        else:
            return 1.0

    def _generate_reasoning(
        self,
        risk_mode: RiskMode,
        dd_metrics: DrawdownMetrics,
        perf_metrics: PerformanceMetrics,
        dd_mult: float,
        perf_mult: float,
        conf_mult: float,
        vol_mult: float,
        heat_mult: float
    ) -> str:
        """Generate human-readable reasoning."""
        reasons = []

        reasons.append(f"Risk mode: {risk_mode.value}")

        if dd_metrics.is_in_drawdown:
            reasons.append(
                f"In {abs(dd_metrics.current_drawdown_pct):.1f}% drawdown "
                f"(reduced size {(1-dd_mult)*100:.0f}%)"
            )

        if perf_metrics.total_trades >= 10:
            reasons.append(
                f"Win rate: {perf_metrics.win_rate:.1f}%, "
                f"Profit factor: {perf_metrics.profit_factor:.2f}"
            )

        if perf_metrics.consecutive_losses >= 3:
            reasons.append(
                f"Consecutive losses: {perf_metrics.consecutive_losses} "
                f"(reduced size)"
            )

        if vol_mult < 1.0:
            reasons.append("High market volatility detected")

        if heat_mult < 1.0:
            reasons.append("High portfolio heat - limiting new positions")

        return "; ".join(reasons)

    def _calculate_recommendation_confidence(
        self,
        num_trades: int,
        in_drawdown: bool,
        volatility: float
    ) -> float:
        """Calculate confidence in position sizing recommendation."""
        confidence = 0.7  # Base confidence

        # More trades = more confidence in performance metrics
        if num_trades >= 50:
            confidence += 0.15
        elif num_trades >= 20:
            confidence += 0.10
        elif num_trades < 10:
            confidence -= 0.15

        # Drawdown reduces confidence
        if in_drawdown:
            confidence -= 0.10

        # High volatility reduces confidence
        if volatility > 0.40:
            confidence -= 0.15

        return max(0.3, min(0.95, confidence))


# Singleton instance
adaptive_position_sizer = AdaptivePositionSizer(
    base_risk_per_trade=0.02,
    max_portfolio_heat=0.10,
    max_drawdown_threshold=-15.0,
    min_win_rate=45.0,
    min_profit_factor=1.2
)
