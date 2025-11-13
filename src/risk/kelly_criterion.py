"""
Kelly Criterion Position Sizing

Implements optimal position sizing using the Kelly Criterion formula,
which maximizes long-term compound growth rate.

Formula: f* = (p * b - q) / b
Where:
- f* = optimal fraction of capital to risk
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio (average win / average loss)

Features:
- Full Kelly and fractional Kelly (for safety)
- Parameter uncertainty adjustment
- Integration with CVaR limits
- Historical win rate estimation
- Dynamic adjustment based on confidence

References:
- Kelly, J. L. (1956). "A New Interpretation of Information Rate"
- Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from loguru import logger as app_logger

from src.data_layer.models import Trade, OrderSide


@dataclass
class KellyParameters:
    """Parameters for Kelly Criterion calculation."""

    win_rate: float  # Probability of winning (0-1)
    avg_win: float  # Average win amount
    avg_loss: float  # Average loss amount
    win_loss_ratio: float  # avg_win / avg_loss
    kelly_fraction: float  # Optimal Kelly fraction (0-1)
    fractional_kelly: float  # Conservative fraction (typically 0.25-0.5)
    confidence: float  # Confidence in parameters (0-1)
    sample_size: int  # Number of trades used for calculation


@dataclass
class KellyPositionSize:
    """Result of Kelly-based position sizing."""

    recommended_fraction: float  # Fraction of portfolio to allocate
    recommended_value: float  # Dollar amount to allocate
    full_kelly: float  # Full Kelly fraction (for reference)
    fractional_kelly: float  # Conservative Kelly fraction (recommended)
    reasoning: str  # Explanation of sizing decision
    confidence: float  # Confidence in recommendation
    risk_adjusted: bool  # Whether CVaR limits were applied


class KellyCriterionCalculator:
    """
    Calculate optimal position sizes using Kelly Criterion.

    The Kelly Criterion maximizes long-term compound growth rate by
    finding the optimal fraction of capital to risk on each trade.
    """

    def __init__(
        self,
        fraction: float = 0.25,
        min_sample_size: int = 30,
        max_fraction: float = 0.25,
    ):
        """
        Initialize Kelly Criterion calculator.

        Args:
            fraction: Fractional Kelly to use (default 0.25 = quarter Kelly)
            min_sample_size: Minimum trades needed for reliable calculation
            max_fraction: Maximum position size as fraction of portfolio (safety limit)
        """
        self.fraction = fraction
        self.min_sample_size = min_sample_size
        self.max_fraction = max_fraction

    def calculate_from_backtest(
        self,
        trades: List[Trade],
        current_confidence: float = 0.8,
    ) -> Optional[KellyParameters]:
        """
        Calculate Kelly parameters from historical trade data.

        Args:
            trades: List of historical trades
            current_confidence: Confidence in current strategy (0-1)

        Returns:
            KellyParameters if sufficient data, None otherwise
        """
        if len(trades) < self.min_sample_size:
            app_logger.warning(
                f"Insufficient trade history ({len(trades)} < {self.min_sample_size})"
            )
            return None

        # Separate wins and losses
        wins = []
        losses = []

        for trade in trades:
            if trade.exit_price is not None:
                pnl = trade.calculate_pnl(trade.exit_price)
                if pnl > 0:
                    wins.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))

        if not wins or not losses:
            app_logger.warning("Need both wins and losses to calculate Kelly")
            return None

        # Calculate parameters
        win_rate = len(wins) / (len(wins) + len(losses))
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Kelly formula: f* = (p * b - q) / b
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio

        kelly_fraction = (p * b - q) / b if b > 0 else 0

        # Clamp to reasonable range
        kelly_fraction = max(0, min(kelly_fraction, 1.0))

        # Apply fractional Kelly for safety
        fractional_kelly = kelly_fraction * self.fraction

        # Adjust for parameter uncertainty (fewer trades = less confidence)
        sample_confidence = min(len(trades) / 100.0, 1.0)  # Cap at 100 trades
        combined_confidence = current_confidence * sample_confidence

        # Further reduce if win rate is close to 50/50 (high uncertainty)
        if 0.45 < win_rate < 0.55:
            combined_confidence *= 0.8

        app_logger.info(
            f"Kelly Criterion: win_rate={win_rate:.1%}, "
            f"win/loss_ratio={win_loss_ratio:.2f}, "
            f"full_kelly={kelly_fraction:.1%}, "
            f"fractional={fractional_kelly:.1%}"
        )

        return KellyParameters(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            kelly_fraction=kelly_fraction,
            fractional_kelly=fractional_kelly,
            confidence=combined_confidence,
            sample_size=len(trades),
        )

    def calculate_position_size(
        self,
        kelly_params: KellyParameters,
        portfolio_value: float,
        signal_confidence: float,
        cvar_limit: Optional[float] = None,
    ) -> KellyPositionSize:
        """
        Calculate recommended position size using Kelly Criterion.

        Args:
            kelly_params: Kelly parameters from historical data
            portfolio_value: Current portfolio value
            signal_confidence: Confidence in current signal (0-1)
            cvar_limit: Optional CVaR limit as fraction of portfolio

        Returns:
            KellyPositionSize with recommendation
        """
        # Start with fractional Kelly
        recommended_fraction = kelly_params.fractional_kelly

        # Adjust for signal confidence
        # High confidence signals get closer to full fractional Kelly
        # Low confidence signals get reduced
        confidence_adjustment = 0.5 + (signal_confidence * 0.5)  # Range: 0.5-1.0
        recommended_fraction *= confidence_adjustment

        # Adjust for parameter confidence
        recommended_fraction *= kelly_params.confidence

        # Apply maximum position limit
        recommended_fraction = min(recommended_fraction, self.max_fraction)

        # Apply CVaR limit if provided
        risk_adjusted = False
        if cvar_limit is not None:
            if recommended_fraction > cvar_limit:
                app_logger.info(
                    f"Kelly position ({recommended_fraction:.1%}) "
                    f"reduced to CVaR limit ({cvar_limit:.1%})"
                )
                recommended_fraction = cvar_limit
                risk_adjusted = True

        # Calculate dollar amount
        recommended_value = portfolio_value * recommended_fraction

        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(
            f"Full Kelly: {kelly_params.kelly_fraction:.1%} "
            f"(win rate={kelly_params.win_rate:.1%}, "
            f"win/loss={kelly_params.win_loss_ratio:.2f})"
        )
        reasoning_parts.append(
            f"Fractional Kelly ({self.fraction}x): {kelly_params.fractional_kelly:.1%}"
        )
        reasoning_parts.append(
            f"Signal confidence adjustment ({signal_confidence:.1%}): "
            f"{kelly_params.fractional_kelly * confidence_adjustment:.1%}"
        )
        reasoning_parts.append(
            f"Parameter confidence adjustment ({kelly_params.confidence:.1%}): "
            f"{recommended_fraction:.1%}"
        )

        if risk_adjusted:
            reasoning_parts.append(f"CVaR limit applied: {cvar_limit:.1%}")

        reasoning = "; ".join(reasoning_parts)

        return KellyPositionSize(
            recommended_fraction=recommended_fraction,
            recommended_value=recommended_value,
            full_kelly=kelly_params.kelly_fraction,
            fractional_kelly=kelly_params.fractional_kelly,
            reasoning=reasoning,
            confidence=kelly_params.confidence * signal_confidence,
            risk_adjusted=risk_adjusted,
        )

    def estimate_from_metrics(
        self,
        sharpe_ratio: float,
        avg_return: float,
        return_std: float,
        current_confidence: float = 0.8,
    ) -> KellyParameters:
        """
        Estimate Kelly parameters from summary statistics (when no trade history).

        Uses the approximation: f* ≈ (Sharpe Ratio) * (avg return / variance)

        Args:
            sharpe_ratio: Sharpe ratio of strategy
            avg_return: Average return per trade
            return_std: Standard deviation of returns
            current_confidence: Confidence in strategy

        Returns:
            Estimated KellyParameters
        """
        if return_std == 0:
            app_logger.warning("Zero volatility, cannot calculate Kelly")
            return KellyParameters(
                win_rate=0.5,
                avg_win=0,
                avg_loss=0,
                win_loss_ratio=1.0,
                kelly_fraction=0,
                fractional_kelly=0,
                confidence=0,
                sample_size=0,
            )

        # Kelly approximation for continuous outcomes
        # f* ≈ μ / σ² where μ = mean, σ² = variance
        variance = return_std**2
        kelly_fraction = avg_return / variance if variance > 0 else 0

        # Clamp to reasonable range
        kelly_fraction = max(0, min(kelly_fraction, 1.0))

        # Apply fractional Kelly
        fractional_kelly = kelly_fraction * self.fraction

        # Estimate win rate from Sharpe ratio
        # Higher Sharpe ≈ higher win rate (very rough approximation)
        estimated_win_rate = 0.5 + min(sharpe_ratio / 10.0, 0.3)
        estimated_win_rate = max(0.1, min(estimated_win_rate, 0.9))

        app_logger.info(
            f"Estimated Kelly: full={kelly_fraction:.1%}, "
            f"fractional={fractional_kelly:.1%} "
            f"(from Sharpe={sharpe_ratio:.2f})"
        )

        return KellyParameters(
            win_rate=estimated_win_rate,
            avg_win=avg_return * estimated_win_rate,
            avg_loss=avg_return * (1 - estimated_win_rate),
            win_loss_ratio=1.0,  # Unknown
            kelly_fraction=kelly_fraction,
            fractional_kelly=fractional_kelly,
            confidence=current_confidence * 0.7,  # Reduce confidence for estimate
            sample_size=0,  # Not from actual trades
        )


# Global singleton instance
kelly_calculator = KellyCriterionCalculator(
    fraction=0.25,  # Quarter Kelly (conservative)
    min_sample_size=30,
    max_fraction=0.25,  # Max 25% of portfolio per position
)
