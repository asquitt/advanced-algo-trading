"""
CVaR-Based Risk Management

Conditional Value at Risk (Expected Shortfall) for tail risk management.

Prevents catastrophic losses through:
1. CVaR calculation and monitoring
2. Tail risk-aware position sizing
3. Dynamic exposure limits based on tail risk
4. Real-time risk budget allocation
5. Portfolio-level risk aggregation

Author: LLM Trading Platform - Institutional Grade
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


@dataclass
class CVaRMetrics:
    """CVaR and tail risk metrics."""
    # VaR metrics
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk

    # CVaR (Expected Shortfall)
    cvar_95: float  # Expected loss if in worst 5%
    cvar_99: float  # Expected loss if in worst 1%

    # Tail risk
    tail_index: float  # Heavy-tailedness measure
    skewness: float  # Return distribution skew
    kurtosis: float  # Fat tails measure (>3 = fat tails)

    # Historical stress
    worst_day: float  # Worst single day loss
    worst_week: float  # Worst week loss
    worst_month: float  # Worst month loss

    # Current exposure
    current_exposure: float  # Current portfolio value at risk
    risk_budget_used: float  # % of CVaR budget used


@dataclass
class RiskLimit:
    """Risk limits for position sizing."""
    max_position_cvar: float  # Max CVaR per position
    max_portfolio_cvar: float  # Max portfolio CVaR
    max_tail_risk_score: float  # Max acceptable tail risk
    max_leverage: float  # Max leverage allowed
    stress_test_required: bool  # Require stress test approval


class CVaRCalculator:
    """Calculate Conditional Value at Risk."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize CVaR calculator.

        Args:
            confidence_level: Confidence level (0.95 = 95%)
        """
        self.confidence_level = confidence_level

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate VaR and CVaR.

        Args:
            returns: Array of returns
            confidence_level: Override default confidence level

        Returns:
            Tuple of (VaR, CVaR)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        # Calculate VaR (percentile)
        var = np.percentile(returns, (1 - confidence_level) * 100)

        # Calculate CVaR (mean of returns below VaR)
        tail_returns = returns[returns <= var]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var

        return var, cvar

    def calculate_parametric_cvar(
        self,
        mean: float,
        std: float,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate parametric CVaR (assumes normal distribution).

        Args:
            mean: Expected return
            std: Standard deviation
            confidence_level: Confidence level

        Returns:
            Tuple of (VaR, CVaR)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)

        # VaR
        var = mean + (z_score * std)

        # CVaR (for normal distribution)
        # CVaR = μ - σ * φ(z) / (1-α)
        # where φ is PDF of standard normal
        pdf_value = stats.norm.pdf(z_score)
        cvar = mean - std * pdf_value / (1 - confidence_level)

        return var, cvar

    def calculate_modified_cvar(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate Cornish-Fisher modified CVaR (accounts for skew and kurtosis).

        More accurate for non-normal distributions.

        Args:
            returns: Array of returns
            confidence_level: Confidence level

        Returns:
            Tuple of (modified VaR, modified CVaR)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        mean = np.mean(returns)
        std = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis

        # Z-score
        z = stats.norm.ppf(1 - confidence_level)

        # Cornish-Fisher expansion
        z_cf = (z +
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        # Modified VaR
        var_modified = mean + z_cf * std

        # Modified CVaR (approximation)
        # For highly non-normal, use historical CVaR
        var_hist, cvar_hist = self.calculate_cvar(returns, confidence_level)

        # Adjust CVaR based on tail heaviness
        adjustment_factor = 1 + (kurt / 10)  # Heavier tails = larger CVaR
        cvar_modified = cvar_hist * adjustment_factor

        return var_modified, cvar_modified


class TailRiskAnalyzer:
    """Analyze tail risk characteristics."""

    def analyze_tail_risk(
        self,
        returns: np.ndarray,
        lookback_days: int = 252
    ) -> CVaRMetrics:
        """
        Comprehensive tail risk analysis.

        Args:
            returns: Array of returns
            lookback_days: Lookback period

        Returns:
            CVaRMetrics with complete tail risk profile
        """
        recent_returns = returns[-lookback_days:] if len(returns) > lookback_days else returns

        calc = CVaRCalculator()

        # Calculate VaR and CVaR at different confidence levels
        var_95, cvar_95 = calc.calculate_modified_cvar(recent_returns, 0.95)
        var_99, cvar_99 = calc.calculate_modified_cvar(recent_returns, 0.99)

        # Tail index (measure of heavy-tailedness)
        # Use Hill estimator
        tail_index = self._calculate_tail_index(recent_returns)

        # Distribution moments
        skewness = float(stats.skew(recent_returns))
        kurtosis = float(stats.kurtosis(recent_returns))  # Excess kurtosis

        # Historical worst periods
        worst_day = np.min(recent_returns)

        # Weekly (5-day) rolling worst
        if len(recent_returns) >= 5:
            weekly_returns = np.convolve(recent_returns, np.ones(5), mode='valid')
            worst_week = np.min(weekly_returns)
        else:
            worst_week = worst_day

        # Monthly (21-day) rolling worst
        if len(recent_returns) >= 21:
            monthly_returns = np.convolve(recent_returns, np.ones(21), mode='valid')
            worst_month = np.min(monthly_returns)
        else:
            worst_month = worst_week

        # Current exposure (would be populated with real portfolio data)
        current_exposure = 0.0
        risk_budget_used = 0.0

        metrics = CVaRMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            tail_index=tail_index,
            skewness=skewness,
            kurtosis=kurtosis,
            worst_day=worst_day,
            worst_week=worst_week,
            worst_month=worst_month,
            current_exposure=current_exposure,
            risk_budget_used=risk_budget_used
        )

        logger.info(
            f"Tail Risk Analysis:\n"
            f"  CVaR 95%: {cvar_95:.2%}\n"
            f"  CVaR 99%: {cvar_99:.2%}\n"
            f"  Tail Index: {tail_index:.2f}\n"
            f"  Skewness: {skewness:.2f}\n"
            f"  Kurtosis: {kurtosis:.2f} {'(FAT TAILS!)' if kurtosis > 3 else ''}\n"
            f"  Worst Day: {worst_day:.2%}"
        )

        return metrics

    def _calculate_tail_index(self, returns: np.ndarray, threshold_percentile: int = 10) -> float:
        """
        Calculate tail index using Hill estimator.

        Higher values = heavier tails.

        Args:
            returns: Return series
            threshold_percentile: Percentile for tail threshold

        Returns:
            Tail index (>0, higher = heavier tails)
        """
        # Focus on negative tail (losses)
        losses = -returns[returns < 0]

        if len(losses) < 10:
            return 1.0  # Not enough data

        # Sort losses
        sorted_losses = np.sort(losses)

        # Use top 10% of losses
        k = max(int(len(sorted_losses) * 0.1), 5)
        tail_losses = sorted_losses[-k:]

        # Hill estimator
        threshold = sorted_losses[-k]
        if threshold == 0:
            return 1.0

        log_excesses = np.log(tail_losses / threshold)
        tail_index = 1.0 / np.mean(log_excesses) if np.mean(log_excesses) > 0 else 1.0

        return min(tail_index, 5.0)  # Cap at 5 for stability


class CVaRPositionSizer:
    """Position sizing based on CVaR limits."""

    def __init__(
        self,
        max_position_cvar: float = 0.02,  # 2% max CVaR per position
        max_portfolio_cvar: float = 0.05,  # 5% max portfolio CVaR
        target_cvar: float = 0.03,  # 3% target CVaR
        confidence_level: float = 0.95
    ):
        """
        Initialize CVaR position sizer.

        Args:
            max_position_cvar: Max CVaR per position (as fraction of portfolio)
            max_portfolio_cvar: Max portfolio-wide CVaR
            target_cvar: Target CVaR to aim for
            confidence_level: Confidence level for CVaR
        """
        self.max_position_cvar = max_position_cvar
        self.max_portfolio_cvar = max_portfolio_cvar
        self.target_cvar = target_cvar
        self.confidence_level = confidence_level

        self.calculator = CVaRCalculator(confidence_level)
        self.tail_analyzer = TailRiskAnalyzer()

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_returns: np.ndarray,
        signal_confidence: float,
        current_portfolio_cvar: float = 0.0,
        max_position_value: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate position size based on CVaR.

        Args:
            portfolio_value: Total portfolio value
            asset_returns: Historical returns for asset
            signal_confidence: Signal strength (0-1)
            current_portfolio_cvar: Current portfolio CVaR
            max_position_value: Optional hard cap on position value

        Returns:
            Tuple of (position_value, details_dict)
        """
        # Analyze tail risk of asset
        tail_metrics = self.tail_analyzer.analyze_tail_risk(asset_returns)

        # Calculate asset CVaR
        _, asset_cvar = self.calculator.calculate_modified_cvar(asset_returns)

        # Determine position CVaR budget
        # Start with target, adjust based on current usage
        remaining_budget = self.max_portfolio_cvar - current_portfolio_cvar
        position_cvar_budget = min(
            self.max_position_cvar,
            remaining_budget * 0.5  # Use max 50% of remaining budget per position
        )

        if position_cvar_budget <= 0:
            logger.warning("No CVaR budget remaining, rejecting position")
            return 0.0, {"reason": "CVaR budget exhausted"}

        # Calculate position size based on CVaR matching
        # Position CVaR = Position Size × Asset CVaR
        # Position Size = Position CVaR Budget / Asset CVaR
        if abs(asset_cvar) > 1e-6:
            base_position_value = portfolio_value * position_cvar_budget / abs(asset_cvar)
        else:
            base_position_value = 0.0

        # Adjust for signal confidence
        adjusted_position_value = base_position_value * signal_confidence

        # Apply tail risk adjustment (reduce size for heavy tails)
        if tail_metrics.kurtosis > 3:  # Fat tails
            tail_adjustment = 1.0 / (1.0 + tail_metrics.kurtosis / 10.0)
            adjusted_position_value *= tail_adjustment
        else:
            tail_adjustment = 1.0

        # Apply skewness adjustment (reduce size for negative skew)
        if tail_metrics.skewness < -0.5:  # Significant negative skew
            skew_adjustment = 1.0 + (tail_metrics.skewness / 2.0)  # Reduces size
            adjusted_position_value *= max(skew_adjustment, 0.5)
        else:
            skew_adjustment = 1.0

        # Apply hard cap if provided
        if max_position_value is not None:
            adjusted_position_value = min(adjusted_position_value, max_position_value)

        # Ensure non-negative
        final_position_value = max(0.0, adjusted_position_value)

        # Calculate resulting position CVaR
        if portfolio_value > 0:
            position_cvar = (final_position_value / portfolio_value) * abs(asset_cvar)
        else:
            position_cvar = 0.0

        details = {
            "base_position_value": base_position_value,
            "adjusted_position_value": final_position_value,
            "position_cvar": position_cvar,
            "position_cvar_budget": position_cvar_budget,
            "remaining_portfolio_cvar": remaining_budget - position_cvar,
            "asset_cvar": asset_cvar,
            "tail_adjustment": tail_adjustment,
            "skew_adjustment": skew_adjustment,
            "signal_confidence": signal_confidence,
            "tail_metrics": tail_metrics
        }

        logger.info(
            f"CVaR Position Sizing:\n"
            f"  Position Value: ${final_position_value:,.0f} ({final_position_value/portfolio_value*100:.1f}% of portfolio)\n"
            f"  Position CVaR: {position_cvar:.2%}\n"
            f"  Asset CVaR 95%: {asset_cvar:.2%}\n"
            f"  Tail Adjustment: {tail_adjustment:.2f}x\n"
            f"  Remaining Portfolio CVaR Budget: {(remaining_budget - position_cvar):.2%}"
        )

        return final_position_value, details

    def check_risk_limits(
        self,
        position_value: float,
        portfolio_value: float,
        position_cvar: float,
        current_portfolio_cvar: float,
        tail_metrics: CVaRMetrics
    ) -> Tuple[bool, str]:
        """
        Check if position meets risk limits.

        Args:
            position_value: Proposed position size
            portfolio_value: Total portfolio value
            position_cvar: Position CVaR
            current_portfolio_cvar: Current portfolio CVaR
            tail_metrics: Tail risk metrics

        Returns:
            Tuple of (approved, reason)
        """
        # Check position CVaR limit
        if position_cvar > self.max_position_cvar:
            return False, f"Position CVaR {position_cvar:.2%} exceeds limit {self.max_position_cvar:.2%}"

        # Check portfolio CVaR limit
        new_portfolio_cvar = current_portfolio_cvar + position_cvar
        if new_portfolio_cvar > self.max_portfolio_cvar:
            return False, f"Portfolio CVaR {new_portfolio_cvar:.2%} exceeds limit {self.max_portfolio_cvar:.2%}"

        # Check concentration limit
        concentration = position_value / portfolio_value if portfolio_value > 0 else 0
        if concentration > 0.25:  # Max 25% in single position
            return False, f"Position concentration {concentration:.1%} too high"

        # Check tail risk limits
        if tail_metrics.kurtosis > 10:  # Extremely fat tails
            return False, f"Excessive kurtosis {tail_metrics.kurtosis:.1f} (>10)"

        if tail_metrics.cvar_99 < -0.20:  # Potential for >20% loss in 1% tail
            return False, f"Excessive tail risk: CVaR 99% = {tail_metrics.cvar_99:.1%}"

        # Stress test requirement
        if tail_metrics.worst_day < -0.10:  # Worst day >10% loss
            logger.warning(f"Large historical loss detected: {tail_metrics.worst_day:.1%}. Stress test recommended.")

        return True, "Approved"


class PortfolioCVaRManager:
    """Manage portfolio-level CVaR."""

    def __init__(
        self,
        max_portfolio_cvar: float = 0.05,
        cvar_confidence: float = 0.95
    ):
        """
        Initialize portfolio CVaR manager.

        Args:
            max_portfolio_cvar: Maximum portfolio CVaR
            cvar_confidence: Confidence level
        """
        self.max_portfolio_cvar = max_portfolio_cvar
        self.cvar_confidence = cvar_confidence
        self.calculator = CVaRCalculator(cvar_confidence)

    def calculate_portfolio_cvar(
        self,
        positions: Dict[str, float],  # {symbol: position_value}
        returns_dict: Dict[str, np.ndarray],  # {symbol: returns}
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate portfolio-level CVaR accounting for correlations.

        Args:
            positions: Dict of position values by symbol
            returns_dict: Dict of return series by symbol
            correlation_matrix: Optional correlation matrix

        Returns:
            Tuple of (portfolio_cvar, details_dict)
        """
        if not positions:
            return 0.0, {}

        symbols = list(positions.keys())
        weights = np.array([positions[s] for s in symbols])
        total_value = np.sum(weights)

        if total_value == 0:
            return 0.0, {}

        weights = weights / total_value  # Normalize to weights

        # Get return series (align dates)
        returns_arrays = [returns_dict[s] for s in symbols]
        min_length = min(len(r) for r in returns_arrays)
        returns_matrix = np.array([r[-min_length:] for r in returns_arrays]).T

        # Calculate portfolio returns
        portfolio_returns = returns_matrix @ weights

        # Calculate portfolio CVaR
        _, portfolio_cvar = self.calculator.calculate_modified_cvar(portfolio_returns)

        # Calculate individual CVaRs
        individual_cvars = {}
        for i, symbol in enumerate(symbols):
            _, cvar = self.calculator.calculate_modified_cvar(returns_matrix[:, i])
            individual_cvars[symbol] = cvar * weights[i]

        # Diversification benefit
        sum_individual_cvars = sum(individual_cvars.values())
        diversification_benefit = sum_individual_cvars - abs(portfolio_cvar)

        details = {
            "portfolio_cvar": portfolio_cvar,
            "individual_cvars": individual_cvars,
            "sum_individual_cvars": sum_individual_cvars,
            "diversification_benefit": diversification_benefit,
            "cvar_utilization": abs(portfolio_cvar) / self.max_portfolio_cvar
        }

        logger.info(
            f"Portfolio CVaR: {portfolio_cvar:.2%}\n"
            f"  Diversification Benefit: {diversification_benefit:.2%}\n"
            f"  CVaR Utilization: {details['cvar_utilization']*100:.1f}%"
        )

        return portfolio_cvar, details

    def allocate_risk_budget(
        self,
        num_positions: int,
        current_cvar: float,
        priority_weights: Optional[List[float]] = None
    ) -> List[float]:
        """
        Allocate remaining CVaR budget across positions.

        Args:
            num_positions: Number of positions to allocate to
            current_cvar: Current portfolio CVaR
            priority_weights: Optional priority weights for allocation

        Returns:
            List of CVaR budgets per position
        """
        remaining_budget = self.max_portfolio_cvar - abs(current_cvar)

        if remaining_budget <= 0:
            return [0.0] * num_positions

        if priority_weights is None:
            # Equal allocation
            budgets = [remaining_budget / num_positions] * num_positions
        else:
            # Weighted allocation
            total_weight = sum(priority_weights)
            budgets = [(w / total_weight) * remaining_budget for w in priority_weights]

        return budgets


# Singleton instances
cvar_position_sizer = CVaRPositionSizer(
    max_position_cvar=0.02,
    max_portfolio_cvar=0.05,
    target_cvar=0.03
)

portfolio_cvar_manager = PortfolioCVaRManager(
    max_portfolio_cvar=0.05
)
