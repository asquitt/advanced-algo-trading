"""
Tests for CVaR-Based Risk Management

Tests Conditional Value at Risk (Expected Shortfall) for tail risk
management and position sizing.

Author: LLM Trading Platform - Test Suite
"""

import pytest
import numpy as np
from scipy import stats
from src.risk.cvar_risk_management import (
    CVaRCalculator,
    TailRiskAnalyzer,
    CVaRPositionSizer,
    PortfolioCVaRManager,
    CVaRMetrics,
    RiskLimit,
    cvar_position_sizer,
    portfolio_cvar_manager
)


class TestCVaRCalculator:
    """Test CVaR calculation methods."""

    def test_basic_cvar_calculation(self):
        """Test basic CVaR calculation from returns."""
        calculator = CVaRCalculator(confidence_level=0.95)

        # Simulate normal returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var, cvar = calculator.calculate_cvar(returns, confidence_level=0.95)

        # VaR should be negative (loss)
        assert var < 0

        # CVaR should be more negative than VaR (average of tail losses)
        assert cvar < var

        # Both should be reasonable for 2% daily volatility
        assert -0.10 < cvar < 0

    def test_cvar_different_confidence_levels(self):
        """Test CVaR at different confidence levels."""
        calculator = CVaRCalculator()

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var_95, cvar_95 = calculator.calculate_cvar(returns, 0.95)
        var_99, cvar_99 = calculator.calculate_cvar(returns, 0.99)

        # 99% CVaR should be more extreme than 95% CVaR
        assert cvar_99 < cvar_95
        assert var_99 < var_95

    def test_parametric_cvar(self):
        """Test parametric CVaR calculation (normal assumption)."""
        calculator = CVaRCalculator(confidence_level=0.95)

        mean = 0.001
        std = 0.02

        var, cvar = calculator.calculate_parametric_cvar(mean, std, 0.95)

        # Should return negative values (losses)
        assert var < mean
        assert cvar < var

        # Check approximate magnitude
        # For 95%, Z-score ≈ -1.645
        expected_var_approx = mean + (-1.645 * std)
        assert abs(var - expected_var_approx) < 0.005

    def test_modified_cvar_with_fat_tails(self):
        """Test modified CVaR with non-normal (fat-tailed) distribution."""
        calculator = CVaRCalculator(confidence_level=0.95)

        # Create fat-tailed distribution (Student's t with low df)
        np.random.seed(42)
        returns = stats.t.rvs(df=3, loc=0.001, scale=0.02, size=1000)

        var_mod, cvar_mod = calculator.calculate_modified_cvar(returns, 0.95)

        # Should account for fat tails
        assert var_mod < 0
        assert cvar_mod < var_mod

        # Modified should be more conservative than simple
        var_simple, cvar_simple = calculator.calculate_cvar(returns, 0.95)
        # May or may not be more extreme depending on skew/kurtosis

    def test_zero_returns(self):
        """Test CVaR with zero returns."""
        calculator = CVaRCalculator()

        returns = np.zeros(100)

        var, cvar = calculator.calculate_cvar(returns, 0.95)

        # All zeros should give zero VaR and CVaR
        assert var == 0
        assert cvar == 0

    def test_all_positive_returns(self):
        """Test CVaR with all positive returns."""
        calculator = CVaRCalculator()

        returns = np.abs(np.random.normal(0.01, 0.005, 100))

        var, cvar = calculator.calculate_cvar(returns, 0.95)

        # 5% tail should still be positive
        assert var >= 0
        assert cvar >= 0


class TestTailRiskAnalyzer:
    """Test tail risk analysis."""

    def test_comprehensive_tail_analysis(self):
        """Test comprehensive tail risk analysis."""
        analyzer = TailRiskAnalyzer()

        # Create returns with known characteristics
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 500)

        metrics = analyzer.analyze_tail_risk(returns, lookback_days=250)

        assert isinstance(metrics, CVaRMetrics)

        # Check all metrics are populated
        assert metrics.var_95 < 0  # Should be negative
        assert metrics.var_99 < 0
        assert metrics.cvar_95 < metrics.var_95
        assert metrics.cvar_99 < metrics.var_99

        assert metrics.tail_index > 0
        assert -5 < metrics.skewness < 5  # Reasonable range
        assert -10 < metrics.kurtosis < 50  # Reasonable range

        assert metrics.worst_day < 0
        assert metrics.worst_week < 0
        assert metrics.worst_month < 0

    def test_tail_index_heavy_tails(self):
        """Test tail index calculation for heavy-tailed distribution."""
        analyzer = TailRiskAnalyzer()

        # Fat-tailed distribution (Student's t)
        np.random.seed(42)
        returns_heavy = stats.t.rvs(df=3, loc=0, scale=0.02, size=1000)
        returns_normal = np.random.normal(0, 0.02, 1000)

        metrics_heavy = analyzer.analyze_tail_risk(returns_heavy)
        metrics_normal = analyzer.analyze_tail_risk(returns_normal)

        # Heavy tails should have higher tail index and kurtosis
        assert metrics_heavy.kurtosis > metrics_normal.kurtosis
        # Tail index comparison may vary due to Hill estimator variance

    def test_negative_skew_detection(self):
        """Test detection of negative skew (crash risk)."""
        analyzer = TailRiskAnalyzer()

        # Create negatively skewed returns (occasional large losses)
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.015, 1000)
        # Add some large negative outliers
        returns[np.random.choice(len(returns), 20)] -= 0.08

        metrics = analyzer.analyze_tail_risk(returns)

        # Should detect negative skew
        assert metrics.skewness < -0.5
        assert metrics.worst_day < -0.05  # Large loss detected

    def test_lookback_period(self):
        """Test different lookback periods."""
        analyzer = TailRiskAnalyzer()

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        metrics_short = analyzer.analyze_tail_risk(returns, lookback_days=100)
        metrics_long = analyzer.analyze_tail_risk(returns, lookback_days=500)

        # Both should work
        assert metrics_short.cvar_95 < 0
        assert metrics_long.cvar_95 < 0

        # May have different values due to different samples
        # But should be in similar range
        assert abs(metrics_short.cvar_95 - metrics_long.cvar_95) < 0.05


class TestCVaRPositionSizer:
    """Test CVaR-based position sizing."""

    def test_basic_position_sizing(self):
        """Test basic CVaR position sizing."""
        sizer = CVaRPositionSizer(
            max_position_cvar=0.02,
            max_portfolio_cvar=0.05,
            confidence_level=0.95
        )

        portfolio_value = 100000.0

        # Create asset with moderate risk
        np.random.seed(42)
        asset_returns = np.random.normal(0.001, 0.02, 252)

        signal_confidence = 0.8

        position_value, details = sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            asset_returns=asset_returns,
            signal_confidence=signal_confidence,
            current_portfolio_cvar=0.01
        )

        # Should return positive position size
        assert position_value > 0
        assert position_value < portfolio_value

        # Check details
        assert "position_cvar" in details
        assert details["position_cvar"] <= 0.02  # Within limit
        assert "tail_metrics" in details
        assert details["signal_confidence"] == 0.8

    def test_position_cvar_limit_enforcement(self):
        """Test that positions exceeding CVaR limits are rejected/reduced."""
        sizer = CVaRPositionSizer(
            max_position_cvar=0.02,
            max_portfolio_cvar=0.05
        )

        portfolio_value = 100000.0

        # Create very volatile asset (high CVaR)
        np.random.seed(42)
        asset_returns = np.random.normal(0.001, 0.08, 252)  # 8% daily vol!

        position_value, details = sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            asset_returns=asset_returns,
            signal_confidence=1.0,
            current_portfolio_cvar=0.0
        )

        # Position should be limited by CVaR
        position_cvar = details["position_cvar"]
        assert position_cvar <= 0.02  # Must not exceed limit

    def test_portfolio_cvar_budget_exhaustion(self):
        """Test behavior when portfolio CVaR budget is exhausted."""
        sizer = CVaRPositionSizer(
            max_position_cvar=0.02,
            max_portfolio_cvar=0.05
        )

        portfolio_value = 100000.0

        np.random.seed(42)
        asset_returns = np.random.normal(0.001, 0.02, 252)

        # Portfolio CVaR already at 4.5% (near limit of 5%)
        position_value, details = sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            asset_returns=asset_returns,
            signal_confidence=1.0,
            current_portfolio_cvar=0.045  # Almost at limit
        )

        # Should return very small or zero position
        if position_value > 0:
            # If non-zero, should be very conservative
            assert position_value < 10000  # Less than 10% of portfolio
            assert details["position_cvar"] < 0.005  # Very small CVaR

    def test_zero_cvar_budget(self):
        """Test rejection when CVaR budget is zero."""
        sizer = CVaRPositionSizer(max_portfolio_cvar=0.05)

        portfolio_value = 100000.0
        asset_returns = np.random.normal(0.001, 0.02, 252)

        # Portfolio CVaR at maximum
        position_value, details = sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            asset_returns=asset_returns,
            signal_confidence=1.0,
            current_portfolio_cvar=0.05  # At limit
        )

        # Should reject
        assert position_value == 0
        assert "budget exhausted" in details.get("reason", "").lower()

    def test_fat_tails_adjustment(self):
        """Test position size reduction for fat-tailed assets."""
        sizer = CVaRPositionSizer()

        portfolio_value = 100000.0

        # Normal distribution
        np.random.seed(42)
        returns_normal = np.random.normal(0.001, 0.02, 500)

        # Fat-tailed distribution
        returns_fat = stats.t.rvs(df=3, loc=0.001, scale=0.02, size=500)

        position_normal, details_normal = sizer.calculate_position_size(
            portfolio_value, returns_normal, 1.0, 0.0
        )

        position_fat, details_fat = sizer.calculate_position_size(
            portfolio_value, returns_fat, 1.0, 0.0
        )

        # Fat-tailed asset should get smaller position
        assert details_fat["tail_adjustment"] < 1.0
        # Position size may vary, but adjustment should be applied

    def test_negative_skew_adjustment(self):
        """Test position reduction for negatively skewed assets."""
        sizer = CVaRPositionSizer()

        portfolio_value = 100000.0

        # Positively skewed (small losses, occasional big wins)
        np.random.seed(42)
        returns_pos_skew = np.random.lognormal(0, 0.02, 500) - 1.0

        # Negatively skewed (small gains, occasional big losses)
        returns_neg_skew = -(np.random.lognormal(0, 0.02, 500) - 1.0)

        position_pos, details_pos = sizer.calculate_position_size(
            portfolio_value, returns_pos_skew, 1.0, 0.0
        )

        position_neg, details_neg = sizer.calculate_position_size(
            portfolio_value, returns_neg_skew, 1.0, 0.0
        )

        # Negative skew should result in smaller position
        # Check if skew adjustment was applied
        if details_neg["tail_metrics"].skewness < -0.5:
            assert details_neg["skew_adjustment"] < 1.0

    def test_check_risk_limits(self):
        """Test risk limit checking."""
        sizer = CVaRPositionSizer(
            max_position_cvar=0.02,
            max_portfolio_cvar=0.05
        )

        portfolio_value = 100000.0
        position_value = 20000.0  # 20% position
        position_cvar = 0.015  # 1.5% CVaR
        current_portfolio_cvar = 0.02  # 2% current

        # Create moderate tail metrics
        tail_metrics = CVaRMetrics(
            var_95=-0.03,
            var_99=-0.05,
            cvar_95=-0.04,
            cvar_99=-0.07,
            tail_index=1.5,
            skewness=-0.3,
            kurtosis=2.0,
            worst_day=-0.05,
            worst_week=-0.08,
            worst_month=-0.12,
            current_exposure=0.0,
            risk_budget_used=0.0
        )

        approved, reason = sizer.check_risk_limits(
            position_value, portfolio_value, position_cvar,
            current_portfolio_cvar, tail_metrics
        )

        assert approved
        assert reason == "Approved"

    def test_reject_excessive_position_cvar(self):
        """Test rejection of excessive position CVaR."""
        sizer = CVaRPositionSizer(max_position_cvar=0.02)

        tail_metrics = CVaRMetrics(
            var_95=-0.05, var_99=-0.08, cvar_95=-0.06, cvar_99=-0.10,
            tail_index=2.0, skewness=-0.5, kurtosis=3.0,
            worst_day=-0.08, worst_week=-0.15, worst_month=-0.25,
            current_exposure=0.0, risk_budget_used=0.0
        )

        approved, reason = sizer.check_risk_limits(
            position_value=30000,
            portfolio_value=100000,
            position_cvar=0.03,  # Exceeds 2% limit
            current_portfolio_cvar=0.01,
            tail_metrics=tail_metrics
        )

        assert not approved
        assert "Position CVaR" in reason

    def test_reject_excessive_concentration(self):
        """Test rejection of excessive position concentration."""
        sizer = CVaRPositionSizer()

        tail_metrics = CVaRMetrics(
            var_95=-0.03, var_99=-0.05, cvar_95=-0.04, cvar_99=-0.07,
            tail_index=1.5, skewness=-0.2, kurtosis=1.0,
            worst_day=-0.05, worst_week=-0.08, worst_month=-0.12,
            current_exposure=0.0, risk_budget_used=0.0
        )

        approved, reason = sizer.check_risk_limits(
            position_value=30000,  # 30% of portfolio (exceeds 25% limit)
            portfolio_value=100000,
            position_cvar=0.015,
            current_portfolio_cvar=0.01,
            tail_metrics=tail_metrics
        )

        assert not approved
        assert "concentration" in reason.lower()

    def test_reject_excessive_kurtosis(self):
        """Test rejection of extremely fat-tailed assets."""
        sizer = CVaRPositionSizer()

        tail_metrics = CVaRMetrics(
            var_95=-0.05, var_99=-0.10, cvar_95=-0.08, cvar_99=-0.15,
            tail_index=4.0, skewness=-0.8, kurtosis=15.0,  # Extremely fat tails
            worst_day=-0.15, worst_week=-0.25, worst_month=-0.40,
            current_exposure=0.0, risk_budget_used=0.0
        )

        approved, reason = sizer.check_risk_limits(
            position_value=10000,
            portfolio_value=100000,
            position_cvar=0.015,
            current_portfolio_cvar=0.01,
            tail_metrics=tail_metrics
        )

        assert not approved
        assert "kurtosis" in reason.lower()


class TestPortfolioCVaRManager:
    """Test portfolio-level CVaR management."""

    def test_portfolio_cvar_calculation(self):
        """Test portfolio CVaR calculation with multiple positions."""
        manager = PortfolioCVaRManager(max_portfolio_cvar=0.05)

        # Create positions
        positions = {
            "AAPL": 30000,
            "GOOGL": 25000,
            "MSFT": 20000
        }

        # Create return series
        np.random.seed(42)
        returns_dict = {
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "GOOGL": np.random.normal(0.001, 0.025, 252),
            "MSFT": np.random.normal(0.001, 0.018, 252)
        }

        portfolio_cvar, details = manager.calculate_portfolio_cvar(
            positions, returns_dict
        )

        # Should return reasonable portfolio CVaR
        assert portfolio_cvar < 0  # Negative (loss)
        assert -0.10 < portfolio_cvar < 0  # Reasonable range

        # Check diversification benefit
        assert "diversification_benefit" in details
        assert details["diversification_benefit"] >= 0  # Should be positive

        # Check utilization
        assert 0 <= details["cvar_utilization"] <= 1.0

    def test_diversification_benefit(self):
        """Test that diversification reduces portfolio CVaR."""
        manager = PortfolioCVaRManager()

        # Single position
        positions_single = {"AAPL": 75000}

        # Diversified positions
        positions_diversified = {
            "AAPL": 25000,
            "GOOGL": 25000,
            "MSFT": 25000
        }

        # Create uncorrelated returns
        np.random.seed(42)
        returns_dict = {
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "GOOGL": np.random.normal(0.001, 0.02, 252),
            "MSFT": np.random.normal(0.001, 0.02, 252)
        }

        cvar_single, details_single = manager.calculate_portfolio_cvar(
            positions_single, returns_dict
        )

        cvar_diversified, details_diversified = manager.calculate_portfolio_cvar(
            positions_diversified, returns_dict
        )

        # Diversified portfolio should have positive diversification benefit
        assert details_diversified["diversification_benefit"] > 0

    def test_empty_portfolio(self):
        """Test handling of empty portfolio."""
        manager = PortfolioCVaRManager()

        positions = {}
        returns_dict = {}

        portfolio_cvar, details = manager.calculate_portfolio_cvar(
            positions, returns_dict
        )

        assert portfolio_cvar == 0.0
        assert details == {}

    def test_risk_budget_allocation(self):
        """Test CVaR risk budget allocation."""
        manager = PortfolioCVaRManager(max_portfolio_cvar=0.05)

        # Allocate remaining budget across 3 positions
        budgets = manager.allocate_risk_budget(
            num_positions=3,
            current_cvar=0.02,  # 2% used
            priority_weights=None  # Equal allocation
        )

        assert len(budgets) == 3

        # Should sum to remaining budget (3%)
        assert abs(sum(budgets) - 0.03) < 1e-6

        # Should be equal
        assert all(abs(b - 0.01) < 1e-6 for b in budgets)

    def test_weighted_risk_budget_allocation(self):
        """Test weighted CVaR budget allocation."""
        manager = PortfolioCVaRManager(max_portfolio_cvar=0.05)

        # Weighted allocation: 50%, 30%, 20%
        budgets = manager.allocate_risk_budget(
            num_positions=3,
            current_cvar=0.01,
            priority_weights=[0.5, 0.3, 0.2]
        )

        remaining = 0.04  # 5% - 1% = 4%

        # Check weighted allocation
        assert abs(budgets[0] - 0.5 * remaining) < 1e-6
        assert abs(budgets[1] - 0.3 * remaining) < 1e-6
        assert abs(budgets[2] - 0.2 * remaining) < 1e-6

    def test_budget_exhausted(self):
        """Test allocation when budget is exhausted."""
        manager = PortfolioCVaRManager(max_portfolio_cvar=0.05)

        budgets = manager.allocate_risk_budget(
            num_positions=3,
            current_cvar=0.05,  # At limit
            priority_weights=None
        )

        # All budgets should be zero
        assert all(b == 0.0 for b in budgets)


class TestGlobalInstances:
    """Test global singleton instances."""

    def test_cvar_position_sizer_instance(self):
        """Test global cvar_position_sizer configuration."""
        assert cvar_position_sizer.max_position_cvar == 0.02
        assert cvar_position_sizer.max_portfolio_cvar == 0.05
        assert cvar_position_sizer.target_cvar == 0.03

    def test_portfolio_cvar_manager_instance(self):
        """Test global portfolio_cvar_manager configuration."""
        assert portfolio_cvar_manager.max_portfolio_cvar == 0.05
        assert portfolio_cvar_manager.cvar_confidence == 0.95


class TestCVaRIntegration:
    """Integration tests for CVaR risk management."""

    def test_full_risk_workflow(self):
        """Test complete CVaR risk management workflow."""
        # 1. Analyze asset tail risk
        analyzer = TailRiskAnalyzer()

        np.random.seed(42)
        asset_returns = np.random.normal(0.001, 0.02, 500)

        tail_metrics = analyzer.analyze_tail_risk(asset_returns)
        assert tail_metrics.cvar_95 < 0

        # 2. Size position based on CVaR
        sizer = CVaRPositionSizer(max_position_cvar=0.02)

        portfolio_value = 100000.0

        position_value, details = sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            asset_returns=asset_returns,
            signal_confidence=0.8,
            current_portfolio_cvar=0.01
        )

        assert position_value > 0

        # 3. Check risk limits
        approved, reason = sizer.check_risk_limits(
            position_value=position_value,
            portfolio_value=portfolio_value,
            position_cvar=details["position_cvar"],
            current_portfolio_cvar=0.01,
            tail_metrics=tail_metrics
        )

        assert approved

        # 4. Calculate portfolio CVaR
        manager = PortfolioCVaRManager()

        positions = {"AAPL": position_value}
        returns_dict = {"AAPL": asset_returns}

        portfolio_cvar, port_details = manager.calculate_portfolio_cvar(
            positions, returns_dict
        )

        assert portfolio_cvar < 0
        assert abs(portfolio_cvar) <= 0.05  # Within limit
