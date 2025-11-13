"""
Tests for Monte Carlo Validation Framework

Tests Monte Carlo simulations for strategy robustness testing using
bootstrap resampling and parameter uncertainty analysis.

Author: LLM Trading Platform - Test Suite
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
from src.validation.monte_carlo_validator import (
    MonteCarloValidator,
    MonteCarloResult,
    MonteCarloSummary,
    MonteCarloValidationResult,
    monte_carlo_validator
)


class TestMonteCarloValidator:
    """Test Monte Carlo validator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = MonteCarloValidator(
            num_simulations=500,
            confidence_level=0.95,
            min_sharpe_ci_lower=0.5,
            parallel=False
        )

        assert validator.num_simulations == 500
        assert validator.confidence_level == 0.95
        assert validator.min_sharpe_ci_lower == 0.5
        assert not validator.parallel

    def test_calculate_sharpe(self):
        """Test Sharpe ratio calculation."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        # Create returns with known Sharpe
        # Daily return=0.001, daily std=0.02
        # Sharpe = 0.001/0.02 * sqrt(252) = 0.793
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        sharpe = validator._calculate_sharpe(returns)

        # Should be positive and reasonable
        assert sharpe > 0
        assert 0.3 < sharpe < 2.0  # Reasonable range

    def test_calculate_sharpe_zero_volatility(self):
        """Test Sharpe calculation with zero volatility."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        returns = pd.Series([0.001] * 100)  # Constant returns

        sharpe = validator._calculate_sharpe(returns)

        assert sharpe == 0.0  # Should return 0 for zero volatility

    def test_calculate_sharpe_empty(self):
        """Test Sharpe calculation with empty series."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        returns = pd.Series([])

        sharpe = validator._calculate_sharpe(returns)

        assert sharpe == 0.0

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        # Create returns with known drawdown
        returns = pd.Series([0.10, -0.20, 0.05, -0.10, 0.15])
        # Cumulative: 1.10, 0.88, 0.924, 0.8316, 0.9564
        # Max so far: 1.10, 1.10, 1.10, 1.10, 1.10
        # Drawdown: 0, -20%, -16%, -24.4%, -13.1%
        # Max drawdown: 24.4%

        max_dd = validator._calculate_max_drawdown(returns)

        assert 0.20 < max_dd < 0.30  # Should be around 24%

    def test_calculate_max_drawdown_no_losses(self):
        """Test drawdown calculation with only gains."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        returns = pd.Series([0.01, 0.02, 0.01, 0.03])

        max_dd = validator._calculate_max_drawdown(returns)

        assert max_dd == 0.0  # No drawdown

    def test_calculate_summary_statistics(self):
        """Test summary statistics calculation."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        # Create sample data
        np.random.seed(42)
        values = list(np.random.normal(1.0, 0.2, 1000))

        summary = validator._calculate_summary(values, "Test Metric")

        assert summary.metric_name == "Test Metric"
        assert summary.num_simulations == 1000

        # Check distribution statistics
        assert 0.9 < summary.mean < 1.1
        assert 0.9 < summary.median < 1.1
        assert 0.15 < summary.std < 0.25

        # Check confidence intervals
        assert summary.ci_95_lower < summary.mean < summary.ci_95_upper
        assert summary.ci_99_lower < summary.ci_95_lower
        assert summary.ci_95_upper < summary.ci_99_upper

        # Check percentiles are ordered
        assert summary.min <= summary.p10 <= summary.p25 <= summary.median
        assert summary.median <= summary.p75 <= summary.p90 <= summary.max

    def test_run_single_simulation(self):
        """Test single Monte Carlo simulation run."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        # Create historical returns
        np.random.seed(42)
        historical_returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        # Mock strategy function (not used in bootstrap)
        strategy_func = Mock()
        params = {}
        data = pd.DataFrame()

        result = validator._run_single_simulation(
            run_id=1,
            historical_returns=historical_returns,
            strategy_func=strategy_func,
            params=params,
            data=data
        )

        assert isinstance(result, MonteCarloResult)
        assert result.run_id == 1
        assert result.sharpe_ratio != 0
        assert -1.0 < result.total_return < 2.0  # Reasonable range
        assert 0 <= result.max_drawdown <= 1.0
        assert 0 <= result.win_rate <= 1.0
        assert result.num_trades == 252
        assert result.final_equity > 0

    def test_bootstrap_resampling_variability(self):
        """Test that bootstrap resampling produces different results."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        historical_returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        strategy_func = Mock()
        params = {}
        data = pd.DataFrame()

        # Run multiple simulations
        results = []
        for i in range(10):
            result = validator._run_single_simulation(
                i, historical_returns, strategy_func, params, data
            )
            results.append(result)

        # Check that results vary (bootstrap creates different paths)
        sharpes = [r.sharpe_ratio for r in results]
        returns = [r.total_return for r in results]

        # Should have some variability
        assert np.std(sharpes) > 0
        assert np.std(returns) > 0

    def test_assess_results_passing(self):
        """Test assessment of passing Monte Carlo results."""
        validator = MonteCarloValidator(
            num_simulations=100,
            min_sharpe_ci_lower=0.5,
            parallel=False
        )

        # Create passing summaries
        sharpe_summary = MonteCarloSummary(
            num_simulations=1000,
            metric_name="Sharpe",
            mean=1.2,
            median=1.15,
            std=0.3,
            min=0.3,
            max=2.5,
            ci_95_lower=0.7,  # Above minimum
            ci_95_upper=1.8,
            ci_99_lower=0.5,
            ci_99_upper=2.0,
            p10=0.8,
            p25=1.0,
            p75=1.4,
            p90=1.6
        )

        return_summary = MonteCarloSummary(
            num_simulations=1000,
            metric_name="Return",
            mean=0.15,  # Positive
            median=0.14,
            std=0.05,
            min=-0.10,
            max=0.40,
            ci_95_lower=0.05,
            ci_95_upper=0.25,
            ci_99_lower=0.02,
            ci_99_upper=0.30,
            p10=0.08,
            p25=0.11,
            p75=0.18,
            p90=0.22
        )

        drawdown_summary = MonteCarloSummary(
            num_simulations=1000,
            metric_name="Drawdown",
            mean=0.12,
            median=0.11,
            std=0.05,
            min=0.02,
            max=0.35,  # Worst case acceptable
            ci_95_lower=0.05,
            ci_95_upper=0.25,  # Below 50%
            ci_99_lower=0.03,
            ci_99_upper=0.30,
            p10=0.06,
            p25=0.08,
            p75=0.15,
            p90=0.20
        )

        passed, confidence, reasons = validator._assess_results(
            sharpe_summary, return_summary, drawdown_summary
        )

        assert passed
        assert confidence > 80
        assert len(reasons) == 0

    def test_assess_results_failing_sharpe(self):
        """Test assessment with failing Sharpe ratio."""
        validator = MonteCarloValidator(
            num_simulations=100,
            min_sharpe_ci_lower=0.5,
            parallel=False
        )

        sharpe_summary = MonteCarloSummary(
            num_simulations=1000,
            metric_name="Sharpe",
            mean=0.6,
            median=0.55,
            std=0.4,
            min=-0.5,
            max=1.8,
            ci_95_lower=0.2,  # Below minimum of 0.5
            ci_95_upper=1.0,
            ci_99_lower=0.1,
            ci_99_upper=1.2,
            p10=0.3,
            p25=0.4,
            p75=0.8,
            p90=1.0
        )

        return_summary = MonteCarloSummary(
            num_simulations=1000, metric_name="Return",
            mean=0.10, median=0.09, std=0.08,
            min=-0.15, max=0.35,
            ci_95_lower=0.02, ci_95_upper=0.20,
            ci_99_lower=0.0, ci_99_upper=0.25,
            p10=0.03, p25=0.06, p75=0.14, p90=0.18
        )

        drawdown_summary = MonteCarloSummary(
            num_simulations=1000, metric_name="Drawdown",
            mean=0.15, median=0.14, std=0.08,
            min=0.02, max=0.45,
            ci_95_lower=0.05, ci_95_upper=0.30,
            ci_99_lower=0.03, ci_99_upper=0.35,
            p10=0.07, p25=0.10, p75=0.20, p90=0.25
        )

        passed, confidence, reasons = validator._assess_results(
            sharpe_summary, return_summary, drawdown_summary
        )

        assert not passed
        assert confidence < 100
        assert len(reasons) > 0
        assert "Sharpe" in reasons[0]

    def test_assess_results_negative_return(self):
        """Test assessment with negative expected return."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        sharpe_summary = MonteCarloSummary(
            num_simulations=1000, metric_name="Sharpe",
            mean=-0.3, median=-0.35, std=0.5,
            min=-1.5, max=0.8,
            ci_95_lower=-1.2, ci_95_upper=0.5,
            ci_99_lower=-1.4, ci_99_upper=0.7,
            p10=-0.9, p25=-0.6, p75=0.0, p90=0.3
        )

        return_summary = MonteCarloSummary(
            num_simulations=1000, metric_name="Return",
            mean=-0.05,  # Negative
            median=-0.06, std=0.10,
            min=-0.30, max=0.15,
            ci_95_lower=-0.20, ci_95_upper=0.08,
            ci_99_lower=-0.25, ci_99_upper=0.12,
            p10=-0.15, p25=-0.10, p75=0.0, p90=0.05
        )

        drawdown_summary = MonteCarloSummary(
            num_simulations=1000, metric_name="Drawdown",
            mean=0.20, median=0.19, std=0.10,
            min=0.05, max=0.50,
            ci_95_lower=0.08, ci_95_upper=0.38,
            ci_99_lower=0.06, ci_99_upper=0.45,
            p10=0.10, p25=0.14, p75=0.26, p90=0.32
        )

        passed, confidence, reasons = validator._assess_results(
            sharpe_summary, return_summary, drawdown_summary
        )

        assert not passed
        assert any("Negative expected return" in r for r in reasons)

    def test_assess_results_excessive_drawdown(self):
        """Test assessment with excessive drawdown."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        sharpe_summary = MonteCarloSummary(
            num_simulations=1000, metric_name="Sharpe",
            mean=1.0, median=0.95, std=0.4,
            min=0.0, max=2.0,
            ci_95_lower=0.6, ci_95_upper=1.5,
            ci_99_lower=0.4, ci_99_upper=1.7,
            p10=0.7, p25=0.8, p75=1.2, p90=1.4
        )

        return_summary = MonteCarloSummary(
            num_simulations=1000, metric_name="Return",
            mean=0.20, median=0.18, std=0.15,
            min=-0.20, max=0.60,
            ci_95_lower=0.0, ci_95_upper=0.45,
            ci_99_lower=-0.05, ci_99_upper=0.52,
            p10=0.05, p25=0.10, p75=0.28, p90=0.38
        )

        drawdown_summary = MonteCarloSummary(
            num_simulations=1000, metric_name="Drawdown",
            mean=0.30, median=0.28, std=0.20,
            min=0.05, max=0.80,
            ci_95_lower=0.10, ci_95_upper=0.65,  # 65% worst case (>50%)
            ci_99_lower=0.08, ci_99_upper=0.75,
            p10=0.12, p25=0.18, p75=0.42, p90=0.55
        )

        passed, confidence, reasons = validator._assess_results(
            sharpe_summary, return_summary, drawdown_summary
        )

        assert any("drawdown" in r.lower() for r in reasons)

    def test_run_monte_carlo_sequential(self):
        """Test full Monte Carlo run in sequential mode."""
        validator = MonteCarloValidator(
            num_simulations=50,  # Small number for fast test
            confidence_level=0.95,
            parallel=False  # Sequential for determinism
        )

        # Create profitable strategy returns
        np.random.seed(42)
        historical_returns = pd.Series(np.random.normal(0.002, 0.015, 252))

        # Mock strategy function (not used in current implementation)
        strategy_func = Mock()
        params = {}
        data = pd.DataFrame()

        result = validator.run_monte_carlo(
            historical_returns, strategy_func, params, data
        )

        assert isinstance(result, MonteCarloValidationResult)
        assert result.num_simulations == 50
        assert result.simulation_time_seconds > 0
        assert len(result.all_runs) == 50

        # Check summaries exist
        assert result.sharpe_summary.num_simulations == 50
        assert result.return_summary.num_simulations == 50
        assert result.drawdown_summary.num_simulations == 50
        assert result.win_rate_summary.num_simulations == 50

        # Check pass/fail assessment
        assert isinstance(result.passed, bool)
        assert 0 <= result.confidence_score <= 100
        assert isinstance(result.failure_reasons, list)

        # Check luck vs skill metrics
        assert 0 <= result.positive_runs_pct <= 100
        assert 0 <= result.skill_score <= 1.0

    def test_luck_vs_skill_metric(self):
        """Test luck vs skill calculation."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        # Create highly profitable returns
        np.random.seed(42)
        historical_returns = pd.Series(np.random.normal(0.003, 0.015, 252))

        strategy_func = Mock()
        result = validator.run_monte_carlo(
            historical_returns, strategy_func, {}, pd.DataFrame()
        )

        # Should have high percentage of positive runs (skill)
        assert result.positive_runs_pct > 70  # Most runs profitable
        assert result.skill_score > 0.3  # Significant skill component

    def test_monte_carlo_with_losing_strategy(self):
        """Test Monte Carlo with losing strategy."""
        validator = MonteCarloValidator(num_simulations=50, parallel=False)

        # Create losing returns
        np.random.seed(42)
        historical_returns = pd.Series(np.random.normal(-0.001, 0.02, 252))

        strategy_func = Mock()
        result = validator.run_monte_carlo(
            historical_returns, strategy_func, {}, pd.DataFrame()
        )

        # Should fail
        assert not result.passed
        assert result.confidence_score < 80
        assert len(result.failure_reasons) > 0

        # Should have low positive runs
        assert result.positive_runs_pct < 60

    def test_global_singleton_instance(self):
        """Test global monte_carlo_validator instance configuration."""
        assert monte_carlo_validator.num_simulations == 1000
        assert monte_carlo_validator.confidence_level == 0.95
        assert monte_carlo_validator.min_sharpe_ci_lower == 0.5
        assert monte_carlo_validator.parallel


class TestMonteCarloIntegration:
    """Integration tests for Monte Carlo validation."""

    def test_full_validation_workflow(self):
        """Test complete Monte Carlo validation workflow."""
        # 1. Create historical strategy returns
        np.random.seed(42)
        # Simulate 1 year of daily returns (profitable strategy)
        daily_returns = np.random.normal(0.0015, 0.018, 252)  # 38% annual, 28% vol
        historical_returns = pd.Series(daily_returns)

        # 2. Initialize validator
        validator = MonteCarloValidator(
            num_simulations=200,
            confidence_level=0.95,
            min_sharpe_ci_lower=0.3,  # Relaxed for test
            parallel=False
        )

        # 3. Run Monte Carlo
        strategy_func = Mock()
        result = validator.run_monte_carlo(
            historical_returns, strategy_func, {}, pd.DataFrame()
        )

        # 4. Verify comprehensive results
        assert result.num_simulations == 200

        # Sharpe ratio
        assert result.sharpe_summary.mean > 0
        assert result.sharpe_summary.ci_95_lower < result.sharpe_summary.mean
        assert result.sharpe_summary.mean < result.sharpe_summary.ci_95_upper

        # Returns
        assert result.return_summary.mean > 0  # Profitable
        assert -0.5 < result.return_summary.min < result.return_summary.max < 2.0

        # Drawdown
        assert 0 <= result.drawdown_summary.mean <= 1.0
        assert result.drawdown_summary.min >= 0

        # Win rate
        assert 0 < result.win_rate_summary.mean < 1.0

        # Should pass with good strategy
        assert result.passed or result.confidence_score > 60

    def test_confidence_interval_coverage(self):
        """Test that confidence intervals have correct coverage."""
        validator = MonteCarloValidator(num_simulations=1000, parallel=False)

        # Create known distribution
        np.random.seed(42)
        historical_returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        strategy_func = Mock()
        result = validator.run_monte_carlo(
            historical_returns, strategy_func, {}, pd.DataFrame()
        )

        # Check 95% CI contains ~95% of observations
        sharpe_values = [r.sharpe_ratio for r in result.all_runs]
        ci_95_lower = result.sharpe_summary.ci_95_lower
        ci_95_upper = result.sharpe_summary.ci_95_upper

        within_ci = sum(1 for s in sharpe_values if ci_95_lower <= s <= ci_95_upper)
        coverage = within_ci / len(sharpe_values)

        # Should be close to 95% (allow some tolerance)
        assert 0.92 < coverage < 0.98

    def test_parameter_robustness(self):
        """Test that similar returns produce consistent results."""
        validator = MonteCarloValidator(num_simulations=100, parallel=False)

        # Run twice with same seed
        np.random.seed(42)
        returns1 = pd.Series(np.random.normal(0.001, 0.02, 252))

        np.random.seed(42)
        returns2 = pd.Series(np.random.normal(0.001, 0.02, 252))

        strategy_func = Mock()

        result1 = validator.run_monte_carlo(returns1, strategy_func, {}, pd.DataFrame())

        result2 = validator.run_monte_carlo(returns2, strategy_func, {}, pd.DataFrame())

        # Results should be similar (same input data)
        assert abs(result1.sharpe_summary.mean - result2.sharpe_summary.mean) < 0.3
        assert abs(result1.return_summary.mean - result2.return_summary.mean) < 0.1

    def test_extreme_volatility_handling(self):
        """Test handling of extremely volatile returns."""
        validator = MonteCarloValidator(num_simulations=50, parallel=False)

        # Create extremely volatile returns
        np.random.seed(42)
        historical_returns = pd.Series(np.random.normal(0.001, 0.10, 252))  # 10% daily vol!

        strategy_func = Mock()
        result = validator.run_monte_carlo(
            historical_returns, strategy_func, {}, pd.DataFrame()
        )

        # Should complete without error
        assert result.num_simulations == 50

        # Should likely fail due to high volatility
        # High volatility usually means negative Sharpe or high drawdowns
        assert result.confidence_score < 90

    def test_rare_events_in_tail(self):
        """Test that Monte Carlo captures rare tail events."""
        validator = MonteCarloValidator(num_simulations=200, parallel=False)

        # Create returns with occasional large losses (fat tails)
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.015, 252)
        # Add some rare large losses
        returns[10] = -0.15
        returns[100] = -0.12
        returns[200] = -0.18

        historical_returns = pd.Series(returns)

        strategy_func = Mock()
        result = validator.run_monte_carlo(
            historical_returns, strategy_func, {}, pd.DataFrame()
        )

        # Should have some runs with large drawdowns
        max_drawdowns = [r.max_drawdown for r in result.all_runs]
        assert max(max_drawdowns) > 0.15  # At least one run captures large loss
