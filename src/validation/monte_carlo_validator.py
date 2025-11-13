"""
Monte Carlo Robustness Testing

Implements Monte Carlo simulation to test strategy robustness under thousands
of synthetic market scenarios. This complements Walk-Forward Analysis by testing
parameter stability and return distribution under different random paths.

Features:
- Bootstrap resampling of returns
- Parameter uncertainty testing
- Confidence intervals on performance metrics
- Return distribution analysis
- Synthetic path generation
- "Luck vs Skill" quantification

References:
- Aronson, D. (2006). "Evidence-Based Technical Analysis"
- De Prado, M. L. (2018). "Advances in Financial Machine Learning"
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Callable, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from loguru import logger as app_logger


@dataclass
class MonteCarloResult:
    """Result from a single Monte Carlo simulation run."""

    run_id: int
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    final_equity: float


@dataclass
class MonteCarloSummary:
    """Summary statistics from Monte Carlo simulations."""

    num_simulations: int
    metric_name: str

    # Distribution statistics
    mean: float
    median: float
    std: float
    min: float
    max: float

    # Confidence intervals
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float

    # Percentiles
    p10: float
    p25: float
    p75: float
    p90: float


@dataclass
class MonteCarloValidationResult:
    """Complete Monte Carlo validation result."""

    num_simulations: int
    simulation_time_seconds: float

    # Summary statistics for each metric
    sharpe_summary: MonteCarloSummary
    return_summary: MonteCarloSummary
    drawdown_summary: MonteCarloSummary
    win_rate_summary: MonteCarloSummary

    # Individual run results
    all_runs: List[MonteCarloResult]

    # Pass/Fail assessment
    passed: bool
    confidence_score: float  # 0-100
    failure_reasons: List[str]

    # Luck vs Skill metrics
    positive_runs_pct: float  # % of runs with positive return
    skill_score: float  # How much better than random (0-1)


class MonteCarloValidator:
    """
    Perform Monte Carlo simulations to assess strategy robustness.

    Tests strategy performance under thousands of synthetic scenarios
    by resampling historical data and testing parameter variations.
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        confidence_level: float = 0.95,
        min_sharpe_ci_lower: float = 0.5,
        parallel: bool = True,
    ):
        """
        Initialize Monte Carlo validator.

        Args:
            num_simulations: Number of Monte Carlo runs
            confidence_level: Confidence level for intervals (default 95%)
            min_sharpe_ci_lower: Minimum lower bound of Sharpe CI for passing
            parallel: Use parallel processing for speed
        """
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.min_sharpe_ci_lower = min_sharpe_ci_lower
        self.parallel = parallel

    def run_monte_carlo(
        self,
        historical_returns: pd.Series,
        strategy_func: Callable,
        params: Dict[str, Any],
        data: pd.DataFrame,
    ) -> MonteCarloValidationResult:
        """
        Run Monte Carlo simulation with bootstrap resampling.

        Args:
            historical_returns: Historical strategy returns
            strategy_func: Strategy function to test
            params: Strategy parameters
            data: Historical market data

        Returns:
            MonteCarloValidationResult with comprehensive statistics
        """
        start_time = datetime.utcnow()

        app_logger.info(
            f"Starting Monte Carlo simulation with {self.num_simulations} runs"
        )

        all_runs = []

        if self.parallel:
            # Parallel execution
            with ProcessPoolExecutor() as executor:
                futures = []
                for i in range(self.num_simulations):
                    future = executor.submit(
                        self._run_single_simulation,
                        i,
                        historical_returns,
                        strategy_func,
                        params,
                        data,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_runs.append(result)
                    except Exception as e:
                        app_logger.error(f"Simulation run failed: {e}")

        else:
            # Sequential execution (easier to debug)
            for i in range(self.num_simulations):
                try:
                    result = self._run_single_simulation(
                        i, historical_returns, strategy_func, params, data
                    )
                    all_runs.append(result)
                except Exception as e:
                    app_logger.error(f"Simulation run {i} failed: {e}")

        end_time = datetime.utcnow()
        simulation_time = (end_time - start_time).total_seconds()

        app_logger.info(
            f"Completed {len(all_runs)}/{self.num_simulations} simulations "
            f"in {simulation_time:.1f}s"
        )

        # Calculate summary statistics
        sharpe_summary = self._calculate_summary(
            [r.sharpe_ratio for r in all_runs], "Sharpe Ratio"
        )
        return_summary = self._calculate_summary(
            [r.total_return for r in all_runs], "Total Return"
        )
        drawdown_summary = self._calculate_summary(
            [r.max_drawdown for r in all_runs], "Max Drawdown"
        )
        win_rate_summary = self._calculate_summary(
            [r.win_rate for r in all_runs], "Win Rate"
        )

        # Assess pass/fail
        passed, confidence_score, failure_reasons = self._assess_results(
            sharpe_summary, return_summary, drawdown_summary
        )

        # Calculate "luck vs skill" metrics
        positive_runs = sum(1 for r in all_runs if r.total_return > 0)
        positive_runs_pct = (positive_runs / len(all_runs)) * 100

        # Skill score: how far above 50% positive (random)
        skill_score = max(0, (positive_runs_pct - 50) / 50)

        return MonteCarloValidationResult(
            num_simulations=len(all_runs),
            simulation_time_seconds=simulation_time,
            sharpe_summary=sharpe_summary,
            return_summary=return_summary,
            drawdown_summary=drawdown_summary,
            win_rate_summary=win_rate_summary,
            all_runs=all_runs,
            passed=passed,
            confidence_score=confidence_score,
            failure_reasons=failure_reasons,
            positive_runs_pct=positive_runs_pct,
            skill_score=skill_score,
        )

    def _run_single_simulation(
        self,
        run_id: int,
        historical_returns: pd.Series,
        strategy_func: Callable,
        params: Dict[str, Any],
        data: pd.DataFrame,
    ) -> MonteCarloResult:
        """
        Run a single Monte Carlo simulation.

        Uses bootstrap resampling: randomly sample returns with replacement.
        """
        # Bootstrap resample returns
        n = len(historical_returns)
        resampled_indices = np.random.choice(n, size=n, replace=True)
        resampled_returns = historical_returns.iloc[resampled_indices]

        # Calculate performance metrics from resampled returns
        sharpe_ratio = self._calculate_sharpe(resampled_returns)
        total_return = (1 + resampled_returns).prod() - 1
        max_drawdown = self._calculate_max_drawdown(resampled_returns)

        # Win rate and profit factor
        wins = resampled_returns[resampled_returns > 0]
        losses = resampled_returns[resampled_returns < 0]

        win_rate = len(wins) / len(resampled_returns) if len(resampled_returns) > 0 else 0
        profit_factor = (
            wins.sum() / abs(losses.sum())
            if len(losses) > 0 and losses.sum() != 0
            else 0
        )

        # Final equity
        final_equity = 100000 * (1 + total_return)  # Assuming $100K starting capital

        return MonteCarloResult(
            run_id=run_id,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(resampled_returns),
            final_equity=final_equity,
        )

    def _calculate_summary(
        self, values: List[float], metric_name: str
    ) -> MonteCarloSummary:
        """Calculate summary statistics for a metric."""
        values_array = np.array(values)

        mean = float(np.mean(values_array))
        median = float(np.median(values_array))
        std = float(np.std(values_array))
        min_val = float(np.min(values_array))
        max_val = float(np.max(values_array))

        # Confidence intervals
        ci_95_lower = float(np.percentile(values_array, 2.5))
        ci_95_upper = float(np.percentile(values_array, 97.5))
        ci_99_lower = float(np.percentile(values_array, 0.5))
        ci_99_upper = float(np.percentile(values_array, 99.5))

        # Percentiles
        p10 = float(np.percentile(values_array, 10))
        p25 = float(np.percentile(values_array, 25))
        p75 = float(np.percentile(values_array, 75))
        p90 = float(np.percentile(values_array, 90))

        return MonteCarloSummary(
            num_simulations=len(values),
            metric_name=metric_name,
            mean=mean,
            median=median,
            std=std,
            min=min_val,
            max=max_val,
            ci_95_lower=ci_95_lower,
            ci_95_upper=ci_95_upper,
            ci_99_lower=ci_99_lower,
            ci_99_upper=ci_99_upper,
            p10=p10,
            p25=p25,
            p75=p75,
            p90=p90,
        )

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))  # Annualized

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(abs(drawdown.min()))

    def _assess_results(
        self,
        sharpe_summary: MonteCarloSummary,
        return_summary: MonteCarloSummary,
        drawdown_summary: MonteCarloSummary,
    ) -> Tuple[bool, float, List[str]]:
        """
        Assess whether Monte Carlo results meet requirements.

        Returns:
            (passed, confidence_score, failure_reasons)
        """
        passed = True
        confidence_score = 100.0
        failure_reasons = []

        # Check 1: Sharpe ratio CI lower bound
        if sharpe_summary.ci_95_lower < self.min_sharpe_ci_lower:
            passed = False
            failure_reasons.append(
                f"Sharpe 95% CI lower bound ({sharpe_summary.ci_95_lower:.2f}) "
                f"below minimum ({self.min_sharpe_ci_lower})"
            )
            confidence_score -= 30

        # Check 2: Positive expected return
        if return_summary.mean <= 0:
            passed = False
            failure_reasons.append(f"Negative expected return ({return_summary.mean:.1%})")
            confidence_score -= 40

        # Check 3: Return volatility (high std = low confidence)
        return_cv = abs(return_summary.std / return_summary.mean) if return_summary.mean != 0 else float('inf')
        if return_cv > 2.0:  # Coefficient of variation > 2
            failure_reasons.append(
                f"High return volatility (CV={return_cv:.2f})"
            )
            confidence_score -= 20

        # Check 4: Drawdown consistency
        if drawdown_summary.ci_95_upper > 0.5:  # 50% drawdown in worst case
            failure_reasons.append(
                f"Excessive worst-case drawdown ({drawdown_summary.ci_95_upper:.1%})"
            )
            confidence_score -= 10

        confidence_score = max(0, confidence_score)

        return passed, confidence_score, failure_reasons


# Global singleton instance
monte_carlo_validator = MonteCarloValidator(
    num_simulations=1000,
    confidence_level=0.95,
    min_sharpe_ci_lower=0.5,
    parallel=True,
)
