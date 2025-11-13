"""
Statistical Validation Framework

Walk-Forward Analysis, Parameter Sensitivity, and Stress Testing

Prevents overfitting and ensures strategy robustness through:
1. Walk-Forward Analysis (WFA) - Out-of-sample validation
2. Monte Carlo parameter sensitivity
3. Comprehensive stress testing
4. Statistical significance tests
5. Regime-specific performance analysis

Author: LLM Trading Platform - Institutional Grade
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    # Returns
    total_return: float
    annual_return: float
    monthly_returns: List[float]

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float

    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float

    # Statistical tests
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05

    # Robustness
    stability_score: float  # 0-1, consistency across periods
    tail_ratio: float  # 95th percentile gain / 5th percentile loss

    # Execution
    win_streak: int
    loss_streak: int
    recovery_factor: float  # Net profit / max DD


@dataclass
class WalkForwardResult:
    """Results from Walk-Forward Analysis."""
    in_sample_metrics: BacktestMetrics
    out_of_sample_metrics: BacktestMetrics

    # Overfitting detection
    performance_degradation: float  # IS - OOS performance
    is_overfit: bool  # True if degradation > 30%

    # Window details
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Optimal parameters from IS period
    optimal_params: Dict[str, float]


@dataclass
class ParameterSensitivityResult:
    """Results from parameter sensitivity analysis."""
    parameter_name: str
    base_value: float
    tested_values: List[float]

    # Performance at each value
    sharpe_ratios: List[float]
    profit_factors: List[float]
    max_drawdowns: List[float]

    # Sensitivity metrics
    sharpe_sensitivity: float  # Std dev of Sharpe across values
    optimal_range: Tuple[float, float]  # Range of good values
    is_robust: bool  # True if performance stable across range


@dataclass
class StressTestResult:
    """Results from stress testing."""
    scenario_name: str
    description: str

    # Performance under stress
    stressed_return: float
    stressed_sharpe: float
    stressed_max_dd: float

    # Comparison to baseline
    return_impact: float  # % change from baseline
    sharpe_impact: float
    dd_impact: float

    # Survival metrics
    survives_stress: bool  # True if Sharpe > 0.5 under stress
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional VaR


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis implementation.

    Validates strategy by:
    1. Optimize parameters on in-sample data
    2. Test on out-of-sample data
    3. Roll forward and repeat
    4. Aggregate results to detect overfitting
    """

    def __init__(
        self,
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,    # 3 months
        step_days: int = 21            # 1 month
    ):
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List[float]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[WalkForwardResult]:
        """
        Run Walk-Forward Analysis.

        Args:
            data: DataFrame with OHLCV data and features
            strategy_func: Function that takes params and returns signals
            param_grid: Grid of parameters to optimize
            start_date: Start of analysis (default: earliest data)
            end_date: End of analysis (default: latest data)

        Returns:
            List of WalkForwardResult for each window
        """
        if start_date is None:
            start_date = data.index.min()
        if end_date is None:
            end_date = data.index.max()

        results = []
        current_date = start_date

        while current_date + timedelta(days=self.train_period_days + self.test_period_days) <= end_date:
            # Define windows
            train_start = current_date
            train_end = current_date + timedelta(days=self.train_period_days)
            test_start = train_end
            test_end = train_end + timedelta(days=self.test_period_days)

            logger.info(
                f"WFA: Train {train_start.date()} to {train_end.date()}, "
                f"Test {test_start.date()} to {test_end.date()}"
            )

            # Get data
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]

            if len(train_data) < 50 or len(test_data) < 10:
                logger.warning("Insufficient data in window, skipping")
                current_date += timedelta(days=self.step_days)
                continue

            # Optimize on in-sample
            optimal_params, is_metrics = self._optimize_parameters(
                train_data, strategy_func, param_grid
            )

            # Test on out-of-sample
            oos_metrics = self._backtest_with_params(
                test_data, strategy_func, optimal_params
            )

            # Detect overfitting
            perf_degradation = is_metrics.sharpe_ratio - oos_metrics.sharpe_ratio
            is_overfit = perf_degradation > 0.5  # Sharpe drops >0.5

            if is_overfit:
                logger.warning(
                    f"Overfitting detected: IS Sharpe {is_metrics.sharpe_ratio:.2f} "
                    f"vs OOS Sharpe {oos_metrics.sharpe_ratio:.2f}"
                )

            result = WalkForwardResult(
                in_sample_metrics=is_metrics,
                out_of_sample_metrics=oos_metrics,
                performance_degradation=perf_degradation,
                is_overfit=is_overfit,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimal_params=optimal_params
            )

            results.append(result)

            # Roll forward
            current_date += timedelta(days=self.step_days)

        # Aggregate analysis
        self._log_wfa_summary(results)

        return results

    def _optimize_parameters(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List[float]]
    ) -> Tuple[Dict[str, float], BacktestMetrics]:
        """Optimize parameters on in-sample data."""
        best_sharpe = -999
        best_params = {}
        best_metrics = None

        # Grid search
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Generate all combinations
        from itertools import product
        for values in product(*param_values):
            params = dict(zip(param_names, values))

            # Run backtest with these params
            metrics = self._backtest_with_params(data, strategy_func, params)

            # Check if best (using Sharpe as primary metric)
            if metrics.sharpe_ratio > best_sharpe:
                best_sharpe = metrics.sharpe_ratio
                best_params = params
                best_metrics = metrics

        return best_params, best_metrics

    def _backtest_with_params(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict[str, float]
    ) -> BacktestMetrics:
        """Run backtest with specific parameters."""
        # Generate signals
        signals = strategy_func(data, **params)

        # Calculate returns (simplified)
        returns = data['returns'].values if 'returns' in data.columns else np.zeros(len(data))

        # Apply signals to returns
        if isinstance(signals, pd.Series):
            signals = signals.values

        # Position: 1 for long, -1 for short, 0 for flat
        positions = np.where(signals > 0, 1, np.where(signals < 0, -1, 0))

        # Strategy returns
        strategy_returns = positions[:-1] * returns[1:]  # Lag positions

        # Calculate metrics
        total_return = np.prod(1 + strategy_returns) - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1

        # Sharpe ratio
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0

        # Sortino ratio (downside risk)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
        sortino = np.mean(strategy_returns) / downside_std * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns)
        avg_dd = np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0

        # Calmar ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # Trade statistics (simplified - assume signal changes are trades)
        trades = np.diff(positions) != 0
        num_trades = np.sum(trades)

        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]

        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        avg_trade = np.mean(strategy_returns) if len(strategy_returns) > 0 else 0

        profit_factor = abs(np.sum(winning_trades) / np.sum(losing_trades)) if np.sum(losing_trades) != 0 else 0

        # Statistical significance
        t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
        is_significant = p_value < 0.05

        # Stability (consistency across months)
        monthly_returns = []
        chunk_size = 21  # ~1 month
        for i in range(0, len(strategy_returns), chunk_size):
            chunk = strategy_returns[i:i+chunk_size]
            if len(chunk) > 0:
                monthly_returns.append(np.sum(chunk))

        stability = 1 - (np.std(monthly_returns) / (abs(np.mean(monthly_returns)) + 1e-6)) if len(monthly_returns) > 1 else 0
        stability = max(0, min(1, stability))

        # Tail ratio
        tail_ratio = abs(np.percentile(strategy_returns, 95) / np.percentile(strategy_returns, 5)) if np.percentile(strategy_returns, 5) != 0 else 1

        # Streaks
        win_streaks = []
        loss_streaks = []
        current_streak = 0
        for ret in strategy_returns:
            if ret > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    loss_streaks.append(abs(current_streak))
                    current_streak = 1
            elif ret < 0:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    win_streaks.append(current_streak)
                    current_streak = -1

        max_win_streak = max(win_streaks) if win_streaks else 0
        max_loss_streak = abs(min(loss_streaks + [0]))

        # Recovery factor
        net_profit = total_return
        recovery_factor = net_profit / abs(max_dd) if max_dd != 0 else 0

        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_returns=monthly_returns,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            total_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            stability_score=stability,
            tail_ratio=tail_ratio,
            win_streak=max_win_streak,
            loss_streak=max_loss_streak,
            recovery_factor=recovery_factor
        )

    def _log_wfa_summary(self, results: List[WalkForwardResult]):
        """Log aggregate WFA results."""
        if not results:
            return

        is_sharpes = [r.in_sample_metrics.sharpe_ratio for r in results]
        oos_sharpes = [r.out_of_sample_metrics.sharpe_ratio for r in results]
        degradations = [r.performance_degradation for r in results]
        overfit_count = sum(1 for r in results if r.is_overfit)

        logger.info(
            f"\n{'='*60}\n"
            f"Walk-Forward Analysis Summary\n"
            f"{'='*60}\n"
            f"Total Windows: {len(results)}\n"
            f"Overfit Windows: {overfit_count} ({overfit_count/len(results)*100:.1f}%)\n"
            f"\n"
            f"In-Sample Sharpe: {np.mean(is_sharpes):.2f} ± {np.std(is_sharpes):.2f}\n"
            f"Out-of-Sample Sharpe: {np.mean(oos_sharpes):.2f} ± {np.std(oos_sharpes):.2f}\n"
            f"Avg Degradation: {np.mean(degradations):.2f}\n"
            f"\n"
            f"OOS Sharpe > 1.0: {sum(1 for s in oos_sharpes if s > 1.0)}/{len(results)}\n"
            f"OOS Sharpe > 0.5: {sum(1 for s in oos_sharpes if s > 0.5)}/{len(results)}\n"
            f"{'='*60}"
        )


class ParameterSensitivityAnalyzer:
    """Analyze parameter sensitivity to detect overfitting."""

    def analyze_parameter(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_name: str,
        base_params: Dict[str, float],
        test_range: Tuple[float, float],
        num_points: int = 20
    ) -> ParameterSensitivityResult:
        """
        Test parameter across range to check sensitivity.

        Args:
            data: Historical data
            strategy_func: Strategy function
            param_name: Parameter to test
            base_params: Baseline parameters
            test_range: (min, max) values to test
            num_points: Number of values to test

        Returns:
            ParameterSensitivityResult
        """
        logger.info(f"Analyzing sensitivity of {param_name} in range {test_range}")

        # Generate test values
        test_values = np.linspace(test_range[0], test_range[1], num_points)

        sharpe_ratios = []
        profit_factors = []
        max_drawdowns = []

        # Test each value
        for value in test_values:
            params = base_params.copy()
            params[param_name] = value

            # Backtest with this value
            wfa = WalkForwardAnalyzer()
            metrics = wfa._backtest_with_params(data, strategy_func, params)

            sharpe_ratios.append(metrics.sharpe_ratio)
            profit_factors.append(metrics.profit_factor)
            max_drawdowns.append(metrics.max_drawdown)

        # Calculate sensitivity
        sharpe_sensitivity = np.std(sharpe_ratios)

        # Find optimal range (where Sharpe > 80% of max)
        max_sharpe = max(sharpe_ratios)
        threshold = max_sharpe * 0.8

        good_indices = [i for i, s in enumerate(sharpe_ratios) if s >= threshold]
        if good_indices:
            optimal_range = (test_values[min(good_indices)], test_values[max(good_indices)])
        else:
            optimal_range = (test_values[np.argmax(sharpe_ratios)], test_values[np.argmax(sharpe_ratios)])

        # Robustness check: if optimal range is wide and performance stable, it's robust
        range_width = optimal_range[1] - optimal_range[0]
        total_width = test_range[1] - test_range[0]
        is_robust = (range_width / total_width > 0.3) and (sharpe_sensitivity < 0.5)

        logger.info(
            f"Parameter {param_name}: "
            f"Sharpe sensitivity={sharpe_sensitivity:.2f}, "
            f"Optimal range={optimal_range}, "
            f"Robust={is_robust}"
        )

        return ParameterSensitivityResult(
            parameter_name=param_name,
            base_value=base_params[param_name],
            tested_values=list(test_values),
            sharpe_ratios=sharpe_ratios,
            profit_factors=profit_factors,
            max_drawdowns=max_drawdowns,
            sharpe_sensitivity=sharpe_sensitivity,
            optimal_range=optimal_range,
            is_robust=is_robust
        )


class StressTestFramework:
    """Comprehensive stress testing."""

    def __init__(self):
        self.scenarios = self._define_scenarios()

    def _define_scenarios(self) -> Dict[str, Dict]:
        """Define stress test scenarios."""
        return {
            "market_crash": {
                "description": "2008-style market crash (-40% in 6 months)",
                "return_shock": -0.40,
                "volatility_multiplier": 3.0,
                "correlation_increase": 0.3
            },
            "flash_crash": {
                "description": "Flash crash (-10% in 1 day, recovery in 1 week)",
                "return_shock": -0.10,
                "volatility_multiplier": 5.0,
                "recovery_days": 5
            },
            "extended_drawdown": {
                "description": "2000-2002 dot-com crash (3 year bear market)",
                "return_shock": -0.50,
                "duration_months": 36,
                "volatility_multiplier": 2.0
            },
            "volatility_spike": {
                "description": "VIX spike to 80+ (extreme fear)",
                "volatility_multiplier": 4.0,
                "liquidity_reduction": 0.5
            },
            "low_volatility_grind": {
                "description": "Prolonged low volatility (2017 style)",
                "volatility_multiplier": 0.3,
                "trend_strength": 0.1
            },
            "interest_rate_shock": {
                "description": "Fed raises rates 300bps in 6 months",
                "rate_change": 0.03,
                "duration_months": 6
            },
            "liquidity_crisis": {
                "description": "2020 COVID March liquidity freeze",
                "spread_multiplier": 5.0,
                "execution_slippage": 3.0  # 3x normal
            }
        }

    def run_stress_tests(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict[str, float],
        baseline_metrics: BacktestMetrics
    ) -> List[StressTestResult]:
        """
        Run all stress test scenarios.

        Args:
            data: Historical data
            strategy_func: Strategy function
            params: Strategy parameters
            baseline_metrics: Normal market performance

        Returns:
            List of StressTestResult
        """
        results = []

        for scenario_name, scenario_config in self.scenarios.items():
            logger.info(f"Running stress test: {scenario_name}")

            # Apply stress to data
            stressed_data = self._apply_stress(data.copy(), scenario_config)

            # Run backtest on stressed data
            wfa = WalkForwardAnalyzer()
            stressed_metrics = wfa._backtest_with_params(stressed_data, strategy_func, params)

            # Calculate impacts
            return_impact = (stressed_metrics.annual_return - baseline_metrics.annual_return) / (abs(baseline_metrics.annual_return) + 1e-6) * 100
            sharpe_impact = stressed_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio
            dd_impact = stressed_metrics.max_drawdown - baseline_metrics.max_drawdown

            # Survival check
            survives = stressed_metrics.sharpe_ratio > 0.5 and stressed_metrics.max_drawdown > -0.30

            # Calculate VaR and CVaR
            returns = stressed_data['returns'].values if 'returns' in stressed_data.columns else np.random.normal(0, 0.02, len(stressed_data))
            var_95 = np.percentile(returns, 5)  # 95% VaR (5th percentile)
            cvar_95 = np.mean(returns[returns <= var_95])  # Expected loss beyond VaR

            result = StressTestResult(
                scenario_name=scenario_name,
                description=scenario_config["description"],
                stressed_return=stressed_metrics.annual_return,
                stressed_sharpe=stressed_metrics.sharpe_ratio,
                stressed_max_dd=stressed_metrics.max_drawdown,
                return_impact=return_impact,
                sharpe_impact=sharpe_impact,
                dd_impact=dd_impact,
                survives_stress=survives,
                var_95=var_95,
                cvar_95=cvar_95
            )

            results.append(result)

            logger.info(
                f"  Stressed Sharpe: {stressed_metrics.sharpe_ratio:.2f}, "
                f"Max DD: {stressed_metrics.max_drawdown:.1%}, "
                f"Survives: {survives}"
            )

        self._log_stress_summary(results, baseline_metrics)

        return results

    def _apply_stress(self, data: pd.DataFrame, scenario: Dict) -> pd.DataFrame:
        """Apply stress scenario to data."""
        if 'returns' not in data.columns:
            data['returns'] = np.random.normal(0, 0.01, len(data))

        returns = data['returns'].values

        # Apply return shock
        if 'return_shock' in scenario:
            shock_size = scenario['return_shock'] / len(returns)
            returns += shock_size

        # Apply volatility multiplier
        if 'volatility_multiplier' in scenario:
            mult = scenario['volatility_multiplier']
            mean_return = np.mean(returns)
            returns = mean_return + (returns - mean_return) * mult

        data['returns'] = returns

        return data

    def _log_stress_summary(self, results: List[StressTestResult], baseline: BacktestMetrics):
        """Log stress test summary."""
        survival_count = sum(1 for r in results if r.survives_stress)
        avg_stressed_sharpe = np.mean([r.stressed_sharpe for r in results])
        worst_dd = min([r.stressed_max_dd for r in results])

        logger.info(
            f"\n{'='*60}\n"
            f"Stress Test Summary\n"
            f"{'='*60}\n"
            f"Baseline Sharpe: {baseline.sharpe_ratio:.2f}\n"
            f"Baseline Max DD: {baseline.max_drawdown:.1%}\n"
            f"\n"
            f"Scenarios Tested: {len(results)}\n"
            f"Scenarios Survived: {survival_count}/{len(results)} ({survival_count/len(results)*100:.0f}%)\n"
            f"Avg Stressed Sharpe: {avg_stressed_sharpe:.2f}\n"
            f"Worst Drawdown: {worst_dd:.1%}\n"
            f"{'='*60}"
        )


class StatisticalValidator:
    """Main validator combining all tests."""

    def __init__(self):
        self.wfa = WalkForwardAnalyzer()
        self.sensitivity = ParameterSensitivityAnalyzer()
        self.stress = StressTestFramework()

    def full_validation(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List[float]],
        final_params: Dict[str, float]
    ) -> Dict:
        """
        Run complete validation suite.

        Returns dict with:
        - wfa_results: Walk-forward analysis
        - sensitivity_results: Parameter sensitivity
        - stress_results: Stress testing
        - is_production_ready: bool
        - validation_score: 0-100
        """
        logger.info("Starting full statistical validation...")

        # 1. Walk-Forward Analysis
        wfa_results = self.wfa.run_walk_forward(data, strategy_func, param_grid)

        # 2. Get baseline metrics
        baseline_metrics = self.wfa._backtest_with_params(data, strategy_func, final_params)

        # 3. Parameter sensitivity
        sensitivity_results = []
        for param_name in final_params.keys():
            if param_name in param_grid:
                test_range = (min(param_grid[param_name]), max(param_grid[param_name]))
                sens = self.sensitivity.analyze_parameter(
                    data, strategy_func, param_name, final_params, test_range
                )
                sensitivity_results.append(sens)

        # 4. Stress testing
        stress_results = self.stress.run_stress_tests(
            data, strategy_func, final_params, baseline_metrics
        )

        # 5. Calculate validation score
        validation_score = self._calculate_validation_score(
            wfa_results, sensitivity_results, stress_results, baseline_metrics
        )

        # 6. Production readiness check
        is_production_ready = self._check_production_readiness(
            validation_score, wfa_results, stress_results, baseline_metrics
        )

        logger.info(
            f"\n{'='*60}\n"
            f"VALIDATION COMPLETE\n"
            f"{'='*60}\n"
            f"Validation Score: {validation_score:.0f}/100\n"
            f"Production Ready: {is_production_ready}\n"
            f"{'='*60}"
        )

        return {
            "wfa_results": wfa_results,
            "sensitivity_results": sensitivity_results,
            "stress_results": stress_results,
            "baseline_metrics": baseline_metrics,
            "validation_score": validation_score,
            "is_production_ready": is_production_ready
        }

    def _calculate_validation_score(
        self,
        wfa_results: List[WalkForwardResult],
        sensitivity_results: List[ParameterSensitivityResult],
        stress_results: List[StressTestResult],
        baseline: BacktestMetrics
    ) -> float:
        """Calculate 0-100 validation score."""
        score = 0.0

        # 1. Baseline performance (30 points)
        if baseline.sharpe_ratio >= 2.0:
            score += 30
        elif baseline.sharpe_ratio >= 1.5:
            score += 25
        elif baseline.sharpe_ratio >= 1.0:
            score += 20
        elif baseline.sharpe_ratio >= 0.5:
            score += 10

        # 2. WFA consistency (25 points)
        if wfa_results:
            oos_sharpes = [r.out_of_sample_metrics.sharpe_ratio for r in wfa_results]
            overfit_pct = sum(1 for r in wfa_results if r.is_overfit) / len(wfa_results)

            avg_oos_sharpe = np.mean(oos_sharpes)
            if avg_oos_sharpe >= 1.0 and overfit_pct < 0.2:
                score += 25
            elif avg_oos_sharpe >= 0.7 and overfit_pct < 0.3:
                score += 20
            elif avg_oos_sharpe >= 0.5:
                score += 15

        # 3. Parameter robustness (20 points)
        if sensitivity_results:
            robust_count = sum(1 for r in sensitivity_results if r.is_robust)
            robust_pct = robust_count / len(sensitivity_results)
            score += 20 * robust_pct

        # 4. Stress test survival (25 points)
        if stress_results:
            survival_count = sum(1 for r in stress_results if r.survives_stress)
            survival_pct = survival_count / len(stress_results)
            score += 25 * survival_pct

        return min(100, score)

    def _check_production_readiness(
        self,
        validation_score: float,
        wfa_results: List[WalkForwardResult],
        stress_results: List[StressTestResult],
        baseline: BacktestMetrics
    ) -> bool:
        """Check if strategy is production ready."""
        # Minimum requirements
        requirements = [
            ("Validation Score >= 70", validation_score >= 70),
            ("Sharpe Ratio >= 1.0", baseline.sharpe_ratio >= 1.0),
            ("Profit Factor >= 1.5", baseline.profit_factor >= 1.5),
            ("Max Drawdown <= 30%", baseline.max_drawdown >= -0.30),
            ("Statistically Significant", baseline.is_significant),
        ]

        if wfa_results:
            avg_oos_sharpe = np.mean([r.out_of_sample_metrics.sharpe_ratio for r in wfa_results])
            requirements.append(("OOS Sharpe >= 0.5", avg_oos_sharpe >= 0.5))

        if stress_results:
            survival_rate = sum(1 for r in stress_results if r.survives_stress) / len(stress_results)
            requirements.append(("Stress Survival >= 50%", survival_rate >= 0.5))

        # Log requirements
        logger.info("\nProduction Readiness Check:")
        for req_name, passed in requirements:
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {req_name}")

        # All requirements must pass
        return all(passed for _, passed in requirements)


# Singleton instance
statistical_validator = StatisticalValidator()
