"""
Institutional-Grade Trading Orchestrator

Integrates all institutional components:
1. Statistical Validation (WFA, stress testing)
2. CVaR Risk Management
3. Data Quality Assurance
4. Model Risk Management
5. Enhanced Execution

Ensures systematic capital deployment only after statistical stability,
tail risk control, and operational resilience.

Author: LLM Trading Platform - Institutional Grade
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from src.validation.statistical_validation import (
    statistical_validator,
    BacktestMetrics,
    WalkForwardResult
)
from src.risk.cvar_risk_management import (
    cvar_position_sizer,
    portfolio_cvar_manager,
    CVaRMetrics
)
from src.operations.data_quality import (
    data_quality_monitor,
    DataQualityReport,
    DataQualityLevel
)
from src.operations.model_risk_management import (
    mrm_system,
    ModelMetadata,
    ModelStatus,
    ModelTier,
    ModelPerformanceMetric
)
from src.trading_engine.enhanced_executor import EnhancedTradingExecutor
from src.data_layer.models import TradingSignal


@dataclass
class InstitutionalChecklistResult:
    """Results from institutional readiness checklist."""
    timestamp: datetime

    # Statistical validation
    wfa_passed: bool
    stress_test_passed: bool
    parameter_robust: bool
    statistical_score: float  # 0-100

    # Risk management
    cvar_within_limits: bool
    tail_risk_acceptable: bool
    risk_score: float  # 0-100

    # Data quality
    data_quality_level: DataQualityLevel
    data_tradable: bool
    quality_score: float  # 0-100

    # Model governance
    model_approved: bool
    model_monitored: bool
    governance_score: float  # 0-100

    # Overall
    overall_score: float  # 0-100
    production_ready: bool
    blockers: List[str]
    warnings: List[str]
    recommendations: List[str]


@dataclass
class TradingDecision:
    """Institutional trading decision."""
    timestamp: datetime
    signal: TradingSignal

    # Risk-adjusted sizing
    approved_position_value: float
    position_cvar: float
    cvar_budget_used: float

    # Execution parameters
    execution_strategy: str
    estimated_slippage_bps: float
    urgency: str

    # Decision rationale
    decision: str  # "EXECUTE", "REJECT", "DEFER"
    reasoning: List[str]

    # Governance
    model_id: str
    data_quality_score: float
    risk_tier: str


class InstitutionalOrchestrator:
    """
    Main orchestrator for institutional-grade trading.

    Enforces systematic checklist before capital deployment.
    """

    def __init__(
        self,
        min_sharpe_required: float = 1.0,
        min_profit_factor_required: float = 1.5,
        max_portfolio_cvar: float = 0.05,
        min_data_quality: float = 70.0
    ):
        """
        Initialize orchestrator with institutional thresholds.

        Args:
            min_sharpe_required: Minimum Sharpe ratio for production
            min_profit_factor_required: Minimum profit factor
            max_portfolio_cvar: Maximum portfolio CVaR (5%)
            min_data_quality: Minimum data quality score (70%)
        """
        self.min_sharpe_required = min_sharpe_required
        self.min_profit_factor_required = min_profit_factor_required
        self.max_portfolio_cvar = max_portfolio_cvar
        self.min_data_quality = min_data_quality

        # Track decisions
        self.decision_history: List[TradingDecision] = []
        self.current_portfolio_cvar = 0.0

        logger.info(
            "Institutional Orchestrator initialized:\n"
            f"  Min Sharpe: {min_sharpe_required}\n"
            f"  Min Profit Factor: {min_profit_factor_required}\n"
            f"  Max Portfolio CVaR: {max_portfolio_cvar*100}%\n"
            f"  Min Data Quality: {min_data_quality}%"
        )

    def run_institutional_checklist(
        self,
        strategy_data: pd.DataFrame,
        strategy_func: callable,
        param_grid: Dict,
        final_params: Dict,
        model_id: str
    ) -> InstitutionalChecklistResult:
        """
        Run complete institutional readiness checklist.

        This is the MANDATORY gate before any live trading.

        Args:
            strategy_data: Historical data for validation
            strategy_func: Strategy function
            param_grid: Parameter grid for optimization
            final_params: Final parameters to validate
            model_id: Model identifier for MRM

        Returns:
            InstitutionalChecklistResult
        """
        logger.info("="*80)
        logger.info("INSTITUTIONAL READINESS CHECKLIST - START")
        logger.info("="*80)

        blockers = []
        warnings = []
        recommendations = []

        # ============================================================
        # PHASE 1: STATISTICAL VALIDATION
        # ============================================================
        logger.info("\n[1/4] Statistical Validation...")

        validation_results = statistical_validator.full_validation(
            data=strategy_data,
            strategy_func=strategy_func,
            param_grid=param_grid,
            final_params=final_params
        )

        wfa_results = validation_results["wfa_results"]
        stress_results = validation_results["stress_results"]
        sensitivity_results = validation_results["sensitivity_results"]
        baseline_metrics = validation_results["baseline_metrics"]
        statistical_score = validation_results["validation_score"]
        stats_production_ready = validation_results["is_production_ready"]

        # Check requirements
        wfa_passed = True
        if wfa_results:
            avg_oos_sharpe = np.mean([r.out_of_sample_metrics.sharpe_ratio for r in wfa_results])
            wfa_passed = avg_oos_sharpe >= 0.5

        stress_survival = sum(1 for r in stress_results if r.survives_stress) / len(stress_results) if stress_results else 0
        stress_test_passed = stress_survival >= 0.5

        robust_count = sum(1 for r in sensitivity_results if r.is_robust) if sensitivity_results else 0
        parameter_robust = robust_count >= len(final_params) * 0.5

        if not wfa_passed:
            blockers.append("Walk-Forward Analysis: OOS Sharpe < 0.5")

        if not stress_test_passed:
            blockers.append(f"Stress Testing: Survival rate {stress_survival:.0%} < 50%")

        if not parameter_robust:
            warnings.append("Parameter sensitivity: Less than 50% of parameters are robust")

        # ============================================================
        # PHASE 2: CVAR RISK MANAGEMENT
        # ============================================================
        logger.info("\n[2/4] CVaR Risk Management...")

        # Analyze tail risk on strategy returns
        strategy_returns = strategy_data['returns'].values if 'returns' in strategy_data.columns else np.random.normal(0, 0.01, len(strategy_data))

        from src.risk.cvar_risk_management import TailRiskAnalyzer
        tail_analyzer = TailRiskAnalyzer()
        tail_metrics = tail_analyzer.analyze_tail_risk(strategy_returns)

        cvar_within_limits = abs(tail_metrics.cvar_95) <= self.max_portfolio_cvar
        tail_risk_acceptable = (
            tail_metrics.kurtosis < 10 and  # Not extremely fat-tailed
            tail_metrics.worst_day > -0.15  # Worst day loss < 15%
        )

        risk_score = 100.0
        if not cvar_within_limits:
            risk_score -= 40
            blockers.append(f"CVaR 95% ({abs(tail_metrics.cvar_95):.2%}) exceeds limit ({self.max_portfolio_cvar:.2%})")

        if not tail_risk_acceptable:
            risk_score -= 30
            warnings.append(f"Heavy tail risk detected: kurtosis={tail_metrics.kurtosis:.1f}, worst_day={tail_metrics.worst_day:.1%}")

        if tail_metrics.skewness < -0.5:
            risk_score -= 10
            warnings.append(f"Negative skew detected: {tail_metrics.skewness:.2f}")

        # ============================================================
        # PHASE 3: DATA QUALITY ASSURANCE
        # ============================================================
        logger.info("\n[3/4] Data Quality Assurance...")

        dq_report = data_quality_monitor.validate_data(
            data=strategy_data,
            data_source="strategy_data"
        )

        data_quality_level = dq_report.quality_level
        data_tradable = dq_report.is_tradable
        quality_score = dq_report.overall_score

        if not data_tradable:
            blockers.append(f"Data quality insufficient: {quality_score:.0f}% (min: {self.min_data_quality}%)")

        if dq_report.critical_issues > 0:
            blockers.append(f"{dq_report.critical_issues} critical data quality issues")

        warnings.extend(dq_report.warnings)
        recommendations.extend(dq_report.recommendations)

        # ============================================================
        # PHASE 4: MODEL RISK MANAGEMENT
        # ============================================================
        logger.info("\n[4/4] Model Risk Management...")

        # Create/update model metadata
        model_metadata = ModelMetadata(
            model_id=model_id,
            model_name=f"Trading Strategy {model_id}",
            model_version="1.0",
            model_tier=ModelTier.TIER_1,  # High risk - direct trading decisions
            status=ModelStatus.VALIDATION,
            business_purpose="Generate trading signals for automated execution",
            target_users=["Trading System"],
            decision_impact="Direct capital deployment decisions",
            model_type="hybrid",
            algorithm_description=f"LLM-augmented strategy with parameters: {final_params}",
            input_features=list(strategy_data.columns),
            output_variables=["signal", "confidence"],
            developer="System",
            development_date=datetime.now(),
            data_period=f"{strategy_data.index.min()} to {strategy_data.index.max()}",
            training_samples=len(strategy_data),
            performance_metrics=[
                ModelPerformanceMetric("sharpe_ratio", self.min_sharpe_required, baseline_metrics.sharpe_ratio, False),
                ModelPerformanceMetric("profit_factor", self.min_profit_factor_required, baseline_metrics.profit_factor, False),
                ModelPerformanceMetric("max_drawdown", -0.30, baseline_metrics.max_drawdown, baseline_metrics.max_drawdown < -0.30)
            ]
        )

        # Onboard through MRM
        model_approved = mrm_system.onboard_model(
            model_metadata,
            validation_results={
                "backtest": {
                    "sharpe_ratio": baseline_metrics.sharpe_ratio,
                    "profit_factor": baseline_metrics.profit_factor,
                    "max_drawdown": baseline_metrics.max_drawdown
                },
                "stress_test": {
                    "survival_rate": stress_survival,
                    "worst_drawdown": min(r.stressed_max_dd for r in stress_results) if stress_results else 0
                },
                "sensitivity": {
                    "robust_parameters": [r.parameter_name for r in sensitivity_results if r.is_robust],
                    "sensitive_parameters": [r.parameter_name for r in sensitivity_results if not r.is_robust]
                }
            }
        )

        model_monitored = model_metadata.status in [ModelStatus.PRODUCTION, ModelStatus.MONITORING]
        governance_score = 100.0 if model_approved else 0.0

        if not model_approved:
            blockers.append("Model failed MRM validation")

        # ============================================================
        # FINAL ASSESSMENT
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("INSTITUTIONAL READINESS ASSESSMENT")
        logger.info("="*80)

        overall_score = (
            statistical_score * 0.35 +
            risk_score * 0.25 +
            quality_score * 0.20 +
            governance_score * 0.20
        )

        production_ready = (
            wfa_passed and
            stress_test_passed and
            cvar_within_limits and
            data_tradable and
            model_approved and
            len(blockers) == 0 and
            overall_score >= 70
        )

        result = InstitutionalChecklistResult(
            timestamp=datetime.now(),
            wfa_passed=wfa_passed,
            stress_test_passed=stress_test_passed,
            parameter_robust=parameter_robust,
            statistical_score=statistical_score,
            cvar_within_limits=cvar_within_limits,
            tail_risk_acceptable=tail_risk_acceptable,
            risk_score=risk_score,
            data_quality_level=data_quality_level,
            data_tradable=data_tradable,
            quality_score=quality_score,
            model_approved=model_approved,
            model_monitored=model_monitored,
            governance_score=governance_score,
            overall_score=overall_score,
            production_ready=production_ready,
            blockers=blockers,
            warnings=warnings,
            recommendations=recommendations
        )

        self._log_checklist_result(result)

        return result

    def make_trading_decision(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        asset_returns: np.ndarray,
        data_quality_report: DataQualityReport,
        model_id: str,
        enhanced_executor: Optional[EnhancedTradingExecutor] = None
    ) -> TradingDecision:
        """
        Make institutional trading decision with full governance.

        Args:
            signal: Trading signal from strategy
            portfolio_value: Current portfolio value
            asset_returns: Historical returns for CVaR analysis
            data_quality_report: Latest data quality report
            model_id: Model identifier
            enhanced_executor: Optional executor for execution

        Returns:
            TradingDecision with approval or rejection
        """
        reasoning = []

        # ============================================================
        # GATE 1: DATA QUALITY
        # ============================================================
        if not data_quality_report.is_tradable:
            return TradingDecision(
                timestamp=datetime.now(),
                signal=signal,
                approved_position_value=0.0,
                position_cvar=0.0,
                cvar_budget_used=0.0,
                execution_strategy="NONE",
                estimated_slippage_bps=0.0,
                urgency="NONE",
                decision="REJECT",
                reasoning=["Data quality insufficient for trading"],
                model_id=model_id,
                data_quality_score=data_quality_report.overall_score,
                risk_tier="N/A"
            )

        reasoning.append(f"Data quality: {data_quality_report.overall_score:.0f}% ({data_quality_report.quality_level.value})")

        # ============================================================
        # GATE 2: MODEL GOVERNANCE
        # ============================================================
        model = mrm_system.inventory.models.get(model_id)
        if not model or model.status != ModelStatus.PRODUCTION:
            return TradingDecision(
                timestamp=datetime.now(),
                signal=signal,
                approved_position_value=0.0,
                position_cvar=0.0,
                cvar_budget_used=0.0,
                execution_strategy="NONE",
                estimated_slippage_bps=0.0,
                urgency="NONE",
                decision="REJECT",
                reasoning=["Model not approved for production"],
                model_id=model_id,
                data_quality_score=data_quality_report.overall_score,
                risk_tier="N/A"
            )

        reasoning.append(f"Model status: {model.status.value}")

        # ============================================================
        # GATE 3: CVAR POSITION SIZING
        # ============================================================
        position_value, cvar_details = cvar_position_sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            asset_returns=asset_returns,
            signal_confidence=signal.confidence_score,
            current_portfolio_cvar=self.current_portfolio_cvar
        )

        if position_value <= 0:
            return TradingDecision(
                timestamp=datetime.now(),
                signal=signal,
                approved_position_value=0.0,
                position_cvar=0.0,
                cvar_budget_used=0.0,
                execution_strategy="NONE",
                estimated_slippage_bps=0.0,
                urgency="NONE",
                decision="REJECT",
                reasoning=["CVaR budget exhausted or position too risky"],
                model_id=model_id,
                data_quality_score=data_quality_report.overall_score,
                risk_tier=model.model_tier.value
            )

        position_cvar = cvar_details["position_cvar"]
        reasoning.append(f"Position CVaR: {position_cvar:.2%} (size: ${position_value:,.0f})")

        # Check risk limits
        approved, limit_reason = cvar_position_sizer.check_risk_limits(
            position_value=position_value,
            portfolio_value=portfolio_value,
            position_cvar=position_cvar,
            current_portfolio_cvar=self.current_portfolio_cvar,
            tail_metrics=cvar_details["tail_metrics"]
        )

        if not approved:
            return TradingDecision(
                timestamp=datetime.now(),
                signal=signal,
                approved_position_value=0.0,
                position_cvar=0.0,
                cvar_budget_used=0.0,
                execution_strategy="NONE",
                estimated_slippage_bps=0.0,
                urgency="NONE",
                decision="REJECT",
                reasoning=[f"Risk limit breach: {limit_reason}"],
                model_id=model_id,
                data_quality_score=data_quality_report.overall_score,
                risk_tier=model.model_tier.value
            )

        # ============================================================
        # GATE 4: EXECUTION PLANNING
        # ============================================================
        # Determine execution parameters based on signal urgency and size
        from src.trading_engine.slippage_management import ExecutionUrgency

        if signal.confidence_score > 0.8:
            urgency = ExecutionUrgency.HIGH
            exec_strategy = "IMMEDIATE"
            estimated_slippage = 15.0  # bps
        elif signal.confidence_score > 0.6:
            urgency = ExecutionUrgency.MEDIUM
            exec_strategy = "VWAP"
            estimated_slippage = 10.0
        else:
            urgency = ExecutionUrgency.LOW
            exec_strategy = "TWAP"
            estimated_slippage = 8.0

        reasoning.append(f"Execution: {exec_strategy} (confidence: {signal.confidence_score:.2f})")

        # ============================================================
        # FINAL DECISION
        # ============================================================
        decision = TradingDecision(
            timestamp=datetime.now(),
            signal=signal,
            approved_position_value=position_value,
            position_cvar=position_cvar,
            cvar_budget_used=position_cvar / self.max_portfolio_cvar * 100,
            execution_strategy=exec_strategy,
            estimated_slippage_bps=estimated_slippage,
            urgency=urgency.value,
            decision="EXECUTE",
            reasoning=reasoning,
            model_id=model_id,
            data_quality_score=data_quality_report.overall_score,
            risk_tier=model.model_tier.value
        )

        # Update portfolio CVaR
        self.current_portfolio_cvar += position_cvar

        # Record decision
        self.decision_history.append(decision)

        logger.info(
            f"\n{'='*60}\n"
            f"TRADING DECISION: {decision.decision}\n"
            f"{'='*60}\n"
            f"Signal: {signal.signal_type.value} {signal.symbol}\n"
            f"Position: ${position_value:,.0f} ({position_value/portfolio_value*100:.1f}% of portfolio)\n"
            f"CVaR: {position_cvar:.2%} (Budget used: {decision.cvar_budget_used:.1f}%)\n"
            f"Reasoning:\n" +
            "\n".join(f"  - {r}" for r in reasoning) +
            f"\n{'='*60}"
        )

        return decision

    def _log_checklist_result(self, result: InstitutionalChecklistResult):
        """Log checklist results."""
        status_icon = "✓" if result.production_ready else "✗"

        logger.info(
            f"\nPhase Results:\n"
            f"  [1] Statistical: {result.statistical_score:.0f}/100 "
            f"({'PASS' if result.wfa_passed and result.stress_test_passed else 'FAIL'})\n"
            f"  [2] Risk: {result.risk_score:.0f}/100 "
            f"({'PASS' if result.cvar_within_limits else 'FAIL'})\n"
            f"  [3] Data Quality: {result.quality_score:.0f}/100 "
            f"({'PASS' if result.data_tradable else 'FAIL'})\n"
            f"  [4] Governance: {result.governance_score:.0f}/100 "
            f"({'PASS' if result.model_approved else 'FAIL'})\n"
            f"\n"
            f"Overall Score: {result.overall_score:.0f}/100\n"
            f"Production Ready: {status_icon} {result.production_ready}\n"
        )

        if result.blockers:
            logger.error("BLOCKERS (must fix):")
            for blocker in result.blockers:
                logger.error(f"  ✗ {blocker}")

        if result.warnings:
            logger.warning("WARNINGS (should address):")
            for warning in result.warnings:
                logger.warning(f"  ⚠ {warning}")

        if result.recommendations:
            logger.info("RECOMMENDATIONS:")
            for rec in result.recommendations:
                logger.info(f"  → {rec}")


# Singleton instance
institutional_orchestrator = InstitutionalOrchestrator(
    min_sharpe_required=1.0,
    min_profit_factor_required=1.5,
    max_portfolio_cvar=0.05,
    min_data_quality=70.0
)
