"""
Model Risk Management (MRM) Framework

Comprehensive model governance per regulatory requirements (SR 11-7).

Key Components:
1. Model inventory and documentation
2. Model validation and backtesting
3. Performance monitoring
4. Model limitations assessment
5. Change management
6. Annual review and recertification

Author: LLM Trading Platform - Institutional Grade
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum
import json
from loguru import logger


class ModelStatus(Enum):
    """Model lifecycle status."""
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    APPROVED = "approved"
    PRODUCTION = "production"
    MONITORING = "monitoring"
    REVIEW_REQUIRED = "review_required"
    DEPRECATED = "deprecated"


class ModelTier(Enum):
    """Model risk tier (regulatory classification)."""
    TIER_1 = "tier_1"  # High risk - significant impact
    TIER_2 = "tier_2"  # Moderate risk - material impact
    TIER_3 = "tier_3"  # Low risk - limited impact


@dataclass
class ModelLimitation:
    """Known model limitation."""
    limitation_id: str
    description: str
    impact: str  # "high", "medium", "low"
    mitigation: str
    residual_risk: str


@dataclass
class ModelAssumption:
    """Model assumption that must hold."""
    assumption_id: str
    description: str
    validation_method: str
    is_critical: bool
    last_validated: datetime


@dataclass
class ModelPerformanceMetric:
    """Model performance tracking."""
    metric_name: str
    target_value: float
    actual_value: float
    threshold_breach: bool
    breach_count: int = 0
    last_breach_date: Optional[datetime] = None


@dataclass
class ModelValidationResult:
    """Model validation outcome."""
    validation_date: datetime
    validator_name: str
    validation_type: str  # "initial", "annual", "ongoing"

    # Test results
    backtesting_passed: bool
    stress_testing_passed: bool
    sensitivity_passed: bool
    documentation_complete: bool

    # Overall
    approved: bool
    expiry_date: datetime
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ModelMetadata:
    """Complete model documentation."""
    model_id: str
    model_name: str
    model_version: str
    model_tier: ModelTier
    status: ModelStatus

    # Business context
    business_purpose: str
    target_users: List[str]
    decision_impact: str  # What decisions does this model drive

    # Technical details
    model_type: str  # "LLM", "statistical", "ML", "hybrid"
    algorithm_description: str
    input_features: List[str]
    output_variables: List[str]

    # Development
    developer: str
    development_date: datetime
    data_period: str  # e.g., "2020-01-01 to 2023-12-31"
    training_samples: int

    # Validation
    validator: Optional[str] = None
    validation_date: Optional[datetime] = None
    validation_result: Optional[ModelValidationResult] = None
    next_review_date: Optional[datetime] = None

    # Performance
    performance_metrics: List[ModelPerformanceMetric] = field(default_factory=list)

    # Risk
    limitations: List[ModelLimitation] = field(default_factory=list)
    assumptions: List[ModelAssumption] = field(default_factory=list)
    known_biases: List[str] = field(default_factory=list)

    # Monitoring
    monitoring_frequency: str = "daily"
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    last_monitored: Optional[datetime] = None

    # Change log
    change_history: List[Dict] = field(default_factory=list)


class ModelInventory:
    """Central model registry."""

    def __init__(self):
        self.models: Dict[str, ModelMetadata] = {}
        logger.info("Model inventory initialized")

    def register_model(self, model: ModelMetadata) -> bool:
        """Register new model in inventory."""
        if model.model_id in self.models:
            logger.warning(f"Model {model.model_id} already registered")
            return False

        self.models[model.model_id] = model
        logger.info(f"Model registered: {model.model_id} ({model.model_name})")

        # Trigger validation requirement
        if model.status == ModelStatus.DEVELOPMENT:
            logger.info(f"Model {model.model_id} requires validation before production use")

        return True

    def update_model_status(
        self,
        model_id: str,
        new_status: ModelStatus,
        reason: str
    ) -> bool:
        """Update model status."""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False

        model = self.models[model_id]
        old_status = model.status

        # Validate status transition
        valid_transition = self._validate_status_transition(old_status, new_status)
        if not valid_transition:
            logger.error(
                f"Invalid status transition for {model_id}: "
                f"{old_status.value} -> {new_status.value}"
            )
            return False

        model.status = new_status
        model.change_history.append({
            "timestamp": datetime.now().isoformat(),
            "change_type": "status_change",
            "old_value": old_status.value,
            "new_value": new_status.value,
            "reason": reason
        })

        logger.info(
            f"Model {model_id} status updated: "
            f"{old_status.value} -> {new_status.value}"
        )

        return True

    def _validate_status_transition(
        self,
        old_status: ModelStatus,
        new_status: ModelStatus
    ) -> bool:
        """Check if status transition is allowed."""
        valid_transitions = {
            ModelStatus.DEVELOPMENT: [ModelStatus.VALIDATION, ModelStatus.DEPRECATED],
            ModelStatus.VALIDATION: [ModelStatus.APPROVED, ModelStatus.DEVELOPMENT],
            ModelStatus.APPROVED: [ModelStatus.PRODUCTION],
            ModelStatus.PRODUCTION: [ModelStatus.MONITORING, ModelStatus.REVIEW_REQUIRED, ModelStatus.DEPRECATED],
            ModelStatus.MONITORING: [ModelStatus.REVIEW_REQUIRED, ModelStatus.PRODUCTION],
            ModelStatus.REVIEW_REQUIRED: [ModelStatus.APPROVED, ModelStatus.DEPRECATED],
            ModelStatus.DEPRECATED: []  # Terminal state
        }

        return new_status in valid_transitions.get(old_status, [])

    def get_production_models(self) -> List[ModelMetadata]:
        """Get all models in production."""
        return [
            m for m in self.models.values()
            if m.status == ModelStatus.PRODUCTION
        ]

    def get_models_needing_review(self) -> List[ModelMetadata]:
        """Get models requiring review."""
        now = datetime.now()
        needing_review = []

        for model in self.models.values():
            if model.status == ModelStatus.REVIEW_REQUIRED:
                needing_review.append(model)
            elif model.next_review_date and model.next_review_date <= now:
                needing_review.append(model)

        return needing_review


class ModelValidator:
    """Model validation procedures."""

    def __init__(self):
        self.validation_history: List[ModelValidationResult] = []

    def validate_model(
        self,
        model: ModelMetadata,
        backtest_results: Dict,
        stress_test_results: Dict,
        sensitivity_results: Dict
    ) -> ModelValidationResult:
        """
        Perform comprehensive model validation.

        Args:
            model: Model to validate
            backtest_results: Results from backtesting
            stress_test_results: Results from stress testing
            sensitivity_results: Results from sensitivity analysis

        Returns:
            ModelValidationResult
        """
        logger.info(f"Starting validation for model {model.model_id}")

        findings = []
        recommendations = []

        # 1. Backtesting validation
        backtesting_passed = self._validate_backtesting(
            backtest_results, findings, recommendations
        )

        # 2. Stress testing validation
        stress_testing_passed = self._validate_stress_testing(
            stress_test_results, findings, recommendations
        )

        # 3. Sensitivity analysis validation
        sensitivity_passed = self._validate_sensitivity(
            sensitivity_results, findings, recommendations
        )

        # 4. Documentation completeness
        documentation_complete = self._validate_documentation(
            model, findings, recommendations
        )

        # Overall approval
        approved = (
            backtesting_passed and
            stress_testing_passed and
            sensitivity_passed and
            documentation_complete
        )

        # Set expiry date (annual review required)
        expiry_date = datetime.now() + timedelta(days=365)

        result = ModelValidationResult(
            validation_date=datetime.now(),
            validator_name="System Validator",
            validation_type="initial",
            backtesting_passed=backtesting_passed,
            stress_testing_passed=stress_testing_passed,
            sensitivity_passed=sensitivity_passed,
            documentation_complete=documentation_complete,
            approved=approved,
            expiry_date=expiry_date,
            findings=findings,
            recommendations=recommendations
        )

        self.validation_history.append(result)

        if approved:
            logger.info(f"✓ Model {model.model_id} validation PASSED")
        else:
            logger.warning(f"✗ Model {model.model_id} validation FAILED")
            logger.warning(f"Findings: {', '.join(findings)}")

        return result

    def _validate_backtesting(
        self,
        results: Dict,
        findings: List[str],
        recommendations: List[str]
    ) -> bool:
        """Validate backtesting results."""
        # Check for minimum performance
        sharpe_ratio = results.get("sharpe_ratio", 0)
        profit_factor = results.get("profit_factor", 0)
        max_dd = results.get("max_drawdown", -1)

        passed = True

        if sharpe_ratio < 1.0:
            passed = False
            findings.append(f"Sharpe ratio {sharpe_ratio:.2f} below minimum 1.0")
            recommendations.append("Improve strategy performance or adjust risk parameters")

        if profit_factor < 1.5:
            passed = False
            findings.append(f"Profit factor {profit_factor:.2f} below minimum 1.5")
            recommendations.append("Review win rate and risk/reward ratio")

        if max_dd < -0.30:  # More than 30% drawdown
            findings.append(f"Maximum drawdown {max_dd:.1%} exceeds 30%")
            recommendations.append("Implement stronger drawdown controls")
            # Don't fail on this, but note it

        return passed

    def _validate_stress_testing(
        self,
        results: Dict,
        findings: List[str],
        recommendations: List[str]
    ) -> bool:
        """Validate stress testing results."""
        survival_rate = results.get("survival_rate", 0)
        worst_case_dd = results.get("worst_drawdown", -1)

        passed = True

        if survival_rate < 0.5:  # Must survive at least 50% of scenarios
            passed = False
            findings.append(f"Stress test survival rate {survival_rate:.0%} below 50%")
            recommendations.append("Add risk controls to survive extreme scenarios")

        if worst_case_dd < -0.50:  # Worst case >50% loss
            findings.append(f"Worst case drawdown {worst_case_dd:.1%} exceeds 50%")
            recommendations.append("Implement emergency stop-loss at -40% portfolio level")
            # Don't fail, but flag for monitoring

        return passed

    def _validate_sensitivity(
        self,
        results: Dict,
        findings: List[str],
        recommendations: List[str]
    ) -> bool:
        """Validate parameter sensitivity."""
        robust_params = results.get("robust_parameters", [])
        sensitive_params = results.get("sensitive_parameters", [])

        passed = True

        if len(sensitive_params) > len(robust_params):
            passed = False
            findings.append(
                f"Too many sensitive parameters: {len(sensitive_params)} sensitive vs "
                f"{len(robust_params)} robust"
            )
            recommendations.append(
                "Reduce parameter count or find more robust parameter values"
            )

        return passed

    def _validate_documentation(
        self,
        model: ModelMetadata,
        findings: List[str],
        recommendations: List[str]
    ) -> bool:
        """Validate documentation completeness."""
        required_fields = [
            ("business_purpose", "Business purpose not documented"),
            ("algorithm_description", "Algorithm not described"),
            ("input_features", "Input features not listed"),
            ("limitations", "Model limitations not documented"),
            ("assumptions", "Model assumptions not documented"),
        ]

        missing = []
        for field, message in required_fields:
            value = getattr(model, field, None)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing.append(message)

        if missing:
            findings.extend(missing)
            recommendations.append("Complete model documentation before production use")
            return False

        return True


class ModelMonitor:
    """Ongoing model performance monitoring."""

    def __init__(self):
        self.performance_history: Dict[str, List[Dict]] = {}

    def monitor_model(
        self,
        model: ModelMetadata,
        actual_performance: Dict[str, float]
    ) -> List[str]:
        """
        Monitor model performance against thresholds.

        Args:
            model: Model to monitor
            actual_performance: Dict of actual metric values

        Returns:
            List of alerts (empty if all good)
        """
        alerts = []

        # Check each performance metric
        for metric in model.performance_metrics:
            actual = actual_performance.get(metric.metric_name)

            if actual is None:
                alerts.append(
                    f"Missing performance data for {metric.metric_name}"
                )
                continue

            # Update metric
            metric.actual_value = actual

            # Check for threshold breach
            # Determine if breach based on metric type
            is_breach = False

            if "sharpe" in metric.metric_name.lower():
                is_breach = actual < metric.target_value
            elif "drawdown" in metric.metric_name.lower():
                is_breach = actual < metric.target_value  # More negative = worse
            elif "win_rate" in metric.metric_name.lower():
                is_breach = actual < metric.target_value
            else:
                # Generic: actual should be >= target
                is_breach = actual < metric.target_value

            if is_breach:
                metric.threshold_breach = True
                metric.breach_count += 1
                metric.last_breach_date = datetime.now()

                alerts.append(
                    f"Threshold breach: {metric.metric_name} = {actual:.2f} "
                    f"(target: {metric.target_value:.2f}, breaches: {metric.breach_count})"
                )

                # Trigger review if persistent breaches
                if metric.breach_count >= 3:
                    alerts.append(
                        f"CRITICAL: {metric.metric_name} breached 3+ times - "
                        f"model review required"
                    )
            else:
                metric.threshold_breach = False

        # Record performance snapshot
        if model.model_id not in self.performance_history:
            self.performance_history[model.model_id] = []

        self.performance_history[model.model_id].append({
            "timestamp": datetime.now().isoformat(),
            "performance": actual_performance.copy(),
            "alerts": alerts.copy()
        })

        # Update model last monitored
        model.last_monitored = datetime.now()

        if alerts:
            logger.warning(
                f"Model {model.model_id} monitoring alerts:\n" +
                "\n".join(f"  - {alert}" for alert in alerts)
            )

        return alerts


class ModelRiskManagementSystem:
    """Main MRM orchestrator."""

    def __init__(self):
        self.inventory = ModelInventory()
        self.validator = ModelValidator()
        self.monitor = ModelMonitor()

        logger.info("Model Risk Management System initialized")

    def onboard_model(
        self,
        model_metadata: ModelMetadata,
        validation_results: Dict
    ) -> bool:
        """
        Onboard new model through full MRM process.

        Args:
            model_metadata: Model documentation
            validation_results: Dict with backtest/stress/sensitivity results

        Returns:
            True if approved for production
        """
        logger.info(f"Onboarding model: {model_metadata.model_id}")

        # 1. Register in inventory
        registered = self.inventory.register_model(model_metadata)
        if not registered:
            return False

        # 2. Run validation
        validation_result = self.validator.validate_model(
            model_metadata,
            backtest_results=validation_results.get("backtest", {}),
            stress_test_results=validation_results.get("stress_test", {}),
            sensitivity_results=validation_results.get("sensitivity", {})
        )

        # 3. Update model with validation results
        model_metadata.validation_result = validation_result
        model_metadata.validation_date = validation_result.validation_date
        model_metadata.next_review_date = validation_result.expiry_date

        # 4. Update status based on approval
        if validation_result.approved:
            self.inventory.update_model_status(
                model_metadata.model_id,
                ModelStatus.APPROVED,
                "Passed initial validation"
            )
            logger.info(f"✓ Model {model_metadata.model_id} approved for production")
            return True
        else:
            self.inventory.update_model_status(
                model_metadata.model_id,
                ModelStatus.DEVELOPMENT,
                f"Validation failed: {', '.join(validation_result.findings)}"
            )
            logger.warning(f"✗ Model {model_metadata.model_id} not approved")
            return False

    def monitor_production_models(
        self,
        performance_data: Dict[str, Dict[str, float]]
    ):
        """
        Monitor all production models.

        Args:
            performance_data: Dict mapping model_id to performance metrics
        """
        production_models = self.inventory.get_production_models()

        logger.info(f"Monitoring {len(production_models)} production models")

        for model in production_models:
            if model.model_id not in performance_data:
                logger.warning(f"No performance data for model {model.model_id}")
                continue

            alerts = self.monitor.monitor_model(
                model,
                performance_data[model.model_id]
            )

            # Trigger review if critical alerts
            critical_alerts = [a for a in alerts if "CRITICAL" in a]
            if critical_alerts:
                self.inventory.update_model_status(
                    model.model_id,
                    ModelStatus.REVIEW_REQUIRED,
                    "Critical performance degradation detected"
                )

    def generate_compliance_report(self) -> Dict:
        """Generate MRM compliance report."""
        all_models = list(self.inventory.models.values())
        production_models = self.inventory.get_production_models()
        review_needed = self.inventory.get_models_needing_review()

        report = {
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_models": len(all_models),
                "production_models": len(production_models),
                "models_needing_review": len(review_needed),
                "high_risk_models": sum(1 for m in all_models if m.model_tier == ModelTier.TIER_1)
            },
            "models_by_status": {},
            "models_by_tier": {},
            "review_schedule": []
        }

        # Count by status
        for status in ModelStatus:
            count = sum(1 for m in all_models if m.status == status)
            report["models_by_status"][status.value] = count

        # Count by tier
        for tier in ModelTier:
            count = sum(1 for m in all_models if m.model_tier == tier)
            report["models_by_tier"][tier.value] = count

        # Review schedule
        for model in review_needed:
            report["review_schedule"].append({
                "model_id": model.model_id,
                "model_name": model.model_name,
                "tier": model.model_tier.value,
                "next_review_date": model.next_review_date.isoformat() if model.next_review_date else None
            })

        logger.info(
            f"\n{'='*60}\n"
            f"MRM Compliance Report\n"
            f"{'='*60}\n"
            f"Total Models: {report['summary']['total_models']}\n"
            f"Production: {report['summary']['production_models']}\n"
            f"Review Needed: {report['summary']['models_needing_review']}\n"
            f"High Risk (Tier 1): {report['summary']['high_risk_models']}\n"
            f"{'='*60}"
        )

        return report


# Singleton instance
mrm_system = ModelRiskManagementSystem()
