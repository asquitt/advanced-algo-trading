"""
Data Quality Assurance Framework

Ensures data integrity and prevents garbage-in-garbage-out scenarios.

Key Features:
1. Real-time data validation
2. Anomaly detection
3. Completeness checks
4. Consistency validation
5. Timeliness monitoring
6. Data reconciliation

Author: LLM Trading Platform - Institutional Grade
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class DataQualityLevel(Enum):
    """Data quality classification."""
    EXCELLENT = "excellent"  # >95% quality score
    GOOD = "good"  # 85-95%
    ACCEPTABLE = "acceptable"  # 70-85%
    POOR = "poor"  # 50-70%
    UNACCEPTABLE = "unacceptable"  # <50%


class ValidationSeverity(Enum):
    """Severity of validation failures."""
    CRITICAL = "critical"  # Trading must stop
    HIGH = "high"  # Alert immediately
    MEDIUM = "medium"  # Log and monitor
    LOW = "low"  # Track for reporting


@dataclass
class DataQualityIssue:
    """Individual data quality issue."""
    timestamp: datetime
    issue_type: str
    severity: ValidationSeverity
    field_name: str
    expected_value: Any
    actual_value: Any
    description: str
    auto_corrected: bool = False


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    timestamp: datetime
    data_source: str
    total_records: int

    # Quality scores (0-100)
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    overall_score: float
    quality_level: DataQualityLevel

    # Issues
    issues: List[DataQualityIssue]
    critical_issues: int
    high_issues: int

    # Recommendations
    is_tradable: bool  # Safe to trade with this data
    warnings: List[str]
    recommendations: List[str]


class DataValidator:
    """Base validator for data quality checks."""

    def __init__(self, field_name: str):
        self.field_name = field_name
        self.issues: List[DataQualityIssue] = []

    def validate(self, data: pd.DataFrame) -> bool:
        """Override in subclasses."""
        raise NotImplementedError

    def add_issue(
        self,
        issue_type: str,
        severity: ValidationSeverity,
        expected: Any,
        actual: Any,
        description: str,
        auto_corrected: bool = False
    ):
        """Record a data quality issue."""
        issue = DataQualityIssue(
            timestamp=datetime.now(),
            issue_type=issue_type,
            severity=severity,
            field_name=self.field_name,
            expected_value=expected,
            actual_value=actual,
            description=description,
            auto_corrected=auto_corrected
        )
        self.issues.append(issue)

        if severity == ValidationSeverity.CRITICAL:
            logger.error(f"CRITICAL DATA ISSUE: {description}")
        elif severity == ValidationSeverity.HIGH:
            logger.warning(f"HIGH PRIORITY: {description}")


class CompletenessValidator(DataValidator):
    """Validate data completeness (no missing values)."""

    def __init__(self, field_name: str, max_missing_pct: float = 0.01):
        super().__init__(field_name)
        self.max_missing_pct = max_missing_pct

    def validate(self, data: pd.DataFrame) -> bool:
        """Check for missing values."""
        if self.field_name not in data.columns:
            self.add_issue(
                "missing_column",
                ValidationSeverity.CRITICAL,
                f"Column {self.field_name}",
                "Not found",
                f"Required column {self.field_name} is missing from data"
            )
            return False

        series = data[self.field_name]
        total = len(series)
        missing = series.isna().sum()
        missing_pct = missing / total if total > 0 else 0

        if missing_pct > self.max_missing_pct:
            self.add_issue(
                "missing_values",
                ValidationSeverity.HIGH if missing_pct < 0.05 else ValidationSeverity.CRITICAL,
                f"<{self.max_missing_pct*100:.1f}% missing",
                f"{missing_pct*100:.1f}% missing ({missing}/{total})",
                f"{self.field_name} has {missing_pct*100:.1f}% missing values"
            )
            return False

        return True


class RangeValidator(DataValidator):
    """Validate values are within expected range."""

    def __init__(
        self,
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_outliers_pct: float = 0.01
    ):
        super().__init__(field_name)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_outliers_pct = allow_outliers_pct

    def validate(self, data: pd.DataFrame) -> bool:
        """Check values are in valid range."""
        if self.field_name not in data.columns:
            return False

        series = data[self.field_name].dropna()
        if len(series) == 0:
            return True

        outliers = 0
        total = len(series)

        if self.min_value is not None:
            below_min = (series < self.min_value).sum()
            outliers += below_min

            if below_min > 0:
                self.add_issue(
                    "below_minimum",
                    ValidationSeverity.HIGH,
                    f">={self.min_value}",
                    f"{below_min} values < {self.min_value}",
                    f"{self.field_name} has {below_min} values below minimum {self.min_value}"
                )

        if self.max_value is not None:
            above_max = (series > self.max_value).sum()
            outliers += above_max

            if above_max > 0:
                self.add_issue(
                    "above_maximum",
                    ValidationSeverity.HIGH,
                    f"<={self.max_value}",
                    f"{above_max} values > {self.max_value}",
                    f"{self.field_name} has {above_max} values above maximum {self.max_value}"
                )

        outlier_pct = outliers / total
        return outlier_pct <= self.allow_outliers_pct


class AnomalyDetector(DataValidator):
    """Detect statistical anomalies in data."""

    def __init__(
        self,
        field_name: str,
        zscore_threshold: float = 5.0,
        max_anomalies_pct: float = 0.02
    ):
        super().__init__(field_name)
        self.zscore_threshold = zscore_threshold
        self.max_anomalies_pct = max_anomalies_pct

    def validate(self, data: pd.DataFrame) -> bool:
        """Detect anomalies using z-score."""
        if self.field_name not in data.columns:
            return False

        series = data[self.field_name].dropna()
        if len(series) < 10:
            return True  # Not enough data

        # Calculate z-scores
        mean = series.mean()
        std = series.std()

        if std == 0:
            return True  # No variance

        z_scores = np.abs((series - mean) / std)
        anomalies = z_scores > self.zscore_threshold
        num_anomalies = anomalies.sum()
        anomaly_pct = num_anomalies / len(series)

        if anomaly_pct > self.max_anomalies_pct:
            max_zscore = z_scores.max()
            self.add_issue(
                "statistical_anomaly",
                ValidationSeverity.MEDIUM,
                f"<{self.max_anomalies_pct*100:.1f}% anomalies",
                f"{anomaly_pct*100:.1f}% anomalies (max z-score: {max_zscore:.1f})",
                f"{self.field_name} has {num_anomalies} statistical anomalies"
            )
            return False

        return True


class TimelinessValidator(DataValidator):
    """Validate data freshness/timeliness."""

    def __init__(
        self,
        timestamp_field: str = "timestamp",
        max_age_seconds: int = 300  # 5 minutes
    ):
        super().__init__(timestamp_field)
        self.max_age_seconds = max_age_seconds

    def validate(self, data: pd.DataFrame) -> bool:
        """Check data is fresh."""
        if self.field_name not in data.columns:
            return False

        # Get latest timestamp
        latest = pd.to_datetime(data[self.field_name]).max()
        now = pd.Timestamp.now()

        age_seconds = (now - latest).total_seconds()

        if age_seconds > self.max_age_seconds:
            self.add_issue(
                "stale_data",
                ValidationSeverity.CRITICAL if age_seconds > self.max_age_seconds * 2 else ValidationSeverity.HIGH,
                f"<{self.max_age_seconds}s old",
                f"{age_seconds:.0f}s old",
                f"Data is {age_seconds:.0f} seconds old (max: {self.max_age_seconds}s)"
            )
            return False

        return True


class ConsistencyValidator:
    """Validate consistency across related fields."""

    def __init__(self):
        self.issues: List[DataQualityIssue] = []

    def validate_ohlc_consistency(self, data: pd.DataFrame) -> bool:
        """Validate OHLC relationships."""
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return False

        issues = 0

        # High >= Open, Close, Low
        invalid_high = (data['high'] < data['open']) | (data['high'] < data['close']) | (data['high'] < data['low'])
        issues += invalid_high.sum()

        # Low <= Open, Close, High
        invalid_low = (data['low'] > data['open']) | (data['low'] > data['close']) | (data['low'] > data['high'])
        issues += invalid_low.sum()

        if issues > 0:
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type="ohlc_inconsistency",
                severity=ValidationSeverity.CRITICAL,
                field_name="ohlc",
                expected_value="Low <= Open, Close <= High",
                actual_value=f"{issues} inconsistent records",
                description=f"Found {issues} records with invalid OHLC relationships"
            )
            self.issues.append(issue)
            logger.error(f"OHLC inconsistency detected: {issues} records")
            return False

        return True

    def validate_price_volume_relationship(self, data: pd.DataFrame) -> bool:
        """Check for suspicious price-volume patterns."""
        if 'close' not in data.columns or 'volume' not in data.columns:
            return True

        # Large price moves with zero volume are suspicious
        price_changes = data['close'].pct_change().abs()
        zero_volume = data['volume'] == 0

        suspicious = ((price_changes > 0.05) & zero_volume).sum()  # >5% move with no volume

        if suspicious > 0:
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type="price_volume_inconsistency",
                severity=ValidationSeverity.HIGH,
                field_name="price_volume",
                expected_value="Volume > 0 for large price moves",
                actual_value=f"{suspicious} records with large moves and zero volume",
                description=f"Found {suspicious} suspicious price-volume patterns"
            )
            self.issues.append(issue)
            return False

        return True


class DataQualityMonitor:
    """Main data quality monitoring system."""

    def __init__(self):
        self.validators: Dict[str, List[DataValidator]] = {}
        self.consistency_validator = ConsistencyValidator()

    def add_validator(self, data_source: str, validator: DataValidator):
        """Add validator for a data source."""
        if data_source not in self.validators:
            self.validators[data_source] = []
        self.validators[data_source].append(validator)

    def setup_market_data_validators(self):
        """Setup standard validators for market data."""
        validators = [
            # Price fields
            CompletenessValidator("close", max_missing_pct=0.0),
            RangeValidator("close", min_value=0.01, max_value=1000000),
            AnomalyDetector("close", zscore_threshold=5.0),

            CompletenessValidator("volume", max_missing_pct=0.01),
            RangeValidator("volume", min_value=0),

            # Timestamp
            TimelinessValidator("timestamp", max_age_seconds=300),
        ]

        for validator in validators:
            self.add_validator("market_data", validator)

        logger.info("Market data validators configured")

    def validate_data(
        self,
        data: pd.DataFrame,
        data_source: str = "market_data"
    ) -> DataQualityReport:
        """
        Run all validations on data.

        Args:
            data: DataFrame to validate
            data_source: Source identifier

        Returns:
            DataQualityReport with results
        """
        logger.info(f"Validating {len(data)} records from {data_source}")

        all_issues = []

        # Run field validators
        completeness_passed = 0
        accuracy_passed = 0
        total_validators = 0

        if data_source in self.validators:
            for validator in self.validators[data_source]:
                passed = validator.validate(data)
                all_issues.extend(validator.issues)

                if isinstance(validator, CompletenessValidator):
                    if passed:
                        completeness_passed += 1
                elif isinstance(validator, (RangeValidator, AnomalyDetector)):
                    if passed:
                        accuracy_passed += 1

                total_validators += 1

        # Run consistency validators
        ohlc_valid = self.consistency_validator.validate_ohlc_consistency(data)
        pv_valid = self.consistency_validator.validate_price_volume_relationship(data)
        all_issues.extend(self.consistency_validator.issues)

        consistency_score = (
            (100 if ohlc_valid else 50) +
            (100 if pv_valid else 50)
        ) / 2

        # Calculate scores
        completeness_score = (completeness_passed / max(1, total_validators // 2)) * 100
        accuracy_score = (accuracy_passed / max(1, total_validators // 2)) * 100

        # Timeliness score (check if data is fresh)
        timeliness_score = 100.0  # Default
        for issue in all_issues:
            if issue.issue_type == "stale_data":
                timeliness_score = 50.0 if issue.severity == ValidationSeverity.HIGH else 25.0

        # Overall score (weighted average)
        overall_score = (
            completeness_score * 0.3 +
            accuracy_score * 0.3 +
            consistency_score * 0.2 +
            timeliness_score * 0.2
        )

        # Determine quality level
        if overall_score >= 95:
            quality_level = DataQualityLevel.EXCELLENT
        elif overall_score >= 85:
            quality_level = DataQualityLevel.GOOD
        elif overall_score >= 70:
            quality_level = DataQualityLevel.ACCEPTABLE
        elif overall_score >= 50:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.UNACCEPTABLE

        # Count issues by severity
        critical_issues = sum(1 for issue in all_issues if issue.severity == ValidationSeverity.CRITICAL)
        high_issues = sum(1 for issue in all_issues if issue.severity == ValidationSeverity.HIGH)

        # Determine if safe to trade
        is_tradable = (
            critical_issues == 0 and
            overall_score >= 70 and
            quality_level not in [DataQualityLevel.UNACCEPTABLE, DataQualityLevel.POOR]
        )

        # Generate warnings and recommendations
        warnings = []
        recommendations = []

        if critical_issues > 0:
            warnings.append(f"CRITICAL: {critical_issues} critical data quality issues detected")
            recommendations.append("Stop trading immediately until data issues resolved")

        if high_issues > 0:
            warnings.append(f"WARNING: {high_issues} high-priority data issues")
            recommendations.append("Review data pipeline and consider reducing position sizes")

        if overall_score < 85:
            recommendations.append("Enhance data quality monitoring and add redundant data sources")

        if not is_tradable:
            warnings.append("Data quality insufficient for safe trading")

        report = DataQualityReport(
            timestamp=datetime.now(),
            data_source=data_source,
            total_records=len(data),
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            overall_score=overall_score,
            quality_level=quality_level,
            issues=all_issues,
            critical_issues=critical_issues,
            high_issues=high_issues,
            is_tradable=is_tradable,
            warnings=warnings,
            recommendations=recommendations
        )

        self._log_report(report)

        return report

    def _log_report(self, report: DataQualityReport):
        """Log data quality report."""
        logger.info(
            f"\n{'='*60}\n"
            f"Data Quality Report - {report.data_source}\n"
            f"{'='*60}\n"
            f"Records: {report.total_records}\n"
            f"Quality Level: {report.quality_level.value.upper()}\n"
            f"\n"
            f"Scores:\n"
            f"  Completeness: {report.completeness_score:.1f}%\n"
            f"  Accuracy: {report.accuracy_score:.1f}%\n"
            f"  Consistency: {report.consistency_score:.1f}%\n"
            f"  Timeliness: {report.timeliness_score:.1f}%\n"
            f"  Overall: {report.overall_score:.1f}%\n"
            f"\n"
            f"Issues: {len(report.issues)} ({report.critical_issues} critical, {report.high_issues} high)\n"
            f"Tradable: {'YES' if report.is_tradable else 'NO'}\n"
            f"{'='*60}"
        )

        if report.warnings:
            logger.warning("Warnings:")
            for warning in report.warnings:
                logger.warning(f"  - {warning}")

        if report.recommendations:
            logger.info("Recommendations:")
            for rec in report.recommendations:
                logger.info(f"  - {rec}")


# Singleton instance
data_quality_monitor = DataQualityMonitor()
data_quality_monitor.setup_market_data_validators()
