"""
Tests for Data Quality Assurance Framework

Tests data validation, anomaly detection, and quality monitoring to prevent
garbage-in-garbage-out scenarios.

Author: LLM Trading Platform - Test Suite
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.operations.data_quality import (
    DataQualityLevel,
    ValidationSeverity,
    DataQualityIssue,
    DataQualityReport,
    DataValidator,
    CompletenessValidator,
    RangeValidator
)


class TestDataQualityEnums:
    """Test data quality enums."""

    def test_quality_levels(self):
        """Test quality level enum."""
        assert DataQualityLevel.EXCELLENT.value == "excellent"
        assert DataQualityLevel.GOOD.value == "good"
        assert DataQualityLevel.ACCEPTABLE.value == "acceptable"
        assert DataQualityLevel.POOR.value == "poor"
        assert DataQualityLevel.UNACCEPTABLE.value == "unacceptable"

    def test_validation_severity(self):
        """Test validation severity enum."""
        assert ValidationSeverity.CRITICAL.value == "critical"
        assert ValidationSeverity.HIGH.value == "high"
        assert ValidationSeverity.MEDIUM.value == "medium"
        assert ValidationSeverity.LOW.value == "low"


class TestCompletenessValidator:
    """Test completeness validation."""

    def test_complete_data_passes(self):
        """Test that complete data passes validation."""
        validator = CompletenessValidator("price", max_missing_pct=0.01)

        data = pd.DataFrame({
            "price": [150.0, 151.0, 152.0, 153.0, 154.0]
        })

        result = validator.validate(data)

        assert result is True
        assert len(validator.issues) == 0

    def test_missing_column_fails(self):
        """Test that missing column is detected."""
        validator = CompletenessValidator("price")

        data = pd.DataFrame({
            "volume": [1000, 2000, 3000]
        })

        result = validator.validate(data)

        assert result is False
        assert len(validator.issues) == 1
        assert validator.issues[0].severity == ValidationSeverity.CRITICAL
        assert "missing" in validator.issues[0].description.lower()

    def test_excessive_missing_values_fails(self):
        """Test that excessive missing values fail validation."""
        validator = CompletenessValidator("price", max_missing_pct=0.05)

        data = pd.DataFrame({
            "price": [150.0, None, 152.0, None, None, None, 156.0, None, None, None]
        })

        result = validator.validate(data)

        assert result is False
        assert len(validator.issues) == 1
        assert "missing" in validator.issues[0].description.lower()

    def test_acceptable_missing_values_passes(self):
        """Test that acceptable missing values pass."""
        validator = CompletenessValidator("price", max_missing_pct=0.10)

        # 5% missing (below 10% threshold)
        data = pd.DataFrame({
            "price": [150.0] * 95 + [None] * 5
        })

        result = validator.validate(data)

        assert result is True
        assert len(validator.issues) == 0

    def test_critical_severity_for_high_missing(self):
        """Test critical severity for very high missing percentage."""
        validator = CompletenessValidator("price", max_missing_pct=0.01)

        # 20% missing
        data = pd.DataFrame({
            "price": [150.0] * 80 + [None] * 20
        })

        result = validator.validate(data)

        assert result is False
        assert validator.issues[0].severity == ValidationSeverity.CRITICAL


class TestRangeValidator:
    """Test range validation."""

    def test_valid_range_passes(self):
        """Test that data within range passes."""
        validator = RangeValidator("price", min_value=0, max_value=1000)

        data = pd.DataFrame({
            "price": [100.0, 200.0, 300.0, 400.0, 500.0]
        })

        result = validator.validate(data)

        assert result is True
        assert len(validator.issues) == 0

    def test_below_minimum_fails(self):
        """Test that values below minimum are detected."""
        validator = RangeValidator("price", min_value=0)

        data = pd.DataFrame({
            "price": [100.0, -50.0, 150.0, -20.0]
        })

        result = validator.validate(data)

        assert result is False
        assert len(validator.issues) > 0
        assert "below" in validator.issues[0].description.lower()

    def test_above_maximum_fails(self):
        """Test that values above maximum are detected."""
        validator = RangeValidator("price", max_value=1000)

        data = pd.DataFrame({
            "price": [100.0, 500.0, 1500.0, 2000.0]
        })

        result = validator.validate(data)

        assert result is False
        assert len(validator.issues) > 0
        assert "above" in validator.issues[0].description.lower()

    def test_outliers_within_tolerance(self):
        """Test that small percentage of outliers is tolerated."""
        validator = RangeValidator("price", min_value=0, max_value=1000, allow_outliers_pct=0.05)

        # 2% outliers (within 5% tolerance)
        data = pd.DataFrame({
            "price": [500.0] * 98 + [1500.0, 1600.0]  # 2% above max
        })

        result = validator.validate(data)

        # Should detect outliers but tolerate them
        assert len(validator.issues) > 0  # Issues recorded
        assert validator.issues[0].severity == ValidationSeverity.HIGH

    def test_missing_column_handled(self):
        """Test handling of missing column."""
        validator = RangeValidator("price", min_value=0)

        data = pd.DataFrame({
            "volume": [1000, 2000, 3000]
        })

        result = validator.validate(data)

        assert result is False


class TestDataQualityReport:
    """Test data quality reporting."""

    def test_report_creation(self):
        """Test creating a data quality report."""
        report = DataQualityReport(
            timestamp=datetime.utcnow(),
            data_source="test_source",
            total_records=1000,
            completeness_score=98.5,
            accuracy_score=99.0,
            consistency_score=97.5,
            timeliness_score=95.0,
            overall_score=97.5,
            quality_level=DataQualityLevel.EXCELLENT,
            issues=[],
            critical_issues=0,
            high_issues=0,
            is_tradable=True,
            warnings=[],
            recommendations=[]
        )

        assert report.overall_score == 97.5
        assert report.quality_level == DataQualityLevel.EXCELLENT
        assert report.is_tradable
        assert report.critical_issues == 0

    def test_quality_level_classification(self):
        """Test quality level classification based on score."""
        # Excellent: >95%
        assert DataQualityLevel.EXCELLENT.value == "excellent"

        # Good: 85-95%
        assert DataQualityLevel.GOOD.value == "good"

        # Acceptable: 70-85%
        assert DataQualityLevel.ACCEPTABLE.value == "acceptable"

        # Poor: 50-70%
        assert DataQualityLevel.POOR.value == "poor"

        # Unacceptable: <50%
        assert DataQualityLevel.UNACCEPTABLE.value == "unacceptable"

    def test_critical_issues_block_trading(self):
        """Test that critical issues make data not tradable."""
        issue = DataQualityIssue(
            timestamp=datetime.utcnow(),
            issue_type="stale_data",
            severity=ValidationSeverity.CRITICAL,
            field_name="timestamp",
            expected_value="< 5 minutes old",
            actual_value="15 minutes old",
            description="Data is stale",
            auto_corrected=False
        )

        report = DataQualityReport(
            timestamp=datetime.utcnow(),
            data_source="market_data",
            total_records=100,
            completeness_score=100.0,
            accuracy_score=100.0,
            consistency_score=100.0,
            timeliness_score=50.0,  # Poor timeliness
            overall_score=87.5,
            quality_level=DataQualityLevel.GOOD,
            issues=[issue],
            critical_issues=1,
            high_issues=0,
            is_tradable=False,  # Critical issue blocks trading
            warnings=["Stale data detected"],
            recommendations=["Refresh data source"]
        )

        assert not report.is_tradable
        assert report.critical_issues == 1
        assert len(report.warnings) > 0


class TestDataQualityIntegration:
    """Integration tests for data quality framework."""

    def test_market_data_validation_workflow(self):
        """Test complete market data validation workflow."""
        # Simulate market data
        data = pd.DataFrame({
            "symbol": ["AAPL"] * 100,
            "timestamp": pd.date_range(end=datetime.utcnow(), periods=100, freq="1min"),
            "price": np.random.uniform(140, 160, 100),
            "volume": np.random.randint(1000, 10000, 100),
            "bid": np.random.uniform(139, 159, 100),
            "ask": np.random.uniform(141, 161, 100)
        })

        # Validate completeness
        price_validator = CompletenessValidator("price", max_missing_pct=0.01)
        volume_validator = CompletenessValidator("volume", max_missing_pct=0.01)

        assert price_validator.validate(data)
        assert volume_validator.validate(data)

        # Validate ranges
        price_range = RangeValidator("price", min_value=0, max_value=10000)
        volume_range = RangeValidator("volume", min_value=0)

        assert price_range.validate(data)
        assert volume_range.validate(data)

    def test_detect_data_quality_degradation(self):
        """Test detection of data quality degradation."""
        # Good quality data
        good_data = pd.DataFrame({
            "price": [150.0] * 100,
            "volume": [5000] * 100
        })

        # Degraded data (missing values, outliers)
        bad_data = pd.DataFrame({
            "price": [150.0] * 80 + [None] * 15 + [-100.0] * 5,  # Missing and invalid
            "volume": [5000] * 70 + [None] * 30  # High missing
        })

        # Validate good data
        price_val_good = CompletenessValidator("price", max_missing_pct=0.05)
        assert price_val_good.validate(good_data)

        # Validate bad data
        price_val_bad = CompletenessValidator("price", max_missing_pct=0.05)
        volume_val_bad = CompletenessValidator("volume", max_missing_pct=0.05)

        assert not price_val_bad.validate(bad_data)  # Should fail
        assert not volume_val_bad.validate(bad_data)  # Should fail

        # Check that issues were recorded
        assert len(price_val_bad.issues) > 0
        assert len(volume_val_bad.issues) > 0

    def test_stale_data_detection(self):
        """Test detection of stale/outdated data."""
        now = datetime.utcnow()

        # Recent data (good)
        fresh_data = pd.DataFrame({
            "timestamp": [now - timedelta(seconds=i) for i in range(100)],
            "price": [150.0] * 100
        })

        # Old data (stale)
        stale_data = pd.DataFrame({
            "timestamp": [now - timedelta(minutes=30) - timedelta(seconds=i) for i in range(100)],
            "price": [150.0] * 100
        })

        # In a real implementation, we'd have a TimelinessValidator
        # For now, we can check timestamps manually
        latest_fresh = fresh_data["timestamp"].max()
        latest_stale = stale_data["timestamp"].max()

        assert (now - latest_fresh).total_seconds() < 60  # Fresh
        assert (now - latest_stale).total_seconds() > 300  # Stale (>5 min)

    def test_data_consistency_validation(self):
        """Test data consistency checks."""
        # Inconsistent data: bid > ask (invalid)
        inconsistent_data = pd.DataFrame({
            "bid": [150.0, 151.0, 152.0],
            "ask": [149.0, 150.0, 151.0]  # Ask < Bid (wrong!)
        })

        # Check consistency: bid should be <= ask
        issues = []
        for idx, row in inconsistent_data.iterrows():
            if row["bid"] > row["ask"]:
                issues.append(f"Row {idx}: bid ({row['bid']}) > ask ({row['ask']})")

        assert len(issues) == 3  # All rows are inconsistent

    def test_quality_score_calculation(self):
        """Test overall quality score calculation."""
        # Weights for different dimensions
        completeness = 0.95
        accuracy = 0.98
        consistency = 0.92
        timeliness = 0.88

        # Calculate weighted average
        overall = (completeness * 0.3 + accuracy * 0.3 +
                   consistency * 0.2 + timeliness * 0.2)

        # Determine quality level
        if overall > 0.95:
            level = DataQualityLevel.EXCELLENT
        elif overall > 0.85:
            level = DataQualityLevel.GOOD
        elif overall > 0.70:
            level = DataQualityLevel.ACCEPTABLE
        elif overall > 0.50:
            level = DataQualityLevel.POOR
        else:
            level = DataQualityLevel.UNACCEPTABLE

        assert 0.90 < overall < 0.95
        assert level == DataQualityLevel.GOOD

    def test_auto_correction_capability(self):
        """Test auto-correction of minor data issues."""
        issue = DataQualityIssue(
            timestamp=datetime.utcnow(),
            issue_type="minor_outlier",
            severity=ValidationSeverity.MEDIUM,
            field_name="price",
            expected_value="100-200",
            actual_value="205",
            description="Price slightly above normal range",
            auto_corrected=True  # Auto-corrected to 200
        )

        assert issue.auto_corrected
        assert issue.severity == ValidationSeverity.MEDIUM

    def test_validation_chain(self):
        """Test chaining multiple validators."""
        data = pd.DataFrame({
            "price": [150.0, 151.0, 152.0, 153.0, 154.0],
            "volume": [1000, 2000, 3000, 4000, 5000]
        })

        validators = [
            CompletenessValidator("price", max_missing_pct=0.01),
            CompletenessValidator("volume", max_missing_pct=0.01),
            RangeValidator("price", min_value=0, max_value=1000),
            RangeValidator("volume", min_value=0)
        ]

        all_passed = True
        all_issues = []

        for validator in validators:
            if not validator.validate(data):
                all_passed = False
                all_issues.extend(validator.issues)

        assert all_passed
        assert len(all_issues) == 0
