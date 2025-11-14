"""
Exercise 1: Online Learning and Model Drift Detection

Test and implement the online learning system for continuous model improvement.

Tasks:
1. Implement the 20 TODOs in features/online_learning.py
2. Set up a model monitoring pipeline
3. Simulate model drift and detect it
4. Trigger automatic model retraining
5. Validate new models before deployment
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from features.online_learning import (
    OnlineLearningEngine,
    ModelMonitor,
    DriftDetectionConfig,
    ModelMetrics,
    DriftType
)


def generate_sample_data(n_samples: int = 1000, drift: bool = False) -> pd.DataFrame:
    """Generate sample trading data with optional drift."""
    np.random.seed(42 if not drift else 123)

    # Features
    returns_lag1 = np.random.randn(n_samples) * 0.02
    returns_lag2 = np.random.randn(n_samples) * 0.02
    volume = np.random.lognormal(15, 1, n_samples)
    volatility = np.random.uniform(0.01, 0.05, n_samples)

    if drift:
        # Introduce concept drift: relationship changes
        returns_lag1 = returns_lag1 * 1.5  # Stronger momentum
        volatility = volatility * 1.3  # Higher volatility regime

    data = pd.DataFrame({
        'returns_lag1': returns_lag1,
        'returns_lag2': returns_lag2,
        'volume': volume,
        'volatility': volatility,
        'target': (returns_lag1 > 0).astype(int)  # Predict direction
    })

    return data


class SimpleModel:
    """Simple model for testing."""

    def __init__(self):
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.coefficients = None

    def fit(self, X, y):
        """Fit simple logistic regression."""
        # Simple implementation for testing
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X, y)
        self.coefficients = model.coef_
        return self

    def predict(self, X):
        """Make predictions."""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.coef_ = self.coefficients
        return model.predict(X)


def test_model_versioning():
    """Test model saving and loading with versions."""
    print("\n" + "="*60)
    print("Test 1: Model Versioning")
    print("="*60)

    # TODO: Implement this test
    # 1. Create OnlineLearningEngine
    # 2. Train a simple model
    # 3. Save model with metrics
    # 4. Load model by version
    # 5. Verify model works correctly

    print("✓ Test model versioning")


def test_drift_detection():
    """Test drift detection mechanisms."""
    print("\n" + "="*60)
    print("Test 2: Drift Detection")
    print("="*60)

    # TODO: Implement this test
    # 1. Generate baseline data
    # 2. Generate data with drift
    # 3. Run drift detection
    # 4. Verify drift is detected
    # 5. Check drift scores

    print("✓ Test drift detection")


def test_performance_monitoring():
    """Test performance metric tracking."""
    print("\n" + "="*60)
    print("Test 3: Performance Monitoring")
    print("="*60)

    # TODO: Implement this test
    # 1. Create engine and load model
    # 2. Make predictions on test data
    # 3. Update metrics
    # 4. Check metrics are tracked correctly
    # 5. Verify performance calculation

    print("✓ Test performance monitoring")


def test_automatic_retraining():
    """Test automatic retraining workflow."""
    print("\n" + "="*60)
    print("Test 4: Automatic Retraining")
    print("="*60)

    # TODO: Implement this test
    # 1. Set up engine with drift detection
    # 2. Simulate performance degradation
    # 3. Trigger drift detection
    # 4. Verify should_retrain returns True
    # 5. Execute retraining
    # 6. Validate new model

    print("✓ Test automatic retraining")


def test_model_rollback():
    """Test model rollback capability."""
    print("\n" + "="*60)
    print("Test 5: Model Rollback")
    print("="*60)

    # TODO: Implement this test
    # 1. Save multiple model versions
    # 2. Deploy latest version
    # 3. Rollback to previous version
    # 4. Verify correct version is active

    print("✓ Test model rollback")


def run_full_workflow():
    """Run complete online learning workflow."""
    print("\n" + "="*60)
    print("Full Workflow: Online Learning Pipeline")
    print("="*60)

    # Step 1: Initialize engine
    print("\n1. Initializing online learning engine...")
    config = DriftDetectionConfig(
        window_size=100,
        drift_threshold=0.15,
        performance_threshold=0.10,
        min_samples=50
    )

    engine = OnlineLearningEngine(
        model_name="direction_predictor",
        model_dir=Path("models/"),
        drift_config=config
    )

    # Step 2: Initial model training
    print("2. Training initial model...")
    baseline_data = generate_sample_data(n_samples=1000, drift=False)
    X_train = baseline_data.drop('target', axis=1)
    y_train = baseline_data['target']

    # TODO: Complete the workflow
    # - Train and save initial model
    # - Generate recent data with drift
    # - Make predictions and track performance
    # - Run drift detection
    # - Retrain if drift detected
    # - Deploy new model

    print("\n✓ Full workflow completed successfully!")


def test_model_monitor():
    """Test the model monitoring system."""
    print("\n" + "="*60)
    print("Test 6: Model Monitoring System")
    print("="*60)

    # TODO: Implement this test
    # 1. Create multiple OnlineLearningEngine instances
    # 2. Initialize ModelMonitor
    # 3. Start monitoring
    # 4. Get status dashboard
    # 5. Verify all models are being monitored

    print("✓ Test model monitoring")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 1: Online Learning and Model Drift Detection")
    print("="*60)

    # Run tests
    test_model_versioning()
    test_drift_detection()
    test_performance_monitoring()
    test_automatic_retraining()
    test_model_rollback()
    test_model_monitor()

    # Run full workflow
    run_full_workflow()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the implementation in features/online_learning.py")
    print("2. Implement all 20 TODOs")
    print("3. Run this test file to verify your implementation")
    print("4. Set up continuous monitoring for production models")
    print("5. Configure drift detection thresholds for your use case")
