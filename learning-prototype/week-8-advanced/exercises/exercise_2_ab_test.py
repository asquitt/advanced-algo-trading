"""
Exercise 2: A/B Testing Framework

Test and implement the A/B testing system for comparing trading strategies.

Tasks:
1. Implement the 15 TODOs in features/ab_testing.py
2. Design an A/B test for strategy comparison
3. Run traffic allocation and tracking
4. Calculate statistical significance
5. Make data-driven deployment decisions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from features.ab_testing import (
    ABTestFramework,
    ABTestConfig,
    Variant,
    VariantType,
    MultiArmedBandit
)


def simulate_strategy_performance(variant_config: dict, n_trades: int = 100) -> list:
    """Simulate trading performance for a variant."""
    np.random.seed(hash(variant_config['strategy']) % 2**32)

    # Different strategies have different characteristics
    if variant_config.get('lookback') == 20:
        # Control: moderate performance
        win_rate = 0.52
        avg_win = 0.015
        avg_loss = -0.010
    else:
        # Treatment: slightly better performance
        win_rate = 0.55
        avg_win = 0.016
        avg_loss = -0.010

    results = []
    for _ in range(n_trades):
        is_win = np.random.random() < win_rate
        pnl = np.random.normal(avg_win, 0.005) if is_win else np.random.normal(avg_loss, 0.004)
        results.append(pnl)

    return results


def test_variant_assignment():
    """Test consistent variant assignment."""
    print("\n" + "="*60)
    print("Test 1: Variant Assignment")
    print("="*60)

    # TODO: Implement this test
    # 1. Create ABTestFramework with 2 variants
    # 2. Assign same identifier multiple times
    # 3. Verify assignment is consistent (same variant each time)
    # 4. Check traffic allocation percentages are respected

    print("✓ Test variant assignment")


def test_impression_tracking():
    """Test impression tracking."""
    print("\n" + "="*60)
    print("Test 2: Impression Tracking")
    print("="*60)

    # TODO: Implement this test
    # 1. Create test framework
    # 2. Track impressions for various identifiers
    # 3. Verify impression counts increase correctly
    # 4. Check per-variant impression tracking

    print("✓ Test impression tracking")


def test_conversion_tracking():
    """Test conversion tracking."""
    print("\n" + "="*60)
    print("Test 3: Conversion Tracking")
    print("="*60)

    # TODO: Implement this test
    # 1. Create test framework
    # 2. Track conversions with values
    # 3. Verify conversion counts and values
    # 4. Calculate conversion rates

    print("✓ Test conversion tracking")


def test_statistical_significance():
    """Test statistical significance calculation."""
    print("\n" + "="*60)
    print("Test 4: Statistical Significance")
    print("="*60)

    # TODO: Implement this test
    # 1. Create test with known data
    # 2. Set up control and treatment variants
    # 3. Track enough conversions for significance
    # 4. Calculate statistical significance
    # 5. Verify p-value and effect size

    print("✓ Test statistical significance")


def test_confidence_intervals():
    """Test confidence interval calculation."""
    print("\n" + "="*60)
    print("Test 5: Confidence Intervals")
    print("="*60)

    # TODO: Implement this test
    # 1. Create test with sample data
    # 2. Calculate confidence intervals
    # 3. Verify CI bounds are reasonable
    # 4. Check CI width decreases with more samples

    print("✓ Test confidence intervals")


def test_winner_determination():
    """Test winner determination logic."""
    print("\n" + "="*60)
    print("Test 6: Winner Determination")
    print("="*60)

    # TODO: Implement this test
    # 1. Create test with clear winner
    # 2. Run enough samples
    # 3. Determine winner
    # 4. Verify correct variant is selected

    print("✓ Test winner determination")


def test_multi_armed_bandit():
    """Test multi-armed bandit optimization."""
    print("\n" + "="*60)
    print("Test 7: Multi-Armed Bandit")
    print("="*60)

    # TODO: Implement this test
    # 1. Create MultiArmedBandit with 3 variants
    # 2. Simulate different success rates
    # 3. Update bandit with results
    # 4. Verify it converges to best variant

    print("✓ Test multi-armed bandit")


def run_full_ab_test():
    """Run complete A/B test workflow."""
    print("\n" + "="*60)
    print("Full Workflow: A/B Test for Strategy Optimization")
    print("="*60)

    # Step 1: Define variants
    print("\n1. Defining test variants...")
    control = Variant(
        name="momentum_20d",
        variant_type=VariantType.CONTROL,
        traffic_allocation=0.5,
        config={"strategy": "momentum", "lookback": 20}
    )

    treatment = Variant(
        name="momentum_30d",
        variant_type=VariantType.TREATMENT,
        traffic_allocation=0.5,
        config={"strategy": "momentum", "lookback": 30}
    )

    # Step 2: Create test
    print("2. Creating A/B test...")
    test_config = ABTestConfig(
        name="momentum_lookback_optimization",
        description="Compare 20-day vs 30-day momentum lookback",
        start_date=datetime.now(),
        min_sample_size=1000,
        confidence_level=0.95,
        minimum_detectable_effect=0.05
    )

    ab_test = ABTestFramework(test_config)
    ab_test.add_variant(control)
    ab_test.add_variant(treatment)
    ab_test.start_test()

    # Step 3: Simulate traffic
    print("3. Simulating trading activity...")
    n_trades = 2000

    for trade_id in range(n_trades):
        identifier = f"trade_{trade_id}"

        # Assign variant
        variant = ab_test.assign_variant(identifier)
        config = ab_test.get_variant_config(identifier)

        # Track impression
        ab_test.track_impression(identifier)

        # Simulate trade result
        pnl = simulate_strategy_performance(config, n_trades=1)[0]

        # Track conversion (profitable trade)
        if pnl > 0:
            ab_test.track_conversion(identifier, value=pnl)

        # Track PnL as custom metric
        ab_test.track_metric(identifier, "pnl", pnl)

    # Step 4: Analyze results
    print("\n4. Analyzing results...")

    # TODO: Complete the analysis
    # - Calculate performance metrics for each variant
    # - Determine statistical significance
    # - Calculate confidence intervals
    # - Determine winner
    # - Generate report
    # - Export results

    print("\n✓ Full A/B test workflow completed!")


def run_dynamic_allocation_test():
    """Test dynamic traffic allocation with bandit."""
    print("\n" + "="*60)
    print("Bonus: Dynamic Traffic Allocation")
    print("="*60)

    # TODO: Implement dynamic allocation
    # 1. Start with equal traffic split
    # 2. Use MultiArmedBandit to adjust allocation
    # 3. Show traffic shifts to better variant
    # 4. Compare to fixed allocation

    print("✓ Test dynamic allocation")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 2: A/B Testing Framework")
    print("="*60)

    # Run tests
    test_variant_assignment()
    test_impression_tracking()
    test_conversion_tracking()
    test_statistical_significance()
    test_confidence_intervals()
    test_winner_determination()
    test_multi_armed_bandit()

    # Run full workflow
    run_full_ab_test()

    # Bonus: dynamic allocation
    run_dynamic_allocation_test()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the implementation in features/ab_testing.py")
    print("2. Implement all 15 TODOs")
    print("3. Run this test file to verify your implementation")
    print("4. Design real A/B tests for your strategies")
    print("5. Set up continuous A/B testing pipeline")
    print("\nBest practices:")
    print("- Always have a clear hypothesis")
    print("- Calculate required sample size before starting")
    print("- Don't stop tests too early")
    print("- Consider sequential testing for faster decisions")
    print("- Monitor for novelty effects")
