"""
Exercise 3: Portfolio Optimization

Test and implement the portfolio optimization system for multi-strategy allocation.

Tasks:
1. Implement the 20 TODOs in features/portfolio_optimizer.py
2. Optimize portfolio using different methods
3. Calculate efficient frontier
4. Implement dynamic rebalancing
5. Run stress tests and Monte Carlo simulations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from features.portfolio_optimizer import (
    PortfolioOptimizer,
    Strategy,
    OptimizationConfig
)


def generate_strategy_returns(n_days: int = 252) -> pd.DataFrame:
    """Generate synthetic strategy returns."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # Correlated returns for different strategies
    mean_returns = np.array([0.0006, 0.0005, 0.0003, 0.0007])  # Daily returns
    volatilities = np.array([0.015, 0.012, 0.008, 0.020])

    # Correlation matrix
    corr_matrix = np.array([
        [1.00, 0.30, 0.10, 0.40],
        [0.30, 1.00, 0.15, 0.25],
        [0.10, 0.15, 1.00, 0.05],
        [0.40, 0.25, 0.05, 1.00]
    ])

    # Covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)

    df = pd.DataFrame(
        returns,
        columns=['momentum', 'mean_reversion', 'market_making', 'sentiment'],
        index=dates
    )

    return df


def test_covariance_calculation():
    """Test covariance matrix calculation."""
    print("\n" + "="*60)
    print("Test 1: Covariance Matrix Calculation")
    print("="*60)

    # TODO: Implement this test
    # 1. Create strategies with historical returns
    # 2. Calculate covariance matrix
    # 3. Verify matrix is symmetric
    # 4. Check diagonal elements (variances)

    print("✓ Test covariance calculation")


def test_sharpe_optimization():
    """Test Sharpe ratio optimization."""
    print("\n" + "="*60)
    print("Test 2: Sharpe Ratio Optimization")
    print("="*60)

    # TODO: Implement this test
    # 1. Create optimizer with sample strategies
    # 2. Run Sharpe optimization
    # 3. Verify allocations sum to 1
    # 4. Check Sharpe ratio is maximized

    print("✓ Test Sharpe optimization")


def test_minimum_variance():
    """Test minimum variance optimization."""
    print("\n" + "="*60)
    print("Test 3: Minimum Variance Optimization")
    print("="*60)

    # TODO: Implement this test
    # 1. Create optimizer
    # 2. Run minimum variance optimization
    # 3. Verify portfolio variance is minimized
    # 4. Compare to equal-weight portfolio

    print("✓ Test minimum variance")


def test_risk_parity():
    """Test risk parity allocation."""
    print("\n" + "="*60)
    print("Test 4: Risk Parity Allocation")
    print("="*60)

    # TODO: Implement this test
    # 1. Create optimizer
    # 2. Run risk parity optimization
    # 3. Verify each strategy contributes equally to risk
    # 4. Check total risk allocation

    print("✓ Test risk parity")


def test_efficient_frontier():
    """Test efficient frontier calculation."""
    print("\n" + "="*60)
    print("Test 5: Efficient Frontier")
    print("="*60)

    # TODO: Implement this test
    # 1. Create optimizer
    # 2. Calculate efficient frontier
    # 3. Verify return increases with risk
    # 4. Plot frontier (optional)

    print("✓ Test efficient frontier")


def test_portfolio_metrics():
    """Test portfolio metrics calculation."""
    print("\n" + "="*60)
    print("Test 6: Portfolio Metrics")
    print("="*60)

    # TODO: Implement this test
    # 1. Create portfolio with known allocations
    # 2. Calculate metrics
    # 3. Verify expected return, volatility, Sharpe
    # 4. Check VaR and CVaR

    print("✓ Test portfolio metrics")


def test_rebalancing():
    """Test portfolio rebalancing logic."""
    print("\n" + "="*60)
    print("Test 7: Portfolio Rebalancing")
    print("="*60)

    # TODO: Implement this test
    # 1. Set current and target allocations
    # 2. Calculate rebalancing trades
    # 3. Verify transaction costs
    # 4. Check should_rebalance logic

    print("✓ Test rebalancing")


def test_constraints():
    """Test allocation constraints."""
    print("\n" + "="*60)
    print("Test 8: Allocation Constraints")
    print("="*60)

    # TODO: Implement this test
    # 1. Set min/max allocation constraints
    # 2. Run optimization
    # 3. Verify constraints are respected
    # 4. Test leverage limit

    print("✓ Test constraints")


def test_monte_carlo():
    """Test Monte Carlo simulation."""
    print("\n" + "="*60)
    print("Test 9: Monte Carlo Simulation")
    print("="*60)

    # TODO: Implement this test
    # 1. Create optimizer with allocations
    # 2. Run Monte Carlo simulation
    # 3. Analyze distribution of outcomes
    # 4. Check percentiles (5th, 50th, 95th)

    print("✓ Test Monte Carlo simulation")


def test_stress_testing():
    """Test portfolio stress testing."""
    print("\n" + "="*60)
    print("Test 10: Stress Testing")
    print("="*60)

    # TODO: Implement this test
    # 1. Define stress scenarios (crash, volatility spike, etc.)
    # 2. Run stress test
    # 3. Analyze portfolio impact
    # 4. Identify worst-case scenario

    print("✓ Test stress testing")


def run_full_optimization():
    """Run complete portfolio optimization workflow."""
    print("\n" + "="*60)
    print("Full Workflow: Multi-Strategy Portfolio Optimization")
    print("="*60)

    # Step 1: Define strategies
    print("\n1. Defining strategies...")
    strategies = [
        Strategy(
            name="momentum",
            expected_return=0.15,
            volatility=0.20,
            sharpe_ratio=0.75,
            max_drawdown=-0.25,
            min_allocation=0.1,
            max_allocation=0.5
        ),
        Strategy(
            name="mean_reversion",
            expected_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.80,
            max_drawdown=-0.18,
            min_allocation=0.1,
            max_allocation=0.5
        ),
        Strategy(
            name="market_making",
            expected_return=0.08,
            volatility=0.10,
            sharpe_ratio=0.80,
            max_drawdown=-0.12,
            min_allocation=0.0,
            max_allocation=0.4
        ),
        Strategy(
            name="sentiment",
            expected_return=0.18,
            volatility=0.25,
            sharpe_ratio=0.72,
            max_drawdown=-0.30,
            min_allocation=0.0,
            max_allocation=0.3
        )
    ]

    # Step 2: Generate historical returns
    print("2. Generating historical returns...")
    returns_df = generate_strategy_returns(n_days=252)

    # Add returns to strategies
    for strategy in strategies:
        if strategy.name in returns_df.columns:
            strategy.historical_returns = returns_df[strategy.name].tolist()

    # Step 3: Create optimizer
    print("3. Creating portfolio optimizer...")
    config = OptimizationConfig(
        objective="sharpe",
        risk_free_rate=0.02,
        max_leverage=1.0
    )

    optimizer = PortfolioOptimizer(strategies, config)

    # Step 4: Calculate inputs
    print("4. Calculating covariance and returns...")
    # TODO: Implement
    # - Calculate covariance matrix
    # - Calculate expected returns

    # Step 5: Run optimizations
    print("\n5. Running optimizations...")
    # TODO: Implement
    # - Max Sharpe ratio
    # - Minimum variance
    # - Risk parity
    # - Compare results

    # Step 6: Calculate efficient frontier
    print("\n6. Calculating efficient frontier...")
    # TODO: Implement

    # Step 7: Backtest best allocation
    print("\n7. Backtesting optimal allocation...")
    # TODO: Implement

    # Step 8: Stress test
    print("\n8. Running stress tests...")
    # TODO: Implement

    # Step 9: Generate report
    print("\n9. Generating allocation report...")
    # TODO: Implement

    print("\n✓ Full optimization workflow completed!")


def compare_optimization_methods():
    """Compare different optimization methods."""
    print("\n" + "="*60)
    print("Bonus: Compare Optimization Methods")
    print("="*60)

    # TODO: Implement comparison
    # 1. Run all optimization methods on same data
    # 2. Compare allocations
    # 3. Compare risk/return metrics
    # 4. Analyze differences
    # 5. Recommend best method for different goals

    print("✓ Comparison completed")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 3: Portfolio Optimization")
    print("="*60)

    # Run tests
    test_covariance_calculation()
    test_sharpe_optimization()
    test_minimum_variance()
    test_risk_parity()
    test_efficient_frontier()
    test_portfolio_metrics()
    test_rebalancing()
    test_constraints()
    test_monte_carlo()
    test_stress_testing()

    # Run full workflow
    run_full_optimization()

    # Bonus
    compare_optimization_methods()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the implementation in features/portfolio_optimizer.py")
    print("2. Implement all 20 TODOs")
    print("3. Run this test file to verify your implementation")
    print("4. Apply optimization to your real strategies")
    print("5. Set up automated rebalancing")
    print("\nKey concepts:")
    print("- Modern Portfolio Theory (MPT)")
    print("- Sharpe ratio maximization")
    print("- Risk parity vs equal weight")
    print("- Transaction costs matter")
    print("- Diversification reduces risk")
    print("- Rebalance frequency is a trade-off")
