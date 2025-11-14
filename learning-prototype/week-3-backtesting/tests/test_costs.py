"""
Unit tests for transaction costs module.

Run with: pytest tests/test_costs.py -v
"""

import sys
sys.path.append('../starter-code')
sys.path.append('../solutions')

import pytest
import pandas as pd
import numpy as np
from transaction_costs_complete import (
    TransactionCostModel,
    TransactionCostConfig,
    CostBreakdown,
    bps_to_percentage,
    estimate_bid_ask_spread
)


@pytest.fixture
def default_config():
    """Create default cost configuration."""
    return TransactionCostConfig(
        commission_fixed=1.0,
        commission_pct=0.001,
        spread_bps=2.0,
        slippage_bps=5.0,
        slippage_volatility_factor=0.1,
        market_impact_factor=0.01
    )


@pytest.fixture
def cost_model(default_config):
    """Create a TransactionCostModel instance."""
    return TransactionCostModel(default_config)


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    prices = pd.Series(100 * (1 + np.random.randn(len(dates)) * 0.01).cumprod(), index=dates)
    return prices


class TestTransactionCostModel:
    """Test suite for TransactionCostModel class."""

    def test_initialization(self, cost_model, default_config):
        """Test that cost model initializes correctly."""
        assert cost_model is not None
        assert cost_model.config == default_config

    def test_initialization_with_defaults(self):
        """Test initialization with default config."""
        model = TransactionCostModel()
        assert model.config is not None
        assert isinstance(model.config, TransactionCostConfig)

    def test_calculate_commission(self, cost_model):
        """Test commission calculation."""
        trade_value = 10000.0
        commission = cost_model.calculate_commission(trade_value)

        # Should be fixed + percentage
        expected = 1.0 + (10000 * 0.001)
        assert abs(commission - expected) < 0.01

    def test_calculate_spread_cost(self, cost_model):
        """Test spread cost calculation."""
        trade_value = 10000.0
        spread = cost_model.calculate_spread_cost(trade_value)

        # 2 bps = 0.02% = 0.0002
        expected = 10000 * (2.0 / 10000)
        assert abs(spread - expected) < 0.01

    def test_calculate_slippage(self, cost_model):
        """Test slippage calculation."""
        trade_value = 10000.0

        # Low volatility
        slippage_low = cost_model.calculate_slippage(trade_value, volatility=0.01)
        assert slippage_low > 0

        # High volatility
        slippage_high = cost_model.calculate_slippage(trade_value, volatility=0.05)
        assert slippage_high > slippage_low  # Higher vol = more slippage

    def test_calculate_market_impact(self, cost_model):
        """Test market impact calculation."""
        trade_value = 10000.0

        # Small order (0.1% of volume)
        impact_small = cost_model.calculate_market_impact(
            trade_value, volume_participation=0.001
        )
        assert impact_small > 0

        # Large order (10% of volume)
        impact_large = cost_model.calculate_market_impact(
            trade_value, volume_participation=0.10
        )
        assert impact_large > impact_small  # Larger orders = more impact

    def test_calculate_total_cost(self, cost_model):
        """Test total cost calculation."""
        trade_value = 10000.0
        costs = cost_model.calculate_total_cost(
            trade_value=trade_value,
            volatility=0.02,
            volume_participation=0.01
        )

        assert isinstance(costs, CostBreakdown)
        assert costs.commission > 0
        assert costs.spread > 0
        assert costs.slippage > 0
        assert costs.market_impact > 0

        # Total should be sum of components
        expected_total = (
            costs.commission +
            costs.spread +
            costs.slippage +
            costs.market_impact
        )
        assert abs(costs.total_cost - expected_total) < 0.01

        # Total percentage should be correct
        expected_pct = costs.total_cost / trade_value
        assert abs(costs.total_cost_pct - expected_pct) < 0.0001

    def test_apply_costs_to_returns(self, cost_model, sample_prices):
        """Test applying costs to returns."""
        returns = sample_prices.pct_change().dropna()
        positions = pd.Series(1, index=returns.index)  # Always long

        adjusted_returns = cost_model.apply_costs_to_returns(
            returns, positions, sample_prices[1:], initial_capital=100000
        )

        assert len(adjusted_returns) == len(returns)
        # Adjusted returns should be slightly lower due to costs
        assert adjusted_returns.sum() <= returns.sum()

    def test_estimate_annual_costs(self, cost_model):
        """Test annual cost estimation."""
        costs = cost_model.estimate_annual_costs(
            avg_trade_value=10000,
            trades_per_year=100
        )

        assert isinstance(costs, dict)
        assert 'cost_per_trade' in costs
        assert 'trades_per_year' in costs
        assert 'annual_cost' in costs
        assert 'cost_percentage' in costs
        assert 'breakeven_return' in costs

        # Annual cost should be cost_per_trade * trades_per_year
        assert abs(costs['annual_cost'] - costs['cost_per_trade'] * 100) < 0.01

    def test_compare_strategies(self, cost_model):
        """Test strategy comparison."""
        strategies = {
            'Buy & Hold': {'avg_trade_value': 100000, 'trades_per_year': 2},
            'Day Trading': {'avg_trade_value': 10000, 'trades_per_year': 500}
        }

        comparison = cost_model.compare_strategies(strategies)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'strategy' in comparison.columns
        assert 'annual_cost' in comparison.columns

        # Day trading should have higher annual costs
        buy_hold_cost = comparison[comparison['strategy'] == 'Buy & Hold']['annual_cost'].iloc[0]
        day_trade_cost = comparison[comparison['strategy'] == 'Day Trading']['annual_cost'].iloc[0]
        assert day_trade_cost > buy_hold_cost


class TestCostConfiguration:
    """Test different cost configurations."""

    def test_retail_costs(self):
        """Test retail trader cost configuration."""
        retail_config = TransactionCostConfig(
            commission_fixed=1.0,
            commission_pct=0.001,
            spread_bps=5.0,
            slippage_bps=10.0
        )
        model = TransactionCostModel(retail_config)

        costs = model.calculate_total_cost(10000)
        assert costs.total_cost > 20  # Should be significant

    def test_professional_costs(self):
        """Test professional trader cost configuration."""
        pro_config = TransactionCostConfig(
            commission_fixed=0.0,
            commission_pct=0.0002,
            spread_bps=1.0,
            slippage_bps=2.0
        )
        model = TransactionCostModel(pro_config)

        costs = model.calculate_total_cost(10000)
        assert costs.total_cost < 10  # Should be lower

    def test_zero_costs(self):
        """Test with zero costs (theoretical)."""
        zero_config = TransactionCostConfig(
            commission_fixed=0.0,
            commission_pct=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            market_impact_factor=0.0
        )
        model = TransactionCostModel(zero_config)

        costs = model.calculate_total_cost(10000)
        assert costs.total_cost == 0.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_bps_to_percentage(self):
        """Test basis points to percentage conversion."""
        assert abs(bps_to_percentage(100) - 0.01) < 0.0001  # 100 bps = 1%
        assert abs(bps_to_percentage(50) - 0.005) < 0.0001  # 50 bps = 0.5%
        assert abs(bps_to_percentage(1) - 0.0001) < 0.00001  # 1 bp = 0.01%

    def test_estimate_bid_ask_spread(self, sample_prices):
        """Test bid-ask spread estimation."""
        spread = estimate_bid_ask_spread(sample_prices, window=20)

        assert isinstance(spread, pd.Series)
        assert len(spread) == len(sample_prices)
        assert (spread >= 0).all()  # Spread is always positive


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_trade_value(self, cost_model):
        """Test with zero trade value."""
        costs = cost_model.calculate_total_cost(0.0)
        assert costs.total_cost >= 0  # Should handle gracefully

    def test_very_large_trade(self, cost_model):
        """Test with very large trade value."""
        costs = cost_model.calculate_total_cost(1000000000)  # $1 billion
        assert costs.total_cost > 0
        assert costs.total_cost < 1000000000  # Costs shouldn't exceed trade value

    def test_negative_volatility_handling(self, cost_model):
        """Test handling of negative volatility (shouldn't happen but handle it)."""
        # Should not crash
        slippage = cost_model.calculate_slippage(10000, volatility=-0.01)
        assert slippage >= 0  # Should treat as absolute value or zero

    def test_zero_volume_participation(self, cost_model):
        """Test with zero volume participation."""
        impact = cost_model.calculate_market_impact(10000, volume_participation=0.0)
        assert impact == 0.0

    def test_no_position_changes(self, cost_model, sample_prices):
        """Test applying costs when there are no position changes."""
        returns = sample_prices.pct_change().dropna()
        positions = pd.Series(1, index=returns.index)  # Constant position (no changes)
        positions.iloc[0] = 1  # Initial position

        # Make sure no position changes after first position
        adjusted_returns = cost_model.apply_costs_to_returns(
            returns, positions, sample_prices[1:], initial_capital=100000
        )

        # First period should have costs (initial position)
        # But subsequent periods should have minimal cost impact
        assert len(adjusted_returns) == len(returns)


class TestCostImpact:
    """Test the impact of costs on strategy returns."""

    def test_cost_impact_on_high_frequency_strategy(self, cost_model, sample_prices):
        """Test that costs significantly impact high-frequency strategies."""
        returns = sample_prices.pct_change().dropna()

        # High-frequency: change position every day
        positions = pd.Series([-1, 1] * (len(returns) // 2), index=returns.index[:len(returns)])
        if len(positions) < len(returns):
            positions = pd.concat([positions, pd.Series([1], index=[returns.index[-1]])])

        adjusted_returns = cost_model.apply_costs_to_returns(
            returns, positions[:len(returns)], sample_prices[1:len(returns)+1], initial_capital=100000
        )

        # Total cost should be significant
        cost_impact = returns.sum() - adjusted_returns.sum()
        assert cost_impact > 0  # Costs reduce returns

    def test_cost_impact_on_buy_and_hold(self, cost_model, sample_prices):
        """Test that costs minimally impact buy-and-hold."""
        returns = sample_prices.pct_change().dropna()

        # Buy and hold: position = 1 throughout
        positions = pd.Series(1, index=returns.index)

        adjusted_returns = cost_model.apply_costs_to_returns(
            returns, positions, sample_prices[1:], initial_capital=100000
        )

        # Total cost should be minimal (only initial buy)
        cost_impact = returns.sum() - adjusted_returns.sum()
        assert cost_impact >= 0  # Some cost from initial position
        assert cost_impact < returns.sum() * 0.01  # But < 1% of total returns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
