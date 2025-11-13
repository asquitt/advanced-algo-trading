"""
Tests for Kelly Criterion Position Sizing

Tests optimal position sizing using Kelly Criterion for maximizing
long-term compound growth rate.

Author: LLM Trading Platform - Test Suite
"""

import pytest
import numpy as np
from datetime import datetime
from src.risk.kelly_criterion import (
    KellyCriterionCalculator,
    KellyParameters,
    KellyPositionSize,
    kelly_calculator
)
from src.data_layer.models import Trade, OrderSide, OrderStatus


class TestKellyCriterionCalculator:
    """Test Kelly Criterion calculator."""

    def test_basic_kelly_calculation(self):
        """Test basic Kelly fraction calculation from trades."""
        calculator = KellyCriterionCalculator(fraction=0.25, min_sample_size=10)

        # Create profitable trading history: 60% win rate, 1.5:1 win/loss ratio
        trades = []
        for i in range(30):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=160.0 if i < 18 else 145.0,  # 18 wins, 12 losses
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades.append(trade)

        params = calculator.calculate_from_backtest(trades, current_confidence=0.8)

        assert params is not None
        assert 0.55 <= params.win_rate <= 0.65  # ~60% win rate
        assert params.win_loss_ratio > 1.0  # Wins larger than losses
        assert 0 < params.kelly_fraction <= 1.0
        assert params.fractional_kelly == params.kelly_fraction * 0.25
        assert params.sample_size == 30
        assert 0 < params.confidence <= 1.0

    def test_insufficient_sample_size(self):
        """Test behavior with insufficient trade history."""
        calculator = KellyCriterionCalculator(min_sample_size=30)

        # Only 10 trades (below minimum of 30)
        trades = []
        for i in range(10):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=160.0 if i < 6 else 145.0,
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades.append(trade)

        params = calculator.calculate_from_backtest(trades)

        assert params is None  # Should return None for insufficient data

    def test_only_wins(self):
        """Test behavior when all trades are winners."""
        calculator = KellyCriterionCalculator(min_sample_size=10)

        # All winning trades
        trades = []
        for i in range(20):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=160.0,  # All wins
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades.append(trade)

        params = calculator.calculate_from_backtest(trades)

        # Should return None (need both wins and losses)
        assert params is None

    def test_only_losses(self):
        """Test behavior when all trades are losers."""
        calculator = KellyCriterionCalculator(min_sample_size=10)

        # All losing trades
        trades = []
        for i in range(20):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=140.0,  # All losses
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades.append(trade)

        params = calculator.calculate_from_backtest(trades)

        # Should return None (need both wins and losses)
        assert params is None

    def test_position_size_calculation(self):
        """Test position size calculation using Kelly parameters."""
        calculator = KellyCriterionCalculator(fraction=0.25, max_fraction=0.25)

        # Simulated Kelly parameters
        kelly_params = KellyParameters(
            win_rate=0.60,
            avg_win=1000.0,
            avg_loss=667.0,
            win_loss_ratio=1.5,
            kelly_fraction=0.20,  # 20% Kelly
            fractional_kelly=0.05,  # 5% fractional Kelly (0.25x)
            confidence=0.7,
            sample_size=50
        )

        portfolio_value = 100000.0
        signal_confidence = 0.85

        result = calculator.calculate_position_size(
            kelly_params=kelly_params,
            portfolio_value=portfolio_value,
            signal_confidence=signal_confidence,
            cvar_limit=None
        )

        assert isinstance(result, KellyPositionSize)
        assert result.full_kelly == 0.20
        assert result.fractional_kelly == 0.05
        assert 0 < result.recommended_fraction <= 0.25  # Max 25%
        assert result.recommended_value == result.recommended_fraction * portfolio_value
        assert result.confidence > 0
        assert not result.risk_adjusted  # No CVaR limit applied
        assert len(result.reasoning) > 0

    def test_position_size_with_cvar_limit(self):
        """Test position sizing with CVaR limit enforcement."""
        calculator = KellyCriterionCalculator(fraction=0.25, max_fraction=0.25)

        kelly_params = KellyParameters(
            win_rate=0.60,
            avg_win=1000.0,
            avg_loss=667.0,
            win_loss_ratio=1.5,
            kelly_fraction=0.30,
            fractional_kelly=0.075,  # 7.5% fractional Kelly
            confidence=0.8,
            sample_size=100
        )

        portfolio_value = 100000.0
        signal_confidence = 1.0
        cvar_limit = 0.03  # 3% CVaR limit (below Kelly recommendation)

        result = calculator.calculate_position_size(
            kelly_params=kelly_params,
            portfolio_value=portfolio_value,
            signal_confidence=signal_confidence,
            cvar_limit=cvar_limit
        )

        # Should be capped at CVaR limit
        assert result.recommended_fraction <= cvar_limit
        assert result.risk_adjusted  # CVaR limit was applied
        assert "CVaR limit" in result.reasoning

    def test_confidence_adjustment(self):
        """Test that signal confidence affects position size."""
        calculator = KellyCriterionCalculator(fraction=0.25)

        kelly_params = KellyParameters(
            win_rate=0.60,
            avg_win=1000.0,
            avg_loss=667.0,
            win_loss_ratio=1.5,
            kelly_fraction=0.20,
            fractional_kelly=0.05,
            confidence=0.8,
            sample_size=50
        )

        portfolio_value = 100000.0

        # High confidence
        result_high = calculator.calculate_position_size(
            kelly_params, portfolio_value, signal_confidence=0.95
        )

        # Low confidence
        result_low = calculator.calculate_position_size(
            kelly_params, portfolio_value, signal_confidence=0.50
        )

        # High confidence should result in larger position
        assert result_high.recommended_value > result_low.recommended_value
        assert result_high.confidence > result_low.confidence

    def test_max_fraction_limit(self):
        """Test maximum position fraction enforcement."""
        calculator = KellyCriterionCalculator(
            fraction=1.0,  # Full Kelly
            max_fraction=0.15  # But cap at 15%
        )

        kelly_params = KellyParameters(
            win_rate=0.70,
            avg_win=2000.0,
            avg_loss=500.0,
            win_loss_ratio=4.0,
            kelly_fraction=0.50,  # 50% Kelly (very aggressive)
            fractional_kelly=0.50,  # Full Kelly
            confidence=0.9,
            sample_size=100
        )

        portfolio_value = 100000.0

        result = calculator.calculate_position_size(
            kelly_params, portfolio_value, signal_confidence=1.0
        )

        # Should be capped at max_fraction
        assert result.recommended_fraction <= 0.15
        assert result.recommended_value <= 15000.0

    def test_estimate_from_metrics(self):
        """Test Kelly estimation from summary statistics."""
        calculator = KellyCriterionCalculator(fraction=0.25)

        # Good Sharpe ratio strategy
        sharpe_ratio = 2.0
        avg_return = 0.02  # 2% average return
        return_std = 0.01  # 1% standard deviation

        params = calculator.estimate_from_metrics(
            sharpe_ratio=sharpe_ratio,
            avg_return=avg_return,
            return_std=return_std,
            current_confidence=0.8
        )

        assert isinstance(params, KellyParameters)
        assert params.kelly_fraction >= 0
        assert params.fractional_kelly == params.kelly_fraction * 0.25
        assert params.confidence < 0.8  # Reduced confidence for estimation
        assert params.sample_size == 0  # No actual trades
        assert 0.4 <= params.win_rate <= 0.8  # Estimated from Sharpe

    def test_estimate_zero_volatility(self):
        """Test estimation with zero volatility."""
        calculator = KellyCriterionCalculator(fraction=0.25)

        params = calculator.estimate_from_metrics(
            sharpe_ratio=1.0,
            avg_return=0.01,
            return_std=0.0,  # Zero volatility
            current_confidence=0.8
        )

        assert params.kelly_fraction == 0
        assert params.fractional_kelly == 0
        assert params.confidence == 0

    def test_estimate_negative_sharpe(self):
        """Test estimation with negative Sharpe ratio."""
        calculator = KellyCriterionCalculator(fraction=0.25)

        params = calculator.estimate_from_metrics(
            sharpe_ratio=-1.0,  # Losing strategy
            avg_return=-0.01,
            return_std=0.02,
            current_confidence=0.8
        )

        # Kelly should be 0 or very small for losing strategies
        assert params.kelly_fraction >= 0
        assert params.kelly_fraction < 0.1

    def test_win_rate_uncertainty_penalty(self):
        """Test that win rates near 50% reduce confidence."""
        calculator = KellyCriterionCalculator(min_sample_size=10)

        # 50/50 win rate (high uncertainty)
        trades_50_50 = []
        for i in range(50):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=160.0 if i < 25 else 140.0,  # Exactly 50% win rate
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades_50_50.append(trade)

        # 65% win rate (lower uncertainty)
        trades_65 = []
        for i in range(50):
            trade = Trade(
                symbol="GOOGL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=160.0 if i < 33 else 140.0,  # 66% win rate
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades_65.append(trade)

        params_50 = calculator.calculate_from_backtest(trades_50_50, 0.8)
        params_65 = calculator.calculate_from_backtest(trades_65, 0.8)

        # 50/50 should have lower confidence due to uncertainty penalty
        assert params_50.confidence < params_65.confidence

    def test_sample_size_confidence_scaling(self):
        """Test that confidence scales with sample size."""
        calculator = KellyCriterionCalculator(min_sample_size=10)

        # Small sample
        trades_small = []
        for i in range(30):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=160.0 if i < 18 else 140.0,
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades_small.append(trade)

        # Large sample
        trades_large = []
        for i in range(150):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=160.0 if i < 90 else 140.0,  # Same win rate
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades_large.append(trade)

        params_small = calculator.calculate_from_backtest(trades_small, 0.8)
        params_large = calculator.calculate_from_backtest(trades_large, 0.8)

        # Larger sample should have higher confidence
        assert params_large.confidence > params_small.confidence

    def test_global_singleton_instance(self):
        """Test that global kelly_calculator instance is configured correctly."""
        assert kelly_calculator.fraction == 0.25
        assert kelly_calculator.min_sample_size == 30
        assert kelly_calculator.max_fraction == 0.25

    def test_trades_without_exit_price(self):
        """Test handling of trades without exit prices."""
        calculator = KellyCriterionCalculator(min_sample_size=10)

        trades = []
        for i in range(20):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=None,  # No exit price (open position)
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades.append(trade)

        params = calculator.calculate_from_backtest(trades)

        # Should return None (no closed trades to analyze)
        assert params is None

    def test_kelly_formula_correctness(self):
        """Test that Kelly formula is calculated correctly."""
        calculator = KellyCriterionCalculator(fraction=1.0, min_sample_size=10)

        # Known parameters: p=0.6, b=1.5
        # Kelly formula: f* = (p*b - q) / b = (0.6*1.5 - 0.4) / 1.5 = 0.4
        trades = []
        for i in range(50):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=100.0,
                # 30 wins of $15, 20 losses of $10
                exit_price=115.0 if i < 30 else 90.0,
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades.append(trade)

        params = calculator.calculate_from_backtest(trades, current_confidence=1.0)

        # Win rate should be 60%
        assert abs(params.win_rate - 0.6) < 0.01

        # Win/loss ratio should be 1.5
        assert abs(params.win_loss_ratio - 1.5) < 0.1

        # Kelly fraction should be approximately 0.4
        # f* = (0.6 * 1.5 - 0.4) / 1.5 = 0.4
        expected_kelly = (0.6 * 1.5 - 0.4) / 1.5
        assert abs(params.kelly_fraction - expected_kelly) < 0.05


class TestKellyIntegration:
    """Integration tests for Kelly Criterion in trading context."""

    def test_full_workflow(self):
        """Test complete Kelly Criterion workflow."""
        calculator = KellyCriterionCalculator(fraction=0.25, min_sample_size=20)

        # 1. Build trade history
        trades = []
        for i in range(40):
            trade = Trade(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=150.0,
                exit_price=160.0 if i < 24 else 145.0,  # 60% win rate
                status=OrderStatus.FILLED,
                order_id=f"order_{i}",
                timestamp=datetime.utcnow()
            )
            trades.append(trade)

        # 2. Calculate Kelly parameters
        params = calculator.calculate_from_backtest(trades, current_confidence=0.8)
        assert params is not None

        # 3. Calculate position size
        portfolio_value = 100000.0
        signal_confidence = 0.85
        cvar_limit = 0.025  # 2.5% CVaR limit

        result = calculator.calculate_position_size(
            kelly_params=params,
            portfolio_value=portfolio_value,
            signal_confidence=signal_confidence,
            cvar_limit=cvar_limit
        )

        # 4. Verify constraints
        assert 0 < result.recommended_value < portfolio_value
        assert result.recommended_fraction <= 0.25  # Max fraction
        assert result.recommended_fraction <= cvar_limit  # CVaR limit
        assert 0 < result.confidence <= 1.0

        # 5. Verify sizing makes sense
        position_pct = result.recommended_value / portfolio_value
        assert 0.01 <= position_pct <= 0.05  # Reasonable range for quarter Kelly

    def test_risk_progression(self):
        """Test that position size decreases with decreasing confidence."""
        calculator = KellyCriterionCalculator(fraction=0.25, min_sample_size=10)

        kelly_params = KellyParameters(
            win_rate=0.60,
            avg_win=1000.0,
            avg_loss=667.0,
            win_loss_ratio=1.5,
            kelly_fraction=0.20,
            fractional_kelly=0.05,
            confidence=0.8,
            sample_size=50
        )

        portfolio_value = 100000.0

        # Test different confidence levels
        confidences = [1.0, 0.8, 0.6, 0.4, 0.2]
        position_values = []

        for conf in confidences:
            result = calculator.calculate_position_size(
                kelly_params, portfolio_value, signal_confidence=conf
            )
            position_values.append(result.recommended_value)

        # Position values should decrease monotonically
        for i in range(len(position_values) - 1):
            assert position_values[i] > position_values[i + 1]
