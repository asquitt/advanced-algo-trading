"""
Performance Analysis Module

Calculates comprehensive performance metrics for backtesting results.
Includes risk-adjusted returns, drawdowns, and trading statistics.

Author: LLM Trading Platform
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""

    # Return metrics
    total_return: float
    annual_return: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int

    # Value at Risk
    value_at_risk_95: float
    value_at_risk_99: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float

    # Trading statistics
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float

    # Additional metrics
    recovery_factor: float  # Total return / Max DD
    ulcer_index: float  # RMS of drawdowns
    avg_trade_duration: float  # In days

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'value_at_risk_95': self.value_at_risk_95,
            'value_at_risk_99': self.value_at_risk_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'recovery_factor': self.recovery_factor,
            'ulcer_index': self.ulcer_index,
            'avg_trade_duration': self.avg_trade_duration
        }


class PerformanceAnalyzer:
    """
    Calculates comprehensive performance metrics for backtesting.

    Computes risk-adjusted returns, drawdown analysis, Value at Risk,
    and trading statistics.
    """

    def __init__(self):
        """Initialize performance analyzer."""
        logger.info("Initialized PerformanceAnalyzer")

    def calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> BacktestMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Series of strategy returns
            equity_curve: Series of portfolio equity over time
            trades: DataFrame of completed trades
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            BacktestMetrics with all calculated metrics
        """
        logger.info("Calculating performance metrics...")

        # Return metrics
        total_return = self._calculate_total_return(equity_curve)
        annual_return = self._calculate_annual_return(returns)

        # Risk metrics
        volatility = self._calculate_volatility(returns)
        downside_deviation = self._calculate_downside_deviation(returns)
        max_dd, avg_dd, max_dd_duration = self._calculate_drawdown_metrics(equity_curve)

        # Risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = self._calculate_sortino_ratio(returns, risk_free_rate)
        calmar_ratio = self._calculate_calmar_ratio(annual_return, max_dd)

        # Value at Risk
        var_95 = self._calculate_var(returns, confidence=0.95)
        var_99 = self._calculate_var(returns, confidence=0.99)
        cvar_95 = self._calculate_cvar(returns, confidence=0.95)
        cvar_99 = self._calculate_cvar(returns, confidence=0.99)

        # Trading statistics
        if len(trades) > 0:
            trade_stats = self._calculate_trade_statistics(trades)
        else:
            trade_stats = {
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_trade_duration': 0.0
            }

        # Additional metrics
        recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0.0
        ulcer_index = self._calculate_ulcer_index(equity_curve)

        logger.info(f"Metrics calculated: Sharpe={sharpe_ratio:.2f}, Total Return={total_return:.2%}")

        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            **trade_stats
        )

    def _calculate_total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return from equity curve."""
        if len(equity_curve) == 0:
            return 0.0
        return (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0

        # Assume daily returns, 252 trading days per year
        total_return = (1 + returns).prod() - 1
        num_years = len(returns) / 252.0

        if num_years > 0:
            annual_return = (1 + total_return) ** (1 / num_years) - 1
        else:
            annual_return = 0.0

        return annual_return

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        return returns.std() * np.sqrt(252)

    def _calculate_downside_deviation(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate downside deviation (semi-deviation)."""
        if len(returns) == 0:
            return 0.0

        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0.0

        return downside_returns.std() * np.sqrt(252)

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate

        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of volatility)."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        downside_dev = self._calculate_downside_deviation(returns) / np.sqrt(252)  # Daily

        if downside_dev == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_dev * np.sqrt(252)
        return sortino

    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if max_drawdown == 0:
            return 0.0
        return annual_return / abs(max_drawdown)

    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> tuple:
        """
        Calculate drawdown metrics.

        Returns:
            (max_drawdown, avg_drawdown, max_drawdown_duration)
        """
        if len(equity_curve) == 0:
            return 0.0, 0.0, 0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Drawdown series
        drawdowns = (equity_curve - running_max) / running_max

        # Max drawdown
        max_dd = drawdowns.min()

        # Average drawdown
        avg_dd = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0

        # Max drawdown duration
        underwater = (equity_curve < running_max).astype(int)

        # Find consecutive underwater periods
        underwater_periods = []
        current_period = 0

        for is_underwater in underwater:
            if is_underwater:
                current_period += 1
            else:
                if current_period > 0:
                    underwater_periods.append(current_period)
                current_period = 0

        # Don't forget last period if still underwater
        if current_period > 0:
            underwater_periods.append(current_period)

        max_dd_duration = max(underwater_periods) if underwater_periods else 0

        return max_dd, avg_dd, max_dd_duration

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk at given confidence level.

        VaR is the maximum expected loss over a time period at a given confidence level.
        """
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.
        """
        if len(returns) == 0:
            return 0.0

        var = self._calculate_var(returns, confidence)

        # Average of all returns below VaR threshold
        cvar = returns[returns <= var].mean()

        return cvar if not np.isnan(cvar) else 0.0

    def _calculate_ulcer_index(self, equity_curve: pd.Series) -> float:
        """
        Calculate Ulcer Index (RMS of drawdowns).

        Measures the depth and duration of drawdowns.
        """
        if len(equity_curve) == 0:
            return 0.0

        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max

        # Root mean square of drawdowns
        ulcer = np.sqrt((drawdowns ** 2).mean())

        return ulcer

    def _calculate_trade_statistics(self, trades: pd.DataFrame) -> dict:
        """Calculate trading statistics from trades DataFrame."""
        if len(trades) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_trade_duration': 0.0
            }

        # Separate winning and losing trades
        winning_trades = trades[trades['pnl_pct'] > 0]
        losing_trades = trades[trades['pnl_pct'] < 0]

        # Basic stats
        num_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)

        win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0.0

        # Win/Loss amounts
        avg_win = winning_trades['pnl_pct'].mean() if num_wins > 0 else 0.0
        avg_loss = losing_trades['pnl_pct'].mean() if num_losses > 0 else 0.0
        largest_win = winning_trades['pnl_pct'].max() if num_wins > 0 else 0.0
        largest_loss = losing_trades['pnl_pct'].min() if num_losses > 0 else 0.0

        # Profit factor: Total wins / Total losses
        total_wins = winning_trades['pnl_pct'].sum() if num_wins > 0 else 0.0
        total_losses = abs(losing_trades['pnl_pct'].sum()) if num_losses > 0 else 0.0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

        # Expectancy: Average P&L per trade
        expectancy = trades['pnl_pct'].mean()

        # Average trade duration
        avg_trade_duration = trades['duration'].mean() if 'duration' in trades.columns else 0.0

        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade_duration': avg_trade_duration
        }
