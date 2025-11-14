"""
Vectorized Backtesting Engine

Fast backtesting using NumPy/Pandas vectorized operations.
Supports multiple strategies, realistic costs, and comprehensive performance metrics.

Author: LLM Trading Platform
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from loguru import logger

from src.backtesting.transaction_cost_model import TransactionCostModel
from src.backtesting.performance_analyzer import PerformanceAnalyzer, BacktestMetrics


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 0.1% commission
    slippage_bps: float = 5.0  # 5 basis points slippage
    enable_shorting: bool = False
    max_leverage: float = 1.0
    position_sizing_method: str = "equal_weight"  # equal_weight, volatility, kelly
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    compound_returns: bool = True


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Performance metrics
    metrics: BacktestMetrics

    # Time series data
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame

    # Signal data
    signals: pd.DataFrame

    # Drawdown analysis
    drawdowns: pd.Series
    underwater_curve: pd.Series

    # Configuration
    config: BacktestConfig
    start_date: datetime
    end_date: datetime

    def summary(self) -> str:
        """Generate summary report."""
        return f"""
Backtest Summary
================
Period: {self.start_date.date()} to {self.end_date.date()}
Duration: {(self.end_date - self.start_date).days} days

Performance Metrics:
-------------------
Total Return: {self.metrics.total_return:.2%}
Annual Return: {self.metrics.annual_return:.2%}
Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}
Sortino Ratio: {self.metrics.sortino_ratio:.2f}
Calmar Ratio: {self.metrics.calmar_ratio:.2f}
Max Drawdown: {self.metrics.max_drawdown:.2%}
Win Rate: {self.metrics.win_rate:.2%}

Risk Metrics:
------------
Volatility (Annual): {self.metrics.volatility:.2%}
Downside Deviation: {self.metrics.downside_deviation:.2%}
VaR (95%): {self.metrics.value_at_risk_95:.2%}
CVaR (95%): {self.metrics.cvar_95:.2%}

Trading Statistics:
------------------
Total Trades: {self.metrics.num_trades}
Winning Trades: {int(self.metrics.num_trades * self.metrics.win_rate / 100)}
Losing Trades: {int(self.metrics.num_trades * (1 - self.metrics.win_rate / 100))}
Average Win: {self.metrics.avg_win:.2%}
Average Loss: {self.metrics.avg_loss:.2%}
Profit Factor: {self.metrics.profit_factor:.2f}
Expectancy: {self.metrics.expectancy:.2%}

Final Portfolio Value: ${self.equity_curve.iloc[-1]:,.2f}
        """


class VectorizedBacktester:
    """
    High-performance vectorized backtesting engine.

    Uses NumPy/Pandas for fast computation without loops.
    Supports realistic transaction costs, position sizing, and multiple strategies.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.cost_model = TransactionCostModel(
            commission_pct=self.config.commission_pct,
            slippage_bps=self.config.slippage_bps
        )
        self.performance_analyzer = PerformanceAnalyzer()

        logger.info(f"Initialized vectorized backtester: initial_capital=${self.config.initial_capital:,.2f}")

    def run_backtest(
        self,
        data: pd.DataFrame,
        signal_function: Callable[[pd.DataFrame], pd.Series],
        **kwargs
    ) -> BacktestResult:
        """
        Run vectorized backtest on historical data.

        Args:
            data: DataFrame with OHLCV data, indexed by timestamp
            signal_function: Function that takes data and returns signals (-1, 0, 1)
            **kwargs: Additional parameters for signal function

        Returns:
            BacktestResult with complete analysis
        """
        logger.info(f"Running backtest from {data.index[0]} to {data.index[-1]}")
        logger.info(f"Data points: {len(data)}")

        # Generate signals
        signals = signal_function(data, **kwargs)
        signals = signals.fillna(0)  # Fill NaN signals with 0 (no position)

        # Calculate returns
        data['returns'] = data['close'].pct_change()

        # Calculate positions (signal shifted by 1 to avoid look-ahead bias)
        positions = signals.shift(1).fillna(0)

        # Apply position sizing
        if self.config.position_sizing_method == "equal_weight":
            sized_positions = positions
        elif self.config.position_sizing_method == "volatility":
            sized_positions = self._volatility_sizing(positions, data)
        elif self.config.position_sizing_method == "kelly":
            sized_positions = self._kelly_sizing(positions, data)
        else:
            sized_positions = positions

        # Calculate strategy returns (position * market return)
        strategy_returns = sized_positions * data['returns']

        # Apply transaction costs
        strategy_returns_after_costs = self._apply_transaction_costs(
            strategy_returns,
            sized_positions,
            data
        )

        # Calculate equity curve
        if self.config.compound_returns:
            equity_curve = (1 + strategy_returns_after_costs).cumprod() * self.config.initial_capital
        else:
            equity_curve = (strategy_returns_after_costs.cumsum() + 1) * self.config.initial_capital

        # Calculate drawdowns
        drawdowns = self._calculate_drawdowns(equity_curve)
        underwater_curve = self._calculate_underwater_curve(equity_curve)

        # Generate trades DataFrame
        trades = self._generate_trades(sized_positions, data)

        # Calculate performance metrics
        metrics = self.performance_analyzer.calculate_metrics(
            returns=strategy_returns_after_costs,
            equity_curve=equity_curve,
            trades=trades
        )

        # Create positions DataFrame
        positions_df = pd.DataFrame({
            'signal': signals,
            'position': sized_positions,
            'price': data['close'],
            'returns': strategy_returns_after_costs
        })

        # Create signals DataFrame
        signals_df = pd.DataFrame({
            'signal': signals,
            'close': data['close'],
            'volume': data['volume'] if 'volume' in data else None
        })

        logger.info(f"Backtest complete: Total Return={metrics.total_return:.2%}, Sharpe={metrics.sharpe_ratio:.2f}")

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            returns=strategy_returns_after_costs,
            positions=positions_df,
            trades=trades,
            signals=signals_df,
            drawdowns=drawdowns,
            underwater_curve=underwater_curve,
            config=self.config,
            start_date=data.index[0],
            end_date=data.index[-1]
        )

    def _volatility_sizing(self, positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Position sizing based on inverse volatility.

        Higher volatility = smaller position size.
        """
        # Calculate rolling volatility (20-day window)
        volatility = data['returns'].rolling(20).std()

        # Inverse volatility weighting
        weights = 1 / volatility
        weights = weights / weights.rolling(20).mean()  # Normalize

        # Apply to positions
        sized = positions * weights.shift(1).fillna(1.0)

        # Cap at max leverage
        sized = sized.clip(-self.config.max_leverage, self.config.max_leverage)

        return sized

    def _kelly_sizing(self, positions: pd.Series, data: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """
        Position sizing using Kelly Criterion.

        Kelly fraction = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
        """
        sized = positions.copy()

        # Calculate rolling win rate and average win/loss
        for i in range(lookback, len(data)):
            recent_returns = data['returns'].iloc[i-lookback:i]
            wins = recent_returns[recent_returns > 0]
            losses = recent_returns[recent_returns < 0]

            if len(wins) > 0 and len(losses) > 0:
                win_rate = len(wins) / lookback
                avg_win = wins.mean()
                avg_loss = abs(losses.mean())

                # Kelly fraction
                kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

                # Use fractional Kelly (half Kelly for safety)
                kelly_f = kelly_f * 0.5

                # Apply to position
                sized.iloc[i] = positions.iloc[i] * max(0, min(kelly_f, self.config.max_leverage))

        return sized

    def _apply_transaction_costs(
        self,
        returns: pd.Series,
        positions: pd.Series,
        data: pd.DataFrame
    ) -> pd.Series:
        """Apply realistic transaction costs to returns."""
        # Detect position changes (trades)
        position_changes = positions.diff().abs()

        # Calculate costs
        costs = self.cost_model.calculate_costs(
            position_changes=position_changes,
            prices=data['close'],
            volumes=data['volume'] if 'volume' in data else None
        )

        # Subtract costs from returns
        returns_after_costs = returns - costs

        return returns_after_costs

    def _calculate_drawdowns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdowns from equity curve."""
        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Drawdown is current value minus running max
        drawdowns = (equity_curve - running_max) / running_max

        return drawdowns

    def _calculate_underwater_curve(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate underwater curve (time below previous peak)."""
        # Find peak equity
        running_max = equity_curve.expanding().max()

        # Underwater when below peak
        underwater = (equity_curve < running_max).astype(int)

        return underwater

    def _generate_trades(self, positions: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trades DataFrame from position changes.

        Each trade has entry and exit.
        """
        trades = []

        # Detect position changes
        position_changes = positions.diff()

        in_trade = False
        entry_idx = None
        entry_price = None
        entry_size = None

        for i, (idx, pos_change) in enumerate(position_changes.items()):
            current_pos = positions.iloc[i]

            # Entry: position goes from 0 to non-zero
            if not in_trade and current_pos != 0:
                in_trade = True
                entry_idx = idx
                entry_price = data['close'].loc[idx]
                entry_size = current_pos

            # Exit: position goes to 0 or reverses
            elif in_trade and (current_pos == 0 or np.sign(current_pos) != np.sign(entry_size)):
                exit_price = data['close'].loc[idx]

                # Calculate P&L
                pnl = (exit_price - entry_price) / entry_price * entry_size

                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': entry_size,
                    'pnl_pct': pnl,
                    'pnl_dollar': pnl * self.config.initial_capital,  # Approximate
                    'duration': (idx - entry_idx).days if hasattr(idx - entry_idx, 'days') else 1,
                    'side': 'long' if entry_size > 0 else 'short'
                })

                # Check if we're entering a new position immediately
                if current_pos != 0:
                    entry_idx = idx
                    entry_price = exit_price
                    entry_size = current_pos
                else:
                    in_trade = False

        return pd.DataFrame(trades)

    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        signal_function: Callable,
        train_period_days: int = 252,
        test_period_days: int = 63,
        step_days: int = 21,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Walk-forward analysis for out-of-sample validation.

        Args:
            data: Historical data
            signal_function: Strategy signal function
            train_period_days: Training window size
            test_period_days: Testing window size
            step_days: Step size between windows
            **kwargs: Parameters for signal function

        Returns:
            Dictionary with in-sample and out-of-sample results
        """
        logger.info("Starting walk-forward analysis...")

        in_sample_results = []
        out_of_sample_results = []

        start_idx = 0
        while start_idx + train_period_days + test_period_days <= len(data):
            # Split data
            train_data = data.iloc[start_idx:start_idx + train_period_days]
            test_data = data.iloc[start_idx + train_period_days:start_idx + train_period_days + test_period_days]

            # Run backtest on training data
            train_result = self.run_backtest(train_data, signal_function, **kwargs)
            in_sample_results.append(train_result)

            # Run backtest on test data
            test_result = self.run_backtest(test_data, signal_function, **kwargs)
            out_of_sample_results.append(test_result)

            logger.info(f"Window {len(in_sample_results)}: "
                       f"In-sample Sharpe={train_result.metrics.sharpe_ratio:.2f}, "
                       f"Out-of-sample Sharpe={test_result.metrics.sharpe_ratio:.2f}")

            # Move window forward
            start_idx += step_days

        # Aggregate results
        avg_in_sample_sharpe = np.mean([r.metrics.sharpe_ratio for r in in_sample_results])
        avg_out_sample_sharpe = np.mean([r.metrics.sharpe_ratio for r in out_of_sample_results])

        logger.info(f"Walk-forward complete: "
                   f"Avg in-sample Sharpe={avg_in_sample_sharpe:.2f}, "
                   f"Avg out-of-sample Sharpe={avg_out_sample_sharpe:.2f}")

        return {
            'in_sample_results': in_sample_results,
            'out_of_sample_results': out_of_sample_results,
            'avg_in_sample_sharpe': avg_in_sample_sharpe,
            'avg_out_of_sample_sharpe': avg_out_sample_sharpe,
            'degradation': avg_in_sample_sharpe - avg_out_sample_sharpe,
            'num_windows': len(in_sample_results)
        }

    def parameter_optimization(
        self,
        data: pd.DataFrame,
        signal_function: Callable,
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.

        Args:
            data: Historical data
            signal_function: Strategy signal function
            param_grid: Dictionary of parameter names and values to test
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)

        Returns:
            Best parameters and results
        """
        from itertools import product

        logger.info(f"Starting parameter optimization on {len(data)} bars...")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations...")

        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                result = self.run_backtest(data, signal_function, **params)
                metric_value = getattr(result.metrics, metric)

                results.append({
                    'params': params,
                    'result': result,
                    'metric_value': metric_value
                })

                logger.debug(f"Params {params}: {metric}={metric_value:.2f}")

            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue

        # Find best parameters
        if not results:
            raise ValueError("No valid parameter combinations found")

        best = max(results, key=lambda x: x['metric_value'])

        logger.info(f"Best parameters: {best['params']} ({metric}={best['metric_value']:.2f})")

        return {
            'best_params': best['params'],
            'best_result': best['result'],
            'best_metric_value': best['metric_value'],
            'all_results': results
        }
