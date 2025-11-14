"""
Week 3: Backtesting Engine - Starter Code

Your mission: Build a complete vectorized backtesting framework from scratch!

Total TODOs: 40
Estimated time: 6 hours

Hint levels:
🟢 Easy: Direct implementation
🟡 Medium: Requires understanding of concept
🔴 Hard: Complex calculation or logic

Author: LLM Trading Platform Learning Lab
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from performance_metrics import PerformanceAnalyzer, BacktestMetrics
from transaction_costs import TransactionCostModel, TransactionCostConfig


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    No TODOs here - just configure your backtest!
    """
    initial_capital: float = 100000.0
    position_size: float = 1.0  # Fraction of capital per position (0.0-1.0)

    # Cost modeling
    enable_costs: bool = True
    cost_config: Optional[TransactionCostConfig] = None

    # Risk management
    max_position_size: float = 1.0  # Maximum position as fraction of capital
    stop_loss_pct: Optional[float] = None  # Stop loss percentage (e.g., 0.02 for 2%)
    take_profit_pct: Optional[float] = None  # Take profit percentage

    # Rebalancing
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'


@dataclass
class BacktestResult:
    """
    Complete backtest results.

    Contains all the information about your backtest!
    """
    # Configuration
    config: BacktestConfig

    # Time series data
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame

    # Performance metrics
    metrics: BacktestMetrics

    # Metadata
    start_date: datetime
    end_date: datetime
    total_days: int


class VectorizedBacktester:
    """
    Vectorized backtesting engine using NumPy/Pandas.

    Why vectorized?
    - 10-100x faster than loop-based backtesting
    - Uses NumPy's optimized C code
    - Can backtest years of data in seconds!

    Key principle: NO LOOPS!
    - Use pandas operations: .shift(), .rolling(), etc.
    - Use numpy operations: np.where(), np.maximum(), etc.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtesting engine.

        Args:
            config: Backtest configuration

        🟢 Easy TODO #1: Initialize the backtester
        """
        # TODO #1: Store config and initialize components
        # HINT: self.config = config if config else BacktestConfig()
        # HINT: self.cost_model = TransactionCostModel(self.config.cost_config) if self.config.enable_costs else None
        # HINT: self.metrics_analyzer = PerformanceAnalyzer()
        # YOUR CODE HERE
        pass

        print("VectorizedBacktester initialized")
        print(f"  Initial capital: ${self.config.initial_capital:,.0f}")
        print(f"  Transaction costs: {'Enabled' if self.config.enable_costs else 'Disabled'}")

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate data for backtesting.

        Expected columns: open, high, low, close, volume
        Will add: returns, log_returns

        Args:
            data: OHLCV data

        Returns:
            Prepared data

        🟡 Medium TODO #2-5: Prepare data
        """
        # TODO #2: Create a copy of the data
        # HINT: df = data.copy()
        df = None  # YOUR CODE HERE

        # TODO #3: Calculate returns
        # HINT: df['returns'] = df['close'].pct_change()
        # YOUR CODE HERE

        # TODO #4: Calculate log returns (for some metrics)
        # HINT: df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        # YOUR CODE HERE

        # TODO #5: Drop NaN values
        # HINT: df = df.dropna()
        # YOUR CODE HERE

        return df

    def generate_signals(
        self,
        data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame], pd.Series]
    ) -> pd.Series:
        """
        Generate trading signals using a strategy function.

        Signals:
        - 1 = Long (buy)
        - 0 = Flat (no position)
        - -1 = Short (sell)

        Args:
            data: Prepared price data
            strategy_func: Function that takes data and returns signals

        Returns:
            Signal series

        🟢 Easy TODO #6: Generate signals
        """
        # TODO #6: Call the strategy function
        # HINT: signals = strategy_func(data)
        # YOUR CODE HERE
        pass

    def signals_to_positions(
        self,
        signals: pd.Series,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Convert signals to positions with proper timing.

        CRITICAL: Avoid look-ahead bias!
        - Signal generated on day T
        - Trade executed on day T+1
        - Return realized on day T+1

        Args:
            signals: Trading signals (1, 0, -1)
            data: Price data

        Returns:
            Position series (shifted signals)

        🟡 Medium TODO #7: Convert signals to positions
        """
        # TODO #7: Shift signals by 1 to avoid look-ahead bias
        # HINT: positions = signals.shift(1)
        # HINT: Fill first NaN with 0 (no position initially)
        # YOUR CODE HERE
        pass

    def calculate_position_returns(
        self,
        positions: pd.Series,
        returns: pd.Series
    ) -> pd.Series:
        """
        Calculate returns from positions.

        Formula: Position * Market_Return

        Example:
        - Position: 1 (long)
        - Market return: 2%
        - Strategy return: 1 * 2% = 2%

        Args:
            positions: Position series
            returns: Market returns

        Returns:
            Strategy returns

        🟢 Easy TODO #8: Calculate position returns
        """
        # TODO #8: Multiply positions by returns
        # HINT: strategy_returns = positions * returns
        # YOUR CODE HERE
        pass

    def calculate_equity_curve(
        self,
        returns: pd.Series,
        initial_capital: float
    ) -> pd.Series:
        """
        Calculate equity curve from returns.

        Equity curve = How much money you have over time

        Formula: Initial_Capital * (1 + returns).cumprod()

        Args:
            returns: Strategy returns
            initial_capital: Starting capital

        Returns:
            Equity curve

        🟢 Easy TODO #9: Calculate equity curve
        """
        # TODO #9: Calculate cumulative equity
        # HINT: equity = initial_capital * (1 + returns).cumprod()
        # YOUR CODE HERE
        pass

    def detect_trades(
        self,
        positions: pd.Series
    ) -> pd.DataFrame:
        """
        Detect individual trades from position series.

        A trade occurs when position changes.

        Args:
            positions: Position series

        Returns:
            DataFrame with trade information

        🔴 Hard TODO #10-14: Detect trades
        """
        trades = []

        # TODO #10: Calculate position changes
        # HINT: position_changes = positions.diff()
        position_changes = None  # YOUR CODE HERE

        # TODO #11: Find where positions change (trades)
        # HINT: trade_indices = position_changes[position_changes != 0].index
        # YOUR CODE HERE

        # TODO #12-14: Loop through trades and record them
        # HINT: For each trade, record: entry_date, exit_date, direction, pnl
        # YOUR CODE HERE (this is one of the few places we use a loop!)
        for i, idx in enumerate(trade_indices):
            if i < len(trade_indices) - 1:
                # Trade from this index to next change
                entry_idx = idx
                exit_idx = trade_indices[i + 1]

                # TODO #12: Get entry and exit positions
                # YOUR CODE HERE

                # TODO #13: Calculate trade PnL
                # YOUR CODE HERE

                # TODO #14: Append trade to list
                # YOUR CODE HERE

        return pd.DataFrame(trades)

    def apply_transaction_costs(
        self,
        returns: pd.Series,
        positions: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        """
        Apply transaction costs to returns.

        Args:
            returns: Strategy returns
            positions: Position series
            prices: Price series

        Returns:
            Returns with costs applied

        🟡 Medium TODO #15: Apply transaction costs
        """
        if not self.config.enable_costs or self.cost_model is None:
            return returns

        # TODO #15: Use cost model to apply costs
        # HINT: Use self.cost_model.apply_costs_to_returns()
        # YOUR CODE HERE
        pass

    def apply_stop_loss_take_profit(
        self,
        data: pd.DataFrame,
        positions: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        """
        Apply stop-loss and take-profit rules.

        Exit position if:
        - Loss exceeds stop_loss_pct
        - Profit exceeds take_profit_pct

        Args:
            data: Price data
            positions: Current positions
            entry_prices: Entry prices for each position

        Returns:
            Adjusted positions

        🔴 Hard TODO #16-18: Apply risk management
        """
        if self.config.stop_loss_pct is None and self.config.take_profit_pct is None:
            return positions

        # TODO #16: Create a copy of positions
        adjusted_positions = None  # YOUR CODE HERE

        # TODO #17: Calculate P&L from entry
        # HINT: pnl_pct = (data['close'] - entry_prices) / entry_prices * positions
        # YOUR CODE HERE

        # TODO #18: Apply stop-loss and take-profit
        # HINT: Use np.where() to exit positions that hit stops
        # YOUR CODE HERE

        return adjusted_positions

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame], pd.Series]
    ) -> BacktestResult:
        """
        Run a complete backtest!

        This is the main method that orchestrates everything.

        Args:
            data: OHLCV data
            strategy_func: Strategy function

        Returns:
            BacktestResult with all results

        🟡 Medium TODO #19-25: Run complete backtest
        """
        print("🚀 Running backtest...")
        print("-" * 50)

        # TODO #19: Prepare data
        # HINT: df = self.prepare_data(data)
        df = None  # YOUR CODE HERE

        # TODO #20: Generate signals
        # HINT: signals = self.generate_signals(df, strategy_func)
        signals = None  # YOUR CODE HERE

        # TODO #21: Convert to positions
        # HINT: positions = self.signals_to_positions(signals, df)
        positions = None  # YOUR CODE HERE

        # TODO #22: Calculate returns
        # HINT: strategy_returns = self.calculate_position_returns(positions, df['returns'])
        strategy_returns = None  # YOUR CODE HERE

        # TODO #23: Apply transaction costs
        # HINT: if self.config.enable_costs: strategy_returns = self.apply_transaction_costs(...)
        # YOUR CODE HERE

        # TODO #24: Calculate equity curve
        # HINT: equity_curve = self.calculate_equity_curve(strategy_returns, self.config.initial_capital)
        equity_curve = None  # YOUR CODE HERE

        # TODO #25: Detect trades
        # HINT: trades = self.detect_trades(positions)
        trades = None  # YOUR CODE HERE

        # Calculate metrics
        metrics = self.metrics_analyzer.calculate_metrics(
            returns=strategy_returns,
            equity_curve=equity_curve,
            trades=trades
        )

        # Create result
        result = BacktestResult(
            config=self.config,
            equity_curve=equity_curve,
            returns=strategy_returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
            start_date=df.index[0],
            end_date=df.index[-1],
            total_days=len(df)
        )

        print("✅ Backtest complete!")
        print(f"  Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"  Total return: {result.metrics.total_return:.2%}")
        print(f"  Sharpe ratio: {result.metrics.sharpe_ratio:.2f}")
        print(f"  Max drawdown: {result.metrics.max_drawdown:.2%}")

        return result


# ============================================================================
# Walk-Forward Analysis
# ============================================================================

class WalkForwardAnalyzer:
    """
    Implement walk-forward analysis to avoid overfitting.

    Walk-forward process:
    1. Split data into train/test windows
    2. Optimize on train window
    3. Test on test window
    4. Move window forward
    5. Repeat

    This prevents curve-fitting to historical data!
    """

    def __init__(
        self,
        train_period_days: int = 252,
        test_period_days: int = 63,
        step_days: int = 63
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            train_period_days: Training window size (default: 1 year)
            test_period_days: Testing window size (default: 1 quarter)
            step_days: Step size for moving window (default: 1 quarter)

        🟢 Easy TODO #26: Initialize walk-forward analyzer
        """
        # TODO #26: Store parameters
        # YOUR CODE HERE
        pass

        print(f"WalkForwardAnalyzer initialized")
        print(f"  Train: {train_period_days} days")
        print(f"  Test: {test_period_days} days")
        print(f"  Step: {step_days} days")

    def split_train_test(
        self,
        data: pd.DataFrame,
        start_idx: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.

        Args:
            data: Full dataset
            start_idx: Starting index for this window

        Returns:
            (train_data, test_data)

        🟡 Medium TODO #27-28: Split data
        """
        # TODO #27: Extract train data
        # HINT: train_end = start_idx + self.train_period_days
        # HINT: train_data = data.iloc[start_idx:train_end]
        train_data = None  # YOUR CODE HERE

        # TODO #28: Extract test data
        # HINT: test_end = train_end + self.test_period_days
        # HINT: test_data = data.iloc[train_end:test_end]
        test_data = None  # YOUR CODE HERE

        return train_data, test_data

    def optimize_parameters(
        self,
        train_data: pd.DataFrame,
        param_grid: Dict[str, List],
        strategy_func_template: Callable
    ) -> Dict:
        """
        Optimize strategy parameters on training data.

        Uses grid search to find best parameters.

        Args:
            train_data: Training data
            param_grid: Parameter grid to search
            strategy_func_template: Strategy function template

        Returns:
            Best parameters

        🔴 Hard TODO #29-31: Optimize parameters
        """
        # TODO #29: Initialize best parameters
        best_params = None
        best_sharpe = -np.inf
        # YOUR CODE HERE

        # TODO #30: Grid search over parameters
        # HINT: Use itertools.product() to generate all combinations
        # YOUR CODE HERE

        # TODO #31: Return best parameters
        # YOUR CODE HERE
        pass

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Optional[Dict[str, List]] = None
    ) -> pd.DataFrame:
        """
        Run complete walk-forward analysis.

        Args:
            data: Full dataset
            strategy_func: Strategy function
            param_grid: Parameter grid for optimization

        Returns:
            DataFrame with walk-forward results

        🔴 Hard TODO #32-35: Run walk-forward analysis
        """
        results = []
        backtester = VectorizedBacktester()

        # TODO #32: Calculate number of windows
        # HINT: num_windows = (len(data) - self.train_period_days) // self.step_days
        num_windows = None  # YOUR CODE HERE

        print(f"Running walk-forward analysis: {num_windows} windows")

        # TODO #33: Loop through windows
        for i in range(num_windows):
            # TODO #34: Split data
            start_idx = None  # YOUR CODE HERE
            train_data, test_data = None, None  # YOUR CODE HERE

            if len(test_data) < 10:  # Skip if test set too small
                continue

            # TODO #35: Run backtest on test data
            # YOUR CODE HERE

            # Store results
            # YOUR CODE HERE

        return pd.DataFrame(results)


# ============================================================================
# Parameter Optimization
# ============================================================================

class ParameterOptimizer:
    """
    Optimize strategy parameters using grid search.

    Warning: Easy to overfit! Always validate out-of-sample.
    """

    def __init__(self, backtester: VectorizedBacktester):
        """
        Initialize optimizer.

        Args:
            backtester: Backtesting engine

        🟢 Easy TODO #36: Initialize optimizer
        """
        # TODO #36: Store backtester
        # YOUR CODE HERE
        pass

    def grid_search(
        self,
        data: pd.DataFrame,
        strategy_template: Callable,
        param_grid: Dict[str, List],
        metric: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Run grid search over parameters.

        Args:
            data: Price data
            strategy_template: Strategy function template
            param_grid: Parameter grid
            metric: Metric to optimize

        Returns:
            DataFrame with all results

        🔴 Hard TODO #37-40: Implement grid search
        """
        # TODO #37: Generate all parameter combinations
        # HINT: Use itertools.product()
        # YOUR CODE HERE

        # TODO #38: Test each combination
        results = []
        # YOUR CODE HERE

        # TODO #39: Create results DataFrame
        # YOUR CODE HERE

        # TODO #40: Sort by metric
        # YOUR CODE HERE
        pass


# ============================================================================
# Example Strategy Functions
# ============================================================================

def moving_average_crossover(data: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Simple moving average crossover strategy.

    Signals:
    - Buy when fast MA crosses above slow MA
    - Sell when fast MA crosses below slow MA

    Args:
        data: Price data
        fast: Fast MA period
        slow: Slow MA period

    Returns:
        Signal series
    """
    fast_ma = data['close'].rolling(fast).mean()
    slow_ma = data['close'].rolling(slow).mean()

    signals = pd.Series(0, index=data.index)
    signals[fast_ma > slow_ma] = 1  # Long
    signals[fast_ma < slow_ma] = -1  # Short

    return signals


def rsi_strategy(data: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """
    RSI mean reversion strategy.

    Signals:
    - Buy when RSI < oversold
    - Sell when RSI > overbought

    Args:
        data: Price data
        period: RSI period
        oversold: Oversold threshold
        overbought: Overbought threshold

    Returns:
        Signal series
    """
    # Calculate RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Generate signals
    signals = pd.Series(0, index=data.index)
    signals[rsi < oversold] = 1  # Buy
    signals[rsi > overbought] = -1  # Sell

    return signals


# ============================================================================
# Testing Your Implementation
# ============================================================================

def test_your_implementation():
    """
    Quick test to see if your implementation works!

    Run this file to test: python backtesting_engine.py
    """
    print("🧪 Testing your implementation...")
    print("-" * 50)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

    # Simulate price data with trend
    returns = np.random.randn(len(dates)) * 0.01 + 0.0002  # Slight upward trend
    prices = 100 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)

    # Create backtester
    config = BacktestConfig(
        initial_capital=100000,
        enable_costs=True
    )
    backtester = VectorizedBacktester(config)

    try:
        # Run backtest
        result = backtester.run_backtest(
            data=data,
            strategy_func=lambda df: moving_average_crossover(df, fast=20, slow=50)
        )

        print("\n✅ SUCCESS! Your implementation works!")
        print("\nBacktest Results:")
        print(f"  Total Return: {result.metrics.total_return:.2%}")
        print(f"  Annual Return: {result.metrics.annual_return:.2%}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
        print(f"  Number of Trades: {result.metrics.num_trades}")
        print("\n🎉 Great job! Now run the full test suite:")
        print("   pytest tests/test_integration.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Debug tips:")
        print("  1. Check that all TODOs are filled in")
        print("  2. Make sure performance_metrics.py is complete")
        print("  3. Make sure transaction_costs.py is complete")
        print("  4. Use print() to debug intermediate values")


if __name__ == "__main__":
    test_your_implementation()
