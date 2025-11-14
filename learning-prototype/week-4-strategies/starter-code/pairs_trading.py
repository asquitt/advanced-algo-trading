"""
Pairs Trading Strategy - Starter Code with TODOs

Your mission: Implement a statistical arbitrage pairs trading strategy!

This strategy trades two cointegrated assets, betting on mean reversion when
the spread between them diverges from its historical average.

Difficulty levels:
🟢 Easy: Basic implementation
🟡 Medium: Requires some thinking
🔴 Hard: Advanced concepts

Author: Learning Lab Week 4
"""

from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint, adfuller


@dataclass
class PairsTradingConfig:
    """Configuration for pairs trading strategy."""
    lookback_period: int = 60  # Days to calculate spread statistics
    entry_zscore: float = 2.0  # Z-score threshold for entry
    exit_zscore: float = 0.5  # Z-score threshold for exit
    stop_loss_zscore: float = 3.5  # Z-score for stop loss
    cointegration_pvalue: float = 0.05  # Max p-value for cointegration
    min_half_life: int = 1  # Minimum half-life in days
    max_half_life: int = 30  # Maximum half-life in days


class PairsTradingStrategy:
    """
    Statistical arbitrage pairs trading strategy.

    Core concepts:
    1. Find two cointegrated assets (long-term relationship)
    2. Calculate spread = price1 - hedge_ratio * price2
    3. Trade when spread deviates from mean
    4. Exit when spread reverts to mean
    """

    def __init__(self, config: Optional[PairsTradingConfig] = None):
        """Initialize pairs trading strategy."""
        self.config = config or PairsTradingConfig()
        self.current_position = 0  # 1 = long spread, -1 = short spread, 0 = flat

    # ========================================
    # Part 1: Cointegration Testing
    # ========================================

    def test_cointegration(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> Tuple[bool, float, float]:
        """
        Test if two price series are cointegrated using Engle-Granger test.

        🟢 TODO #1: Implement cointegration test

        Steps:
        1. Use statsmodels.tsa.stattools.coint() to test cointegration
        2. Extract p-value from result
        3. Calculate hedge ratio using linear regression
        4. Return (is_cointegrated, p_value, hedge_ratio)

        HINT: coint() returns (test_stat, pvalue, critical_values)
        HINT: hedge_ratio = slope from regressing price1 on price2
        HINT: Compare p_value with self.config.cointegration_pvalue

        Args:
            price1: First asset price series
            price2: Second asset price series

        Returns:
            (is_cointegrated, p_value, hedge_ratio)
        """
        # YOUR CODE HERE
        pass

    def calculate_hedge_ratio(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> float:
        """
        Calculate optimal hedge ratio using linear regression.

        🟢 TODO #2: Calculate hedge ratio

        Formula: price1 = hedge_ratio * price2 + intercept

        HINT: Use sklearn.linear_model.LinearRegression
        HINT: Reshape price2 to column vector: price2.values.reshape(-1, 1)
        HINT: hedge_ratio = model.coef_[0]

        Args:
            price1: First asset prices
            price2: Second asset prices

        Returns:
            Hedge ratio (beta coefficient)
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 2: Spread Calculation
    # ========================================

    def calculate_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """
        Calculate the spread between two assets.

        🟢 TODO #3: Calculate spread

        Formula: spread = price1 - hedge_ratio * price2

        HINT: Simple subtraction!

        Args:
            price1: First asset prices
            price2: Second asset prices
            hedge_ratio: Hedge ratio from regression

        Returns:
            Spread series
        """
        # YOUR CODE HERE
        pass

    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling z-score of spread.

        🟢 TODO #4: Calculate z-score

        Formula: z = (spread - rolling_mean) / rolling_std

        HINT: Use pd.Series.rolling(window).mean() and .std()
        HINT: Use self.config.lookback_period if lookback is None

        Args:
            spread: Spread series
            lookback: Lookback period for rolling statistics

        Returns:
            Z-score series
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 3: Mean Reversion Metrics
    # ========================================

    def calculate_half_life(
        self,
        spread: pd.Series
    ) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck.

        🟡 TODO #5: Calculate half-life

        The Ornstein-Uhlenbeck process: dS = θ(μ - S)dt + σdW
        Half-life = -ln(2) / θ

        Steps:
        1. Calculate spread_lag = spread.shift(1)
        2. Calculate spread_diff = spread - spread_lag
        3. Run OLS: spread_diff ~ spread_lag
        4. Extract θ (theta) = coefficient
        5. half_life = -np.log(2) / θ

        HINT: Use statsmodels.regression.linear_model.OLS
        HINT: Drop NaN values before regression
        HINT: theta = model.fit().params[0]

        Args:
            spread: Spread series

        Returns:
            Half-life in periods (days)
        """
        # YOUR CODE HERE
        pass

    def is_spread_stationary(
        self,
        spread: pd.Series,
        significance: float = 0.05
    ) -> bool:
        """
        Test if spread is stationary using Augmented Dickey-Fuller test.

        🟡 TODO #6: Test stationarity

        HINT: Use statsmodels.tsa.stattools.adfuller()
        HINT: adfuller() returns (test_stat, pvalue, ...)
        HINT: Stationary if p_value < significance

        Args:
            spread: Spread series
            significance: Significance level (default 0.05)

        Returns:
            True if spread is stationary
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 4: Signal Generation
    # ========================================

    def generate_entry_signals(
        self,
        zscore: pd.Series
    ) -> pd.Series:
        """
        Generate entry signals based on z-score.

        🟢 TODO #7: Generate entry signals

        Logic:
        - If z > entry_threshold: Signal = -1 (short spread)
        - If z < -entry_threshold: Signal = 1 (long spread)
        - Otherwise: Signal = 0 (no entry)

        HINT: Use self.config.entry_zscore as threshold
        HINT: Use pd.Series with same index as zscore

        Args:
            zscore: Z-score series

        Returns:
            Entry signals (-1, 0, or 1)
        """
        # YOUR CODE HERE
        pass

    def generate_exit_signals(
        self,
        zscore: pd.Series,
        position: pd.Series
    ) -> pd.Series:
        """
        Generate exit signals based on z-score and current position.

        🟡 TODO #8: Generate exit signals

        Logic:
        - If long (position=1) and z > -exit_threshold: Exit
        - If short (position=-1) and z < exit_threshold: Exit
        - If abs(z) > stop_loss_threshold: Stop loss exit
        - Otherwise: Hold

        HINT: Exit signal = 1 when should exit, 0 when should hold
        HINT: Check self.config.exit_zscore and self.config.stop_loss_zscore

        Args:
            zscore: Z-score series
            position: Current position series

        Returns:
            Exit signals (1 = exit, 0 = hold)
        """
        # YOUR CODE HERE
        pass

    def combine_signals(
        self,
        entry_signals: pd.Series,
        exit_signals: pd.Series
    ) -> pd.Series:
        """
        Combine entry and exit signals into position series.

        🟡 TODO #9: Combine signals

        Logic:
        - Start with position = 0
        - On entry signal: position = entry_signal value
        - On exit signal: position = 0
        - Otherwise: maintain previous position

        HINT: Use forward fill to maintain positions
        HINT: Exit signals should set position to 0

        Args:
            entry_signals: Entry signals (-1, 0, 1)
            exit_signals: Exit signals (0, 1)

        Returns:
            Position series
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 5: Main Strategy Logic
    # ========================================

    def run_strategy(
        self,
        price1: pd.Series,
        price2: pd.Series,
        validate_pair: bool = True
    ) -> Dict:
        """
        Run complete pairs trading strategy.

        🔴 TODO #10: Implement main strategy

        Steps:
        1. Test cointegration (if validate_pair=True)
        2. Calculate hedge ratio
        3. Calculate spread
        4. Calculate z-score
        5. Calculate half-life
        6. Generate entry signals
        7. Generate exit signals
        8. Combine signals
        9. Return results dict

        HINT: Use all the functions you implemented above
        HINT: If not cointegrated, return empty signals

        Args:
            price1: First asset price series
            price2: Second asset price series
            validate_pair: Whether to test cointegration first

        Returns:
            Dictionary with:
            - 'is_cointegrated': bool
            - 'p_value': float
            - 'hedge_ratio': float
            - 'spread': pd.Series
            - 'zscore': pd.Series
            - 'half_life': float
            - 'signals': pd.Series (position signals)
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 6: Backtesting Integration
    # ========================================

    def calculate_strategy_returns(
        self,
        price1: pd.Series,
        price2: pd.Series,
        signals: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """
        Calculate strategy returns from signals.

        🟡 TODO #11: Calculate returns

        Steps:
        1. Calculate asset returns (pct_change)
        2. Calculate spread return = return1 - hedge_ratio * return2
        3. Strategy return = signal (shifted by 1) * spread_return

        HINT: Shift signals by 1 to avoid look-ahead bias
        HINT: align() or reindex() to ensure same index

        Args:
            price1: First asset prices
            price2: Second asset prices
            signals: Position signals
            hedge_ratio: Hedge ratio

        Returns:
            Strategy returns series
        """
        # YOUR CODE HERE
        pass

    def backtest_signal_function(
        self,
        data: pd.DataFrame,
        symbol1: str = 'asset1',
        symbol2: str = 'asset2',
        **kwargs
    ) -> pd.Series:
        """
        Signal function for use with VectorizedBacktester.

        🔴 TODO #12: Implement backtest signal function

        This function should work with the backtesting engine from Week 3.

        HINT: Extract price1 and price2 from data DataFrame
        HINT: Use run_strategy() to get signals
        HINT: Return position series with same index as data

        Args:
            data: DataFrame with prices for both assets
            symbol1: Column name for first asset
            symbol2: Column name for second asset
            **kwargs: Additional parameters

        Returns:
            Position series (-1, 0, 1)
        """
        # YOUR CODE HERE
        pass


# ========================================
# Self-Test Function
# ========================================

def test_implementation():
    """
    Test your pairs trading implementation!

    🎯 Run this function to check if your code works:
        python pairs_trading.py
    """
    print("🧪 Testing Pairs Trading Implementation...\n")

    # Generate synthetic cointegrated pair
    np.random.seed(42)
    n = 250  # 1 year of daily data

    # Asset 1: random walk
    returns1 = np.random.normal(0.0005, 0.02, n)
    price1 = pd.Series(100 * np.exp(np.cumsum(returns1)))

    # Asset 2: cointegrated with asset 1
    hedge_ratio_true = 1.5
    noise = np.random.normal(0, 0.01, n)
    price2 = pd.Series((price1 + noise) / hedge_ratio_true)

    # Create strategy
    strategy = PairsTradingStrategy()

    # Test 1: Cointegration
    print("Test 1: Cointegration Test")
    try:
        is_coint, pvalue, hedge_ratio = strategy.test_cointegration(price1, price2)
        print(f"✅ Cointegrated: {is_coint}, p-value: {pvalue:.4f}, hedge_ratio: {hedge_ratio:.4f}")
    except:
        print("❌ test_cointegration() not implemented")

    # Test 2: Spread
    print("\nTest 2: Spread Calculation")
    try:
        spread = strategy.calculate_spread(price1, price2, hedge_ratio_true)
        print(f"✅ Spread calculated: mean={spread.mean():.4f}, std={spread.std():.4f}")
    except:
        print("❌ calculate_spread() not implemented")

    # Test 3: Z-score
    print("\nTest 3: Z-Score Calculation")
    try:
        zscore = strategy.calculate_zscore(spread, lookback=60)
        print(f"✅ Z-score calculated: min={zscore.min():.2f}, max={zscore.max():.2f}")
    except:
        print("❌ calculate_zscore() not implemented")

    # Test 4: Half-life
    print("\nTest 4: Half-Life Calculation")
    try:
        half_life = strategy.calculate_half_life(spread)
        print(f"✅ Half-life: {half_life:.2f} days")
    except:
        print("❌ calculate_half_life() not implemented")

    # Test 5: Signals
    print("\nTest 5: Signal Generation")
    try:
        entry = strategy.generate_entry_signals(zscore)
        print(f"✅ Entry signals generated: {entry.abs().sum():.0f} entries")
    except:
        print("❌ generate_entry_signals() not implemented")

    # Test 6: Full strategy
    print("\nTest 6: Full Strategy")
    try:
        result = strategy.run_strategy(price1, price2)
        signals = result['signals']
        print(f"✅ Strategy run successfully!")
        print(f"   Trades: {signals.diff().abs().sum() / 2:.0f}")
        print(f"   Time in market: {(signals != 0).sum() / len(signals) * 100:.1f}%")
    except:
        print("❌ run_strategy() not implemented")

    # Test 7: Returns
    print("\nTest 7: Strategy Returns")
    try:
        returns = strategy.calculate_strategy_returns(price1, price2, signals, hedge_ratio_true)
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        print(f"✅ Returns calculated!")
        print(f"   Total return: {total_return*100:.2f}%")
        print(f"   Sharpe ratio: {sharpe:.2f}")
    except:
        print("❌ calculate_strategy_returns() not implemented")

    print("\n" + "="*50)
    print("🎉 Testing complete!")
    print("="*50)


if __name__ == "__main__":
    test_implementation()
