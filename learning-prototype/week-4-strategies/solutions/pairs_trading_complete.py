"""
Pairs Trading Strategy - Complete Solution

This is a complete implementation of a statistical arbitrage pairs trading strategy.
All functions are fully implemented and production-ready.

Author: Learning Lab Week 4
"""

from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm


@dataclass
class PairsTradingConfig:
    """Configuration for pairs trading strategy."""
    lookback_period: int = 60
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_loss_zscore: float = 3.5
    cointegration_pvalue: float = 0.05
    min_half_life: int = 1
    max_half_life: int = 30


class PairsTradingStrategy:
    """Statistical arbitrage pairs trading strategy - Complete Implementation."""

    def __init__(self, config: Optional[PairsTradingConfig] = None):
        """Initialize pairs trading strategy."""
        self.config = config or PairsTradingConfig()
        self.current_position = 0

    def test_cointegration(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> Tuple[bool, float, float]:
        """Test if two price series are cointegrated using Engle-Granger test."""
        # Run cointegration test
        score, pvalue, _ = coint(price1, price2)

        # Calculate hedge ratio
        hedge_ratio = self.calculate_hedge_ratio(price1, price2)

        # Check if cointegrated
        is_cointegrated = pvalue < self.config.cointegration_pvalue

        return is_cointegrated, pvalue, hedge_ratio

    def calculate_hedge_ratio(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> float:
        """Calculate optimal hedge ratio using linear regression."""
        # Prepare data for regression
        X = price2.values.reshape(-1, 1)
        y = price1.values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Return hedge ratio (beta coefficient)
        return model.coef_[0]

    def calculate_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """Calculate the spread between two assets."""
        spread = price1 - hedge_ratio * price2
        return spread

    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """Calculate rolling z-score of spread."""
        if lookback is None:
            lookback = self.config.lookback_period

        # Calculate rolling mean and std
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()

        # Calculate z-score
        zscore = (spread - rolling_mean) / rolling_std

        return zscore

    def calculate_half_life(
        self,
        spread: pd.Series
    ) -> float:
        """Calculate half-life of mean reversion using Ornstein-Uhlenbeck."""
        # Calculate lagged spread
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag

        # Drop NaN values
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()

        # Align indices
        spread_lag = spread_lag[spread_diff.index]

        # Run OLS regression: spread_diff ~ spread_lag
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_diff, X)
        result = model.fit()

        # Extract theta (mean reversion speed)
        theta = result.params[1]

        # Calculate half-life
        if theta < 0:
            half_life = -np.log(2) / theta
        else:
            half_life = np.inf

        return half_life

    def is_spread_stationary(
        self,
        spread: pd.Series,
        significance: float = 0.05
    ) -> bool:
        """Test if spread is stationary using Augmented Dickey-Fuller test."""
        # Run ADF test
        result = adfuller(spread.dropna())
        pvalue = result[1]

        # Stationary if p-value < significance
        return pvalue < significance

    def generate_entry_signals(
        self,
        zscore: pd.Series
    ) -> pd.Series:
        """Generate entry signals based on z-score."""
        signals = pd.Series(0, index=zscore.index)

        # Short spread when z > entry threshold
        signals[zscore > self.config.entry_zscore] = -1

        # Long spread when z < -entry threshold
        signals[zscore < -self.config.entry_zscore] = 1

        return signals

    def generate_exit_signals(
        self,
        zscore: pd.Series,
        position: pd.Series
    ) -> pd.Series:
        """Generate exit signals based on z-score and current position."""
        exit_signals = pd.Series(0, index=zscore.index)

        # Exit long positions
        long_exit = (position == 1) & (zscore > -self.config.exit_zscore)
        exit_signals[long_exit] = 1

        # Exit short positions
        short_exit = (position == -1) & (zscore < self.config.exit_zscore)
        exit_signals[short_exit] = 1

        # Stop loss exits
        stop_loss = (
            ((position == 1) & (zscore < -self.config.stop_loss_zscore)) |
            ((position == -1) & (zscore > self.config.stop_loss_zscore))
        )
        exit_signals[stop_loss] = 1

        return exit_signals

    def combine_signals(
        self,
        entry_signals: pd.Series,
        exit_signals: pd.Series
    ) -> pd.Series:
        """Combine entry and exit signals into position series."""
        positions = pd.Series(0, index=entry_signals.index)

        # Start with no position
        current_position = 0

        for i in range(len(positions)):
            # Check for exit first
            if exit_signals.iloc[i] == 1:
                current_position = 0
            # Then check for entry
            elif entry_signals.iloc[i] != 0:
                current_position = entry_signals.iloc[i]

            positions.iloc[i] = current_position

        return positions

    def run_strategy(
        self,
        price1: pd.Series,
        price2: pd.Series,
        validate_pair: bool = True
    ) -> Dict:
        """Run complete pairs trading strategy."""
        # Test cointegration
        if validate_pair:
            is_coint, pvalue, hedge_ratio = self.test_cointegration(price1, price2)

            if not is_coint:
                # Return empty results if not cointegrated
                return {
                    'is_cointegrated': False,
                    'p_value': pvalue,
                    'hedge_ratio': hedge_ratio,
                    'spread': pd.Series(0, index=price1.index),
                    'zscore': pd.Series(0, index=price1.index),
                    'half_life': np.inf,
                    'signals': pd.Series(0, index=price1.index)
                }
        else:
            hedge_ratio = self.calculate_hedge_ratio(price1, price2)
            is_coint = True
            pvalue = 0.0

        # Calculate spread
        spread = self.calculate_spread(price1, price2, hedge_ratio)

        # Calculate z-score
        zscore = self.calculate_zscore(spread)

        # Calculate half-life
        half_life = self.calculate_half_life(spread)

        # Generate signals
        entry_signals = self.generate_entry_signals(zscore)
        exit_signals = self.generate_exit_signals(zscore, entry_signals)
        signals = self.combine_signals(entry_signals, exit_signals)

        # Forward fill signals to maintain positions
        signals = signals.replace(0, np.nan).ffill().fillna(0)

        return {
            'is_cointegrated': is_coint,
            'p_value': pvalue,
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'zscore': zscore,
            'half_life': half_life,
            'signals': signals
        }

    def calculate_strategy_returns(
        self,
        price1: pd.Series,
        price2: pd.Series,
        signals: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """Calculate strategy returns from signals."""
        # Calculate returns
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()

        # Calculate spread return
        spread_return = returns1 - hedge_ratio * returns2

        # Lag signals to avoid look-ahead bias
        lagged_signals = signals.shift(1)

        # Strategy returns
        strategy_returns = lagged_signals * spread_return

        return strategy_returns.fillna(0)

    def backtest_signal_function(
        self,
        data: pd.DataFrame,
        symbol1: str = 'asset1',
        symbol2: str = 'asset2',
        **kwargs
    ) -> pd.Series:
        """Signal function for use with VectorizedBacktester."""
        # Extract prices
        price1 = data[symbol1]
        price2 = data[symbol2]

        # Run strategy
        result = self.run_strategy(price1, price2, validate_pair=True)

        # Return signals
        return result['signals']


if __name__ == "__main__":
    print("Pairs Trading Strategy - Complete Solution")
    print("=" * 50)
    print("\nThis is a production-ready implementation with all functions complete.")
    print("\nTo test this strategy, run:")
    print("  python ../exercises/exercise_1_pairs.py")
    print("\nOr import and use in your own code:")
    print("  from solutions.pairs_trading_complete import PairsTradingStrategy")
