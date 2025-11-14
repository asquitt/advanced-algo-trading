"""
Pairs Trading Strategy (Mean Reversion)

Statistical arbitrage strategy that trades cointegrated pairs of stocks.
When the spread between pairs deviates from its mean, the strategy bets
on mean reversion.

Key Concepts:
- Cointegration: Two non-stationary series that have a stationary linear combination
- Z-score: Standardized deviation from mean (measures how "far" spread has moved)
- Mean reversion: Spread tends to return to its historical mean

Author: LLM Trading Platform
"""

from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
from loguru import logger
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression


class PairsTradingStrategy:
    """
    Statistical arbitrage strategy using cointegration and mean reversion.

    The strategy:
    1. Identifies cointegrated pairs (using Engle-Granger test)
    2. Calculates the spread between pairs
    3. Generates signals when z-score exceeds entry thresholds
    4. Exits when spread reverts to mean
    """

    def __init__(
        self,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        lookback_period: int = 60,
        min_half_life: int = 1,
        max_half_life: int = 30,
        cointegration_pvalue: float = 0.05
    ):
        """
        Initialize pairs trading strategy.

        Args:
            entry_z_score: Z-score threshold for entry (e.g., 2.0 = 2 std devs)
            exit_z_score: Z-score threshold for exit (e.g., 0.5 = 0.5 std devs)
            lookback_period: Period for calculating mean and std of spread
            min_half_life: Minimum acceptable half-life (days)
            max_half_life: Maximum acceptable half-life (days)
            cointegration_pvalue: P-value threshold for cointegration test
        """
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.lookback_period = lookback_period
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.cointegration_pvalue = cointegration_pvalue

        logger.info(
            f"PairsTradingStrategy initialized: "
            f"entry_z={entry_z_score}, "
            f"exit_z={exit_z_score}, "
            f"lookback={lookback_period}"
        )

    def test_cointegration(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> Tuple[bool, float, float]:
        """
        Test if two price series are cointegrated using Engle-Granger test.

        Args:
            price1: First price series
            price2: Second price series

        Returns:
            (is_cointegrated, p_value, hedge_ratio)
        """
        # Remove NaN values
        df = pd.DataFrame({'p1': price1, 'p2': price2}).dropna()

        if len(df) < 30:  # Need minimum data points
            return False, 1.0, 0.0

        # Engle-Granger cointegration test
        _, p_value, _ = coint(df['p1'], df['p2'])

        # Calculate hedge ratio using linear regression
        X = df['p2'].values.reshape(-1, 1)
        y = df['p1'].values
        model = LinearRegression()
        model.fit(X, y)
        hedge_ratio = model.coef_[0]

        is_cointegrated = p_value < self.cointegration_pvalue

        logger.debug(
            f"Cointegration test: p-value={p_value:.4f}, "
            f"hedge_ratio={hedge_ratio:.4f}, "
            f"cointegrated={is_cointegrated}"
        )

        return is_cointegrated, p_value, hedge_ratio

    def calculate_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate spread between two price series.

        Spread = price1 - hedge_ratio * price2

        Args:
            price1: First price series
            price2: Second price series
            hedge_ratio: Hedge ratio from cointegration test (if None, will calculate)

        Returns:
            Spread series
        """
        if hedge_ratio is None:
            # Calculate hedge ratio using linear regression
            df = pd.DataFrame({'p1': price1, 'p2': price2}).dropna()
            X = df['p2'].values.reshape(-1, 1)
            y = df['p1'].values
            model = LinearRegression()
            model.fit(X, y)
            hedge_ratio = model.coef_[0]

        spread = price1 - hedge_ratio * price2

        return spread

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion.

        Half-life is the expected time for the spread to move halfway
        back to its mean. Used to validate mean reversion speed.

        Args:
            spread: Spread series

        Returns:
            Half-life in periods (days)
        """
        spread_lag = spread.shift(1)
        spread_delta = spread - spread_lag

        # Remove NaN
        df = pd.DataFrame({'spread_lag': spread_lag, 'spread_delta': spread_delta}).dropna()

        if len(df) < 30:
            return np.inf

        # Regression: spread_delta = alpha + beta * spread_lag
        X = df['spread_lag'].values.reshape(-1, 1)
        y = df['spread_delta'].values
        model = LinearRegression()
        model.fit(X, y)

        beta = model.coef_[0]

        # Half-life = -log(2) / log(1 + beta)
        if beta >= 0:
            return np.inf  # No mean reversion

        half_life = -np.log(2) / np.log(1 + beta)

        return half_life

    def calculate_z_score(
        self,
        spread: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate z-score of spread.

        Z-score = (spread - rolling_mean) / rolling_std

        Args:
            spread: Spread series
            lookback: Lookback period for rolling statistics

        Returns:
            Z-score series
        """
        if lookback is None:
            lookback = self.lookback_period

        rolling_mean = spread.rolling(lookback).mean()
        rolling_std = spread.rolling(lookback).std()

        z_score = (spread - rolling_mean) / rolling_std

        return z_score

    def generate_signals(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Generate trading signals based on z-score thresholds.

        Signals:
        - 1: Long spread (buy stock1, sell stock2) when z-score < -entry_threshold
        - -1: Short spread (sell stock1, buy stock2) when z-score > entry_threshold
        - 0: Flat when |z-score| < exit_threshold

        Args:
            price1: First stock prices
            price2: Second stock prices
            hedge_ratio: Hedge ratio (if None, will calculate)

        Returns:
            Signal series (-1, 0, 1)
        """
        # Calculate spread
        spread = self.calculate_spread(price1, price2, hedge_ratio)

        # Calculate z-score
        z_score = self.calculate_z_score(spread)

        # Generate signals based on z-score
        signals = pd.Series(0, index=price1.index)

        # Entry signals
        signals[z_score < -self.entry_z_score] = 1  # Long spread
        signals[z_score > self.entry_z_score] = -1  # Short spread

        # Exit signals (flatten position when spread reverts)
        signals[z_score.abs() < self.exit_z_score] = 0

        # Forward fill to maintain position until exit
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)

        logger.debug(
            f"Generated signals: "
            f"long={len(signals[signals == 1])}, "
            f"short={len(signals[signals == -1])}, "
            f"flat={len(signals[signals == 0])}"
        )

        return signals

    def backtest_signal_function(
        self,
        data: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        **kwargs
    ) -> pd.Series:
        """
        Signal function for use with VectorizedBacktester.

        This is a wrapper around generate_signals() that extracts
        prices from a multi-column DataFrame.

        Args:
            data: DataFrame with columns like 'close_AAPL', 'close_MSFT'
            symbol1: First symbol (e.g., 'AAPL')
            symbol2: Second symbol (e.g., 'MSFT')
            **kwargs: Additional parameters

        Returns:
            Signal series
        """
        # Extract prices for each symbol
        price1_col = f'close_{symbol1}'
        price2_col = f'close_{symbol2}'

        if price1_col not in data.columns or price2_col not in data.columns:
            # Fallback: assume data has 'close' column for single pair
            if 'close' in data.columns:
                # For single pair, we need both prices in separate columns
                # This is a limitation - backtester expects single instrument
                logger.warning("Pairs trading requires multi-asset data")
                return pd.Series(0, index=data.index)

            raise ValueError(f"Missing price columns: {price1_col} or {price2_col}")

        price1 = data[price1_col]
        price2 = data[price2_col]

        # Test cointegration
        is_coint, p_value, hedge_ratio = self.test_cointegration(price1, price2)

        if not is_coint:
            logger.warning(
                f"Pair {symbol1}/{symbol2} not cointegrated (p={p_value:.4f})"
            )
            return pd.Series(0, index=data.index)

        # Check half-life
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        half_life = self.calculate_half_life(spread)

        if half_life < self.min_half_life or half_life > self.max_half_life:
            logger.warning(
                f"Half-life {half_life:.1f} outside acceptable range "
                f"[{self.min_half_life}, {self.max_half_life}]"
            )
            return pd.Series(0, index=data.index)

        logger.info(
            f"Valid pair {symbol1}/{symbol2}: "
            f"p-value={p_value:.4f}, "
            f"hedge_ratio={hedge_ratio:.4f}, "
            f"half_life={half_life:.1f}"
        )

        # Generate signals
        signals = self.generate_signals(price1, price2, hedge_ratio)

        return signals

    def find_cointegrated_pairs(
        self,
        prices: Dict[str, pd.Series]
    ) -> List[Tuple[str, str, float, float]]:
        """
        Find all cointegrated pairs from a universe of stocks.

        Args:
            prices: Dictionary of {symbol: price_series}

        Returns:
            List of (symbol1, symbol2, p_value, hedge_ratio) tuples
        """
        symbols = list(prices.keys())
        cointegrated_pairs = []

        logger.info(f"Testing {len(symbols)} symbols for cointegration...")

        # Test all pairs
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]

                is_coint, p_value, hedge_ratio = self.test_cointegration(
                    prices[sym1],
                    prices[sym2]
                )

                if is_coint:
                    # Check half-life
                    spread = self.calculate_spread(prices[sym1], prices[sym2], hedge_ratio)
                    half_life = self.calculate_half_life(spread)

                    if self.min_half_life <= half_life <= self.max_half_life:
                        cointegrated_pairs.append((sym1, sym2, p_value, hedge_ratio))
                        logger.info(
                            f"Found pair: {sym1}/{sym2} "
                            f"(p={p_value:.4f}, hedge={hedge_ratio:.4f}, "
                            f"half_life={half_life:.1f})"
                        )

        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")

        return cointegrated_pairs

    def get_portfolio_weights(
        self,
        signal: float,
        hedge_ratio: float
    ) -> Tuple[float, float]:
        """
        Convert signal to portfolio weights for the two stocks.

        Args:
            signal: Trading signal (-1, 0, 1)
            hedge_ratio: Hedge ratio from cointegration

        Returns:
            (weight1, weight2) tuple
        """
        if signal == 0:
            return 0.0, 0.0

        # Long spread: buy stock1, sell stock2
        if signal == 1:
            weight1 = 1.0
            weight2 = -hedge_ratio
        # Short spread: sell stock1, buy stock2
        else:
            weight1 = -1.0
            weight2 = hedge_ratio

        # Normalize so total absolute weight = 1
        total_abs_weight = abs(weight1) + abs(weight2)
        weight1 = weight1 / total_abs_weight
        weight2 = weight2 / total_abs_weight

        return weight1, weight2
