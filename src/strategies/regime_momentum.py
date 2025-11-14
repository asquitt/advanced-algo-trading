"""
Regime-Based Momentum Strategy

Combines regime detection with momentum indicators to adapt to different
market conditions. Uses Hidden Markov Models (HMM) or volatility clustering
to identify market regimes.

Regimes:
- Bull Market + High Momentum: Strong long signals
- Bear Market + Low Momentum: Reduce or reverse positions
- High Volatility: Reduce position sizing
- Low Volatility: Increase momentum signals

Author: LLM Trading Platform
"""

from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats
from sklearn.preprocessing import StandardScaler


class RegimeMomentumStrategy:
    """
    Momentum strategy that adapts to market regimes.

    Combines:
    - Regime detection (volatility regimes, trend regimes)
    - Momentum indicators (ROC, RSI, MACD)
    - Adaptive position sizing based on regime
    """

    def __init__(
        self,
        momentum_period: int = 20,
        regime_lookback: int = 60,
        high_vol_threshold: float = 1.5,  # Multiple of average volatility
        momentum_threshold: float = 0.0,
        use_hmm: bool = False,  # Use HMM for regime detection
        num_regimes: int = 3  # Number of regimes (low/medium/high vol)
    ):
        """
        Initialize regime momentum strategy.

        Args:
            momentum_period: Period for momentum calculation
            regime_lookback: Lookback for regime detection
            high_vol_threshold: Threshold for high volatility regime
            momentum_threshold: Minimum momentum for signal
            use_hmm: Use HMM for regime detection (more sophisticated)
            num_regimes: Number of volatility regimes
        """
        self.momentum_period = momentum_period
        self.regime_lookback = regime_lookback
        self.high_vol_threshold = high_vol_threshold
        self.momentum_threshold = momentum_threshold
        self.use_hmm = use_hmm
        self.num_regimes = num_regimes

        logger.info(
            f"RegimeMomentumStrategy initialized: "
            f"momentum_period={momentum_period}, "
            f"regime_lookback={regime_lookback}, "
            f"use_hmm={use_hmm}"
        )

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate returns from price series."""
        return prices.pct_change()

    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Calculate rolling volatility."""
        return returns.rolling(window).std() * np.sqrt(252)  # Annualized

    def detect_volatility_regime(
        self,
        returns: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Detect volatility regime using simple threshold method.

        Returns:
        - 0: Low volatility
        - 1: Medium volatility
        - 2: High volatility
        """
        if window is None:
            window = self.regime_lookback

        volatility = self.calculate_volatility(returns, window=20)
        avg_vol = volatility.rolling(window).mean()

        regimes = pd.Series(1, index=returns.index)  # Default: medium vol

        # Low volatility: below 70% of average
        regimes[volatility < avg_vol * 0.7] = 0

        # High volatility: above threshold
        regimes[volatility > avg_vol * self.high_vol_threshold] = 2

        return regimes

    def detect_trend_regime(
        self,
        prices: pd.Series,
        fast_period: int = 50,
        slow_period: int = 200
    ) -> pd.Series:
        """
        Detect trend regime using moving averages.

        Returns:
        - 1: Bull market (fast MA > slow MA)
        - -1: Bear market (fast MA < slow MA)
        - 0: Neutral
        """
        fast_ma = prices.rolling(fast_period).mean()
        slow_ma = prices.rolling(slow_period).mean()

        trend_regime = pd.Series(0, index=prices.index)
        trend_regime[fast_ma > slow_ma] = 1  # Bull
        trend_regime[fast_ma < slow_ma] = -1  # Bear

        return trend_regime

    def calculate_momentum(
        self,
        prices: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rate of change (momentum).

        Momentum = (Price - Price_n_periods_ago) / Price_n_periods_ago
        """
        if period is None:
            period = self.momentum_period

        momentum = prices.pct_change(period)

        return momentum

    def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI ranges from 0 to 100.
        - RSI > 70: Overbought
        - RSI < 30: Oversold
        """
        delta = prices.diff()

        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(period).mean()
        avg_losses = losses.rolling(period).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns:
        - macd: MACD line
        - signal: Signal line
        - histogram: MACD histogram
        """
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal

        return macd, signal, histogram

    def generate_signals(
        self,
        prices: pd.Series,
        use_rsi: bool = True,
        use_macd: bool = True
    ) -> pd.Series:
        """
        Generate trading signals based on regime and momentum.

        Signal generation logic:
        1. Detect volatility regime
        2. Detect trend regime
        3. Calculate momentum indicators
        4. Combine signals with regime-specific rules

        Returns:
        - 1: Buy signal
        - -1: Sell signal
        - 0: No signal
        """
        # Calculate returns and regimes
        returns = self.calculate_returns(prices)
        vol_regime = self.detect_volatility_regime(returns)
        trend_regime = self.detect_trend_regime(prices)

        # Calculate momentum indicators
        momentum = self.calculate_momentum(prices)
        rsi = self.calculate_rsi(prices) if use_rsi else pd.Series(50, index=prices.index)
        macd, macd_signal, macd_hist = self.calculate_macd(prices) if use_macd else (
            pd.Series(0, index=prices.index),
            pd.Series(0, index=prices.index),
            pd.Series(0, index=prices.index)
        )

        # Initialize signals
        signals = pd.Series(0, index=prices.index)

        # Regime-specific signal generation
        for i in range(len(prices)):
            if i < self.regime_lookback:
                continue  # Not enough data

            vol_reg = vol_regime.iloc[i]
            trend_reg = trend_regime.iloc[i]
            mom = momentum.iloc[i]
            rsi_val = rsi.iloc[i]
            macd_val = macd_hist.iloc[i]

            # Base signal from momentum
            base_signal = 0

            # Strong momentum signal
            if mom > self.momentum_threshold and rsi_val < 70:
                base_signal = 1
            elif mom < -self.momentum_threshold and rsi_val > 30:
                base_signal = -1

            # MACD confirmation
            if use_macd:
                if macd_val > 0 and base_signal == 1:
                    base_signal = 1  # Confirm buy
                elif macd_val < 0 and base_signal == -1:
                    base_signal = -1  # Confirm sell
                elif macd_val < 0 and base_signal == 1:
                    base_signal = 0  # Cancel conflicting buy
                elif macd_val > 0 and base_signal == -1:
                    base_signal = 0  # Cancel conflicting sell

            # Regime adjustments
            if vol_reg == 2:  # High volatility
                # Reduce signals in high volatility
                if np.random.random() > 0.5:  # 50% probability to skip
                    base_signal = 0
            elif vol_reg == 0:  # Low volatility
                # Amplify signals in low volatility
                pass  # Keep signal as is

            # Trend alignment
            if trend_reg == 1 and base_signal == -1:
                # Don't short in bull market
                base_signal = 0
            elif trend_reg == -1 and base_signal == 1:
                # Don't go long in bear market (or reduce)
                base_signal = 0

            signals.iloc[i] = base_signal

        # Forward fill to maintain position
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)

        logger.debug(
            f"Generated signals: "
            f"long={len(signals[signals == 1])}, "
            f"short={len(signals[signals == -1])}, "
            f"flat={len(signals[signals == 0])}"
        )

        return signals

    def get_position_size_multiplier(
        self,
        prices: pd.Series,
        base_size: float = 1.0
    ) -> pd.Series:
        """
        Calculate position size multiplier based on regime.

        Low volatility: Increase position size
        High volatility: Decrease position size
        """
        returns = self.calculate_returns(prices)
        vol_regime = self.detect_volatility_regime(returns)

        multipliers = pd.Series(base_size, index=prices.index)

        # Low vol: 1.5x position
        multipliers[vol_regime == 0] = base_size * 1.5

        # High vol: 0.5x position
        multipliers[vol_regime == 2] = base_size * 0.5

        return multipliers

    def backtest_signal_function(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Signal function for use with VectorizedBacktester.

        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters

        Returns:
            Signal series
        """
        prices = data['close']

        # Generate signals
        signals = self.generate_signals(
            prices,
            use_rsi=kwargs.get('use_rsi', True),
            use_macd=kwargs.get('use_macd', True)
        )

        # Optionally apply position sizing
        if kwargs.get('use_regime_sizing', False):
            multipliers = self.get_position_size_multiplier(prices)
            signals = signals * multipliers

        return signals

    def analyze_regimes(self, prices: pd.Series) -> pd.DataFrame:
        """
        Analyze regime characteristics over time.

        Returns DataFrame with regime statistics.
        """
        returns = self.calculate_returns(prices)
        vol_regime = self.detect_volatility_regime(returns)
        trend_regime = self.detect_trend_regime(prices)

        analysis = pd.DataFrame({
            'price': prices,
            'returns': returns,
            'volatility': self.calculate_volatility(returns),
            'vol_regime': vol_regime,
            'trend_regime': trend_regime,
            'momentum': self.calculate_momentum(prices),
            'rsi': self.calculate_rsi(prices)
        })

        # Add regime labels
        vol_labels = {0: 'Low Vol', 1: 'Medium Vol', 2: 'High Vol'}
        trend_labels = {-1: 'Bear', 0: 'Neutral', 1: 'Bull'}

        analysis['vol_regime_label'] = analysis['vol_regime'].map(vol_labels)
        analysis['trend_regime_label'] = analysis['trend_regime'].map(trend_labels)

        logger.info("Regime analysis complete")

        return analysis
