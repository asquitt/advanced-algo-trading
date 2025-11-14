"""
Sentiment-Driven Intraday Trading Strategy - Complete Solution

This is a complete implementation of a sentiment-based intraday trading strategy.
All functions are fully implemented and production-ready.

Author: Learning Lab Week 4
"""

from typing import Optional, Dict
from dataclasses import dataclass
from datetime import time
import pandas as pd
import numpy as np


@dataclass
class SentimentConfig:
    """Configuration for sentiment intraday strategy."""
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    trading_start: time = time(9, 35)
    exit_time: time = time(15, 55)
    sentiment_threshold: float = 0.3
    lookback_minutes: int = 60
    news_weight: float = 0.5
    social_weight: float = 0.3
    analyst_weight: float = 0.2
    use_technical_filter: bool = True
    volume_multiplier: float = 1.5
    price_momentum_threshold: float = 0.001
    max_position_size: float = 1.0
    scale_by_confidence: bool = True
    max_trades_per_day: int = 3


class SentimentIntradayStrategy:
    """Sentiment-driven intraday trading strategy - Complete Implementation."""

    def __init__(self, config: Optional[SentimentConfig] = None):
        """Initialize sentiment intraday strategy."""
        self.config = config or SentimentConfig()

    def is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is during market hours."""
        t = timestamp.time()
        return self.config.market_open <= t <= self.config.market_close

    def is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is during valid trading hours."""
        t = timestamp.time()
        return self.config.trading_start <= t <= self.config.exit_time

    def should_exit_all_positions(self, timestamp: pd.Timestamp) -> bool:
        """Check if we should exit all positions (near market close)."""
        return timestamp.time() >= self.config.exit_time

    def normalize_sentiment(
        self,
        sentiment: pd.Series,
        method: str = 'tanh'
    ) -> pd.Series:
        """Normalize sentiment scores to [-1, 1] range."""
        if method == 'tanh':
            return np.tanh(sentiment)
        elif method == 'clip':
            return sentiment.clip(-1, 1)
        else:  # minmax
            s_min, s_max = sentiment.min(), sentiment.max()
            if s_max - s_min > 0:
                return 2 * (sentiment - s_min) / (s_max - s_min) - 1
            return sentiment

    def aggregate_sentiment_sources(
        self,
        news_sentiment: pd.Series,
        social_sentiment: pd.Series,
        analyst_sentiment: pd.Series
    ) -> pd.Series:
        """Aggregate sentiment from multiple sources using weights."""
        # Fill missing values
        news_sentiment = news_sentiment.fillna(0)
        social_sentiment = social_sentiment.fillna(0)
        analyst_sentiment = analyst_sentiment.fillna(0)

        # Weighted sum
        aggregated = (
            self.config.news_weight * news_sentiment +
            self.config.social_weight * social_sentiment +
            self.config.analyst_weight * analyst_sentiment
        )

        return aggregated

    def calculate_rolling_sentiment(
        self,
        sentiment: pd.Series,
        window_minutes: Optional[int] = None
    ) -> pd.Series:
        """Calculate rolling average sentiment."""
        if window_minutes is None:
            window_minutes = self.config.lookback_minutes

        rolling = sentiment.rolling(window=window_minutes).mean()
        return rolling

    def calculate_sentiment_momentum(
        self,
        sentiment: pd.Series,
        lookback: int = 5
    ) -> pd.Series:
        """Calculate rate of change in sentiment."""
        momentum = sentiment - sentiment.shift(lookback)
        return momentum

    def calculate_volume_ratio(
        self,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """Calculate volume relative to average."""
        avg_volume = volume.rolling(window=lookback).mean()
        volume_ratio = volume / avg_volume
        return volume_ratio

    def calculate_price_momentum(
        self,
        prices: pd.Series,
        lookback: int = 5
    ) -> pd.Series:
        """Calculate short-term price momentum."""
        momentum = prices / prices.shift(lookback) - 1
        return momentum

    def check_technical_confirmation(
        self,
        prices: pd.Series,
        volume: pd.Series,
        sentiment_signal: pd.Series
    ) -> pd.Series:
        """Check if technical indicators confirm sentiment signal."""
        # Calculate indicators
        volume_ratio = self.calculate_volume_ratio(volume)
        price_momentum = self.calculate_price_momentum(prices)

        # Check confirmation
        confirmed = pd.Series(False, index=prices.index)

        # For long signals (sentiment > 0)
        long_confirmed = (
            (sentiment_signal > 0) &
            (price_momentum > self.config.price_momentum_threshold) &
            (volume_ratio > self.config.volume_multiplier)
        )

        # For short signals (sentiment < 0)
        short_confirmed = (
            (sentiment_signal < 0) &
            (price_momentum < -self.config.price_momentum_threshold) &
            (volume_ratio > self.config.volume_multiplier)
        )

        confirmed = long_confirmed | short_confirmed

        return confirmed

    def generate_sentiment_signals(
        self,
        sentiment: pd.Series
    ) -> pd.Series:
        """Generate raw signals from sentiment."""
        signals = pd.Series(0, index=sentiment.index)

        # Long when sentiment > threshold
        signals[sentiment > self.config.sentiment_threshold] = 1

        # Short when sentiment < -threshold
        signals[sentiment < -self.config.sentiment_threshold] = -1

        return signals

    def filter_signals_by_time(
        self,
        signals: pd.Series,
        timestamps: pd.DatetimeIndex
    ) -> pd.Series:
        """Filter signals to only trading hours."""
        filtered = signals.copy()

        for i, ts in enumerate(timestamps):
            if not self.is_trading_hours(ts):
                filtered.iloc[i] = 0

        return filtered

    def apply_intraday_exit_rule(
        self,
        signals: pd.Series,
        timestamps: pd.DatetimeIndex
    ) -> pd.Series:
        """Apply rule to exit all positions near market close."""
        adjusted = signals.copy()

        for i, ts in enumerate(timestamps):
            if self.should_exit_all_positions(ts):
                adjusted.iloc[i] = 0

        return adjusted

    def calculate_confidence_score(
        self,
        sentiment: pd.Series,
        sentiment_momentum: pd.Series,
        technical_confirmed: pd.Series
    ) -> pd.Series:
        """Calculate confidence score for position sizing."""
        # Base confidence from sentiment strength
        confidence = abs(sentiment)

        # Boost for positive sentiment momentum
        momentum_boost = sentiment_momentum.clip(-0.5, 0.5) + 0.5
        confidence = confidence * momentum_boost

        # Boost for technical confirmation
        confirmation_boost = technical_confirmed.astype(float) * 0.5 + 0.5
        confidence = confidence * confirmation_boost

        # Normalize to [0, 1]
        confidence = confidence.clip(0, 1)

        return confidence

    def calculate_position_size(
        self,
        signals: pd.Series,
        confidence: pd.Series
    ) -> pd.Series:
        """Calculate position size based on signals and confidence."""
        if self.config.scale_by_confidence:
            position_size = signals * confidence * self.config.max_position_size
        else:
            position_size = signals * self.config.max_position_size

        return position_size

    def enforce_max_trades_per_day(
        self,
        signals: pd.Series,
        timestamps: pd.DatetimeIndex
    ) -> pd.Series:
        """Enforce maximum number of trades per day."""
        adjusted = signals.copy()

        # Group by date
        dates = pd.Series([ts.date() for ts in timestamps], index=timestamps)

        for date in dates.unique():
            date_mask = dates == date
            date_signals = signals[date_mask]

            # Count signal changes (new trades)
            signal_changes = date_signals.diff().abs()
            trade_count = 0

            for i in date_signals.index:
                if signal_changes[i] > 0:
                    trade_count += 1
                    if trade_count > self.config.max_trades_per_day:
                        # Block this trade
                        adjusted[i] = adjusted[date_signals.index[date_signals.index.get_loc(i) - 1]]

        return adjusted

    def calculate_intraday_stop_loss(
        self,
        entry_price: float,
        direction: int,
        atr: float,
        multiplier: float = 2.0
    ) -> float:
        """Calculate intraday stop loss level."""
        if direction == 1:  # Long
            stop = entry_price - multiplier * atr
        else:  # Short
            stop = entry_price + multiplier * atr

        return stop

    def run_strategy(
        self,
        prices: pd.Series,
        volume: pd.Series,
        sentiment_data: Dict[str, pd.Series],
        timestamps: pd.DatetimeIndex
    ) -> Dict:
        """Run complete sentiment intraday strategy."""
        # Normalize sentiment from each source
        news_norm = self.normalize_sentiment(sentiment_data['news'])
        social_norm = self.normalize_sentiment(sentiment_data['social'])
        analyst_norm = self.normalize_sentiment(sentiment_data['analyst'])

        # Aggregate sentiment
        sentiment = self.aggregate_sentiment_sources(news_norm, social_norm, analyst_norm)

        # Calculate rolling sentiment
        rolling_sentiment = self.calculate_rolling_sentiment(sentiment)

        # Calculate sentiment momentum
        sentiment_momentum = self.calculate_sentiment_momentum(rolling_sentiment)

        # Generate base signals
        base_signals = self.generate_sentiment_signals(rolling_sentiment)

        # Check technical confirmation
        if self.config.use_technical_filter:
            technical_confirmed = self.check_technical_confirmation(prices, volume, base_signals)
            # Filter signals by technical confirmation
            signals = base_signals.copy()
            signals[~technical_confirmed] = 0
        else:
            technical_confirmed = pd.Series(True, index=prices.index)
            signals = base_signals

        # Filter by trading hours
        signals = self.filter_signals_by_time(signals, timestamps)

        # Apply intraday exit rule
        signals = self.apply_intraday_exit_rule(signals, timestamps)

        # Calculate confidence scores
        confidence = self.calculate_confidence_score(
            rolling_sentiment,
            sentiment_momentum,
            technical_confirmed
        )

        # Calculate position sizes
        position_size = self.calculate_position_size(signals, confidence)

        # Enforce max trades per day
        position_size = self.enforce_max_trades_per_day(position_size, timestamps)

        return {
            'signals': np.sign(position_size),
            'position_size': position_size,
            'sentiment': sentiment,
            'sentiment_momentum': sentiment_momentum,
            'confidence': confidence,
            'technical_confirmed': technical_confirmed
        }

    def calculate_strategy_returns(
        self,
        prices: pd.Series,
        position_size: pd.Series
    ) -> pd.Series:
        """Calculate strategy returns."""
        # Calculate returns
        returns = prices.pct_change()

        # Lag position size to avoid look-ahead bias
        lagged_position = position_size.shift(1)

        # Strategy returns
        strategy_returns = lagged_position * returns

        return strategy_returns.fillna(0)


if __name__ == "__main__":
    print("Sentiment-Driven Intraday Strategy - Complete Solution")
    print("=" * 50)
    print("\nThis is a production-ready implementation with all functions complete.")
    print("\nTo test this strategy, run:")
    print("  python ../exercises/exercise_3_sentiment.py")
