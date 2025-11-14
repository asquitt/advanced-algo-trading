"""
Sentiment-Driven Intraday Strategy

Combines sentiment analysis from news and social media with technical indicators
for intraday trading. Uses real-time sentiment scores to generate trading signals.

Key Components:
- Sentiment aggregation from multiple sources (news, Twitter, Reddit)
- Technical confirmation (volume, volatility, support/resistance)
- Intraday position management (entry/exit within same day)
- Sentiment momentum tracking

Author: LLM Trading Platform
"""

from typing import Optional, Dict, List, Tuple
from datetime import datetime, time
import pandas as pd
import numpy as np
from loguru import logger


class SentimentIntradayStrategy:
    """
    Intraday trading strategy driven by real-time sentiment analysis.

    Strategy rules:
    1. Only trade during market hours (9:30 AM - 4:00 PM ET)
    2. Enter positions based on sentiment spikes + technical confirmation
    3. Exit positions before market close
    4. Use sentiment momentum for position sizing
    """

    def __init__(
        self,
        sentiment_threshold: float = 0.6,  # Minimum sentiment score for entry
        sentiment_change_threshold: float = 0.2,  # Minimum sentiment change
        volume_multiplier: float = 1.5,  # Required volume vs average
        max_holding_hours: int = 4,  # Maximum holding period
        market_open_hour: int = 9,  # Market open hour (ET)
        market_open_minute: int = 30,
        market_close_hour: int = 16,  # Market close hour (ET)
        use_technical_confirmation: bool = True
    ):
        """
        Initialize sentiment intraday strategy.

        Args:
            sentiment_threshold: Minimum absolute sentiment score for entry
            sentiment_change_threshold: Minimum sentiment change for signal
            volume_multiplier: Required volume multiple vs average
            max_holding_hours: Maximum time to hold position
            market_open_hour: Market open hour
            market_open_minute: Market open minute
            market_close_hour: Market close hour
            use_technical_confirmation: Require technical confirmation
        """
        self.sentiment_threshold = sentiment_threshold
        self.sentiment_change_threshold = sentiment_change_threshold
        self.volume_multiplier = volume_multiplier
        self.max_holding_hours = max_holding_hours
        self.market_open = time(market_open_hour, market_open_minute)
        self.market_close = time(market_close_hour, 0)
        self.use_technical_confirmation = use_technical_confirmation

        logger.info(
            f"SentimentIntradayStrategy initialized: "
            f"sentiment_threshold={sentiment_threshold}, "
            f"volume_multiplier={volume_multiplier}"
        )

    def aggregate_sentiment(
        self,
        news_sentiment: Optional[pd.Series] = None,
        twitter_sentiment: Optional[pd.Series] = None,
        reddit_sentiment: Optional[pd.Series] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Aggregate sentiment from multiple sources.

        Args:
            news_sentiment: Sentiment scores from news articles
            twitter_sentiment: Sentiment scores from Twitter
            reddit_sentiment: Sentiment scores from Reddit
            weights: Weights for each source (default: equal weight)

        Returns:
            Aggregated sentiment score
        """
        if weights is None:
            weights = {'news': 0.5, 'twitter': 0.3, 'reddit': 0.2}

        # Collect available sentiment series
        sentiments = []
        sentiment_weights = []

        if news_sentiment is not None:
            sentiments.append(news_sentiment)
            sentiment_weights.append(weights.get('news', 0.5))

        if twitter_sentiment is not None:
            sentiments.append(twitter_sentiment)
            sentiment_weights.append(weights.get('twitter', 0.3))

        if reddit_sentiment is not None:
            sentiments.append(reddit_sentiment)
            sentiment_weights.append(weights.get('reddit', 0.2))

        if not sentiments:
            # No sentiment data available
            return pd.Series(0.0)

        # Normalize weights
        total_weight = sum(sentiment_weights)
        normalized_weights = [w / total_weight for w in sentiment_weights]

        # Weighted average
        aggregated = sum(
            s * w for s, w in zip(sentiments, normalized_weights)
        )

        return aggregated

    def calculate_sentiment_momentum(
        self,
        sentiment: pd.Series,
        window: int = 10
    ) -> pd.Series:
        """
        Calculate sentiment momentum (rate of change).

        Args:
            sentiment: Sentiment score series
            window: Lookback window for momentum

        Returns:
            Sentiment momentum
        """
        sentiment_ma = sentiment.rolling(window).mean()
        momentum = sentiment - sentiment_ma

        return momentum

    def is_market_hours(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is during market hours.

        Args:
            timestamp: Datetime to check

        Returns:
            True if during market hours
        """
        current_time = timestamp.time()
        return self.market_open <= current_time < self.market_close

    def calculate_volume_surge(
        self,
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Detect volume surges (volume > average * multiplier).

        Args:
            volume: Volume series
            window: Lookback window for average

        Returns:
            Volume surge indicator (True/False)
        """
        avg_volume = volume.rolling(window).mean()
        volume_surge = volume > (avg_volume * self.volume_multiplier)

        return volume_surge

    def check_technical_confirmation(
        self,
        prices: pd.Series,
        volume: pd.Series,
        sentiment_signal: int
    ) -> bool:
        """
        Check technical confirmation for sentiment signal.

        Confirmation criteria:
        - Volume surge
        - Price moving in direction of sentiment
        - Not overbought/oversold

        Args:
            prices: Price series
            volume: Volume series
            sentiment_signal: Sentiment signal (1=bullish, -1=bearish)

        Returns:
            True if technically confirmed
        """
        if len(prices) < 20:
            return False

        # Volume confirmation
        volume_surge = self.calculate_volume_surge(volume)
        if not volume_surge.iloc[-1]:
            return False

        # Price momentum confirmation
        price_change = prices.pct_change(5).iloc[-1]

        if sentiment_signal == 1 and price_change < 0:
            # Bullish sentiment but price declining
            return False
        elif sentiment_signal == -1 and price_change > 0:
            # Bearish sentiment but price rising
            return False

        return True

    def generate_entry_signals(
        self,
        data: pd.DataFrame,
        sentiment: pd.Series
    ) -> pd.Series:
        """
        Generate entry signals based on sentiment and technicals.

        Args:
            data: OHLCV DataFrame
            sentiment: Sentiment score series

        Returns:
            Entry signals (1=long, -1=short, 0=no signal)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate sentiment momentum
        sent_momentum = self.calculate_sentiment_momentum(sentiment)

        # Iterate through data
        for i in range(len(data)):
            timestamp = data.index[i]

            # Only trade during market hours
            if not self.is_market_hours(timestamp):
                signals.iloc[i] = 0
                continue

            if i < 20:  # Need minimum data
                continue

            sent_score = sentiment.iloc[i]
            sent_change = sent_momentum.iloc[i]

            # Bullish signal
            if (sent_score > self.sentiment_threshold and
                sent_change > self.sentiment_change_threshold):

                # Technical confirmation
                if self.use_technical_confirmation:
                    confirmed = self.check_technical_confirmation(
                        data['close'].iloc[:i+1],
                        data['volume'].iloc[:i+1],
                        sentiment_signal=1
                    )
                    if not confirmed:
                        continue

                signals.iloc[i] = 1

            # Bearish signal
            elif (sent_score < -self.sentiment_threshold and
                  sent_change < -self.sentiment_change_threshold):

                # Technical confirmation
                if self.use_technical_confirmation:
                    confirmed = self.check_technical_confirmation(
                        data['close'].iloc[:i+1],
                        data['volume'].iloc[:i+1],
                        sentiment_signal=-1
                    )
                    if not confirmed:
                        continue

                signals.iloc[i] = -1

        return signals

    def manage_positions(
        self,
        data: pd.DataFrame,
        entry_signals: pd.Series,
        sentiment: pd.Series
    ) -> pd.Series:
        """
        Manage intraday positions with entry and exit logic.

        Exit rules:
        1. Sentiment reversal
        2. Maximum holding period exceeded
        3. Approaching market close (exit 30 min before close)

        Args:
            data: OHLCV DataFrame
            entry_signals: Entry signals
            sentiment: Sentiment scores

        Returns:
            Position series (1=long, -1=short, 0=flat)
        """
        positions = pd.Series(0, index=data.index)

        current_position = 0
        entry_time = None
        entry_sentiment = 0.0

        for i in range(len(data)):
            timestamp = data.index[i]
            current_time = timestamp.time()

            # Force close before market close
            close_time = time(self.market_close.hour - 1, 30)  # 30 min before close
            if current_time >= close_time:
                current_position = 0
                entry_time = None
                positions.iloc[i] = 0
                continue

            # Check for entry signal
            if entry_signals.iloc[i] != 0 and current_position == 0:
                current_position = entry_signals.iloc[i]
                entry_time = timestamp
                entry_sentiment = sentiment.iloc[i]
                positions.iloc[i] = current_position
                continue

            # Manage existing position
            if current_position != 0:
                # Exit on sentiment reversal
                if (current_position == 1 and sentiment.iloc[i] < 0) or \
                   (current_position == -1 and sentiment.iloc[i] > 0):
                    current_position = 0
                    entry_time = None
                    positions.iloc[i] = 0
                    continue

                # Exit on holding period exceeded
                if entry_time is not None:
                    holding_hours = (timestamp - entry_time).seconds / 3600
                    if holding_hours > self.max_holding_hours:
                        current_position = 0
                        entry_time = None
                        positions.iloc[i] = 0
                        continue

                # Maintain position
                positions.iloc[i] = current_position
            else:
                positions.iloc[i] = 0

        return positions

    def backtest_signal_function(
        self,
        data: pd.DataFrame,
        sentiment_column: str = 'sentiment',
        **kwargs
    ) -> pd.Series:
        """
        Signal function for use with VectorizedBacktester.

        Args:
            data: DataFrame with OHLCV and sentiment columns
            sentiment_column: Name of sentiment column
            **kwargs: Additional parameters

        Returns:
            Position series
        """
        if sentiment_column not in data.columns:
            logger.warning(f"Sentiment column '{sentiment_column}' not found in data")
            return pd.Series(0, index=data.index)

        sentiment = data[sentiment_column]

        # Generate entry signals
        entry_signals = self.generate_entry_signals(data, sentiment)

        # Manage positions
        positions = self.manage_positions(data, entry_signals, sentiment)

        logger.debug(
            f"Sentiment strategy: "
            f"{len(positions[positions != 0])} position periods, "
            f"{len(entry_signals[entry_signals != 0])} signals"
        )

        return positions

    def calculate_sentiment_score(
        self,
        text: str,
        method: str = 'simple'
    ) -> float:
        """
        Calculate sentiment score from text.

        This is a placeholder - in production, use a proper sentiment
        analysis model (VADER, TextBlob, transformer models, etc.)

        Args:
            text: Text to analyze
            method: Sentiment analysis method

        Returns:
            Sentiment score (-1 to 1)
        """
        # Placeholder implementation
        # In production, integrate with:
        # - VADER sentiment analyzer
        # - FinBERT for financial sentiment
        # - GPT-4 for complex analysis
        # - Claude API for nuanced understanding

        positive_words = ['bullish', 'profit', 'growth', 'strong', 'buy', 'gain']
        negative_words = ['bearish', 'loss', 'decline', 'weak', 'sell', 'drop']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0

        score = (positive_count - negative_count) / total_count

        return score
