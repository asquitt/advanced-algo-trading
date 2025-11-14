"""
Sentiment-Driven Intraday Trading Strategy - Starter Code with TODOs

Your mission: Implement an intraday strategy that trades based on news sentiment!

This strategy:
1. Aggregates sentiment from multiple sources
2. Combines with technical indicators
3. Enters positions during market hours
4. Exits all positions at market close
5. Manages risk with position limits

Difficulty levels:
🟢 Easy: Basic implementation
🟡 Medium: Requires some thinking
🔴 Hard: Advanced concepts

Author: Learning Lab Week 4
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import time, datetime
from enum import Enum
import pandas as pd
import numpy as np


class SentimentSource(Enum):
    """Sentiment data sources."""
    NEWS = "news"
    SOCIAL = "social"
    ANALYST = "analyst"


@dataclass
class SentimentConfig:
    """Configuration for sentiment intraday strategy."""
    # Market hours
    market_open: time = time(9, 30)   # 9:30 AM
    market_close: time = time(16, 0)  # 4:00 PM
    trading_start: time = time(9, 35)  # Wait 5 min after open
    exit_time: time = time(15, 55)    # Exit 5 min before close

    # Sentiment parameters
    sentiment_threshold: float = 0.3  # Min absolute sentiment to trade
    lookback_minutes: int = 60        # Sentiment aggregation window

    # Sentiment source weights
    news_weight: float = 0.5
    social_weight: float = 0.3
    analyst_weight: float = 0.2

    # Technical confirmation
    use_technical_filter: bool = True
    volume_multiplier: float = 1.5    # Require volume > avg * multiplier
    price_momentum_threshold: float = 0.001  # 0.1% momentum required

    # Position management
    max_position_size: float = 1.0
    scale_by_confidence: bool = True  # Scale position by sentiment strength
    max_trades_per_day: int = 3


class SentimentIntradayStrategy:
    """
    Sentiment-driven intraday trading strategy.

    Core concepts:
    1. Aggregate sentiment from multiple sources
    2. Confirm with technical indicators
    3. Enter during market hours
    4. Exit all positions at close
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        """Initialize sentiment intraday strategy."""
        self.config = config or SentimentConfig()
        self.trades_today = 0
        self.current_date = None

    # ========================================
    # Part 1: Time Management
    # ========================================

    def is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp is during market hours.

        🟢 TODO #1: Check market hours

        HINT: Extract time from timestamp using .time()
        HINT: Compare with self.config.market_open and market_close

        Args:
            timestamp: Timestamp to check

        Returns:
            True if during market hours
        """
        # YOUR CODE HERE
        pass

    def is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp is during valid trading hours.

        🟢 TODO #2: Check trading hours

        Trading hours are between trading_start and exit_time
        (not the full market hours).

        HINT: Similar to is_market_hours()
        HINT: Use self.config.trading_start and exit_time

        Args:
            timestamp: Timestamp to check

        Returns:
            True if during trading hours
        """
        # YOUR CODE HERE
        pass

    def should_exit_all_positions(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if we should exit all positions (near market close).

        🟢 TODO #3: Check exit time

        HINT: Compare timestamp.time() with self.config.exit_time
        HINT: Return True if >= exit_time

        Args:
            timestamp: Current timestamp

        Returns:
            True if should exit all positions
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 2: Sentiment Aggregation
    # ========================================

    def normalize_sentiment(
        self,
        sentiment: pd.Series,
        method: str = 'tanh'
    ) -> pd.Series:
        """
        Normalize sentiment scores to [-1, 1] range.

        🟡 TODO #4: Normalize sentiment

        Methods:
        - 'tanh': Use tanh function (smooth)
        - 'clip': Clip to [-1, 1]
        - 'minmax': Scale using min-max normalization

        HINT: np.tanh() for tanh method
        HINT: pd.Series.clip(-1, 1) for clip method

        Args:
            sentiment: Raw sentiment scores
            method: Normalization method

        Returns:
            Normalized sentiment in [-1, 1]
        """
        # YOUR CODE HERE
        pass

    def aggregate_sentiment_sources(
        self,
        news_sentiment: pd.Series,
        social_sentiment: pd.Series,
        analyst_sentiment: pd.Series
    ) -> pd.Series:
        """
        Aggregate sentiment from multiple sources using weights.

        🟢 TODO #5: Aggregate sentiment

        Formula: weighted_sum = Σ(weight_i * sentiment_i)

        HINT: Use self.config weights (news_weight, social_weight, analyst_weight)
        HINT: Make sure weights sum to 1.0
        HINT: Handle missing values (use fillna(0))

        Args:
            news_sentiment: News sentiment scores
            social_sentiment: Social media sentiment scores
            analyst_sentiment: Analyst sentiment scores

        Returns:
            Aggregated sentiment series
        """
        # YOUR CODE HERE
        pass

    def calculate_rolling_sentiment(
        self,
        sentiment: pd.Series,
        window_minutes: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling average sentiment.

        🟢 TODO #6: Calculate rolling sentiment

        HINT: Use pd.Series.rolling(window).mean()
        HINT: Use self.config.lookback_minutes if window_minutes is None
        HINT: For intraday data, window is in minutes

        Args:
            sentiment: Sentiment time series
            window_minutes: Rolling window in minutes

        Returns:
            Rolling sentiment series
        """
        # YOUR CODE HERE
        pass

    def calculate_sentiment_momentum(
        self,
        sentiment: pd.Series,
        lookback: int = 5
    ) -> pd.Series:
        """
        Calculate rate of change in sentiment.

        🟡 TODO #7: Calculate sentiment momentum

        Formula: momentum = sentiment - sentiment.shift(lookback)

        This captures whether sentiment is improving or deteriorating.

        HINT: Use pd.Series.shift()

        Args:
            sentiment: Sentiment series
            lookback: Lookback periods

        Returns:
            Sentiment momentum series
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 3: Technical Confirmation
    # ========================================

    def calculate_volume_ratio(
        self,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Calculate volume relative to average.

        🟢 TODO #8: Calculate volume ratio

        Formula: volume_ratio = volume / rolling_avg_volume

        HINT: Use pd.Series.rolling(lookback).mean()

        Args:
            volume: Volume series
            lookback: Lookback for average

        Returns:
            Volume ratio series
        """
        # YOUR CODE HERE
        pass

    def calculate_price_momentum(
        self,
        prices: pd.Series,
        lookback: int = 5
    ) -> pd.Series:
        """
        Calculate short-term price momentum.

        🟢 TODO #9: Calculate price momentum

        Formula: momentum = (price / price.shift(lookback)) - 1

        HINT: Similar to regime momentum calculation

        Args:
            prices: Price series
            lookback: Lookback periods

        Returns:
            Price momentum series
        """
        # YOUR CODE HERE
        pass

    def check_technical_confirmation(
        self,
        prices: pd.Series,
        volume: pd.Series,
        sentiment_signal: pd.Series
    ) -> pd.Series:
        """
        Check if technical indicators confirm sentiment signal.

        🟡 TODO #10: Check technical confirmation

        Confirmation rules:
        - For long (sentiment > 0): Need positive price momentum AND high volume
        - For short (sentiment < 0): Need negative price momentum AND high volume

        HINT: Calculate volume_ratio and price_momentum
        HINT: Use self.config.volume_multiplier and price_momentum_threshold
        HINT: Return boolean series

        Args:
            prices: Price series
            volume: Volume series
            sentiment_signal: Sentiment-based signal

        Returns:
            Boolean series (True if confirmed)
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 4: Signal Generation
    # ========================================

    def generate_sentiment_signals(
        self,
        sentiment: pd.Series
    ) -> pd.Series:
        """
        Generate raw signals from sentiment.

        🟢 TODO #11: Generate sentiment signals

        Logic:
        - If sentiment > threshold: Signal = 1 (long)
        - If sentiment < -threshold: Signal = -1 (short)
        - Otherwise: Signal = 0 (no trade)

        HINT: Use self.config.sentiment_threshold

        Args:
            sentiment: Aggregated sentiment series

        Returns:
            Signal series (-1, 0, 1)
        """
        # YOUR CODE HERE
        pass

    def filter_signals_by_time(
        self,
        signals: pd.Series,
        timestamps: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Filter signals to only trading hours.

        🟡 TODO #12: Filter by time

        Logic:
        - Keep signals during trading hours
        - Set signals to 0 outside trading hours

        HINT: Use is_trading_hours() for each timestamp
        HINT: Use pd.Series.apply() or list comprehension

        Args:
            signals: Trading signals
            timestamps: Timestamp index

        Returns:
            Filtered signals
        """
        # YOUR CODE HERE
        pass

    def apply_intraday_exit_rule(
        self,
        signals: pd.Series,
        timestamps: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Apply rule to exit all positions near market close.

        🟡 TODO #13: Apply exit rule

        Logic:
        - After exit_time: Force all signals to 0
        - Before exit_time: Keep signals as-is

        HINT: Use should_exit_all_positions()
        HINT: Iterate through signals and set to 0 after exit time

        Args:
            signals: Trading signals
            timestamps: Timestamp index

        Returns:
            Signals with exit rule applied
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 5: Position Sizing
    # ========================================

    def calculate_confidence_score(
        self,
        sentiment: pd.Series,
        sentiment_momentum: pd.Series,
        technical_confirmed: pd.Series
    ) -> pd.Series:
        """
        Calculate confidence score for position sizing.

        🟡 TODO #14: Calculate confidence

        Confidence factors:
        - Sentiment strength: abs(sentiment)
        - Sentiment momentum: positive momentum = higher confidence
        - Technical confirmation: boolean boost

        Formula (example):
        confidence = abs(sentiment) * (1 + sentiment_momentum) * (1 if confirmed else 0.5)

        HINT: Combine all three factors
        HINT: Normalize to [0, 1] range
        HINT: Clip to reasonable range

        Args:
            sentiment: Sentiment scores
            sentiment_momentum: Sentiment rate of change
            technical_confirmed: Technical confirmation boolean

        Returns:
            Confidence score [0, 1]
        """
        # YOUR CODE HERE
        pass

    def calculate_position_size(
        self,
        signals: pd.Series,
        confidence: pd.Series
    ) -> pd.Series:
        """
        Calculate position size based on signals and confidence.

        🟢 TODO #15: Calculate position size

        Logic:
        - If scale_by_confidence: size = signal * confidence * max_position_size
        - Otherwise: size = signal * max_position_size

        HINT: Use self.config.scale_by_confidence
        HINT: Use self.config.max_position_size

        Args:
            signals: Trading signals (-1, 0, 1)
            confidence: Confidence scores [0, 1]

        Returns:
            Position size series
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 6: Risk Management
    # ========================================

    def enforce_max_trades_per_day(
        self,
        signals: pd.Series,
        timestamps: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Enforce maximum number of trades per day.

        🔴 TODO #16: Enforce trade limit

        Logic:
        - Count number of signal changes per day
        - After reaching max_trades_per_day, no more entries
        - Still allow exits

        HINT: Group by date
        HINT: Track cumulative trades per day
        HINT: This is complex - consider simplified version

        Args:
            signals: Trading signals
            timestamps: Timestamp index

        Returns:
            Signals with trade limit enforced
        """
        # YOUR CODE HERE
        pass

    def calculate_intraday_stop_loss(
        self,
        entry_price: float,
        direction: int,
        atr: float,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate intraday stop loss level.

        🟡 TODO #17: Calculate stop loss

        Formula:
        - Long: stop = entry_price - multiplier * atr
        - Short: stop = entry_price + multiplier * atr

        HINT: ATR (Average True Range) measures volatility
        HINT: direction = 1 for long, -1 for short

        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            atr: Average True Range
            multiplier: Stop loss multiplier

        Returns:
            Stop loss price level
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 7: Main Strategy Logic
    # ========================================

    def run_strategy(
        self,
        prices: pd.Series,
        volume: pd.Series,
        sentiment_data: Dict[str, pd.Series],
        timestamps: pd.DatetimeIndex
    ) -> Dict:
        """
        Run complete sentiment intraday strategy.

        🔴 TODO #18: Implement main strategy

        Steps:
        1. Normalize sentiment from each source
        2. Aggregate multi-source sentiment
        3. Calculate rolling sentiment
        4. Calculate sentiment momentum
        5. Generate base signals from sentiment
        6. Check technical confirmation
        7. Filter signals by trading hours
        8. Apply intraday exit rule
        9. Calculate confidence scores
        10. Calculate position sizes
        11. Enforce max trades per day
        12. Return results dict

        HINT: sentiment_data dict has keys: 'news', 'social', 'analyst'
        HINT: Use all the functions you implemented above

        Args:
            prices: Intraday price series
            volume: Intraday volume series
            sentiment_data: Dict with sentiment from different sources
            timestamps: DatetimeIndex for intraday data

        Returns:
            Dictionary with:
            - 'signals': Final trading signals
            - 'position_size': Position sizes
            - 'sentiment': Aggregated sentiment
            - 'sentiment_momentum': Sentiment momentum
            - 'confidence': Confidence scores
            - 'technical_confirmed': Technical confirmation boolean
        """
        # YOUR CODE HERE
        pass

    def calculate_strategy_returns(
        self,
        prices: pd.Series,
        position_size: pd.Series
    ) -> pd.Series:
        """
        Calculate strategy returns.

        🟢 TODO #19: Calculate returns

        Formula: return = position_size (lagged) * price_return

        HINT: Lag position_size by 1
        HINT: Calculate returns as prices.pct_change()

        Args:
            prices: Price series
            position_size: Position size series

        Returns:
            Strategy returns
        """
        # YOUR CODE HERE
        pass


# ========================================
# Self-Test Function
# ========================================

def test_implementation():
    """
    Test your sentiment intraday implementation!

    🎯 Run this function to check if your code works:
        python sentiment_intraday.py
    """
    print("🧪 Testing Sentiment Intraday Implementation...\n")

    # Generate synthetic intraday data
    np.random.seed(42)
    n = 390  # One trading day (6.5 hours * 60 minutes)

    # Create minute-level timestamps
    start = pd.Timestamp('2024-01-15 09:30:00')
    timestamps = pd.date_range(start, periods=n, freq='1min')

    # Simulate prices
    returns = np.random.normal(0.00001, 0.0005, n)
    prices = pd.Series(150 * np.exp(np.cumsum(returns)), index=timestamps)

    # Simulate volume
    volume = pd.Series(np.random.randint(1000, 10000, n), index=timestamps)

    # Simulate sentiment data
    sentiment_data = {
        'news': pd.Series(np.random.uniform(-1, 1, n), index=timestamps),
        'social': pd.Series(np.random.uniform(-1, 1, n), index=timestamps),
        'analyst': pd.Series(np.random.uniform(-0.5, 0.5, n), index=timestamps)
    }

    # Create strategy
    strategy = SentimentIntradayStrategy()

    # Test 1: Time checks
    print("Test 1: Time Management")
    try:
        is_market = strategy.is_market_hours(timestamps[0])
        is_trading = strategy.is_trading_hours(timestamps[0])
        should_exit = strategy.should_exit_all_positions(timestamps[-1])
        print(f"✅ Time checks work:")
        print(f"   Market hours: {is_market}")
        print(f"   Trading hours: {is_trading}")
        print(f"   Should exit: {should_exit}")
    except:
        print("❌ Time management functions not implemented")

    # Test 2: Sentiment normalization
    print("\nTest 2: Sentiment Normalization")
    try:
        normalized = strategy.normalize_sentiment(sentiment_data['news'])
        print(f"✅ Sentiment normalized: range=[{normalized.min():.2f}, {normalized.max():.2f}]")
    except:
        print("❌ normalize_sentiment() not implemented")

    # Test 3: Sentiment aggregation
    print("\nTest 3: Sentiment Aggregation")
    try:
        aggregated = strategy.aggregate_sentiment_sources(
            sentiment_data['news'],
            sentiment_data['social'],
            sentiment_data['analyst']
        )
        print(f"✅ Sentiment aggregated: mean={aggregated.mean():.3f}")
    except:
        print("❌ aggregate_sentiment_sources() not implemented")

    # Test 4: Rolling sentiment
    print("\nTest 4: Rolling Sentiment")
    try:
        rolling = strategy.calculate_rolling_sentiment(aggregated, window_minutes=30)
        print(f"✅ Rolling sentiment calculated: mean={rolling.mean():.3f}")
    except:
        print("❌ calculate_rolling_sentiment() not implemented")

    # Test 5: Technical indicators
    print("\nTest 5: Technical Indicators")
    try:
        volume_ratio = strategy.calculate_volume_ratio(volume)
        price_mom = strategy.calculate_price_momentum(prices)
        print(f"✅ Technical indicators:")
        print(f"   Volume ratio: mean={volume_ratio.mean():.2f}")
        print(f"   Price momentum: mean={price_mom.mean():.4f}")
    except:
        print("❌ Technical indicator functions not implemented")

    # Test 6: Signal generation
    print("\nTest 6: Signal Generation")
    try:
        signals = strategy.generate_sentiment_signals(aggregated)
        num_signals = (signals != 0).sum()
        print(f"✅ Signals generated: {num_signals} non-zero signals")
    except:
        print("❌ generate_sentiment_signals() not implemented")

    # Test 7: Time filtering
    print("\nTest 7: Time Filtering")
    try:
        filtered = strategy.filter_signals_by_time(signals, timestamps)
        print(f"✅ Signals filtered by time: {(filtered != 0).sum()} active signals")
    except:
        print("❌ filter_signals_by_time() not implemented")

    # Test 8: Full strategy
    print("\nTest 8: Full Strategy")
    try:
        result = strategy.run_strategy(prices, volume, sentiment_data, timestamps)
        final_signals = result['signals']
        num_trades = final_signals.diff().abs().sum() / 2
        print(f"✅ Strategy run successfully!")
        print(f"   Trades: {num_trades:.0f}")
        print(f"   Active signals: {(final_signals != 0).sum()}")
    except:
        print("❌ run_strategy() not implemented")

    # Test 9: Returns
    print("\nTest 9: Strategy Returns")
    try:
        returns = strategy.calculate_strategy_returns(prices, result['position_size'])
        total_return = returns.sum()
        print(f"✅ Returns calculated!")
        print(f"   Total intraday return: {total_return*100:.3f}%")
        print(f"   Num positive: {(returns > 0).sum()}")
        print(f"   Num negative: {(returns < 0).sum()}")
    except:
        print("❌ calculate_strategy_returns() not implemented")

    print("\n" + "="*50)
    print("🎉 Testing complete!")
    print("="*50)


if __name__ == "__main__":
    test_implementation()
