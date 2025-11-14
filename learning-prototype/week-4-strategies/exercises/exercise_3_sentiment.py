"""
Exercise 3: Sentiment-Driven Intraday Trading

Goal: Build and test a sentiment-based intraday trading strategy!

Tasks:
1. Generate synthetic intraday data with sentiment
2. Aggregate multi-source sentiment
3. Combine sentiment with technical indicators
4. Run intraday strategy
5. Analyze performance

Estimated time: 60-75 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import sys
sys.path.append('..')
from starter_code.sentiment_intraday import SentimentIntradayStrategy, SentimentConfig


# ========================================
# Task 1: Generate Intraday Data
# ========================================

def generate_intraday_data(num_days=5):
    """
    Generate synthetic intraday data with sentiment.

    For real implementation:
    - Use Alpaca API for intraday price/volume data
    - Use news APIs (NewsAPI, Alpha Vantage)
    - Use Twitter API for social sentiment
    - Use analyst APIs for professional opinions
    """
    print(f"📊 Generating {num_days} days of intraday data...\n")

    np.random.seed(42)

    all_prices = []
    all_volumes = []
    all_timestamps = []
    all_sentiment = {'news': [], 'social': [], 'analyst': []}

    for day in range(num_days):
        # Create intraday timestamps (9:30 AM - 4:00 PM = 390 minutes)
        date = pd.Timestamp('2024-01-15') + pd.Timedelta(days=day)
        if date.weekday() >= 5:  # Skip weekends
            continue

        timestamps = pd.date_range(
            start=f"{date.date()} 09:30:00",
            periods=390,
            freq='1min'
        )

        # Simulate intraday price pattern
        # Add intraday seasonality (higher vol near open/close)
        intraday_vol = np.concatenate([
            np.linspace(0.001, 0.0005, 100),   # Morning: High to normal
            np.ones(190) * 0.0003,              # Midday: Low
            np.linspace(0.0003, 0.0008, 100)   # Afternoon: Normal to high
        ])

        # Add random trend for the day
        day_trend = np.random.choice([-0.01, 0, 0.01])
        returns = day_trend / 390 + np.random.normal(0, intraday_vol, 390)

        # Start from previous close or 150.0
        if len(all_prices) > 0:
            start_price = all_prices[-1]
        else:
            start_price = 150.0

        prices = start_price * np.exp(np.cumsum(returns))

        # Volume (higher at open and close)
        volume_pattern = np.concatenate([
            np.linspace(5000, 2000, 100),   # Open: High to normal
            np.ones(190) * 1500,             # Midday: Low
            np.linspace(1500, 4000, 100)    # Close: Normal to high
        ]) + np.random.randint(-500, 500, 390)

        # Sentiment data (correlated with price moves)
        price_momentum = returns / intraday_vol

        # News sentiment (periodic spikes)
        news_events = np.random.choice([0, 1], 390, p=[0.97, 0.03])
        news_sent = news_events * np.random.choice([-1, 1], 390) * np.random.uniform(0.5, 1.0, 390)
        news_sent = np.clip(news_sent + price_momentum * 0.3, -1, 1)

        # Social media sentiment (noisier, more reactive)
        social_sent = price_momentum * 0.5 + np.random.normal(0, 0.3, 390)
        social_sent = np.clip(social_sent, -1, 1)

        # Analyst sentiment (smoother, less frequent)
        analyst_sent = np.zeros(390)
        if np.random.random() > 0.7:  # 30% chance of analyst update
            sentiment_val = np.random.choice([-0.5, 0, 0.5])
            analyst_sent[:] = sentiment_val

        # Append to lists
        all_timestamps.extend(timestamps)
        all_prices.extend(prices)
        all_volumes.extend(volume_pattern)
        all_sentiment['news'].extend(news_sent)
        all_sentiment['social'].extend(social_sent)
        all_sentiment['analyst'].extend(analyst_sent)

    # Create series
    timestamps_index = pd.DatetimeIndex(all_timestamps)
    prices = pd.Series(all_prices, index=timestamps_index)
    volumes = pd.Series(all_volumes, index=timestamps_index)

    sentiment_data = {
        'news': pd.Series(all_sentiment['news'], index=timestamps_index),
        'social': pd.Series(all_sentiment['social'], index=timestamps_index),
        'analyst': pd.Series(all_sentiment['analyst'], index=timestamps_index)
    }

    print(f"✅ Generated {len(prices)} minutes of intraday data")
    print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"   Avg volume: {volumes.mean():.0f}\n")

    return prices, volumes, sentiment_data, timestamps_index


# ========================================
# Task 2: Analyze Sentiment
# ========================================

def analyze_sentiment(sentiment_data, timestamps):
    """
    Analyze and visualize sentiment from different sources.
    """
    print("🔍 Analyzing sentiment data...\n")

    strategy = SentimentIntradayStrategy()

    # Normalize each source
    news_norm = strategy.normalize_sentiment(sentiment_data['news'])
    social_norm = strategy.normalize_sentiment(sentiment_data['social'])
    analyst_norm = strategy.normalize_sentiment(sentiment_data['analyst'])

    # Aggregate
    aggregated = strategy.aggregate_sentiment_sources(
        news_norm, social_norm, analyst_norm
    )

    # Rolling sentiment
    rolling = strategy.calculate_rolling_sentiment(aggregated, window_minutes=60)

    # Sentiment momentum
    sent_mom = strategy.calculate_sentiment_momentum(rolling, lookback=5)

    print(f"Sentiment Statistics:")
    print(f"  News: mean={news_norm.mean():.3f}, std={news_norm.std():.3f}")
    print(f"  Social: mean={social_norm.mean():.3f}, std={social_norm.std():.3f}")
    print(f"  Analyst: mean={analyst_norm.mean():.3f}, std={analyst_norm.std():.3f}")
    print(f"  Aggregated: mean={aggregated.mean():.3f}, std={aggregated.std():.3f}")

    # Count strong sentiment signals
    strong_positive = (aggregated > 0.3).sum()
    strong_negative = (aggregated < -0.3).sum()
    print(f"\nStrong Sentiment Signals:")
    print(f"  Positive (>0.3): {strong_positive}")
    print(f"  Negative (<-0.3): {strong_negative}")

    # Visualize
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Individual sources
    axes[0].plot(timestamps, news_norm, label='News', alpha=0.7, linewidth=1)
    axes[0].plot(timestamps, social_norm, label='Social', alpha=0.7, linewidth=1)
    axes[0].plot(timestamps, analyst_norm, label='Analyst', alpha=0.7, linewidth=1)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
    axes[0].axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Individual Sentiment Sources', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Sentiment')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Aggregated sentiment
    axes[1].plot(timestamps, aggregated, linewidth=2, color='purple', label='Aggregated')
    axes[1].fill_between(timestamps, 0, aggregated, where=(aggregated > 0),
                          alpha=0.3, color='green', label='Positive')
    axes[1].fill_between(timestamps, 0, aggregated, where=(aggregated < 0),
                          alpha=0.3, color='red', label='Negative')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Aggregated Sentiment', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Sentiment')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Rolling sentiment
    axes[2].plot(timestamps, rolling, linewidth=2, color='blue', label='60-min Rolling')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Threshold')
    axes[2].axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title('Rolling Sentiment (60-min)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Sentiment')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Sentiment momentum
    axes[3].bar(timestamps, sent_mom, width=0.0007, color='orange', alpha=0.7)
    axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[3].set_title('Sentiment Momentum', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Momentum')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sentiment_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Sentiment analysis chart saved: sentiment_analysis.png\n")

    return aggregated, rolling


# ========================================
# Task 3: Run Intraday Strategy
# ========================================

def run_sentiment_strategy(prices, volumes, sentiment_data, timestamps):
    """
    Run sentiment-driven intraday strategy.
    """
    print("🎯 Running sentiment intraday strategy...\n")

    strategy = SentimentIntradayStrategy(
        SentimentConfig(
            sentiment_threshold=0.3,
            lookback_minutes=60,
            use_technical_filter=True,
            volume_multiplier=1.5,
            max_trades_per_day=3
        )
    )

    # Run strategy
    result = strategy.run_strategy(
        prices, volumes, sentiment_data, timestamps
    )

    # Extract results
    signals = result['signals']
    position_size = result['position_size']
    sentiment = result['sentiment']
    confidence = result['confidence']

    # Calculate statistics
    num_trades = signals.diff().abs().sum() / 2
    num_days = (timestamps[-1] - timestamps[0]).days + 1

    long_signals = (signals > 0).sum()
    short_signals = (signals < 0).sum()
    flat = (signals == 0).sum()

    avg_confidence = confidence[signals != 0].mean()

    print(f"Strategy Statistics:")
    print(f"  Trading days: {num_days}")
    print(f"  Total trades: {num_trades:.0f}")
    print(f"  Avg trades/day: {num_trades/num_days:.1f}")
    print(f"\nPosition Distribution:")
    print(f"  Long: {long_signals} minutes ({long_signals/len(signals)*100:.1f}%)")
    print(f"  Short: {short_signals} minutes ({short_signals/len(signals)*100:.1f}%)")
    print(f"  Flat: {flat} minutes ({flat/len(signals)*100:.1f}%)")
    print(f"\nAverage confidence: {avg_confidence:.2f}")

    return result


# ========================================
# Task 4: Visualize Trades
# ========================================

def visualize_trades(prices, volumes, result, timestamps):
    """
    Visualize intraday trading activity.
    """
    print("\n📈 Visualizing trades...\n")

    signals = result['signals']
    sentiment = result['sentiment']

    # Find entry and exit points
    entries = signals.diff() != 0
    exits = (signals.shift(-1) == 0) & (signals != 0)

    long_entries = entries & (signals > 0)
    short_entries = entries & (signals < 0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Price with signals
    axes[0].plot(timestamps, prices, linewidth=1.5, color='black', alpha=0.7, label='Price')

    # Mark entries
    axes[0].scatter(timestamps[long_entries], prices[long_entries],
                    marker='^', color='green', s=100, label='Long Entry', zorder=5)
    axes[0].scatter(timestamps[short_entries], prices[short_entries],
                    marker='v', color='red', s=100, label='Short Entry', zorder=5)

    # Shade positions
    axes[0].fill_between(timestamps, prices.min(), prices.max(),
                          where=(signals > 0), alpha=0.2, color='green')
    axes[0].fill_between(timestamps, prices.min(), prices.max(),
                          where=(signals < 0), alpha=0.2, color='red')

    axes[0].set_title('Intraday Price with Trading Signals', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Volume
    colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in signals]
    axes[1].bar(timestamps, volumes, width=0.0007, color=colors, alpha=0.6)
    axes[1].axhline(y=volumes.mean() * 1.5, color='blue', linestyle='--',
                    label='High Volume Threshold')
    axes[1].set_title('Volume (colored by position)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volume')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Sentiment vs signals
    axes[2].plot(timestamps, sentiment, linewidth=2, color='purple',
                 alpha=0.7, label='Sentiment')
    axes[2].scatter(timestamps[long_entries], sentiment[long_entries],
                    marker='^', color='green', s=100, zorder=5)
    axes[2].scatter(timestamps[short_entries], sentiment[short_entries],
                    marker='v', color='red', s=100, zorder=5)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
    axes[2].axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title('Sentiment with Entry Points', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Sentiment')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sentiment_trades.png', dpi=150, bbox_inches='tight')
    print(f"✅ Trade visualization saved: sentiment_trades.png\n")


# ========================================
# Task 5: Analyze Performance
# ========================================

def analyze_performance(prices, result):
    """
    Calculate and analyze strategy performance.
    """
    print("📊 Analyzing performance...\n")

    strategy = SentimentIntradayStrategy()

    position_size = result['position_size']

    # Calculate returns
    returns = strategy.calculate_strategy_returns(prices, position_size)

    # Performance metrics
    total_return = returns.sum()
    avg_return = returns.mean()
    std_return = returns.std()

    # Annualize (390 minutes per day, 252 trading days)
    sharpe = avg_return / std_return * np.sqrt(390 * 252) if std_return > 0 else 0

    # Win/loss
    winning_minutes = (returns > 0).sum()
    losing_minutes = (returns < 0).sum()
    win_rate = winning_minutes / (winning_minutes + losing_minutes) if (winning_minutes + losing_minutes) > 0 else 0

    # Largest win/loss
    max_win = returns.max()
    max_loss = returns.min()

    # Daily performance
    daily_returns = returns.resample('D').sum()
    winning_days = (daily_returns > 0).sum()
    losing_days = (daily_returns < 0).sum()
    daily_win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

    print(f"Overall Performance:")
    print(f"  Total return: {total_return*100:.3f}%")
    print(f"  Sharpe ratio: {sharpe:.2f}")
    print(f"  Avg minute return: {avg_return*100:.4f}%")
    print(f"\nWin/Loss Analysis:")
    print(f"  Win rate (minutes): {win_rate*100:.1f}%")
    print(f"  Winning minutes: {winning_minutes}")
    print(f"  Losing minutes: {losing_minutes}")
    print(f"  Max win: {max_win*100:.3f}%")
    print(f"  Max loss: {max_loss*100:.3f}%")
    print(f"\nDaily Performance:")
    print(f"  Winning days: {winning_days}")
    print(f"  Losing days: {losing_days}")
    print(f"  Daily win rate: {daily_win_rate*100:.1f}%")

    # Plot equity curve
    equity = (1 + returns).cumprod() * 10000  # $10k per day

    plt.figure(figsize=(14, 6))
    plt.plot(equity.index, equity, linewidth=2, color='blue')
    plt.axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Starting Capital')
    plt.fill_between(equity.index, 10000, equity,
                     where=(equity >= 10000), alpha=0.3, color='green')
    plt.fill_between(equity.index, 10000, equity,
                     where=(equity < 10000), alpha=0.3, color='red')
    plt.title('Intraday Equity Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sentiment_equity.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Equity curve saved: sentiment_equity.png")


# ========================================
# Main Exercise
# ========================================

def main():
    """
    Run the complete sentiment intraday exercise.
    """
    print("="*60)
    print("📚 Exercise 3: Sentiment-Driven Intraday Trading")
    print("="*60)
    print()

    # Task 1: Generate data
    prices, volumes, sentiment_data, timestamps = generate_intraday_data(num_days=5)

    # Task 2: Analyze sentiment
    aggregated, rolling = analyze_sentiment(sentiment_data, timestamps)

    # Task 3: Run strategy
    result = run_sentiment_strategy(prices, volumes, sentiment_data, timestamps)

    # Task 4: Visualize trades
    visualize_trades(prices, volumes, result, timestamps)

    # Task 5: Analyze performance
    analyze_performance(prices, result)

    print("\n" + "="*60)
    print("✅ Exercise Complete!")
    print("="*60)
    print("\n📖 Key Takeaways:")
    print("  1. Multi-source sentiment is more reliable than single source")
    print("  2. Technical confirmation reduces false signals")
    print("  3. Intraday strategies exit all positions at close")
    print("  4. Sentiment momentum helps filter stale signals")
    print("  5. Volume confirmation is crucial for sentiment trades")
    print()
    print("🎯 Next Steps:")
    print("  - Integrate real news APIs (NewsAPI, Alpha Vantage)")
    print("  - Add Twitter sentiment analysis")
    print("  - Test on real market events (earnings, FOMC)")
    print("  - Implement NLP for sentiment scoring")
    print()


if __name__ == "__main__":
    main()
