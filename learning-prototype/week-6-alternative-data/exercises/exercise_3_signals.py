"""
Exercise 3: Trading Signal Generation
Convert sentiment scores into actionable trading signals

Tasks:
1. Complete the TODOs in signal_generation.py
2. Generate signals from sentiment data
3. Test signal filtering and combination
4. Calculate position sizes based on signals
5. Backtest signal accuracy

Expected Output:
- Trading signals with strength ratings
- Position sizing recommendations
- Signal summary statistics
- Backtest results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starter_code.signal_generation import SignalGenerator, TradingSignal, SignalType
from datetime import datetime
import json


# Sample sentiment data from previous exercise
SAMPLE_SENTIMENT_DATA = {
    "AAPL": {
        "mean_sentiment": 0.65,
        "median_sentiment": 0.70,
        "std_sentiment": 0.15,
        "article_count": 12,
        "bullish_ratio": 0.83
    },
    "TSLA": {
        "mean_sentiment": -0.45,
        "median_sentiment": -0.50,
        "std_sentiment": 0.20,
        "article_count": 8,
        "bullish_ratio": 0.25
    },
    "GOOGL": {
        "mean_sentiment": 0.05,
        "median_sentiment": 0.03,
        "std_sentiment": 0.18,
        "article_count": 10,
        "bullish_ratio": 0.50
    },
    "MSFT": {
        "mean_sentiment": 0.55,
        "median_sentiment": 0.60,
        "std_sentiment": 0.12,
        "article_count": 15,
        "bullish_ratio": 0.80
    },
    "AMZN": {
        "mean_sentiment": -0.25,
        "median_sentiment": -0.20,
        "std_sentiment": 0.25,
        "article_count": 6,
        "bullish_ratio": 0.33
    },
    "NVDA": {
        "mean_sentiment": 0.75,
        "median_sentiment": 0.80,
        "std_sentiment": 0.10,
        "article_count": 20,
        "bullish_ratio": 0.90
    }
}


def test_basic_signal_generation():
    """Test basic signal generation"""
    print("=" * 80)
    print("Testing Basic Signal Generation")
    print("=" * 80)

    # TODO: Initialize signal generator with default parameters
    generator = SignalGenerator(
        sentiment_threshold=0.1,
        min_article_count=3,
        min_confidence=0.5
    )

    print("\nGenerating signals for each symbol:\n")

    # TODO: Generate signal for each symbol
    for symbol, data in SAMPLE_SENTIMENT_DATA.items():
        signal = generator.generate_signal(
            symbol=symbol,
            sentiment_score=data['mean_sentiment'],
            article_count=data['article_count']
        )

        print(f"{symbol}:")
        print(f"  Signal Type: {signal.signal_type.value}")
        print(f"  Strength: {signal.strength:.2f}")
        print(f"  Sentiment: {signal.sentiment_score:.3f}")
        print(f"  Reasons: {', '.join(signal.reasons)}")
        print()


def test_batch_signal_generation():
    """Test batch signal generation"""
    print("=" * 80)
    print("Testing Batch Signal Generation")
    print("=" * 80)

    generator = SignalGenerator()

    # TODO: Generate signals for all symbols at once
    signals = generator.generate_batch_signals(SAMPLE_SENTIMENT_DATA)

    print(f"\nGenerated {len(signals)} signals:\n")
    for signal in signals:
        print(f"{signal.symbol:6s} | {signal.signal_type.value:12s} | "
              f"Strength: {signal.strength:.2f} | "
              f"Sentiment: {signal.sentiment_score:+.3f}")


def test_signal_filtering():
    """Test signal filtering"""
    print("\n" + "=" * 80)
    print("Testing Signal Filtering")
    print("=" * 80)

    generator = SignalGenerator()
    signals = generator.generate_batch_signals(SAMPLE_SENTIMENT_DATA)

    # TODO: Filter for high-confidence signals only
    print("\n1. High-confidence signals (strength >= 0.7):")
    high_conf_signals = generator.filter_signals(
        signals,
        min_strength=0.7
    )
    for signal in high_conf_signals:
        print(f"  {signal.symbol}: {signal.signal_type.value} ({signal.strength:.2f})")

    # TODO: Filter for buy signals only
    print("\n2. Buy signals only:")
    buy_signals = generator.filter_signals(
        signals,
        signal_types=[SignalType.BUY, SignalType.STRONG_BUY]
    )
    for signal in buy_signals:
        print(f"  {signal.symbol}: {signal.signal_type.value} ({signal.strength:.2f})")

    # TODO: Filter for sell signals only
    print("\n3. Sell signals only:")
    sell_signals = generator.filter_signals(
        signals,
        signal_types=[SignalType.SELL, SignalType.STRONG_SELL]
    )
    for signal in sell_signals:
        print(f"  {signal.symbol}: {signal.signal_type.value} ({signal.strength:.2f})")


def test_signal_combination():
    """Test combining sentiment with technical indicators"""
    print("\n" + "=" * 80)
    print("Testing Signal Combination")
    print("=" * 80)

    generator = SignalGenerator()

    # TODO: Generate sentiment signal for AAPL
    sentiment_signal = generator.generate_signal(
        symbol="AAPL",
        sentiment_score=0.65,
        article_count=12
    )

    # TODO: Simulate technical indicators (e.g., RSI, MACD)
    technical_scores = [
        (0.7, "Strong technical bullish"),
        (0.2, "Weak technical bullish"),
        (-0.3, "Moderate technical bearish"),
        (-0.7, "Strong technical bearish")
    ]

    print(f"\nOriginal sentiment signal for AAPL:")
    print(f"  Type: {sentiment_signal.signal_type.value}")
    print(f"  Strength: {sentiment_signal.strength:.2f}")
    print(f"  Sentiment: {sentiment_signal.sentiment_score:.3f}")

    print(f"\nCombining with technical indicators:\n")

    for tech_score, description in technical_scores:
        # TODO: Combine signals
        combined = generator.combine_signals(
            sentiment_signal,
            tech_score,
            sentiment_weight=0.6,
            technical_weight=0.4
        )

        print(f"{description} (score: {tech_score:+.2f}):")
        print(f"  Combined Type: {combined.signal_type.value}")
        print(f"  Combined Strength: {combined.strength:.2f}")
        print()


def test_position_sizing():
    """Test position size calculation"""
    print("=" * 80)
    print("Testing Position Sizing")
    print("=" * 80)

    generator = SignalGenerator()
    signals = generator.generate_batch_signals(SAMPLE_SENTIMENT_DATA)

    # TODO: Calculate position sizes for a $100,000 account
    account_value = 100000
    print(f"\nPosition sizing for ${account_value:,} account:\n")

    for signal in signals:
        # Skip HOLD signals
        if signal.signal_type == SignalType.HOLD:
            continue

        # TODO: Calculate position size
        position = generator.calculate_position_size(
            signal=signal,
            account_value=account_value,
            max_position_pct=0.15,  # Max 15% per position
            risk_per_trade=0.02  # Risk 2% per trade
        )

        print(f"{signal.symbol} ({signal.signal_type.value}):")
        print(f"  Position Value: ${position['position_value']:,.2f}")
        print(f"  Position %: {position['position_pct']*100:.1f}%")
        print(f"  Risk Amount: ${position['risk_amount']:,.2f}")
        print(f"  Signal Strength: {position['signal_strength']:.2f}")
        print()


def test_signal_summary():
    """Test signal summary generation"""
    print("=" * 80)
    print("Testing Signal Summary")
    print("=" * 80)

    generator = SignalGenerator()
    signals = generator.generate_batch_signals(SAMPLE_SENTIMENT_DATA)

    # TODO: Generate summary
    summary = generator.generate_signal_summary(signals)

    print("\nSignal Summary:")
    print(json.dumps(summary, indent=2, default=str))


def test_threshold_sensitivity():
    """Test sensitivity to threshold changes"""
    print("\n" + "=" * 80)
    print("Testing Threshold Sensitivity")
    print("=" * 80)

    # TODO: Test with different threshold settings
    threshold_configs = [
        (0.05, "Very sensitive"),
        (0.10, "Default"),
        (0.20, "Conservative")
    ]

    print("\nComparing signals with different thresholds:\n")

    for threshold, description in threshold_configs:
        generator = SignalGenerator(
            sentiment_threshold=threshold,
            min_article_count=3
        )

        signals = generator.generate_batch_signals(SAMPLE_SENTIMENT_DATA)
        summary = generator.generate_signal_summary(signals)

        print(f"{description} (threshold={threshold}):")
        print(f"  Buy signals: {summary['signal_counts'].get(SignalType.BUY, 0) + summary['signal_counts'].get(SignalType.STRONG_BUY, 0)}")
        print(f"  Sell signals: {summary['signal_counts'].get(SignalType.SELL, 0) + summary['signal_counts'].get(SignalType.STRONG_SELL, 0)}")
        print(f"  Hold signals: {summary['signal_counts'].get(SignalType.HOLD, 0)}")
        print()


def test_backtest_accuracy():
    """Test signal accuracy backtesting"""
    print("=" * 80)
    print("Testing Backtest Accuracy")
    print("=" * 80)

    generator = SignalGenerator()
    signals = generator.generate_batch_signals(SAMPLE_SENTIMENT_DATA)

    # TODO: Simulate actual returns (in real scenario, fetch historical data)
    actual_returns = {
        "AAPL": 0.08,   # +8% (signal was STRONG_BUY)
        "TSLA": -0.06,  # -6% (signal was SELL)
        "GOOGL": 0.01,  # +1% (signal was HOLD/neutral)
        "MSFT": 0.07,   # +7% (signal was BUY)
        "AMZN": -0.03,  # -3% (signal was SELL)
        "NVDA": 0.12    # +12% (signal was STRONG_BUY)
    }

    # TODO: Backtest signals
    results = generator.backtest_signal_accuracy(
        signals,
        actual_returns,
        holding_period_days=5
    )

    print("\nBacktest Results (5-day holding period):")
    print(f"  Accuracy: {results['accuracy']*100:.1f}%")
    print(f"  Correct signals: {results['correct_signals']}/{results['total_signals']}")

    # TODO: Show individual results
    print("\nDetailed Results:")
    for signal in signals:
        if signal.symbol in actual_returns:
            actual_return = actual_returns[signal.symbol]
            predicted_direction = "UP" if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else \
                                 "DOWN" if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] else "NEUTRAL"
            actual_direction = "UP" if actual_return > 0 else "DOWN" if actual_return < 0 else "NEUTRAL"
            correct = "✓" if (predicted_direction == actual_direction or predicted_direction == "NEUTRAL") else "✗"

            print(f"  {correct} {signal.symbol}: Predicted {predicted_direction}, "
                  f"Actual {actual_direction} ({actual_return:+.1%})")


def test_risk_adjusted_signals():
    """Test signals with volatility adjustment"""
    print("\n" + "=" * 80)
    print("Testing Risk-Adjusted Signals")
    print("=" * 80)

    generator = SignalGenerator()

    # TODO: Simulate volatility data for symbols
    volatility_data = {
        "AAPL": 0.25,  # 25% annualized volatility (moderate)
        "TSLA": 0.60,  # 60% volatility (high)
        "NVDA": 0.35   # 35% volatility (moderate-high)
    }

    print("\nSignals with and without volatility adjustment:\n")

    for symbol in ["AAPL", "TSLA", "NVDA"]:
        sentiment = SAMPLE_SENTIMENT_DATA[symbol]['mean_sentiment']
        article_count = SAMPLE_SENTIMENT_DATA[symbol]['article_count']

        # Without volatility
        signal_no_vol = generator.generate_signal(
            symbol, sentiment, article_count
        )

        # With volatility
        signal_with_vol = generator.generate_signal(
            symbol, sentiment, article_count,
            volatility=volatility_data[symbol]
        )

        print(f"{symbol} (volatility: {volatility_data[symbol]*100:.0f}%):")
        print(f"  Without vol adjustment: {signal_no_vol.signal_type.value} (strength: {signal_no_vol.strength:.2f})")
        print(f"  With vol adjustment:    {signal_with_vol.signal_type.value} (strength: {signal_with_vol.strength:.2f})")
        print()


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("EXERCISE 3: TRADING SIGNAL GENERATION")
    print("=" * 80)

    try:
        # TODO: Uncomment tests as you complete the corresponding TODOs
        # test_basic_signal_generation()
        # test_batch_signal_generation()
        # test_signal_filtering()
        # test_signal_combination()
        # test_position_sizing()
        # test_signal_summary()
        # test_threshold_sensitivity()
        # test_backtest_accuracy()
        # test_risk_adjusted_signals()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# CHALLENGE TASKS:
# 1. Implement Kelly Criterion for optimal position sizing
# 2. Add multi-timeframe signal confirmation
# 3. Create a signal decay function (signals lose strength over time)
# 4. Implement portfolio-level constraints (sector limits, correlation)
# 5. Add adaptive thresholds based on market regime
# 6. Create a signal backtest framework with realistic slippage and costs
