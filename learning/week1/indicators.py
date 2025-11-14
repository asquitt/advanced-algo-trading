"""
Week 1 - Day 2: Technical Indicators

Implement common technical indicators used in trading strategies.
Run: python indicators.py
Test: pytest tests/test_day2.py
"""

import pandas as pd
import numpy as np
from typing import List


# ============================================================================
# TODO 1: Implement Simple Moving Average (SMA)
# ============================================================================

def calculate_sma(prices: List[float], period: int) -> List[float]:
    """
    Calculate Simple Moving Average.

    The SMA is the average price over the last N periods.

    Args:
        prices: List of prices
        period: Number of periods to average

    Returns:
        List of SMA values (same length as input, early values are NaN)

    Example:
        prices = [10, 11, 12, 13, 14, 15]
        sma = calculate_sma(prices, period=3)
        # sma[2] = (10+11+12)/3 = 11.0
        # sma[3] = (11+12+13)/3 = 12.0
    """
    # TODO: Implement SMA calculation
    # Hint 1: Use pandas rolling window: pd.Series(prices).rolling(period).mean()
    # Hint 2: Convert result back to list
    # Hint 3: First (period-1) values will be NaN

    sma = pd.Series(prices).rolling(window=period).mean()
    return sma.tolist()


# ============================================================================
# TODO 2: Implement Relative Strength Index (RSI)
# ============================================================================

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index.

    RSI measures momentum on a scale of 0-100.
    - RSI > 70: Overbought (might sell)
    - RSI < 30: Oversold (might buy)

    Formula:
        1. Calculate price changes (gains and losses)
        2. Average gain = avg of positive changes over period
        3. Average loss = avg of negative changes over period (absolute value)
        4. RS = Average Gain / Average Loss
        5. RSI = 100 - (100 / (1 + RS))

    Args:
        prices: List of prices
        period: Lookback period (typically 14)

    Returns:
        List of RSI values (0-100)
    """
    # TODO: Implement RSI calculation
    # Hint 1: Calculate price changes: diff = prices[i] - prices[i-1]
    # Hint 2: Separate into gains (diff > 0) and losses (diff < 0)
    # Hint 3: Use exponential moving average for smoothing
    # Hint 4: Handle division by zero (when avg_loss = 0, RSI = 100)

    prices_series = pd.Series(prices)

    # Calculate price changes
    delta = prices_series.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate average gains and losses using exponential moving average
    avg_gains = gains.rolling(window=period, min_periods=period).mean()
    avg_losses = losses.rolling(window=period, min_periods=period).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses

    # RSI formula
    rsi = 100 - (100 / (1 + rs))

    # Handle special cases
    rsi = rsi.fillna(50)  # Neutral when no data

    return rsi.tolist()


# ============================================================================
# TODO 3: Implement MACD (Moving Average Convergence Divergence)
# ============================================================================

def calculate_macd(prices: List[float],
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> dict:
    """
    Calculate MACD indicator.

    MACD shows the relationship between two moving averages.

    Formula:
        1. MACD Line = EMA(12) - EMA(26)
        2. Signal Line = EMA(9) of MACD Line
        3. Histogram = MACD Line - Signal Line

    Trading Signals:
        - MACD crosses above signal: Bullish (buy)
        - MACD crosses below signal: Bearish (sell)

    Args:
        prices: List of prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Dict with 'macd', 'signal', and 'histogram' lists
    """
    # TODO: Implement MACD calculation
    # Hint 1: Use pandas ewm().mean() for exponential moving average
    # Hint 2: MACD = fast_ema - slow_ema
    # Hint 3: Signal = EMA of MACD line
    # Hint 4: Histogram = MACD - Signal

    prices_series = pd.Series(prices)

    # Calculate EMAs
    fast_ema = prices_series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices_series.ewm(span=slow_period, adjust=False).mean()

    # MACD line
    macd_line = fast_ema - slow_ema

    # Signal line (EMA of MACD)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return {
        "macd": macd_line.tolist(),
        "signal": signal_line.tolist(),
        "histogram": histogram.tolist()
    }


# ============================================================================
# BONUS: Implement Bollinger Bands (Optional)
# ============================================================================

def calculate_bollinger_bands(prices: List[float],
                              period: int = 20,
                              num_std: float = 2.0) -> dict:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of:
    - Middle Band: SMA(period)
    - Upper Band: SMA + (std * num_std)
    - Lower Band: SMA - (std * num_std)

    Trading Signals:
        - Price touches lower band: Oversold (might buy)
        - Price touches upper band: Overbought (might sell)

    Args:
        prices: List of prices
        period: SMA period (default 20)
        num_std: Number of standard deviations (default 2)

    Returns:
        Dict with 'upper', 'middle', 'lower' lists
    """
    # BONUS TODO: Implement if you finish early!
    # Hint 1: Middle band = SMA(period)
    # Hint 2: Calculate rolling standard deviation
    # Hint 3: Upper = middle + (std * num_std)
    # Hint 4: Lower = middle - (std * num_std)

    prices_series = pd.Series(prices)

    # Middle band (SMA)
    middle_band = prices_series.rolling(window=period).mean()

    # Standard deviation
    rolling_std = prices_series.rolling(window=period).std()

    # Upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)

    return {
        "upper": upper_band.tolist(),
        "middle": middle_band.tolist(),
        "lower": lower_band.tolist()
    }


# ============================================================================
# Testing & Visualization
# ============================================================================

def test_indicators():
    """Test indicator calculations with sample data."""
    print("\n=== Testing Technical Indicators ===\n")

    # Sample price data (AAPL-like movement)
    prices = [
        150.0, 151.5, 152.0, 150.5, 149.0,  # Days 1-5
        148.0, 147.5, 149.0, 150.5, 152.0,  # Days 6-10
        153.5, 155.0, 154.5, 153.0, 151.5,  # Days 11-15
        150.0, 151.0, 152.5, 154.0, 155.5,  # Days 16-20
    ]

    print(f"Sample prices (20 days): {prices[:5]}... (first 5 shown)")

    # Test SMA
    print("\n1. Simple Moving Average (SMA)")
    sma_5 = calculate_sma(prices, period=5)
    print(f"   SMA(5) last 5 values: {[round(x, 2) if not pd.isna(x) else None for x in sma_5[-5:]]}")

    # Test RSI
    print("\n2. Relative Strength Index (RSI)")
    rsi = calculate_rsi(prices, period=14)
    print(f"   RSI last value: {rsi[-1]:.2f}")
    if rsi[-1] > 70:
        print("   → OVERBOUGHT (consider selling)")
    elif rsi[-1] < 30:
        print("   → OVERSOLD (consider buying)")
    else:
        print("   → NEUTRAL")

    # Test MACD
    print("\n3. MACD")
    macd = calculate_macd(prices)
    print(f"   MACD last value: {macd['macd'][-1]:.2f}")
    print(f"   Signal last value: {macd['signal'][-1]:.2f}")
    print(f"   Histogram last value: {macd['histogram'][-1]:.2f}")
    if macd['macd'][-1] > macd['signal'][-1]:
        print("   → BULLISH (MACD above signal)")
    else:
        print("   → BEARISH (MACD below signal)")

    # Test Bollinger Bands
    print("\n4. Bollinger Bands")
    bb = calculate_bollinger_bands(prices, period=20)
    current_price = prices[-1]
    print(f"   Current price: ${current_price:.2f}")
    print(f"   Upper band: ${bb['upper'][-1]:.2f}")
    print(f"   Middle band: ${bb['middle'][-1]:.2f}")
    print(f"   Lower band: ${bb['lower'][-1]:.2f}")

    if current_price >= bb['upper'][-1]:
        print("   → Price at upper band (overbought)")
    elif current_price <= bb['lower'][-1]:
        print("   → Price at lower band (oversold)")
    else:
        print("   → Price within bands (normal)")


def visualize_indicators():
    """Create visual plots of indicators (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt

        # Generate sample data
        days = 100
        prices = [150 + np.sin(i/10) * 10 + np.random.randn() * 2 for i in range(days)]

        # Calculate indicators
        sma_20 = calculate_sma(prices, 20)
        rsi = calculate_rsi(prices, 14)
        macd_data = calculate_macd(prices)
        bb = calculate_bollinger_bands(prices, 20)

        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot 1: Price with SMA and Bollinger Bands
        axes[0].plot(prices, label="Price", linewidth=2)
        axes[0].plot(sma_20, label="SMA(20)", linestyle="--")
        axes[0].plot(bb['upper'], label="Upper BB", linestyle=":", color="red")
        axes[0].plot(bb['lower'], label="Lower BB", linestyle=":", color="green")
        axes[0].set_title("Price with SMA and Bollinger Bands")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: RSI
        axes[1].plot(rsi, label="RSI", color="purple", linewidth=2)
        axes[1].axhline(70, color="red", linestyle="--", label="Overbought")
        axes[1].axhline(30, color="green", linestyle="--", label="Oversold")
        axes[1].set_title("Relative Strength Index (RSI)")
        axes[1].legend()
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: MACD
        axes[2].plot(macd_data['macd'], label="MACD", linewidth=2)
        axes[2].plot(macd_data['signal'], label="Signal", linewidth=2)
        axes[2].bar(range(len(macd_data['histogram'])), macd_data['histogram'],
                   label="Histogram", alpha=0.3)
        axes[2].set_title("MACD")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("indicators_visualization.png")
        print("\n✓ Visualization saved to 'indicators_visualization.png'")
        plt.show()

    except ImportError:
        print("\nℹ matplotlib not installed. Skipping visualization.")
        print("  Install with: pip install matplotlib")


def main():
    """Run tests and visualization."""
    print("=" * 60)
    print("Week 1 - Day 2: Technical Indicators")
    print("=" * 60)

    test_indicators()

    print("\n" + "=" * 60)
    print("✓ Indicator tests complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run pytest: pytest tests/test_day2.py -v")
    print("2. (Optional) Visualize: python indicators.py --visualize")
    print("3. Implement strategy.py (Day 3)")

    # Uncomment to generate visualization
    # visualize_indicators()


if __name__ == "__main__":
    import sys
    if "--visualize" in sys.argv:
        visualize_indicators()
    else:
        main()
