"""
Exercise 2: Regime-Adaptive Momentum Strategy

Goal: Implement and test a momentum strategy that adapts to market regimes!

Tasks:
1. Load historical data
2. Detect market regimes (volatility and trend)
3. Run momentum strategy with regime adaptation
4. Compare performance across regimes
5. Backtest and analyze

Estimated time: 60-75 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from starter_code.regime_momentum import RegimeMomentumStrategy, RegimeMomentumConfig


# ========================================
# Task 1: Load Market Data
# ========================================

def load_sample_data():
    """
    Load sample market data with regime changes.

    For real implementation, use:
    - yfinance for historical data
    - alpaca_trade_api for real-time data
    """
    print("📊 Loading market data...\n")

    np.random.seed(42)
    n = 750  # ~3 years of daily data

    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    # Simulate market with regime changes
    # First 250 days: Low vol bull market
    # Next 250 days: High vol bear market
    # Last 250 days: Normal vol recovery

    volatility = np.concatenate([
        np.random.normal(0.01, 0.002, 250),  # Low vol
        np.random.normal(0.03, 0.005, 250),  # High vol
        np.random.normal(0.02, 0.003, 250)   # Normal vol
    ])

    trend = np.concatenate([
        np.linspace(0, 0.3, 250),   # Bull trend
        np.linspace(0.3, -0.2, 250),  # Bear trend
        np.linspace(-0.2, 0.1, 250)   # Recovery
    ])

    returns = trend/n + np.random.normal(0, volatility, n)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name='SPY')

    print(f"✅ Loaded {n} days of data")
    print(f"   Starting price: ${prices.iloc[0]:.2f}")
    print(f"   Ending price: ${prices.iloc[-1]:.2f}")
    print(f"   Total return: {(prices.iloc[-1]/prices.iloc[0] - 1)*100:.2f}%\n")

    return prices


# ========================================
# Task 2: Analyze Regimes
# ========================================

def analyze_regimes(prices):
    """
    Detect and visualize market regimes.
    """
    print("🔍 Analyzing market regimes...\n")

    strategy = RegimeMomentumStrategy()

    # Calculate returns and volatility
    returns = prices.pct_change()
    vol = strategy.calculate_realized_volatility(returns)

    # Detect regimes
    vol_regime = strategy.detect_volatility_regime(vol)
    trend_regime = strategy.detect_trend_regime(prices)
    regimes = strategy.combine_regimes(vol_regime, trend_regime)

    # Print regime statistics
    print("Volatility Regime Distribution:")
    vol_counts = vol_regime.value_counts()
    for regime, count in vol_counts.items():
        pct = count / len(vol_regime) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")

    print("\nTrend Regime Distribution:")
    trend_counts = trend_regime.value_counts()
    for regime, count in trend_counts.items():
        pct = count / len(trend_regime) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")

    print("\nCombined Regime Distribution:")
    combined_counts = regimes['combined_regime'].value_counts().head(5)
    for regime, count in combined_counts.items():
        pct = count / len(regimes) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")

    # Visualize regimes
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Price with trend regime
    axes[0].plot(prices.index, prices, linewidth=2, color='black', alpha=0.7)

    # Color code by trend regime
    bull_mask = trend_regime == 'bull'
    bear_mask = trend_regime == 'bear'
    neutral_mask = trend_regime == 'neutral'

    axes[0].fill_between(prices.index, 0, prices.max(), where=bull_mask,
                          alpha=0.2, color='green', label='Bull')
    axes[0].fill_between(prices.index, 0, prices.max(), where=bear_mask,
                          alpha=0.2, color='red', label='Bear')
    axes[0].fill_between(prices.index, 0, prices.max(), where=neutral_mask,
                          alpha=0.2, color='gray', label='Neutral')

    axes[0].set_title('Price with Trend Regime', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Volatility with regime
    axes[1].plot(vol.index, vol * 100, linewidth=2, color='orange')
    axes[1].axhline(y=15, color='green', linestyle='--', label='Low Vol Threshold')
    axes[1].axhline(y=25, color='red', linestyle='--', label='High Vol Threshold')
    axes[1].set_title('Realized Volatility', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volatility (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Moving averages
    ma_short, ma_long = strategy.calculate_moving_averages(prices)
    axes[2].plot(prices.index, prices, linewidth=2, label='Price', color='black')
    axes[2].plot(ma_short.index, ma_short, linewidth=1.5, label='20-day MA', color='blue')
    axes[2].plot(ma_long.index, ma_long, linewidth=1.5, label='50-day MA', color='red')
    axes[2].set_title('Moving Averages', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Price ($)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('regime_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Regime analysis chart saved: regime_analysis.png\n")

    return regimes


# ========================================
# Task 3: Run Momentum Strategy
# ========================================

def run_momentum_strategy(prices):
    """
    Run regime-adaptive momentum strategy.
    """
    print("🎯 Running momentum strategy...\n")

    strategy = RegimeMomentumStrategy(
        RegimeMomentumConfig(
            lookback_short=20,
            lookback_long=50,
            low_vol_multiplier=1.5,
            high_vol_multiplier=0.5,
            base_stop_loss=0.02
        )
    )

    # Run strategy
    result = strategy.run_strategy(
        prices,
        use_regime_sizing=True,
        use_stop_loss=False  # Simplified for now
    )

    # Extract results
    signals = result['signals']
    position_size = result['position_size']
    vol_regime = result['vol_regime']
    trend_regime = result['trend_regime']

    # Calculate statistics
    num_trades = signals.diff().abs().sum() / 2
    time_in_market = (signals != 0).sum() / len(signals) * 100

    long_trades = (signals.diff() == 1).sum()
    short_trades = (signals.diff() == -1).sum()

    # Average position size by regime
    avg_size_low_vol = position_size[vol_regime == 'low_vol'].mean()
    avg_size_high_vol = position_size[vol_regime == 'high_vol'].mean()

    print(f"Strategy Statistics:")
    print(f"  Total trades: {num_trades:.0f}")
    print(f"  Long trades: {long_trades}")
    print(f"  Short trades: {short_trades}")
    print(f"  Time in market: {time_in_market:.1f}%")
    print(f"\nPosition Sizing:")
    print(f"  Avg size (low vol): {avg_size_low_vol:.2f}x")
    print(f"  Avg size (high vol): {avg_size_high_vol:.2f}x")

    return result


# ========================================
# Task 4: Compare Regime Performance
# ========================================

def analyze_regime_performance(prices, result):
    """
    Compare strategy performance across different regimes.
    """
    print("\n📊 Analyzing performance by regime...\n")

    strategy = RegimeMomentumStrategy()

    signals = result['signals']
    position_size = result['position_size']
    vol_regime = result['vol_regime']
    trend_regime = result['trend_regime']

    # Calculate returns
    returns = strategy.calculate_strategy_returns(prices, signals, position_size)

    # Performance by volatility regime
    print("Performance by Volatility Regime:")
    for regime in ['low_vol', 'normal_vol', 'high_vol']:
        regime_mask = vol_regime == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) > 0:
            total_ret = (1 + regime_returns).prod() - 1
            sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
            win_rate = (regime_returns > 0).sum() / len(regime_returns)

            print(f"  {regime}:")
            print(f"    Total return: {total_ret*100:.2f}%")
            print(f"    Sharpe ratio: {sharpe:.2f}")
            print(f"    Win rate: {win_rate*100:.1f}%")

    # Performance by trend regime
    print("\nPerformance by Trend Regime:")
    for regime in ['bull', 'neutral', 'bear']:
        regime_mask = trend_regime == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) > 0:
            total_ret = (1 + regime_returns).prod() - 1
            sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
            win_rate = (regime_returns > 0).sum() / len(regime_returns)

            print(f"  {regime}:")
            print(f"    Total return: {total_ret*100:.2f}%")
            print(f"    Sharpe ratio: {sharpe:.2f}")
            print(f"    Win rate: {win_rate*100:.1f}%")

    return returns


# ========================================
# Task 5: Backtest Performance
# ========================================

def backtest_performance(prices, result, returns):
    """
    Calculate and visualize backtest performance.
    """
    print("\n📈 Backtesting performance...\n")

    signals = result['signals']

    # Performance metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Calculate equity curve
    equity = (1 + returns).cumprod() * 100000  # $100k starting

    # Drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

    # Buy & Hold comparison
    bnh_returns = prices.pct_change()
    bnh_equity = (1 + bnh_returns).cumprod() * 100000
    bnh_total = (bnh_equity.iloc[-1] / 100000) - 1
    bnh_sharpe = bnh_returns.mean() / bnh_returns.std() * np.sqrt(252) if bnh_returns.std() > 0 else 0

    print(f"Strategy Performance:")
    print(f"  Total return: {total_return*100:.2f}%")
    print(f"  Annual return: {annual_return*100:.2f}%")
    print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"  Max drawdown: {max_drawdown*100:.2f}%")
    print(f"  Win rate: {win_rate*100:.1f}%")

    print(f"\nBuy & Hold Comparison:")
    print(f"  Total return: {bnh_total*100:.2f}%")
    print(f"  Sharpe ratio: {bnh_sharpe:.2f}")
    print(f"  Strategy outperformance: {(total_return - bnh_total)*100:.2f}%")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Equity curves
    axes[0].plot(equity.index, equity, linewidth=2, label='Regime Momentum', color='blue')
    axes[0].plot(bnh_equity.index, bnh_equity, linewidth=2, label='Buy & Hold',
                 color='gray', linestyle='--')
    axes[0].axhline(y=100000, color='black', linestyle=':', alpha=0.5)
    axes[0].set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([equity.min() * 0.95, equity.max() * 1.05])

    # Plot 2: Drawdown
    axes[1].fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    axes[1].plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
    axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Returns distribution
    axes[2].hist(returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[2].set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Daily Return (%)')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('momentum_backtest.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Backtest chart saved: momentum_backtest.png")


# ========================================
# Main Exercise
# ========================================

def main():
    """
    Run the complete momentum strategy exercise.
    """
    print("="*60)
    print("📚 Exercise 2: Regime-Adaptive Momentum")
    print("="*60)
    print()

    # Task 1: Load data
    prices = load_sample_data()

    # Task 2: Analyze regimes
    regimes = analyze_regimes(prices)

    # Task 3: Run strategy
    result = run_momentum_strategy(prices)

    # Task 4: Regime performance
    returns = analyze_regime_performance(prices, result)

    # Task 5: Backtest
    backtest_performance(prices, result, returns)

    print("\n" + "="*60)
    print("✅ Exercise Complete!")
    print("="*60)
    print("\n📖 Key Takeaways:")
    print("  1. Momentum performs differently in different regimes")
    print("  2. High volatility → reduce position size")
    print("  3. Trend regime determines signal direction")
    print("  4. Regime adaptation improves risk-adjusted returns")
    print("  5. Dynamic position sizing is crucial for momentum")
    print()
    print("🎯 Next Steps:")
    print("  - Try with real market data (SPY, QQQ, etc.)")
    print("  - Test different momentum lookback periods")
    print("  - Implement stop losses")
    print("  - Add multiple asset momentum (cross-sectional)")
    print()


if __name__ == "__main__":
    main()
