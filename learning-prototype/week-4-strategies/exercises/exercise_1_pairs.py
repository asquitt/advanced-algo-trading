"""
Exercise 1: Pairs Trading

Goal: Find and trade a real stock pair using the strategy you implemented!

Tasks:
1. Load historical data for two stocks
2. Test if they're cointegrated
3. Run the pairs trading strategy
4. Backtest and analyze performance

Estimated time: 45-60 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from starter_code.pairs_trading import PairsTradingStrategy, PairsTradingConfig


# ========================================
# Task 1: Load Stock Data
# ========================================

def load_sample_data():
    """
    Load sample stock data for a cointegrated pair.

    For real implementation, use:
    - yfinance: pip install yfinance
    - alpaca_trade_api for paper trading
    - Or CSV files with historical data

    This generates synthetic data for demonstration.
    """
    print("📊 Loading stock data...\n")

    np.random.seed(42)
    n = 500  # 2 years of daily data

    # Simulate two stocks in same sector (e.g., KO and PEP)
    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    # Stock 1: Coca-Cola (KO)
    returns1 = np.random.normal(0.0003, 0.015, n)
    ko_prices = pd.Series(45 * np.exp(np.cumsum(returns1)), index=dates, name='KO')

    # Stock 2: PepsiCo (PEP) - cointegrated with KO
    # Long-term relationship with some noise
    hedge_ratio = 1.3
    noise = np.random.normal(0, 0.5, n)
    pep_prices = pd.Series((ko_prices + noise) * hedge_ratio, index=dates, name='PEP')

    print(f"✅ Loaded {n} days of data")
    print(f"   KO: ${ko_prices.iloc[-1]:.2f}")
    print(f"   PEP: ${pep_prices.iloc[-1]:.2f}\n")

    return ko_prices, pep_prices


# ========================================
# Task 2: Test Cointegration
# ========================================

def test_pair_cointegration(price1, price2):
    """
    Test if the pair is suitable for pairs trading.

    TODO: Use your PairsTradingStrategy to test cointegration
    """
    print("🔬 Testing cointegration...\n")

    strategy = PairsTradingStrategy()

    # Test cointegration
    is_coint, pvalue, hedge_ratio = strategy.test_cointegration(price1, price2)

    print(f"Cointegration Results:")
    print(f"  P-value: {pvalue:.6f}")
    print(f"  Cointegrated: {'✅ Yes' if is_coint else '❌ No'}")
    print(f"  Hedge ratio: {hedge_ratio:.4f}")

    if is_coint:
        print(f"\n✅ This pair is suitable for trading!")
    else:
        print(f"\n⚠️  This pair may not be suitable (p-value > 0.05)")

    return is_coint, hedge_ratio


# ========================================
# Task 3: Visualize Spread
# ========================================

def visualize_spread(price1, price2, hedge_ratio):
    """
    Visualize the price spread and z-score.

    TODO: Calculate and plot spread and z-score
    """
    print("\n📈 Visualizing spread...\n")

    strategy = PairsTradingStrategy()

    # Calculate spread
    spread = strategy.calculate_spread(price1, price2, hedge_ratio)

    # Calculate z-score
    zscore = strategy.calculate_zscore(spread)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Both prices (normalized)
    axes[0].plot(price1.index, price1 / price1.iloc[0], label=price1.name, linewidth=2)
    axes[0].plot(price2.index, price2 / price2.iloc[0], label=price2.name, linewidth=2)
    axes[0].set_title('Normalized Prices', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (normalized)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Spread
    axes[1].plot(spread.index, spread, label='Spread', color='purple', linewidth=2)
    axes[1].axhline(y=spread.mean(), color='black', linestyle='--', label='Mean')
    axes[1].axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle='--', alpha=0.5, label='±2σ')
    axes[1].axhline(y=spread.mean() - 2*spread.std(), color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Spread', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Spread Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Z-score
    axes[2].plot(zscore.index, zscore, label='Z-score', color='orange', linewidth=2)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[2].axhline(y=2, color='green', linestyle='--', label='Entry threshold (±2)')
    axes[2].axhline(y=-2, color='green', linestyle='--')
    axes[2].axhline(y=3.5, color='red', linestyle='--', label='Stop loss (±3.5)')
    axes[2].axhline(y=-3.5, color='red', linestyle='--')
    axes[2].fill_between(zscore.index, -2, 2, alpha=0.1, color='yellow', label='No trade zone')
    axes[2].set_title('Z-Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Z-Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pairs_trading_spread.png', dpi=150, bbox_inches='tight')
    print("✅ Chart saved: pairs_trading_spread.png\n")


# ========================================
# Task 4: Run Strategy
# ========================================

def run_pairs_strategy(price1, price2):
    """
    Run the pairs trading strategy and generate signals.

    TODO: Use run_strategy() to get trading signals
    """
    print("🎯 Running pairs trading strategy...\n")

    strategy = PairsTradingStrategy(
        PairsTradingConfig(
            lookback_period=60,
            entry_zscore=2.0,
            exit_zscore=0.5,
            stop_loss_zscore=3.5
        )
    )

    # Run strategy
    result = strategy.run_strategy(price1, price2, validate_pair=True)

    # Extract results
    signals = result['signals']
    zscore = result['zscore']
    half_life = result['half_life']

    # Calculate statistics
    num_trades = signals.diff().abs().sum() / 2
    time_in_market = (signals != 0).sum() / len(signals) * 100
    long_trades = (signals.diff() == 1).sum()
    short_trades = (signals.diff() == -1).sum()

    print(f"Strategy Statistics:")
    print(f"  Half-life: {half_life:.2f} days")
    print(f"  Number of trades: {num_trades:.0f}")
    print(f"  Long trades: {long_trades}")
    print(f"  Short trades: {short_trades}")
    print(f"  Time in market: {time_in_market:.1f}%")

    return result


# ========================================
# Task 5: Backtest Performance
# ========================================

def backtest_performance(price1, price2, result):
    """
    Calculate and analyze backtest performance.

    TODO: Calculate returns and performance metrics
    """
    print("\n📊 Backtesting performance...\n")

    strategy = PairsTradingStrategy()
    signals = result['signals']
    hedge_ratio = result['hedge_ratio']

    # Calculate strategy returns
    returns = strategy.calculate_strategy_returns(price1, price2, signals, hedge_ratio)

    # Performance metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Calculate equity curve
    equity = (1 + returns).cumprod() * 100000  # $100k starting capital

    # Drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    winning_trades = (returns > 0).sum()
    total_trades = (signals.diff() != 0).sum() / 2
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    print(f"Performance Metrics:")
    print(f"  Total return: {total_return*100:.2f}%")
    print(f"  Annual return: {annual_return*100:.2f}%")
    print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"  Max drawdown: {max_drawdown*100:.2f}%")
    print(f"  Win rate: {win_rate*100:.1f}%")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity, linewidth=2, label='Equity Curve')
    plt.axhline(y=100000, color='black', linestyle='--', alpha=0.5, label='Starting Capital')
    plt.fill_between(equity.index, 100000, equity, alpha=0.2)
    plt.title('Pairs Trading Equity Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pairs_trading_equity.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Equity curve saved: pairs_trading_equity.png")

    return returns, equity


# ========================================
# Main Exercise
# ========================================

def main():
    """
    Run the complete pairs trading exercise.
    """
    print("="*60)
    print("📚 Exercise 1: Pairs Trading")
    print("="*60)
    print()

    # Task 1: Load data
    price1, price2 = load_sample_data()

    # Task 2: Test cointegration
    is_coint, hedge_ratio = test_pair_cointegration(price1, price2)

    if not is_coint:
        print("\n⚠️  Pair not suitable for trading. Try another pair!")
        return

    # Task 3: Visualize spread
    visualize_spread(price1, price2, hedge_ratio)

    # Task 4: Run strategy
    result = run_pairs_strategy(price1, price2)

    # Task 5: Backtest performance
    returns, equity = backtest_performance(price1, price2, result)

    print("\n" + "="*60)
    print("✅ Exercise Complete!")
    print("="*60)
    print("\n📖 Key Takeaways:")
    print("  1. Always test cointegration before pairs trading")
    print("  2. Monitor half-life (should be 1-30 days)")
    print("  3. Use z-score for entry/exit timing")
    print("  4. Set stop losses for relationship breakdown")
    print("  5. Pairs trading is market-neutral (beta = 0)")
    print()
    print("🎯 Next Steps:")
    print("  - Try with real stock data (use yfinance)")
    print("  - Test different pairs (KO/PEP, XOM/CVX, etc.)")
    print("  - Optimize entry/exit thresholds")
    print("  - Add transaction costs to backtest")
    print()


if __name__ == "__main__":
    main()
