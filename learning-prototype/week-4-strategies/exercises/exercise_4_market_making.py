"""
Exercise 4: Market Making Strategy

Goal: Implement and test a market making strategy with inventory management!

Tasks:
1. Generate order book data
2. Calculate fair prices and spreads
3. Manage inventory risk
4. Analyze order book imbalance
5. Backtest market making performance

Estimated time: 75-90 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('..')
from starter_code.market_making import MarketMakingStrategy, MarketMakingConfig


# ========================================
# Task 1: Generate Order Book Data
# ========================================

def generate_order_book_data(num_minutes=390):
    """
    Generate synthetic order book data.

    For real implementation:
    - Use exchange WebSocket feeds (Alpaca, Interactive Brokers)
    - Subscribe to L2 order book data
    - Track top-of-book (best bid/ask) updates
    """
    print(f"📊 Generating {num_minutes} minutes of order book data...\n")

    np.random.seed(42)

    timestamps = pd.date_range(
        start='2024-01-15 09:30:00',
        periods=num_minutes,
        freq='1min'
    )

    # Simulate fair value with random walk
    fair_value = 100.0
    fair_values = [fair_value]

    for _ in range(num_minutes - 1):
        change = np.random.normal(0, 0.02)
        fair_value += change
        fair_values.append(fair_value)

    fair_prices = pd.Series(fair_values, index=timestamps)

    # Generate bid-ask spread (varies with volatility)
    base_spread = 0.02  # 2 cents
    spread_variation = np.random.uniform(0.8, 1.2, num_minutes)
    spreads = base_spread * spread_variation

    # Generate bid and ask prices
    bid_prices = fair_prices - spreads / 2
    ask_prices = fair_prices + spreads / 2

    # Generate sizes with realistic patterns
    bid_sizes = np.random.randint(100, 500, num_minutes)
    ask_sizes = np.random.randint(100, 500, num_minutes)

    # Add some imbalance patterns
    for i in range(0, num_minutes, 50):
        # Random imbalance periods
        if np.random.random() > 0.5:
            bid_sizes[i:i+20] *= 2  # Bid heavy
        else:
            ask_sizes[i:i+20] *= 2  # Ask heavy

    bid_sizes = pd.Series(bid_sizes, index=timestamps)
    ask_sizes = pd.Series(ask_sizes, index=timestamps)

    # Trade prices (actual executions, not quotes)
    trade_prices = fair_prices + np.random.normal(0, 0.01, num_minutes)

    print(f"✅ Generated order book data:")
    print(f"   Fair price range: ${fair_prices.min():.2f} - ${fair_prices.max():.2f}")
    print(f"   Avg spread: ${spreads.mean():.4f} ({spreads.mean()/fair_prices.mean()*10000:.1f} bps)")
    print(f"   Avg bid size: {bid_sizes.mean():.0f}")
    print(f"   Avg ask size: {ask_sizes.mean():.0f}\n")

    return trade_prices, bid_prices, ask_prices, bid_sizes, ask_sizes, timestamps


# ========================================
# Task 2: Analyze Order Book
# ========================================

def analyze_order_book(bid_prices, ask_prices, bid_sizes, ask_sizes, timestamps):
    """
    Analyze order book characteristics.
    """
    print("🔍 Analyzing order book...\n")

    strategy = MarketMakingStrategy()

    # Calculate metrics
    fair_prices = []
    microprices = []
    spreads_bps = []
    imbalances = []

    for i in range(len(bid_prices)):
        # Fair price
        fair = strategy.estimate_fair_price(
            bid_prices.iloc[i],
            ask_prices.iloc[i],
            bid_sizes.iloc[i],
            ask_sizes.iloc[i]
        )
        fair_prices.append(fair)

        # Microprice
        micro = strategy.calculate_microprice(
            bid_prices.iloc[i],
            ask_prices.iloc[i],
            bid_sizes.iloc[i],
            ask_sizes.iloc[i]
        )
        microprices.append(micro)

        # Spread in bps
        spread = (ask_prices.iloc[i] - bid_prices.iloc[i]) / fair * 10000
        spreads_bps.append(spread)

        # Imbalance
        imb = strategy.calculate_order_book_imbalance(
            bid_sizes.iloc[i],
            ask_sizes.iloc[i]
        )
        imbalances.append(imb)

    fair_prices = pd.Series(fair_prices, index=timestamps)
    microprices = pd.Series(microprices, index=timestamps)
    spreads_bps = pd.Series(spreads_bps, index=timestamps)
    imbalances = pd.Series(imbalances, index=timestamps)

    print(f"Order Book Statistics:")
    print(f"  Avg spread: {spreads_bps.mean():.2f} bps")
    print(f"  Min spread: {spreads_bps.min():.2f} bps")
    print(f"  Max spread: {spreads_bps.max():.2f} bps")
    print(f"\nImbalance Statistics:")
    print(f"  Avg imbalance: {imbalances.mean():.3f}")
    print(f"  Bid-heavy periods: {(imbalances > 0.2).sum()}")
    print(f"  Ask-heavy periods: {(imbalances < -0.2).sum()}")
    print(f"  Balanced periods: {(abs(imbalances) <= 0.2).sum()}")

    # Visualize
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Bid-Ask spread
    mid_prices = (bid_prices + ask_prices) / 2
    axes[0].plot(timestamps, mid_prices, linewidth=2, color='black',
                 alpha=0.7, label='Mid Price')
    axes[0].fill_between(timestamps, bid_prices, ask_prices,
                          alpha=0.3, color='blue', label='Bid-Ask Spread')
    axes[0].plot(timestamps, fair_prices, linewidth=1, color='red',
                 linestyle='--', label='Fair Price', alpha=0.7)
    axes[0].set_title('Order Book: Bid-Ask Spread', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Spread in bps
    axes[1].plot(timestamps, spreads_bps, linewidth=1.5, color='purple')
    axes[1].axhline(y=spreads_bps.mean(), color='red', linestyle='--',
                    label=f'Avg: {spreads_bps.mean():.1f} bps')
    axes[1].fill_between(timestamps, spreads_bps, spreads_bps.mean(),
                          where=(spreads_bps > spreads_bps.mean()),
                          alpha=0.3, color='red')
    axes[1].fill_between(timestamps, spreads_bps, spreads_bps.mean(),
                          where=(spreads_bps <= spreads_bps.mean()),
                          alpha=0.3, color='green')
    axes[1].set_title('Spread (bps)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Spread (bps)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Order sizes
    axes[2].plot(timestamps, bid_sizes, linewidth=1, color='green',
                 alpha=0.7, label='Bid Size')
    axes[2].plot(timestamps, ask_sizes, linewidth=1, color='red',
                 alpha=0.7, label='Ask Size')
    axes[2].set_title('Order Book Depth', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Size')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Imbalance
    colors = ['green' if imb > 0 else 'red' for imb in imbalances]
    axes[3].bar(timestamps, imbalances, width=0.0007, color=colors, alpha=0.6)
    axes[3].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[3].axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Imbalance Threshold')
    axes[3].axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
    axes[3].set_title('Order Book Imbalance', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Imbalance')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('order_book_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Order book analysis chart saved: order_book_analysis.png\n")

    return fair_prices, microprices, spreads_bps, imbalances


# ========================================
# Task 3: Run Market Making Strategy
# ========================================

def run_market_making_strategy(prices, bid_prices, ask_prices, bid_sizes, ask_sizes,
                               fair_prices, imbalances, timestamps):
    """
    Run market making strategy and generate quotes.
    """
    print("🎯 Running market making strategy...\n")

    strategy = MarketMakingStrategy(
        MarketMakingConfig(
            base_spread_bps=10.0,
            base_quote_size=100,
            max_inventory=1000,
            target_inventory=0,
            inventory_skew_factor=0.5
        )
    )

    # Track quotes and inventory over time
    all_quotes = []
    inventory_history = [0]
    pnl_history = [0.0]

    current_inventory = 0
    realized_pnl = 0.0
    avg_cost = 0.0

    returns = prices.pct_change()
    vol = strategy.calculate_realized_volatility(returns, lookback=20)
    avg_vol = vol

    for i in range(len(prices)):
        if i < 20:  # Need enough data for volatility
            continue

        # Generate quotes
        quotes = strategy.generate_quotes(
            fair_price=fair_prices.iloc[i],
            volatility=vol,
            avg_volatility=avg_vol,
            current_inventory=current_inventory,
            bid_size_book=bid_sizes.iloc[i],
            ask_size_book=ask_sizes.iloc[i]
        )

        quotes['timestamp'] = timestamps[i]
        quotes['inventory'] = current_inventory
        all_quotes.append(quotes)

        # Simulate fills (simplified)
        # Assume we get filled if market crosses our quotes
        if i > 0:
            # Buy fill (someone hits our bid)
            if bid_prices.iloc[i] <= quotes['bid_price']:
                fill_size = min(quotes['bid_size'], 50)  # Partial fill
                fill_price = quotes['bid_price']

                # Update inventory
                current_inventory += fill_size

                # Update average cost
                if current_inventory > 0:
                    avg_cost = (avg_cost * (current_inventory - fill_size) +
                               fill_price * fill_size) / current_inventory

            # Sell fill (someone lifts our ask)
            if ask_prices.iloc[i] >= quotes['ask_price']:
                fill_size = min(quotes['ask_size'], 50)
                fill_price = quotes['ask_price']

                # Update inventory
                current_inventory -= fill_size

                # Realize PnL if reducing position
                if current_inventory >= 0:
                    trade_pnl = fill_size * (fill_price - avg_cost)
                    realized_pnl += trade_pnl

        # Calculate unrealized PnL
        unrealized_pnl = strategy.calculate_inventory_pnl(
            current_inventory, avg_cost, prices.iloc[i]
        )
        total_pnl = realized_pnl + unrealized_pnl

        inventory_history.append(current_inventory)
        pnl_history.append(total_pnl)

    quotes_df = pd.DataFrame(all_quotes)

    print(f"Market Making Results:")
    print(f"  Total quotes: {len(quotes_df)}")
    print(f"  Final inventory: {current_inventory}")
    print(f"  Final PnL: ${pnl_history[-1]:.2f}")
    print(f"  Avg spread quoted: {quotes_df['spread_bps'].mean():.2f} bps")
    print(f"  Avg inventory: {quotes_df['inventory'].mean():.0f}")
    print(f"  Max inventory: {quotes_df['inventory'].max()}")
    print(f"  Min inventory: {quotes_df['inventory'].min()}")

    return quotes_df, inventory_history, pnl_history


# ========================================
# Task 4: Analyze Inventory Management
# ========================================

def analyze_inventory_management(quotes_df, inventory_history, timestamps):
    """
    Analyze how inventory was managed.
    """
    print("\n📊 Analyzing inventory management...\n")

    inventory = pd.Series(inventory_history[1:], index=timestamps[20:])

    # Inventory statistics
    time_long = (inventory > 0).sum() / len(inventory) * 100
    time_short = (inventory < 0).sum() / len(inventory) * 100
    time_flat = (inventory == 0).sum() / len(inventory) * 100

    avg_long_size = inventory[inventory > 0].mean() if (inventory > 0).any() else 0
    avg_short_size = abs(inventory[inventory < 0].mean()) if (inventory < 0).any() else 0

    print(f"Inventory Statistics:")
    print(f"  Time long: {time_long:.1f}%")
    print(f"  Time short: {time_short:.1f}%")
    print(f"  Time flat: {time_flat:.1f}%")
    print(f"  Avg long size: {avg_long_size:.0f}")
    print(f"  Avg short size: {avg_short_size:.0f}")
    print(f"  Max long: {inventory.max()}")
    print(f"  Max short: {inventory.min()}")

    # Analyze quote skewing
    avg_skew_when_long = quotes_df[quotes_df['inventory'] > 0]['inventory_skew'].mean()
    avg_skew_when_short = quotes_df[quotes_df['inventory'] < 0]['inventory_skew'].mean()

    print(f"\nQuote Skewing:")
    print(f"  Avg skew when long: {avg_skew_when_long:.3f}")
    print(f"  Avg skew when short: {avg_skew_when_short:.3f}")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Inventory over time
    axes[0].plot(inventory.index, inventory, linewidth=2, color='blue')
    axes[0].fill_between(inventory.index, 0, inventory,
                          where=(inventory > 0), alpha=0.3, color='green', label='Long')
    axes[0].fill_between(inventory.index, 0, inventory,
                          where=(inventory < 0), alpha=0.3, color='red', label='Short')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0].set_title('Inventory Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Inventory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Quote prices relative to fair
    axes[1].plot(quotes_df['timestamp'], quotes_df['bid_price'] - fair_prices.iloc[20:].values,
                 linewidth=1, color='green', alpha=0.7, label='Bid vs Fair')
    axes[1].plot(quotes_df['timestamp'], quotes_df['ask_price'] - fair_prices.iloc[20:].values,
                 linewidth=1, color='red', alpha=0.7, label='Ask vs Fair')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Quote Prices Relative to Fair Price', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price Difference ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Inventory skew
    axes[2].scatter(quotes_df['inventory'], quotes_df['inventory_skew'],
                    alpha=0.5, s=20, c=quotes_df['inventory'], cmap='RdYlGn_r')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[2].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[2].set_title('Inventory vs Skew Factor', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Inventory')
    axes[2].set_ylabel('Skew Factor')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inventory_management.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Inventory management chart saved: inventory_management.png\n")


# ========================================
# Task 5: Analyze PnL
# ========================================

def analyze_pnl(pnl_history, timestamps):
    """
    Analyze market making PnL.
    """
    print("💰 Analyzing PnL...\n")

    pnl = pd.Series(pnl_history[1:], index=timestamps[20:])

    # PnL statistics
    final_pnl = pnl.iloc[-1]
    max_pnl = pnl.max()
    min_pnl = pnl.min()

    # Drawdown
    running_max = pnl.expanding().max()
    drawdown = pnl - running_max
    max_drawdown = drawdown.min()

    # PnL changes (profits per period)
    pnl_changes = pnl.diff()
    profitable_periods = (pnl_changes > 0).sum()
    losing_periods = (pnl_changes < 0).sum()
    win_rate = profitable_periods / (profitable_periods + losing_periods) if (profitable_periods + losing_periods) > 0 else 0

    print(f"PnL Statistics:")
    print(f"  Final PnL: ${final_pnl:.2f}")
    print(f"  Max PnL: ${max_pnl:.2f}")
    print(f"  Min PnL: ${min_pnl:.2f}")
    print(f"  Max drawdown: ${max_drawdown:.2f}")
    print(f"\nPeriod Analysis:")
    print(f"  Profitable periods: {profitable_periods}")
    print(f"  Losing periods: {losing_periods}")
    print(f"  Win rate: {win_rate*100:.1f}%")

    # Plot PnL
    plt.figure(figsize=(14, 6))
    plt.plot(pnl.index, pnl, linewidth=2, color='blue', label='Cumulative PnL')
    plt.fill_between(pnl.index, 0, pnl,
                     where=(pnl >= 0), alpha=0.3, color='green')
    plt.fill_between(pnl.index, 0, pnl,
                     where=(pnl < 0), alpha=0.3, color='red')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.title('Market Making PnL', fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('PnL ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('market_making_pnl.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ PnL chart saved: market_making_pnl.png")


# ========================================
# Main Exercise
# ========================================

def main():
    """
    Run the complete market making exercise.
    """
    print("="*60)
    print("📚 Exercise 4: Market Making Strategy")
    print("="*60)
    print()

    # Task 1: Generate data
    prices, bid_prices, ask_prices, bid_sizes, ask_sizes, timestamps = \
        generate_order_book_data(num_minutes=390)

    # Task 2: Analyze order book
    fair_prices, microprices, spreads_bps, imbalances = \
        analyze_order_book(bid_prices, ask_prices, bid_sizes, ask_sizes, timestamps)

    # Task 3: Run strategy
    quotes_df, inventory_history, pnl_history = \
        run_market_making_strategy(
            prices, bid_prices, ask_prices, bid_sizes, ask_sizes,
            fair_prices, imbalances, timestamps
        )

    # Task 4: Analyze inventory
    analyze_inventory_management(quotes_df, inventory_history, timestamps)

    # Task 5: Analyze PnL
    analyze_pnl(pnl_history, timestamps)

    print("\n" + "="*60)
    print("✅ Exercise Complete!")
    print("="*60)
    print("\n📖 Key Takeaways:")
    print("  1. Market making profits from bid-ask spread")
    print("  2. Inventory management is crucial to avoid directional risk")
    print("  3. Quote skewing helps manage inventory")
    print("  4. Order book imbalance predicts short-term price moves")
    print("  5. Adverse selection is the biggest risk")
    print()
    print("🎯 Next Steps:")
    print("  - Connect to real exchange data (Alpaca, IB)")
    print("  - Implement multi-level order book analysis")
    print("  - Add adverse selection detection")
    print("  - Test on different assets and liquidity levels")
    print()


if __name__ == "__main__":
    main()
