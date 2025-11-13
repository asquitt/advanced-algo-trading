"""
Simple example: Generate a trading signal and execute a paper trade.

This script demonstrates the basic workflow:
1. Generate a signal using LLM agents
2. Review the signal
3. Execute a paper trade (optional)

Run this after starting the platform with ./scripts/start.sh
"""

import requests
import json
from typing import Dict, Any


def generate_signal(symbol: str, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Generate a trading signal for a symbol.

    Args:
        symbol: Stock ticker (e.g., AAPL, MSFT)
        api_url: Base URL of the trading API

    Returns:
        Trading signal dictionary
    """
    print(f"\n{'='*60}")
    print(f"Generating signal for {symbol}...")
    print(f"{'='*60}\n")

    response = requests.post(
        f"{api_url}/signals/generate",
        params={
            "symbol": symbol,
            "use_cache": True,  # Use cached analysis if available (saves money!)
            "execute": False,  # Don't execute automatically
        }
    )

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

    signal = response.json()
    return signal


def print_signal(signal: Dict[str, Any]):
    """Pretty print a trading signal."""

    print(f"📊 Trading Signal for {signal['symbol']}")
    print(f"{'-'*60}")

    # Signal type with emoji
    signal_type = signal['signal_type']
    emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "🟡"

    print(f"\n{emoji} Signal Type: {signal_type}")
    print(f"📈 AI Conviction Score: {signal['ai_conviction_score']:.2%}")
    print(f"🎯 Confidence: {signal['confidence_score']:.2%}")

    print(f"\n📋 Component Scores:")
    print(f"  💰 Fundamental: {signal['fundamental_score']:.2%}")
    print(f"  📰 Sentiment: {signal['sentiment_score']:.2%}")
    print(f"  📊 Technical: {signal['technical_score']:.2%}")

    print(f"\n💭 AI Reasoning:")
    print(f"{'-'*60}")
    print(signal['reasoning'])

    # Cost information
    if 'metadata' in signal and 'total_cost_usd' in signal['metadata']:
        cost = signal['metadata']['total_cost_usd']
        tokens = signal['metadata']['tokens_used']
        print(f"\n💵 Analysis Cost: ${cost:.4f} ({tokens:,} tokens)")


def execute_trade(symbol: str, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Execute a paper trade for a symbol.

    Args:
        symbol: Stock ticker
        api_url: Base URL of the trading API

    Returns:
        Trade execution result
    """
    print(f"\n{'='*60}")
    print(f"Executing trade for {symbol}...")
    print(f"{'='*60}\n")

    response = requests.post(
        f"{api_url}/trades/execute",
        params={
            "symbol": symbol,
            "generate_signal": True,  # Generate new signal before trading
        }
    )

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

    trade = response.json()
    return trade


def print_trade(trade: Dict[str, Any]):
    """Pretty print a trade execution."""

    print(f"✅ Trade Executed!")
    print(f"{'-'*60}")
    print(f"Symbol: {trade['symbol']}")
    print(f"Side: {trade['side'].upper()}")
    print(f"Quantity: {trade['quantity']} shares")
    print(f"Entry Price: ${trade.get('entry_price', 'pending')}")
    print(f"Status: {trade['status']}")
    print(f"Order ID: {trade.get('order_id', 'N/A')}")


def get_portfolio(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Get current portfolio state."""

    response = requests.get(f"{api_url}/portfolio/summary")

    if response.status_code != 200:
        print(f"❌ Error getting portfolio: {response.status_code}")
        return None

    return response.json()


def print_portfolio(portfolio: Dict[str, Any]):
    """Pretty print portfolio summary."""

    print(f"\n{'='*60}")
    print(f"📊 Portfolio Summary")
    print(f"{'='*60}\n")

    account = portfolio['account']
    state = portfolio['portfolio_state']

    print(f"💵 Cash Balance: ${account['cash']:,.2f}")
    print(f"💼 Portfolio Value: ${account['portfolio_value']:,.2f}")
    print(f"📈 Buying Power: ${account['buying_power']:,.2f}")

    print(f"\n📊 Performance:")
    print(f"  Total P&L: ${state['total_pnl']:,.2f} ({state['total_pnl_percent']:.2%})")
    print(f"  Active Positions: {state['active_positions']}")

    if portfolio['positions']:
        print(f"\n🎯 Open Positions:")
        for pos in portfolio['positions']:
            pnl_color = "🟢" if pos['unrealized_pnl'] > 0 else "🔴"
            print(f"  {pnl_color} {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_entry_price']:.2f}")
            print(f"     Current: ${pos['current_price']:.2f} | P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_percent']:.2%})")


def main():
    """
    Main example workflow.

    This demonstrates:
    1. Generating signals for multiple stocks
    2. Reviewing the signals
    3. (Optional) Executing trades
    4. Checking portfolio
    """

    # Symbols to analyze
    symbols = ["AAPL", "MSFT", "GOOGL"]

    # Generate and review signals
    for symbol in symbols:
        signal = generate_signal(symbol)
        if signal:
            print_signal(signal)
            print("\n")

    # Ask user if they want to execute trades
    print(f"\n{'='*60}")
    response = input("Execute paper trades? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        for symbol in symbols:
            signal = generate_signal(symbol)
            if signal and signal['signal_type'] != 'HOLD':
                trade = execute_trade(symbol)
                if trade:
                    print_trade(trade)
                    print("\n")

        # Show portfolio after trades
        portfolio = get_portfolio()
        if portfolio:
            print_portfolio(portfolio)
    else:
        print("No trades executed (paper trading only anyway!)")

    print(f"\n{'='*60}")
    print("Example complete! 🎉")
    print(f"{'='*60}")
    print("\n💡 Next steps:")
    print("  - View signals in MLflow: http://localhost:5000")
    print("  - Check API docs: http://localhost:8000/docs")
    print("  - Review logs: docker-compose logs -f trading-api")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the trading platform is running:")
        print("  ./scripts/start.sh")
