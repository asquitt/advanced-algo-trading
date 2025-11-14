# Paper Trading with Alpaca

## Table of Contents
1. [What is Paper Trading?](#what-is-paper-trading)
2. [Why Paper Trading Matters](#why-paper-trading-matters)
3. [Introduction to Alpaca](#introduction-to-alpaca)
4. [Getting Started with Alpaca](#getting-started-with-alpaca)
5. [Authentication](#authentication)
6. [Order Types](#order-types)
7. [Position Tracking](#position-tracking)
8. [Risk Management Basics](#risk-management-basics)
9. [Best Practices](#best-practices)
10. [Common Mistakes](#common-mistakes)

---

## What is Paper Trading?

**Paper trading** (also called simulated trading) is the practice of trading with fake money in a real market environment. It allows you to test trading strategies without risking actual capital.

### How It Works

1. You get a **simulated account** with virtual money (e.g., $100,000)
2. You execute trades using **real market data** and prices
3. Orders are **simulated** - they don't affect the real market
4. Your account value changes based on **actual market movements**
5. You can track performance as if it were real trading

### Paper vs Live Trading

| Feature | Paper Trading | Live Trading |
|---------|--------------|--------------|
| **Money** | Virtual ($100k+) | Real (your money) |
| **Risk** | Zero financial risk | Real financial risk |
| **Market data** | Real-time | Real-time |
| **Execution** | Simulated | Real orders |
| **Emotions** | Minimal | High (fear, greed) |
| **Slippage** | Idealized | Real slippage |
| **Purpose** | Learning & testing | Making money |

### Benefits of Paper Trading

- **Risk-Free Learning**: Make mistakes without losing money
- **Strategy Testing**: Validate algorithms before going live
- **API Integration**: Test your code and API connections
- **Performance Tracking**: Understand if your strategy works
- **Build Confidence**: Gain experience before risking capital

---

## Why Paper Trading Matters

### 1. Protect Your Capital

Most new algorithmic traders lose money. Paper trading lets you learn from mistakes without financial consequences:

```python
# Bug that could be costly in live trading
def calculate_quantity(cash, price):
    # BUG: Forgot to handle division by zero!
    return cash / price  # Crashes if price is 0

# In paper trading: You catch the bug
# In live trading: Could result in failed orders or worse
```

### 2. Test Your Strategy

Before risking real money, you need to know:
- Does your strategy actually make money?
- How does it perform in different market conditions?
- What's the maximum drawdown?
- Are there any edge cases that break your code?

### 3. Understand Market Dynamics

Paper trading teaches you:
- How order execution works
- The impact of market hours
- Bid-ask spreads and slippage
- Volatility and price movements
- Position management

### 4. Regulatory Compliance

Some regulations require paper trading before live trading:
- Pattern Day Trading (PDT) rules
- Margin requirements
- Settlement periods (T+2)
- Short selling restrictions

### 5. Build Your System Gradually

You can develop your trading system step by step:
1. **Week 1**: Execute basic trades
2. **Week 2**: Add strategy logic
3. **Week 3**: Implement risk management
4. **Week 4**: Optimize performance
5. **Week 5+**: Go live with confidence

---

## Introduction to Alpaca

**Alpaca** is a commission-free trading platform with a powerful API designed for algorithmic trading. It's one of the most popular platforms for developers building trading systems.

### Why Alpaca?

1. **Free Paper Trading**: Unlimited paper trading accounts
2. **Commission-Free**: No fees for trades (even live trading)
3. **Modern API**: RESTful API with WebSocket streaming
4. **Real Market Data**: Actual market prices and quotes
5. **Easy Integration**: Official Python SDK
6. **Well Documented**: Excellent documentation and examples
7. **Active Community**: Large developer community

### Key Features

- **Paper and Live Trading**: Separate accounts for testing and real trading
- **Real-Time Data**: Live market data via WebSocket
- **Multiple Order Types**: Market, limit, stop, stop-limit orders
- **Fractional Shares**: Trade partial shares (e.g., 0.5 shares of AAPL)
- **Portfolio Management**: Track positions, orders, and account value
- **No Minimum**: Start with any amount (paper or live)

### What You Can Trade

- **US Stocks**: NYSE, NASDAQ, AMEX
- **ETFs**: All major ETFs
- **Crypto** (coming soon): Bitcoin, Ethereum, etc.

### Limitations

- **US Markets Only**: Currently only US stocks/ETFs
- **Market Hours**: Trading during market hours (9:30 AM - 4:00 PM ET)
- **Pattern Day Trading**: Live accounts subject to PDT rules
- **No Options**: Currently no options trading
- **No Forex**: No foreign exchange trading

---

## Getting Started with Alpaca

### Step 1: Create an Account

1. Go to https://alpaca.markets
2. Click "Sign Up"
3. Choose "Paper Trading Only" (for learning)
4. Verify your email
5. Access your dashboard

### Step 2: Get API Keys

```
1. Log into Alpaca dashboard
2. Click "Paper Trading" (top right)
3. Go to "API Keys" section
4. Click "Generate New Key"
5. Copy and save:
   - API Key ID
   - Secret Key
```

**IMPORTANT**: Never share your secret key or commit it to git!

### Step 3: Install Alpaca SDK

```bash
# Install the official Alpaca Python SDK
pip install alpaca-trade-api

# Or use the REST client directly
pip install requests
```

### Step 4: Test Your Connection

```python
import alpaca_trade_api as tradeapi

# Your API credentials
API_KEY = "your_api_key_here"
SECRET_KEY = "your_secret_key_here"
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading

# Initialize API connection
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Test connection - get account info
account = api.get_account()
print(f"Account Status: {account.status}")
print(f"Cash Available: ${account.cash}")
print(f"Portfolio Value: ${account.portfolio_value}")
```

---

## Authentication

### API Keys

Alpaca uses API key authentication. You need two keys:

```python
API_KEY = "PK1234567890"           # Public API Key ID
SECRET_KEY = "sk_abcdef123456"     # Secret Key (keep private!)
```

### Environment Variables (Recommended)

Never hardcode API keys! Use environment variables:

```python
# .env file (DO NOT commit to git!)
ALPACA_API_KEY=PK1234567890
ALPACA_SECRET_KEY=sk_abcdef123456
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")
```

### Secure Authentication

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Use in your code
api = tradeapi.REST(
    settings.alpaca_api_key,
    settings.alpaca_secret_key,
    settings.alpaca_base_url
)
```

---

## Order Types

### 1. Market Order

Executes immediately at the current market price.

```python
# Buy 100 shares of AAPL at market price
api.submit_order(
    symbol='AAPL',
    qty=100,
    side='buy',
    type='market',
    time_in_force='day'
)
```

**Pros**: Guaranteed execution, fast
**Cons**: Price uncertainty, can get bad fills

### 2. Limit Order

Executes only at a specified price or better.

```python
# Buy 100 shares of AAPL at $150 or less
api.submit_order(
    symbol='AAPL',
    qty=100,
    side='buy',
    type='limit',
    time_in_force='day',
    limit_price=150.00
)
```

**Pros**: Price control
**Cons**: May not execute

### 3. Stop Order

Triggers a market order when price reaches a stop price.

```python
# Sell 100 shares if price drops to $145 (stop loss)
api.submit_order(
    symbol='AAPL',
    qty=100,
    side='sell',
    type='stop',
    time_in_force='day',
    stop_price=145.00
)
```

**Pros**: Risk management
**Cons**: No price guarantee after trigger

### 4. Stop-Limit Order

Triggers a limit order when price reaches a stop price.

```python
# If price drops to $145, sell at $144 or better
api.submit_order(
    symbol='AAPL',
    qty=100,
    side='sell',
    type='stop_limit',
    time_in_force='day',
    stop_price=145.00,
    limit_price=144.00
)
```

**Pros**: Price control with stop protection
**Cons**: May not execute

### Time In Force Options

```python
# day: Cancel at end of trading day (default)
time_in_force='day'

# gtc: Good Till Cancelled (stays open until filled or cancelled)
time_in_force='gtc'

# ioc: Immediate Or Cancel (fill immediately or cancel)
time_in_force='ioc'

# fok: Fill Or Kill (fill completely or cancel)
time_in_force='fok'
```

### Practical Example

```python
def execute_trade(symbol, quantity, side, order_type='market', limit_price=None):
    """Execute a trade with proper error handling"""
    try:
        order_params = {
            'symbol': symbol,
            'qty': quantity,
            'side': side,
            'type': order_type,
            'time_in_force': 'day'
        }

        if order_type == 'limit' and limit_price:
            order_params['limit_price'] = limit_price

        order = api.submit_order(**order_params)

        print(f"Order submitted: {order.id}")
        print(f"Status: {order.status}")
        return order

    except Exception as e:
        print(f"Error submitting order: {e}")
        return None
```

---

## Position Tracking

### Get All Positions

```python
# Get all open positions
positions = api.list_positions()

for position in positions:
    print(f"Symbol: {position.symbol}")
    print(f"Quantity: {position.qty}")
    print(f"Market Value: ${position.market_value}")
    print(f"Unrealized P/L: ${position.unrealized_pl}")
    print(f"Unrealized P/L %: {position.unrealized_plpc}%")
    print("---")
```

### Get Specific Position

```python
# Get position for a specific symbol
try:
    position = api.get_position('AAPL')
    print(f"AAPL Position: {position.qty} shares")
    print(f"Average Entry: ${position.avg_entry_price}")
    print(f"Current Price: ${position.current_price}")
    print(f"P/L: ${position.unrealized_pl}")
except Exception:
    print("No position in AAPL")
```

### Track Account Performance

```python
# Get account information
account = api.get_account()

print(f"Portfolio Value: ${account.portfolio_value}")
print(f"Cash: ${account.cash}")
print(f"Buying Power: ${account.buying_power}")
print(f"Equity: ${account.equity}")
print(f"Last Equity: ${account.last_equity}")

# Calculate daily P/L
daily_pl = float(account.equity) - float(account.last_equity)
daily_pl_pct = (daily_pl / float(account.last_equity)) * 100

print(f"Daily P/L: ${daily_pl:.2f} ({daily_pl_pct:.2f}%)")
```

### Get Order History

```python
# Get all orders
orders = api.list_orders(status='all', limit=100)

for order in orders:
    print(f"Order ID: {order.id}")
    print(f"Symbol: {order.symbol}")
    print(f"Side: {order.side}")
    print(f"Quantity: {order.qty}")
    print(f"Status: {order.status}")
    print(f"Submitted: {order.submitted_at}")
    print("---")
```

### Cancel Orders

```python
# Cancel a specific order
api.cancel_order('order_id_here')

# Cancel all orders
api.cancel_all_orders()
```

---

## Risk Management Basics

### 1. Position Sizing

Never risk too much on a single trade:

```python
def calculate_position_size(account_value, risk_percent, entry_price, stop_price):
    """Calculate position size based on risk"""
    risk_amount = account_value * (risk_percent / 100)
    risk_per_share = abs(entry_price - stop_price)
    position_size = int(risk_amount / risk_per_share)
    return position_size

# Risk 1% of $100,000 account
position_size = calculate_position_size(
    account_value=100000,
    risk_percent=1,
    entry_price=150.00,
    stop_price=145.00  # $5 risk per share
)
# Result: 200 shares (1% risk = $1,000 / $5 = 200 shares)
```

### 2. Stop Loss Orders

Always have an exit plan:

```python
def execute_trade_with_stop_loss(symbol, quantity, entry_price, stop_loss_pct=2):
    """Execute trade with automatic stop loss"""
    # Calculate stop price (2% below entry)
    stop_price = entry_price * (1 - stop_loss_pct / 100)

    # Submit buy order
    buy_order = api.submit_order(
        symbol=symbol,
        qty=quantity,
        side='buy',
        type='limit',
        time_in_force='day',
        limit_price=entry_price
    )

    # Submit stop loss order
    stop_order = api.submit_order(
        symbol=symbol,
        qty=quantity,
        side='sell',
        type='stop',
        time_in_force='gtc',
        stop_price=round(stop_price, 2)
    )

    return buy_order, stop_order
```

### 3. Portfolio Limits

Don't over-concentrate in one position:

```python
def check_position_limit(symbol, quantity, max_position_pct=10):
    """Ensure position doesn't exceed portfolio limit"""
    account = api.get_account()
    portfolio_value = float(account.portfolio_value)

    # Get current price
    quote = api.get_latest_trade(symbol)
    price = quote.price

    # Calculate position value
    position_value = quantity * price

    # Calculate percentage
    position_pct = (position_value / portfolio_value) * 100

    if position_pct > max_position_pct:
        raise ValueError(
            f"Position too large: {position_pct:.1f}% "
            f"(max {max_position_pct}%)"
        )

    return True
```

### 4. Daily Loss Limit

Stop trading if you lose too much in a day:

```python
def check_daily_loss_limit(max_daily_loss_pct=3):
    """Check if daily loss limit exceeded"""
    account = api.get_account()

    equity = float(account.equity)
    last_equity = float(account.last_equity)

    daily_loss_pct = ((equity - last_equity) / last_equity) * 100

    if daily_loss_pct < -max_daily_loss_pct:
        # Stop trading for the day
        api.cancel_all_orders()
        raise Exception(
            f"Daily loss limit exceeded: {daily_loss_pct:.2f}%"
        )

    return True
```

---

## Best Practices

### 1. Test Everything in Paper First

```python
# Always use paper trading for development
BASE_URL = "https://paper-api.alpaca.markets"  # Paper
# BASE_URL = "https://api.alpaca.markets"      # Live - only when ready!
```

### 2. Handle Errors Gracefully

```python
def safe_submit_order(symbol, qty, side):
    """Submit order with proper error handling"""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        return order
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 3. Log All Trades

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_trade(symbol, qty, side):
    """Execute trade with logging"""
    logger.info(f"Submitting order: {side} {qty} {symbol}")

    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force='day'
    )

    logger.info(f"Order submitted: {order.id}, Status: {order.status}")
    return order
```

### 4. Check Market Status

```python
def is_market_open():
    """Check if market is currently open"""
    clock = api.get_clock()
    return clock.is_open

# Use before trading
if is_market_open():
    execute_trade("AAPL", 100, "buy")
else:
    print("Market is closed")
```

---

## Common Mistakes

### 1. Not Checking Account Balance

```python
# WRONG - May fail if insufficient funds
api.submit_order(symbol='AAPL', qty=10000, side='buy', type='market')

# RIGHT - Check buying power first
account = api.get_account()
if float(account.buying_power) > estimated_cost:
    api.submit_order(...)
```

### 2. Ignoring Order Status

```python
# WRONG - Assuming order was filled
order = api.submit_order(...)
# Continue without checking if order was filled

# RIGHT - Verify order status
order = api.submit_order(...)
filled_order = api.get_order(order.id)
if filled_order.status == 'filled':
    print("Order successfully filled")
```

### 3. Not Using Stop Losses

```python
# WRONG - No exit strategy
api.submit_order(symbol='AAPL', qty=100, side='buy', type='market')

# RIGHT - Always use stop losses
api.submit_order(symbol='AAPL', qty=100, side='buy', type='market')
api.submit_order(symbol='AAPL', qty=100, side='sell', type='stop', stop_price=145)
```

### 4. Hard-Coding API Keys

```python
# WRONG - Security risk!
API_KEY = "PK1234567890"

# RIGHT - Use environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
```

---

## Summary

Paper trading with Alpaca is the perfect way to learn algorithmic trading without risk. Key takeaways:

1. **Start with Paper**: Never trade live money until your strategy is proven
2. **Use Alpaca**: Free, powerful API, great for beginners
3. **Understand Order Types**: Know when to use market vs limit vs stop orders
4. **Track Everything**: Monitor positions, orders, and account performance
5. **Manage Risk**: Use position sizing, stop losses, and portfolio limits
6. **Handle Errors**: Always expect and handle API errors gracefully
7. **Secure Credentials**: Never hardcode API keys

Paper trading teaches you the mechanics of trading without the emotional and financial stress of real money. Master it before going live!

---

## Next Steps

1. Create your free Alpaca paper trading account
2. Get your API keys and test the connection
3. Execute your first paper trade
4. Build a simple trading bot
5. Track performance for at least a month
6. Read `testing_basics.md` to learn how to test your trading code
7. Only consider live trading after consistent paper trading success

Happy (paper) trading!
