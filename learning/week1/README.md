# Week 1: Foundations 🏗️

**Goal:** Build a solid foundation by understanding trading data structures and implementing basic trading logic.

**Time Estimate:** 8-10 hours over 5 days

## 📚 What You'll Learn

By the end of this week, you'll understand:
- How trading systems represent data (orders, trades, positions)
- How to calculate profit & loss (P&L)
- Basic technical indicators (RSI, moving averages)
- Simple trading strategies
- How to backtest a strategy on historical data

## 🎯 Learning Objectives

- [ ] Create data models for trading signals, orders, and positions
- [ ] Implement P&L calculation
- [ ] Build 3 technical indicators (SMA, RSI, MACD)
- [ ] Create a simple trading strategy
- [ ] Backtest the strategy on historical data
- [ ] Calculate performance metrics (Sharpe ratio, max drawdown)

## 📖 Concepts to Understand

### Trading Basics

**Order**: An instruction to buy or sell
- Market Order: Buy/sell immediately at current price
- Limit Order: Buy/sell only at specified price or better

**Position**: Amount of stock you own
- Long Position: You own the stock (profit when price goes up)
- Short Position: You borrowed and sold (profit when price goes down)

**P&L (Profit & Loss)**:
```python
# For a buy (long) trade:
P&L = (exit_price - entry_price) * quantity

# Example:
# Buy 100 shares at $50 = $5,000
# Sell 100 shares at $55 = $5,500
# P&L = ($55 - $50) * 100 = $500 profit (10%)
```

**Portfolio**: Collection of all your positions
- Portfolio Value = Cash + Sum(position values)
- Return = (Current Value - Starting Value) / Starting Value

### Technical Indicators

**SMA (Simple Moving Average)**:
- Average price over N periods
- SMA(20) = average of last 20 days
- Used to identify trends

**RSI (Relative Strength Index)**:
- Momentum indicator (0-100)
- RSI > 70: Overbought (might sell)
- RSI < 30: Oversold (might buy)

**MACD (Moving Average Convergence Divergence)**:
- Trend-following indicator
- MACD = EMA(12) - EMA(26)
- Signal = EMA(9) of MACD
- When MACD crosses above signal: bullish

## 🛠️ Implementation Tasks

### Task 1: Data Models (starter.py)

Create Pydantic models for:
1. `TradingSignal` - AI/indicator recommendation
2. `Order` - Instruction to buy/sell
3. `Trade` - Executed order
4. `Position` - Current holdings
5. `Portfolio` - Overall state

### Task 2: Technical Indicators (indicators.py)

Implement:
1. `calculate_sma()` - Simple Moving Average
2. `calculate_rsi()` - Relative Strength Index
3. `calculate_macd()` - MACD indicator

### Task 3: Trading Strategy (strategy.py)

Create a simple RSI strategy:
- Buy when RSI < 30 (oversold)
- Sell when RSI > 70 (overbought)
- Hold otherwise

### Task 4: Backtesting (backtest.py)

Implement:
1. Load historical price data
2. Calculate indicators
3. Generate signals
4. Simulate trades
5. Calculate performance metrics

## 📝 Daily Plan

### Day 1: Setup & Data Models (2 hours)
- Read this README
- Set up environment
- Implement data models in `starter.py`
- Run: `python starter.py`
- Tests: `pytest tests/test_day1.py`

### Day 2: Technical Indicators (2-3 hours)
- Read about SMA, RSI, MACD
- Implement in `indicators.py`
- Tests: `pytest tests/test_day2.py`
- Verify with `python test_indicators.py` (visual)

### Day 3: Trading Strategy (2 hours)
- Implement RSI strategy in `strategy.py`
- Tests: `pytest tests/test_day3.py`

### Day 4: Backtesting (2-3 hours)
- Implement backtest engine in `backtest.py`
- Run on historical data
- Tests: `pytest tests/test_day4.py`

### Day 5: Performance & Review (2 hours)
- Calculate Sharpe ratio, max drawdown
- Review all code
- Compare with solutions
- Tests: `pytest tests/ -v` (all should pass)

## 🎓 Exercises

### Exercise 1: Manual P&L Calculation
Calculate P&L for these trades:
```
Trade 1: Buy 50 shares AAPL at $150, sell at $160
Trade 2: Buy 100 shares GOOGL at $2800, sell at $2750
Trade 3: Buy 200 shares MSFT at $380, sell at $395
```

<details>
<summary>Click for Answer</summary>

```python
# Trade 1
pnl_1 = (160 - 150) * 50 = $500 profit

# Trade 2
pnl_2 = (2750 - 2800) * 100 = -$5,000 loss

# Trade 3
pnl_3 = (395 - 380) * 200 = $3,000 profit

# Total P&L = $500 - $5,000 + $3,000 = -$1,500 loss
```
</details>

### Exercise 2: RSI Interpretation
Given RSI values, what should the strategy do?
```
Day 1: RSI = 25
Day 2: RSI = 45
Day 3: RSI = 75
Day 4: RSI = 80
Day 5: RSI = 30
```

<details>
<summary>Click for Answer</summary>

```python
Day 1: RSI = 25 → BUY (oversold, RSI < 30)
Day 2: RSI = 45 → HOLD (neutral)
Day 3: RSI = 75 → SELL (overbought, RSI > 70)
Day 4: RSI = 80 → SELL (very overbought)
Day 5: RSI = 30 → BUY (at oversold threshold)
```
</details>

### Exercise 3: SMA Crossover
If SMA(10) crosses above SMA(50), what does this signal?

<details>
<summary>Click for Answer</summary>

This is a **bullish** signal (golden cross):
- Short-term average (10-day) rising faster than long-term (50-day)
- Indicates upward trend beginning
- Common strategy: Buy on golden cross, sell on death cross (opposite)
</details>

## 🧪 Testing Your Code

### Run All Tests
```bash
# All week 1 tests
pytest tests/test_week1.py -v

# Specific day
pytest tests/test_day1.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Expected Test Results

Day 1 (Data Models):
```
✓ test_create_trading_signal
✓ test_signal_validation
✓ test_order_creation
✓ test_trade_pnl_calculation
✓ test_position_tracking
✓ test_portfolio_value
```

Day 2 (Indicators):
```
✓ test_sma_calculation
✓ test_rsi_calculation
✓ test_rsi_boundaries
✓ test_macd_calculation
```

Day 3 (Strategy):
```
✓ test_rsi_buy_signal
✓ test_rsi_sell_signal
✓ test_rsi_hold_signal
```

Day 4 (Backtest):
```
✓ test_backtest_execution
✓ test_performance_metrics
✓ test_sharpe_ratio
✓ test_max_drawdown
```

## 📚 Additional Resources

### Reading
- [Investopedia: Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [RSI Explained](https://www.investopedia.com/terms/r/rsi.asp)
- [Moving Averages Guide](https://www.investopedia.com/terms/m/movingaverage.asp)

### Videos
- [Technical Indicators Explained (YouTube)](https://www.youtube.com/results?search_query=technical+indicators+explained)
- [Backtesting Basics](https://www.youtube.com/results?search_query=backtesting+trading+strategy)

### Practice
- Try implementing other indicators: Bollinger Bands, Stochastic Oscillator
- Experiment with different RSI thresholds (20/80 instead of 30/70)
- Combine multiple indicators (RSI + MACD)

## 🎯 Success Criteria

You've completed Week 1 when you can:
- [ ] Explain P&L calculation
- [ ] Implement 3 technical indicators correctly
- [ ] Create a simple trading strategy
- [ ] Run a backtest and understand the results
- [ ] Calculate Sharpe ratio and max drawdown
- [ ] All tests pass (`pytest tests/test_week1.py`)

## ⏭️ Next Week Preview

**Week 2: LLM Integration**
- Connect to Claude/GPT APIs
- Prompt engineering for stock analysis
- Parse LLM responses into trading signals
- Combine AI signals with technical indicators

## 💡 Tips

- **Start simple**: Get basic functionality working before optimizing
- **Test frequently**: Run tests after each function you write
- **Use print statements**: Debug by printing intermediate values
- **Compare with solutions**: But only after trying yourself!
- **Take breaks**: Don't code for more than 90 minutes straight

## 🆘 Common Issues

**"ModuleNotFoundError: No module named 'pandas'"**
```bash
pip install -r requirements.txt
```

**"Division by zero in RSI"**
```python
# Check for edge cases:
if gains_avg == 0 and losses_avg == 0:
    return 50  # Neutral when no movement
```

**"Tests pass but results look wrong"**
```python
# Add debug prints:
print(f"RSI calculated: {rsi}")
print(f"Expected range: 0-100")
print(f"Signal generated: {signal}")
```

---

**Ready to code? Open `starter.py` and let's build! 🚀**
