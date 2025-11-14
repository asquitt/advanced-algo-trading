# 📚 Key Concepts - Trading & Finance Fundamentals

Quick reference guide for important trading concepts.

## 🎯 Trading Basics

### Orders

**Market Order**: Buy/sell immediately at current market price
- **Pros**: Guaranteed execution
- **Cons**: Price not guaranteed (slippage)
- **Use when**: You need immediate execution

**Limit Order**: Buy/sell only at specified price or better
- **Pros**: Price control
- **Cons**: May not execute
- **Use when**: You can wait for better price

### Positions

**Long Position**: You own the stock
- **Profit when**: Price goes up
- **Max loss**: 100% (stock goes to $0)
- **Example**: Buy AAPL at $150, sell at $160 = $10 profit per share

**Short Position**: You borrowed and sold stock
- **Profit when**: Price goes down
- **Max loss**: Unlimited (price can rise infinitely)
- **Example**: Sell AAPL at $150, buy back at $140 = $10 profit per share

### P&L (Profit & Loss)

**Formula**:
```
Long:  P&L = (Exit Price - Entry Price) × Quantity
Short: P&L = (Entry Price - Exit Price) × Quantity

Percentage Return = (P&L / Cost Basis) × 100
Cost Basis = Entry Price × Quantity
```

**Example**:
```
Buy 100 shares at $50  → Cost: $5,000
Sell 100 shares at $55 → Revenue: $5,500
P&L = $500 (10% return)
```

## 📊 Technical Indicators

### SMA (Simple Moving Average)

**Purpose**: Identify trends
**Formula**: Average of last N prices
```
SMA(20) = (Price[today] + Price[yesterday] + ... + Price[19 days ago]) / 20
```

**Interpretation**:
- Price above SMA → Uptrend
- Price below SMA → Downtrend
- SMA(50) crosses above SMA(200) → "Golden Cross" (bullish)

### RSI (Relative Strength Index)

**Purpose**: Measure momentum (overbought/oversold)
**Range**: 0-100
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss (over 14 periods typically)
```

**Interpretation**:
- RSI > 70 → Overbought (might sell)
- RSI < 30 → Oversold (might buy)
- RSI = 50 → Neutral

### MACD (Moving Average Convergence Divergence)

**Purpose**: Trend following and momentum
**Components**:
- MACD Line = EMA(12) - EMA(26)
- Signal Line = EMA(9) of MACD
- Histogram = MACD - Signal

**Interpretation**:
- MACD crosses above Signal → Bullish (buy)
- MACD crosses below Signal → Bearish (sell)
- Histogram growing → Trend strengthening

## 🛡️ Risk Management

### Kelly Criterion

**Purpose**: Optimal position sizing
**Formula**:
```
f* = (p × b - q) / b

Where:
- p = win probability (e.g., 0.60 = 60% win rate)
- q = loss probability (1 - p)
- b = win/loss ratio (average win ÷ average loss)
- f* = fraction of capital to risk
```

**Example**:
```
Win rate = 60% (p = 0.6)
Win/loss ratio = 1.5 (wins are 1.5x losses)
q = 1 - 0.6 = 0.4

f* = (0.6 × 1.5 - 0.4) / 1.5
   = (0.9 - 0.4) / 1.5
   = 0.5 / 1.5
   = 0.333 (33.3% of capital)
```

**Important**: Use **fractional Kelly** (0.25x or 0.5x) for safety!

### CVaR (Conditional Value at Risk)

**Purpose**: Measure tail risk (worst-case losses)
**Definition**: Expected loss in worst X% of cases

**Example**:
```
CVaR 95% = -5%

This means: In the worst 5% of scenarios,
you can expect to lose 5% on average.
```

**Usage**:
- Position CVaR limit: ≤ 2% of portfolio
- Portfolio CVaR limit: ≤ 5% of portfolio

### Drawdown

**Definition**: Peak-to-trough decline
```
Drawdown = (Peak Value - Current Value) / Peak Value

Example:
Portfolio was $100,000 (peak)
Now worth $85,000
Drawdown = ($100,000 - $85,000) / $100,000 = 15%
```

**Max Drawdown**: Largest peak-to-trough decline in history
- **Good**: < 10%
- **Acceptable**: 10-20%
- **Concerning**: 20-30%
- **Dangerous**: > 30%

## 📈 Performance Metrics

### Sharpe Ratio

**Purpose**: Risk-adjusted returns
**Formula**:
```
Sharpe = (Average Return - Risk-Free Rate) / Standard Deviation

Higher is better!
```

**Interpretation**:
- < 1.0 → Poor
- 1.0-2.0 → Good
- 2.0-3.0 → Very good
- > 3.0 → Excellent

**Example**:
```
Average return = 15% per year
Risk-free rate = 2% (treasury bonds)
Std deviation = 10%

Sharpe = (15% - 2%) / 10% = 1.3 (Good!)
```

### Win Rate

**Formula**:
```
Win Rate = Number of Winning Trades / Total Trades
```

**Interpretation**:
- 50-55% → Acceptable
- 55-60% → Good
- 60-65% → Very good
- > 65% → Excellent (or overfitted!)

**Note**: Win rate alone doesn't tell the whole story!
- 40% win rate with 3:1 win/loss ratio → Profitable
- 70% win rate with 1:3 win/loss ratio → Losing money

### Profit Factor

**Formula**:
```
Profit Factor = Gross Profit / Gross Loss

Example:
Total wins = $10,000
Total losses = $4,000
Profit Factor = 10,000 / 4,000 = 2.5
```

**Interpretation**:
- < 1.0 → Losing strategy
- 1.0-1.5 → Marginally profitable
- 1.5-2.0 → Good
- > 2.0 → Excellent

## 🤖 LLM Concepts (Week 2)

### Prompt Engineering

**Good Prompt Structure**:
1. Context (what you want analyzed)
2. Data (provide relevant information)
3. Task (specific instruction)
4. Format (how to respond)

**Example**:
```
Context: You are a financial analyst.
Data: AAPL P/E=28, Revenue Growth=12%, Profit Margin=26%
Task: Determine if this is a buy, sell, or hold.
Format: Respond in JSON with score and reasoning.
```

### Temperature

**What it controls**: Randomness of responses
- **0.0**: Deterministic (same answer every time)
- **0.7**: Balanced (some creativity)
- **1.0+**: Creative (more random)

**For trading**: Use 0.0-0.3 for consistency

### Token Limits

**What they are**: Units of text (roughly 4 chars = 1 token)
- Claude 3.5 Sonnet: 200K tokens (~150K words)
- GPT-4: 8K-128K tokens depending on model

**Strategy**: Summarize data to fit limits

## 💡 Tips & Best Practices

### Position Sizing
- Never risk more than 2% per trade
- Use 25% of portfolio maximum per position
- Diversify across at least 5-10 positions

### Backtesting
- Always use out-of-sample data
- Account for transaction costs
- Include slippage estimates
- Test across different market conditions

### Risk Control
- Set stop losses
- Monitor drawdown continuously
- Reduce size during losing streaks
- Never average down on losing trades

### LLM Usage
- Cache responses to save money
- Validate all LLM outputs
- Don't blindly trust AI recommendations
- Combine AI with technical indicators

## 📖 Further Reading

- **Books**:
  - "A Random Walk Down Wall Street" (Burton Malkiel)
  - "The Intelligent Investor" (Benjamin Graham)
  - "Algorithmic Trading" (Ernest Chan)

- **Online**:
  - Investopedia (free education)
  - QuantStart (algorithmic trading blog)
  - Papers on SSRN.com

- **Courses**:
  - Coursera: Financial Markets (Yale)
  - Khan Academy: Finance & Capital Markets

---

**Remember**:
- Start small (paper trading)
- Learn continuously
- Manage risk obsessively
- Never stop testing
