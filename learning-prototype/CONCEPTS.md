# 📖 Key Concepts Glossary

**A comprehensive reference for all concepts covered in the 8-week program**

---

## Trading Concepts

### Algorithmic Trading
**Definition**: Using computer programs to execute trades based on predefined rules and strategies.

**Why It Matters**: Removes emotion, enables 24/7 trading, processes data faster than humans, can execute complex strategies.

**Example**: "If RSI < 30 and MACD crosses above signal line, buy 100 shares."

---

### Paper Trading
**Definition**: Simulated trading with fake money to test strategies without risk.

**Why It Matters**: Learn and test without losing real money. Industry standard before live trading.

**Best Practice**: Paper trade for at least 3-6 months before using real capital.

---

### Signal
**Definition**: A recommendation to buy, sell, or hold a security based on analysis.

**Types**:
- **BUY**: Enter a long position
- **SELL**: Enter a short position or close long
- **HOLD**: No action recommended

**Components**:
- Direction (BUY/SELL/HOLD)
- Confidence (0-1, how sure we are)
- Reasoning (why this signal?)

---

### Position Sizing
**Definition**: Determining how many shares/contracts to trade.

**Methods**:
- **Fixed Dollar**: Always trade $10,000 worth
- **Fixed Percentage**: Always risk 2% of portfolio
- **Kelly Criterion**: Mathematically optimal size
- **Volatility-Adjusted**: Smaller positions in volatile stocks

**Formula (Simple)**:
```
Position Size = (Portfolio Value × Risk Percentage) / Stock Price
```

---

## Risk Management Concepts

### Maximum Drawdown
**Definition**: Largest peak-to-trough decline in portfolio value.

**Formula**:
```
Max Drawdown = (Trough Value - Peak Value) / Peak Value
```

**Example**: Portfolio goes from $100K → $80K → $120K
- Peak: $100K
- Trough: $80K
- Max Drawdown: 20%

**Why It Matters**: Measures worst-case scenario. 20% drawdown requires 25% return to recover.

---

### Sharpe Ratio
**Definition**: Risk-adjusted return metric. Higher is better.

**Formula**:
```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
```

**Interpretation**:
- < 1.0: Poor risk-adjusted returns
- 1.0-2.0: Good
- 2.0-3.0: Very good
- > 3.0: Excellent (rare)

**Example**:
- Return: 20%
- Risk-free rate: 2%
- Volatility: 12%
- Sharpe = (20% - 2%) / 12% = 1.5 (good!)

---

### Value at Risk (VaR)
**Definition**: Maximum expected loss over a time period at a given confidence level.

**Example**:
"95% VaR of $10,000" means:
- 95% of the time, losses won't exceed $10,000
- 5% of the time, losses could be worse

**Types**:
- **Historical VaR**: Based on past returns
- **Parametric VaR**: Assumes normal distribution
- **Monte Carlo VaR**: Simulation-based

---

### Conditional Value at Risk (CVaR)
**Definition**: Expected loss in the worst X% of outcomes. Also called "Expected Shortfall."

**Formula**:
```
CVaR = Average of all losses worse than VaR
```

**Example**:
- VaR (95%) = $10,000 (95th percentile loss)
- CVaR (95%) = $15,000 (average of worst 5%)

**Why Better than VaR**: VaR only tells you the threshold. CVaR tells you how bad it gets beyond that.

**Use in Trading**:
- Limit position CVaR to 2% of portfolio
- Limit portfolio CVaR to 5% of portfolio

---

### Kelly Criterion
**Definition**: Formula for optimal position sizing that maximizes long-term growth.

**Formula**:
```
f* = (p × b - q) / b

Where:
- f* = fraction of capital to bet
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio (avg win / avg loss)
```

**Example**:
- Win rate: 60% (p = 0.6)
- Loss rate: 40% (q = 0.4)
- Win/loss ratio: 2.0 (wins are 2x losses)
- Kelly = (0.6 × 2 - 0.4) / 2 = 0.4 (bet 40% of capital)

**Fractional Kelly**: Use 25-50% of full Kelly for safety
- Full Kelly: 40%
- Half Kelly: 20%
- Quarter Kelly: 10% (recommended)

**Why Important**: Mathematically proven to maximize long-term compound growth.

---

## Technical Analysis Concepts

### RSI (Relative Strength Index)
**Definition**: Momentum oscillator measuring overbought/oversold conditions.

**Range**: 0-100

**Interpretation**:
- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)
- RSI crossing 50: Momentum shift

**Formula**:
```
RSI = 100 - (100 / (1 + RS))
Where RS = Average Gain / Average Loss
```

---

### MACD (Moving Average Convergence Divergence)
**Definition**: Trend-following momentum indicator.

**Components**:
- **MACD Line**: 12-period EMA - 26-period EMA
- **Signal Line**: 9-period EMA of MACD
- **Histogram**: MACD - Signal

**Signals**:
- MACD crosses above signal: Buy
- MACD crosses below signal: Sell
- Histogram growing: Strengthening trend

---

### Bollinger Bands
**Definition**: Volatility bands around a moving average.

**Components**:
- **Middle Band**: 20-period SMA
- **Upper Band**: Middle + (2 × standard deviation)
- **Lower Band**: Middle - (2 × standard deviation)

**Use**:
- Price at upper band: Overbought
- Price at lower band: Oversold
- Bands narrowing: Low volatility (breakout coming)
- Bands widening: High volatility

---

## Machine Learning & LLM Concepts

### Prompt Engineering
**Definition**: Crafting input text to get desired outputs from LLMs.

**Best Practices**:
1. **Be specific**: "Analyze AAPL financials" vs "Analyze this company"
2. **Provide context**: Include relevant data
3. **Use examples**: Show desired format
4. **Set constraints**: "Respond in JSON format"
5. **Iterate**: Test and refine prompts

**Example**:
```
BAD:
"Is AAPL good?"

GOOD:
"Analyze Apple Inc. (AAPL) fundamentals. Consider:
- P/E ratio: 28.5 (sector avg: 22)
- Revenue growth: 8% YoY
- Profit margin: 25%

Return JSON with:
1. Investment thesis (3-5 sentences)
2. Strengths (list)
3. Risks (list)
4. Score (0-100)"
```

---

### Temperature (LLM Parameter)
**Definition**: Controls randomness in LLM outputs.

**Range**: 0.0 - 2.0

**Values**:
- **0.0**: Deterministic, always same answer
- **0.3-0.7**: Balanced (recommended for analysis)
- **1.0-1.5**: Creative, varied outputs
- **2.0**: Very random, unpredictable

**Use in Trading**:
- Analysis: 0.3 (consistent, factual)
- Summarization: 0.5 (some variety)
- Brainstorming: 1.0 (creative ideas)

---

### Tokens
**Definition**: Chunks of text that LLMs process. ~4 characters per token.

**Importance**: APIs charge per token.

**Examples**:
- "Hello" = 1 token
- "Hello world!" = 3 tokens
- "Analyze AAPL" = 3 tokens

**Cost Optimization**:
- Claude: ~$3 per 1M tokens
- Groq: ~$0.0001 per 1M tokens
- Use Groq for simple tasks
- Use Claude for complex reasoning

**Token Limits**:
- Input + Output can't exceed model limit
- Claude Sonnet: 200K tokens
- Groq Mixtral: 32K tokens

---

### Overfitting
**Definition**: Model performs well on training data but poorly on new data.

**Example**:
```
Strategy backtested 2020-2023:
- Sharpe Ratio: 3.5 (amazing!)

Same strategy tested 2024:
- Sharpe Ratio: 0.5 (terrible!)

Why? Overfitted to 2020-2023 data.
```

**Prevention**:
1. **Walk-Forward Analysis**: Rolling train/test
2. **Out-of-Sample Testing**: Reserve data for validation
3. **Parameter Simplicity**: Fewer parameters = less overfitting
4. **Cross-Validation**: Test on multiple time periods

---

## System Design Concepts

### REST API
**Definition**: HTTP-based API for client-server communication.

**HTTP Methods**:
- **GET**: Retrieve data (idempotent)
- **POST**: Create data
- **PUT**: Update data
- **DELETE**: Delete data

**Status Codes**:
- **200**: OK
- **201**: Created
- **400**: Bad Request
- **401**: Unauthorized
- **404**: Not Found
- **500**: Server Error

**Example**:
```
GET /signals/AAPL
→ Returns: {"symbol": "AAPL", "signal": "BUY", "confidence": 0.85}

POST /trades
Body: {"symbol": "AAPL", "side": "buy", "quantity": 10}
→ Returns: {"order_id": "123", "status": "filled"}
```

---

### Async/Await
**Definition**: Non-blocking concurrent programming in Python.

**Why Use**:
- Handle multiple API calls simultaneously
- Don't block while waiting for I/O
- Better performance for I/O-bound tasks

**Example**:
```python
# Synchronous (slow - 9 seconds total)
price1 = get_price("AAPL")  # 3 seconds
price2 = get_price("MSFT")  # 3 seconds
price3 = get_price("GOOGL")  # 3 seconds

# Async (fast - 3 seconds total, runs concurrently)
price1 = await get_price_async("AAPL")  # \
price2 = await get_price_async("MSFT")  #  } All run at once!
price3 = await get_price_async("GOOGL")  # /
```

---

### Microservices
**Definition**: Architecture where application is split into small, independent services.

**Traditional Monolith**:
```
[Single App]
├── Trading Logic
├── Data Fetching
├── Risk Management
└── Order Execution
```

**Microservices**:
```
[Signal Service] ←→ [Data Service]
       ↓                    ↓
[Risk Service] ←→ [Execution Service]
```

**Benefits**:
- Independent scaling
- Easier testing
- Technology flexibility
- Fault isolation

**Our Platform (Week 7)**:
- API Service (FastAPI)
- Data Service (Kafka)
- Risk Service (Separate process)
- Database (PostgreSQL)
- Cache (Redis)
- Monitoring (Prometheus + Grafana)

---

## Statistical Concepts

### Monte Carlo Simulation
**Definition**: Running thousands of random scenarios to understand probability distributions.

**In Trading**:
1. Take historical returns
2. Randomly resample (bootstrap)
3. Run strategy 1,000+ times
4. Analyze distribution of outcomes

**Example**:
```
Run strategy 1,000 times:
- Mean Sharpe: 1.8
- 95% CI: [1.2, 2.4]
- Prob(Sharpe > 1.0): 92%

Conclusion: 92% confident strategy has positive risk-adjusted returns
```

**Use**: Distinguish luck from skill.

---

### P-Value
**Definition**: Probability that results occurred by chance.

**Interpretation**:
- p < 0.05: Statistically significant (< 5% chance of luck)
- p < 0.01: Highly significant (< 1% chance)
- p > 0.05: Not significant (could be random)

**In Trading**:
```
Backtest returns: 15% annual

H0 (Null Hypothesis): Returns = 0 (no edge)
p-value = 0.02

Conclusion: Reject H0. Only 2% chance returns are random.
```

**Warning**: Statistical significance ≠ practical significance. Need both!

---

### Correlation
**Definition**: Measure of linear relationship between two variables.

**Range**: -1 to +1

**Values**:
- **+1**: Perfect positive correlation
- **0**: No correlation
- **-1**: Perfect negative correlation

**In Trading**:
- Portfolio diversification: Want low/negative correlations
- Pairs trading: Trade correlated stocks
- Risk management: Avoid highly correlated positions

**Example**:
```
AAPL vs SPY correlation: 0.75 (high)
AAPL vs GLD correlation: -0.15 (low, good for diversification)
```

---

## Regulatory Concepts

### SR 11-7
**Definition**: Federal Reserve guidance on Model Risk Management.

**Requirements**:
1. **Model Inventory**: Track all models
2. **Validation**: Independent validation before use
3. **Documentation**: Complete model documentation
4. **Monitoring**: Ongoing performance tracking
5. **Review**: Periodic model review

**In Our Platform (Week 6)**:
- Model registration system
- Validation framework
- Performance monitoring
- Audit trail

---

### MiFID II
**Definition**: EU regulation requiring "best execution" for trades.

**Requirements**:
1. **Execution Quality**: Prove you got best price
2. **Transaction Reporting**: Detailed trade records
3. **Transparency**: Disclose execution venues
4. **Monitoring**: Ongoing quality assessment

**Implementation (Week 8)**:
- Transaction Cost Analysis (TCA)
- Execution quality metrics
- Venue comparison
- Regulatory reports

---

## Performance Metrics

### Win Rate
**Definition**: Percentage of profitable trades.

**Formula**:
```
Win Rate = (Winning Trades / Total Trades) × 100%
```

**Example**:
```
Total Trades: 100
Wins: 60
Losses: 40
Win Rate = 60%
```

**Note**: High win rate doesn't mean profitable! A 90% win rate with tiny wins and huge losses = bad.

---

### Profit Factor
**Definition**: Ratio of gross profit to gross loss.

**Formula**:
```
Profit Factor = Total Winning $ / Total Losing $
```

**Interpretation**:
- PF < 1.0: Losing strategy
- PF = 1.0: Breakeven
- PF = 1.5: Decent (make $1.50 for every $1 lost)
- PF = 2.0+: Good
- PF = 3.0+: Excellent

**Example**:
```
Wins: $50,000
Losses: $20,000
Profit Factor = 50,000 / 20,000 = 2.5 (good!)
```

---

## Data Concepts

### OHLC
**Definition**: Open, High, Low, Close - standard price data format.

**Example**:
```
AAPL 2024-11-13:
O: $150.00 (opening price)
H: $152.50 (highest price)
L: $149.00 (lowest price)
C: $151.25 (closing price)
V: 50M shares (volume)
```

**Use**: Candlestick charts, technical analysis, backtesting.

**Validation**:
- H ≥ O, C, L (high is highest)
- L ≤ O, C, H (low is lowest)
- All prices > 0

---

### Bid-Ask Spread
**Definition**: Difference between highest buy price and lowest sell price.

**Example**:
```
AAPL:
Bid: $150.45 (buyers want to pay)
Ask: $150.55 (sellers want to receive)
Spread: $0.10 (0.067%)
```

**Why It Matters**:
- Liquidity indicator (narrow = liquid)
- Trading cost (you pay the spread)
- Market maker profit

**Typical Spreads**:
- Large caps: 0.01-0.05% (tight)
- Small caps: 0.1-0.5% (wider)
- Illiquid: 1%+ (very wide)

---

## Software Engineering Concepts

### Test-Driven Development (TDD)
**Definition**: Write tests before writing code.

**Process**:
1. **Red**: Write failing test
2. **Green**: Write minimal code to pass
3. **Refactor**: Improve code
4. **Repeat**

**Benefits**:
- Better design
- Fewer bugs
- Confidence to refactor
- Living documentation

---

### Code Coverage
**Definition**: Percentage of code executed by tests.

**Types**:
- **Line Coverage**: % of lines run
- **Branch Coverage**: % of if/else paths
- **Function Coverage**: % of functions called

**Targets**:
- Critical code: 100%
- Business logic: 90%+
- Overall: 80%+
- UI code: 60%+ (harder to test)

---

### CI/CD
**Definition**: Continuous Integration / Continuous Deployment

**CI** (Integration):
- Automatically run tests on every commit
- Catch bugs early
- Ensure code quality

**CD** (Deployment):
- Automatically deploy passing code
- Faster releases
- Less manual work

**Our Pipeline (Week 7)**:
```
Code Push → Tests Run → If Pass → Deploy to Staging → Manual Approve → Deploy to Prod
```

---

## Glossary Quick Reference

**Essential Terms to Master Each Week**:

- **Week 1**: API, Endpoint, Model, Validation, Async, Paper Trading
- **Week 2**: Prompt, Token, Temperature, LLM, Cache, JSON
- **Week 3**: Risk, Position Size, VaR, Sharpe, Drawdown, Portfolio
- **Week 4**: Indicator, RSI, MACD, Regime, Backtest, Overfitting
- **Week 5**: Order Book, Spread, Slippage, VWAP, Liquidity, Microstructure
- **Week 6**: CVaR, Kelly, Monte Carlo, WFA, Stress Test, Validation
- **Week 7**: Test, Coverage, Mock, CI/CD, Container, Monitoring
- **Week 8**: TCA, Regime, Reconciliation, Compliance, Optimization

---

**Need deeper explanation of any concept? Check the weekly notes or ask in forums!**

---

*This glossary will be your reference throughout the 8 weeks. Bookmark it!*
