# Week 4: Trading Strategies

Implement 4 production-grade trading strategies from scratch!

## Learning Objectives

By the end of this week, you will:

✅ Understand statistical arbitrage and pairs trading
✅ Implement cointegration testing and mean reversion signals
✅ Build regime-adaptive momentum strategies
✅ Create sentiment-driven intraday trading systems
✅ Implement market making with inventory management
✅ Master position sizing and risk management
✅ Backtest all strategies with realistic costs

## Why Trading Strategies Matter

> "The market is a device for transferring money from the impatient to the patient." - Warren Buffett

Trading strategies are:
- The core of your edge in the market
- How you generate alpha (excess returns)
- Your systematic approach to trading decisions
- The foundation for scaling your trading

**Common mistake**: Following strategies without understanding WHY they work
**Solution**: Understand the economic rationale behind each strategy

## Prerequisites

- Python basics (Week 1)
- Backtesting fundamentals (Week 3)
- Basic statistics (correlation, cointegration)
- Understanding of market microstructure

## Folder Structure

```
week-4-strategies/
├── README.md (you are here)
├── CONCEPTS.md (strategy concepts explained)
├── starter-code/
│   ├── pairs_trading.py      (25 TODOs)
│   ├── regime_momentum.py    (25 TODOs)
│   ├── sentiment_intraday.py (25 TODOs)
│   └── market_making.py      (30 TODOs)
├── exercises/
│   ├── exercise_1_pairs.py
│   ├── exercise_2_momentum.py
│   ├── exercise_3_sentiment.py
│   └── exercise_4_market_making.py
├── solutions/
│   ├── pairs_trading_complete.py
│   ├── regime_momentum_complete.py
│   ├── sentiment_intraday_complete.py
│   └── market_making_complete.py
└── tests/
    ├── test_pairs.py
    ├── test_momentum.py
    ├── test_sentiment.py
    └── test_market_making.py
```

## Learning Path

### Day 1: Pairs Trading (4 hours)

**Morning**: Statistical arbitrage fundamentals
- Read `CONCEPTS.md` - Pairs Trading section
- Understand cointegration vs correlation
- Learn about mean reversion
- Study hedge ratios

**Afternoon**: Implement pairs trading
- Complete `starter-code/pairs_trading.py`
- Fill in 25 TODOs
- Run tests: `pytest tests/test_pairs.py`

**Key concepts**:
- Cointegration testing (Engle-Granger, Johansen)
- Z-score calculation for entry/exit
- Hedge ratio estimation
- Half-life of mean reversion

**Resources**:
- [Cointegration Explained](https://www.quantstart.com/articles/Cointegration-Augmented-Dickey-Fuller-Test-in-R/)
- [Pairs Trading Tutorial](https://www.quantopian.com/lectures/introduction-to-pairs-trading)

### Day 2: Regime Momentum (3-4 hours)

**Morning**: Regime detection
- Read `CONCEPTS.md` - Regime Detection section
- Understand volatility regimes (low/med/high)
- Learn trend regimes (bull/bear/neutral)
- Study adaptive position sizing

**Afternoon**: Implement regime momentum
- Complete `starter-code/regime_momentum.py`
- Fill in 25 TODOs
- Run tests: `pytest tests/test_momentum.py`

**Key insight**: Momentum works differently in different market regimes!

### Day 3: Sentiment Intraday (3-4 hours)

**Morning**: Sentiment analysis
- Read `CONCEPTS.md` - Sentiment Trading section
- Understand sentiment sources (news, social media)
- Learn sentiment aggregation
- Study intraday patterns

**Afternoon**: Implement sentiment strategy
- Complete `starter-code/sentiment_intraday.py`
- Fill in 25 TODOs
- Run tests: `pytest tests/test_sentiment.py`

**Key concepts**:
- Multi-source sentiment aggregation
- Technical confirmation
- Intraday position management
- Market hours enforcement

### Day 4: Market Making (4-5 hours)

**Morning**: Market microstructure
- Read `CONCEPTS.md` - Market Making section
- Understand bid-ask spread
- Learn about inventory risk
- Study order book dynamics

**Afternoon**: Implement market making
- Complete `starter-code/market_making.py`
- Fill in 30 TODOs (most complex strategy!)
- Run tests: `pytest tests/test_market_making.py`

**Key concepts**:
- Quote pricing with spread
- Inventory management and skewing
- Order book imbalance
- Adverse selection risk

### Day 5: Integration & Backtesting (3-4 hours)

**Morning**: Backtest all strategies
```bash
# Backtest each strategy with realistic costs
python exercises/exercise_1_pairs.py
python exercises/exercise_2_momentum.py
python exercises/exercise_3_sentiment.py
python exercises/exercise_4_market_making.py
```

**Afternoon**: Compare strategies
- Analyze performance metrics (Sharpe, drawdown)
- Understand when each strategy works best
- Learn about strategy diversification
- Create a multi-strategy portfolio

## Strategy Overview

### 1. Pairs Trading (Statistical Arbitrage)

**What it is**: Trade two cointegrated assets, betting on mean reversion

**When it works**:
- Low volatility markets
- Stable correlations
- Range-bound markets

**Example**:
```
Coca-Cola (KO) and PepsiCo (PEP)
- Cointegrated: Move together long-term
- Short-term deviations: Trading opportunity
- When spread widens: Buy undervalued, sell overvalued
- When spread narrows: Profit!
```

**Expected Sharpe**: 1.5-2.5

### 2. Regime Momentum

**What it is**: Adaptive momentum that changes behavior based on market regime

**When it works**:
- Trending markets (bull/bear)
- After regime changes
- Low correlation environments

**Example**:
```
High volatility regime detected:
- Reduce position size (more risk)
- Widen stop losses
- Take profits earlier

Low volatility regime:
- Increase position size (less risk)
- Tighter stop losses
- Let winners run longer
```

**Expected Sharpe**: 1.0-2.0

### 3. Sentiment Intraday

**What it is**: Trade based on news sentiment with same-day entry/exit

**When it works**:
- High news flow days
- Earnings season
- Major economic releases

**Example**:
```
9:35 AM: Positive news sentiment for AAPL
- Aggregate sentiment: +0.75
- Technical confirmation: Price breaking resistance
- Enter long at $150.00
3:55 PM: Exit at close
- Price: $151.50
- Profit: +1.0%
```

**Expected Sharpe**: 0.8-1.5

### 4. Market Making

**What it is**: Provide liquidity by quoting both bid and ask, capturing spread

**When it works**:
- Liquid markets
- Stable price action
- Low adverse selection

**Example**:
```
Fair price: $100.00
Spread: 10 bps ($0.10)

Post quotes:
- Bid: $99.95 (size: 100)
- Ask: $100.05 (size: 100)

Both sides fill:
- Buy at $99.95
- Sell at $100.05
- Profit: $0.10 per share = $10 total
```

**Expected Sharpe**: 2.0-3.0 (but requires high frequency)

## Key Concepts

### 1. Cointegration vs Correlation

**Correlation** (DON'T use for pairs trading):
```python
# Correlation measures short-term relationship
correlation = returns1.corr(returns2)
# Problem: Can spuriously correlate without long-term relationship
```

**Cointegration** (DO use for pairs trading):
```python
# Cointegration tests for long-term equilibrium
from statsmodels.tsa.stattools import coint
score, pvalue, _ = coint(price1, price2)
# If pvalue < 0.05: Cointegrated! Can pairs trade
```

### 2. Z-Score for Mean Reversion

```python
# Calculate spread
spread = price1 - hedge_ratio * price2

# Calculate z-score
z_score = (spread - spread.mean()) / spread.std()

# Trading signals
if z_score > 2:  # Spread too wide
    # Short spread: sell asset1, buy asset2
elif z_score < -2:  # Spread too narrow
    # Long spread: buy asset1, sell asset2
```

### 3. Regime Detection

**Volatility Regime**:
```python
vol = returns.rolling(20).std() * np.sqrt(252)
if vol < 0.15:
    regime = "LOW_VOL"  # Trade more aggressively
elif vol < 0.25:
    regime = "NORMAL"
else:
    regime = "HIGH_VOL"  # Trade defensively
```

**Trend Regime**:
```python
sma_20 = prices.rolling(20).mean()
sma_50 = prices.rolling(50).mean()

if sma_20 > sma_50 * 1.02:
    regime = "BULL"  # Long bias
elif sma_20 < sma_50 * 0.98:
    regime = "BEAR"  # Short bias
else:
    regime = "NEUTRAL"  # Mean reversion
```

### 4. Position Sizing

**Equal Weight** (simplest):
```python
position_size = capital / num_positions
```

**Volatility-Weighted** (better):
```python
# Allocate inversely to volatility
target_vol = 0.15
position_vol = returns.std() * np.sqrt(252)
position_size = (capital * target_vol) / position_vol
```

**Kelly Criterion** (optimal but risky):
```python
# f* = (p*b - q) / b
# where p = win_rate, q = 1-p, b = avg_win/avg_loss
win_rate = 0.55
avg_win = 0.02
avg_loss = 0.01
kelly_fraction = (win_rate * avg_win - (1-win_rate)) / avg_win
position_size = capital * kelly_fraction * 0.5  # Use half-Kelly
```

## Common Pitfalls

### Pitfall #1: Ignoring Economic Rationale
**Problem**: Trading strategies that work in backtest but have no logical reason to work
**Solution**: Always ask "WHY should this strategy work?" If you can't answer, don't trade it

### Pitfall #2: Over-Fitting Parameters
**Problem**: Optimizing parameters until backtest looks perfect
**Solution**: Use walk-forward analysis, limit parameters, demand robust performance

### Pitfall #3: Ignoring Transaction Costs
**Problem**: High-frequency strategies that are unprofitable after costs
**Solution**: Always model realistic costs (see Week 3)

### Pitfall #4: Not Adapting to Regimes
**Problem**: Using same strategy parameters in all market conditions
**Solution**: Detect regimes and adapt position sizing, stop losses, etc.

### Pitfall #5: Correlation Breakdown
**Problem**: Pairs trading when correlation breaks down
**Solution**: Monitor cointegration continuously, exit if relationship breaks

## Success Criteria

You've mastered Week 4 when you can:

✅ Implement all 4 strategies from scratch
✅ Explain the economic rationale for each strategy
✅ Test cointegration and calculate z-scores
✅ Detect volatility and trend regimes
✅ Aggregate multi-source sentiment
✅ Calculate market making quotes with inventory skew
✅ Backtest all strategies with realistic costs
✅ Compare strategies and understand when each works
✅ Pass all integration tests

## Testing Your Knowledge

```bash
# Run all strategy tests
pytest week-4-strategies/tests/ -v

# Expected output:
# ✓ test_cointegration_test
# ✓ test_hedge_ratio_calculation
# ✓ test_zscore_signals
# ✓ test_regime_detection
# ✓ test_volatility_regime
# ✓ test_trend_regime
# ✓ test_sentiment_aggregation
# ✓ test_market_making_quotes
# ✓ test_inventory_management
#
# 20+ passed in 3.45s
```

## Performance Expectations

Realistic expectations for each strategy:

| Strategy | Sharpe Ratio | Max Drawdown | Win Rate | Best Market |
|----------|-------------|--------------|----------|-------------|
| Pairs Trading | 1.5-2.5 | -8% to -15% | 55-65% | Low vol, range-bound |
| Regime Momentum | 1.0-2.0 | -15% to -25% | 40-50% | Trending markets |
| Sentiment Intraday | 0.8-1.5 | -10% to -20% | 50-60% | High news flow |
| Market Making | 2.0-3.0 | -5% to -10% | 50-55% | Liquid, stable |

**Multi-strategy portfolio**:
- Sharpe: 2.0-2.5 (diversification benefit!)
- Max Drawdown: -12% to -18%
- More stable returns

## Next Steps

After completing Week 4:
- **Week 5**: Deploy strategies with production infrastructure
- **Build your own**: Create a custom strategy
- **Paper trade**: Test in real-time with Alpaca paper trading
- **Read**: "Algorithmic Trading" by Ernest Chan

## Resources

### Articles
- [Pairs Trading in the 21st Century](https://www.quantopian.com/posts/pairs-trading-in-the-21st-century)
- [Momentum Strategies](https://www.quantstart.com/articles/Momentum-Strategies/)
- [Sentiment Analysis for Trading](https://www.investopedia.com/articles/active-trading/041015/how-sentiment-analysis-used-finance.asp)

### Books
- "Quantitative Trading" by Ernest Chan (Chapters 3-5)
- "Algorithmic Trading" by Ernest Chan
- "Statistical Arbitrage" by Andrew Pole

### Papers
- ["Pairs Trading: Performance of a Relative-Value Arbitrage Rule"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615)
- ["Time Series Momentum"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463)

## Questions?

**Q: Which strategy is best for beginners?**
A: Start with pairs trading - it's market-neutral and easier to understand.

**Q: Can I combine multiple strategies?**
A: Yes! Multi-strategy portfolios have better risk-adjusted returns due to diversification.

**Q: How much capital do I need?**
A: For paper trading (learning): $0. For live trading: Start with $10k-25k minimum.

**Q: What if my backtest doesn't match the expected Sharpe ratio?**
A: That's normal! Focus on understanding WHY the strategy works, not hitting specific numbers.

**Q: Should I trade all 4 strategies at once?**
A: Start with one, master it, then add others gradually.

**Happy strategy building! 📈**
