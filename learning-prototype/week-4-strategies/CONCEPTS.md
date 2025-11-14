# Trading Strategy Concepts

Deep dive into the theory and practice of quantitative trading strategies.

## Table of Contents

1. [Pairs Trading](#pairs-trading)
2. [Regime Momentum](#regime-momentum)
3. [Sentiment Trading](#sentiment-trading)
4. [Market Making](#market-making)
5. [Position Sizing](#position-sizing)
6. [Risk Management](#risk-management)

---

## Pairs Trading

### Overview

Pairs trading is a **market-neutral** statistical arbitrage strategy that trades two cointegrated securities.

**Key principle**: If two assets move together over the long term, temporary divergences create trading opportunities.

### Cointegration Explained

**Cointegration** ≠ **Correlation**

- **Correlation**: Measures short-term linear relationship between **returns**
- **Cointegration**: Tests whether two **price series** have a stable long-term relationship

**Example**:
```
Two random walks can be highly correlated but NOT cointegrated
Two stocks (e.g., KO and PEP) can be cointegrated
```

### Engle-Granger Test

The most common cointegration test:

1. Run regression: `price1 = β * price2 + ε`
2. Extract residuals (the "spread"): `spread = price1 - β * price2`
3. Test if residuals are stationary (ADF test)
4. If p-value < 0.05: **Cointegrated!**

**Python**:
```python
from statsmodels.tsa.stattools import coint
score, pvalue, _ = coint(price1, price2)
if pvalue < 0.05:
    print("Cointegrated!")
```

### Hedge Ratio

The β coefficient from regression is the **hedge ratio**:
- Tells you how many units of asset2 to trade for each unit of asset1
- Minimizes variance of the spread
- Must be recalculated periodically

**Example**:
```
If hedge_ratio = 1.5:
- To be market-neutral: Buy 100 shares of asset1, sell 150 shares of asset2
```

### Z-Score Trading

Once you have the spread, convert to z-score:

```python
spread = price1 - hedge_ratio * price2
z_score = (spread - spread.mean()) / spread.std()
```

**Trading rules**:
```
Entry signals:
- z > +2.0: Short the spread (sell asset1, buy asset2)
- z < -2.0: Long the spread (buy asset1, sell asset2)

Exit signals:
- z crosses 0: Close position (mean reversion)
- z reaches ±3: Stop loss (relationship breakdown)
```

### Half-Life of Mean Reversion

How long does it take for the spread to revert?

**Calculate**:
```python
from statsmodels.regression.linear_model import OLS

# Ornstein-Uhlenbeck process: dS = θ(μ - S)dt + σdW
# Half-life = -ln(2) / θ

spread_lag = spread.shift(1)
spread_diff = spread - spread_lag
model = OLS(spread_diff.dropna(), spread_lag.dropna())
theta = model.fit().params[0]
half_life = -np.log(2) / theta
```

**Interpretation**:
- Half-life = 5 days: Wait ~5 days for spread to halve
- Half-life < 1 day: Good for intraday pairs trading
- Half-life > 30 days: Too slow, avoid trading

### When Pairs Trading Works

✅ **Best in**:
- Low volatility markets
- Range-bound markets
- Stable correlations
- High liquidity (tight spreads)

❌ **Avoid in**:
- High volatility
- Trending markets
- Correlation breakdown
- Illiquid assets

### Common Pairs

**Sector pairs** (more stable):
- KO vs PEP (beverages)
- AAPL vs MSFT (tech)
- XOM vs CVX (energy)
- GLD vs SLV (precious metals)

**ETF pairs**:
- SPY vs DIA
- QQQ vs IWM
- EWU vs EWG (UK vs Germany)

---

## Regime Momentum

### Overview

Momentum strategies exploit the tendency of assets to continue moving in the same direction. **Regime-adaptive** momentum adjusts behavior based on market conditions.

### Why Momentum Works

**Behavioral explanations**:
- **Underreaction**: Investors are slow to react to news
- **Herding**: Investors follow the crowd
- **Disposition effect**: Investors hold losers too long

**Result**: Trends persist longer than they should

### Volatility Regimes

Markets cycle through volatility regimes:

1. **Low Volatility** (VIX < 15):
   - Stable, predictable
   - Trade more aggressively
   - Tighter stops

2. **Normal Volatility** (VIX 15-25):
   - Standard parameters
   - Normal position sizing

3. **High Volatility** (VIX > 25):
   - Unpredictable, risky
   - Reduce position size
   - Wider stops

**Detection**:
```python
volatility = returns.rolling(20).std() * np.sqrt(252)
avg_vol = volatility.rolling(60).mean()

if volatility < avg_vol * 0.7:
    regime = "LOW"
elif volatility < avg_vol * 1.3:
    regime = "NORMAL"
else:
    regime = "HIGH"
```

### Trend Regimes

**Three trend states**:

1. **Bull Market** (uptrend):
   - Long bias
   - Ride winners longer
   - Tighter stop losses

2. **Bear Market** (downtrend):
   - Short bias or stay out
   - Quick exits
   - Defensive

3. **Neutral Market** (sideways):
   - Mean reversion
   - Range trading
   - No strong momentum

**Detection (SMA crossover)**:
```python
sma_fast = prices.rolling(20).mean()
sma_slow = prices.rolling(50).mean()

if sma_fast > sma_slow * 1.02:
    trend_regime = "BULL"
elif sma_fast < sma_slow * 0.98:
    trend_regime = "BEAR"
else:
    trend_regime = "NEUTRAL"
```

### Momentum Indicators

**Rate of Change (ROC)**:
```python
roc = (price / price.shift(n) - 1) * 100
# Buy if ROC > threshold (e.g., 5%)
```

**RSI (Relative Strength Index)**:
```python
# RSI = 100 - (100 / (1 + RS))
# RS = Average Gain / Average Loss
# Buy if RSI < 30 (oversold)
# Sell if RSI > 70 (overbought)
```

**MACD (Moving Average Convergence Divergence)**:
```python
ema_fast = prices.ewm(span=12).mean()
ema_slow = prices.ewm(span=26).mean()
macd = ema_fast - ema_slow
signal = macd.ewm(span=9).mean()
# Buy when MACD crosses above signal
```

### Adaptive Position Sizing

Adjust position size based on regime:

```python
base_size = 100_000  # $100k base position

if vol_regime == "LOW" and trend_regime == "BULL":
    size = base_size * 1.5  # Aggressive
elif vol_regime == "HIGH":
    size = base_size * 0.5  # Defensive
else:
    size = base_size  # Normal
```

### When Momentum Works

✅ **Best in**:
- Trending markets
- After breakouts
- Following major news
- Low correlation environments

❌ **Avoid in**:
- Range-bound markets
- High volatility
- Frequent reversals

---

## Sentiment Trading

### Overview

Sentiment trading uses natural language processing (NLP) to extract trading signals from news, social media, and other text sources.

### Sentiment Sources

**News**:
- Financial news (Bloomberg, Reuters)
- Company announcements
- Earnings releases
- Economic data

**Social Media**:
- Twitter/X (real-time)
- Reddit (WallStreetBets, etc.)
- StockTwits
- Seeking Alpha comments

**Alternative**:
- SEC filings (10-K, 8-K)
- Earnings call transcripts
- Analyst reports

### Sentiment Extraction

**Lexicon-based** (simple):
```python
positive_words = ['bullish', 'growth', 'profit', 'beat']
negative_words = ['bearish', 'loss', 'miss', 'decline']

score = (positive_count - negative_count) / total_words
```

**ML-based** (advanced):
```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis",
                     model="ProsusAI/finbert")
result = sentiment("Company reports strong earnings")
# {'label': 'positive', 'score': 0.95}
```

### Sentiment Aggregation

Combine multiple sources:

```python
def aggregate_sentiment(articles):
    scores = [a.sentiment_score for a in articles]

    # Weighted by recency (newer = more important)
    now = datetime.now()
    weights = [1.0 / (1 + (now - a.published_at).days)
               for a in articles]

    # Weighted average
    agg_sentiment = np.average(scores, weights=weights)

    return agg_sentiment
```

### Sentiment Signals

**Simple threshold**:
```python
if sentiment > 0.5:
    signal = "BUY"
elif sentiment < -0.5:
    signal = "SELL"
else:
    signal = "HOLD"
```

**With change (momentum)**:
```python
sentiment_change = sentiment - sentiment_prev

if sentiment > 0.3 and sentiment_change > 0.2:
    signal = "STRONG_BUY"  # Improving sentiment
elif sentiment < -0.3 and sentiment_change < -0.2:
    signal = "STRONG_SELL"  # Deteriorating sentiment
```

### Technical Confirmation

Don't trade on sentiment alone! Confirm with technicals:

```python
def check_technical_confirmation(sentiment, price, volume):
    if sentiment > 0:
        # For bullish sentiment, need:
        # 1. Price breaking resistance
        # 2. Volume surge
        if price > resistance and volume > avg_volume * 1.5:
            return True
    return False
```

### Intraday Patterns

Sentiment is most effective **intraday**:

- **9:30-10:00 AM**: Morning news reaction
- **2:00-3:00 PM**: Afternoon repositioning
- **3:50-4:00 PM**: Close-of-day positioning

**Exit**: Always close by end of day (sentiment decays quickly)

### When Sentiment Trading Works

✅ **Best in**:
- Earnings season
- Major news events
- High news flow stocks
- Large cap liquid stocks

❌ **Avoid in**:
- Low news flow periods
- Illiquid stocks
- Overnight positions

---

## Market Making

### Overview

Market making provides liquidity by simultaneously quoting buy (bid) and sell (ask) prices, profiting from the bid-ask spread.

### Bid-Ask Spread

**Example**:
```
Bid: $99.95 (willing to buy)
Ask: $100.05 (willing to sell)
Spread: $0.10 (1 cent = 10 bps)
```

**Profit**:
- Buy at bid: $99.95
- Sell at ask: $100.05
- Profit: $0.10 per share

### Fair Price

Market makers quote around a "fair price":

**Methods**:

1. **Mid-price** (simplest):
```python
fair = (bid + ask) / 2
```

2. **VWAP** (volume-weighted):
```python
fair = (price * volume).sum() / volume.sum()
```

3. **Microprice** (order book weighted):
```python
# Weight by opposite side size
fair = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
```

### Quote Pricing

Set quotes around fair price:

```python
spread_bps = 10  # 10 basis points
spread = fair * spread_bps / 10000

bid_price = fair - spread / 2
ask_price = fair + spread / 2
```

### Inventory Management

**Problem**: Getting stuck with a large long or short position

**Solution**: Skew quotes based on inventory

```python
# If long (positive inventory):
# - Widen ask (easier to buy)
# - Tighten bid (harder to sell to you)

inventory_pct = current_inventory / max_inventory
skew = inventory_pct * spread * 0.5

bid_price = fair - spread / 2 - skew
ask_price = fair + spread / 2 - skew
```

**Example**:
```
No inventory:
- Bid: $99.95, Ask: $100.05

After buying 500 shares (75% of max):
- Bid: $99.90 (wider, less attractive)
- Ask: $100.00 (tighter, more attractive)
Result: Encourages selling to reduce inventory
```

### Order Book Imbalance

Adjust quotes based on supply/demand:

```python
bid_volume = sum(size for price, size in bid_levels)
ask_volume = sum(size for price, size in ask_levels)

imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

# If more bids (imbalance > 0):
# - Price likely to rise
# - Widen bid (less aggressive buying)
# - Tighten ask (more aggressive selling)
```

### Adverse Selection

**Risk**: Trading with informed traders who know more than you

**Example**:
```
You quote:
- Bid: $100.00
- Ask: $100.10

Informed trader knows stock is about to drop to $99.00
- They hit your bid at $100.00
- You lose $1.00 per share
```

**Mitigation**:
- Widen spreads during news
- Monitor fill rates (too high = adverse selection)
- Use order book imbalance
- Cancel quotes quickly when market moves

### Position Limits

Always enforce strict inventory limits:

```python
max_inventory = 1000  # shares
current_inventory = 800

# Don't allow buy if would exceed limit
if side == "buy" and current_inventory + qty > max_inventory:
    return False  # Don't post bid

# Don't allow sell if would exceed limit
if side == "sell" and current_inventory - qty < -max_inventory:
    return False  # Don't post ask
```

### When Market Making Works

✅ **Best in**:
- Highly liquid markets
- Stable price action
- Low adverse selection
- Tight spreads already exist

❌ **Avoid in**:
- Illiquid markets
- High volatility
- News events
- Wide existing spreads

---

## Position Sizing

### Equal Weight

**Simplest approach**:
```python
capital = 100_000
num_positions = 5
position_size = capital / num_positions  # $20k each
```

**Pros**: Simple, diversified
**Cons**: Ignores risk differences

### Volatility-Weighted

**Better approach** - allocate inversely to volatility:

```python
target_vol = 0.15  # 15% annual volatility
asset_vol = returns.std() * np.sqrt(252)

# Scale position to achieve target volatility
position_size = (capital * target_vol) / asset_vol
```

**Result**: High-vol assets get smaller positions, low-vol assets get larger

### Kelly Criterion

**Optimal sizing** for maximizing log wealth:

```
f* = (p * b - q) / b

where:
- f* = fraction of capital to risk
- p = probability of win
- q = 1 - p
- b = ratio of win to loss (avg_win / avg_loss)
```

**Example**:
```python
win_rate = 0.55  # 55% winners
avg_win = 0.02   # 2% average win
avg_loss = 0.01  # 1% average loss

b = avg_win / avg_loss  # = 2
kelly = (win_rate * b - (1 - win_rate)) / b
# kelly = (0.55 * 2 - 0.45) / 2 = 0.325 = 32.5%

# Use HALF Kelly (less risky)
position_size = capital * kelly * 0.5  # 16.25%
```

**Warning**: Full Kelly is very aggressive! Use 0.25-0.5x Kelly in practice.

### Risk Parity

Allocate so each position contributes equally to portfolio risk:

```python
# Each asset contributes same to portfolio variance
weights = 1 / volatilities
weights = weights / weights.sum()  # Normalize
```

---

## Risk Management

### Stop Losses

**Fixed percentage**:
```python
entry_price = 100
stop_loss = entry_price * 0.98  # 2% stop
```

**ATR-based** (adapts to volatility):
```python
atr = ta.ATR(high, low, close, timeperiod=14)
stop_loss = entry_price - 2 * atr  # 2 ATR stop
```

### Take Profits

**Fixed target**:
```python
take_profit = entry_price * 1.06  # 6% target (3:1 reward:risk)
```

**Trailing stop**:
```python
# Lock in profits as price rises
trailing_stop = max(entry_price * 0.98,  # Initial stop
                    current_high * 0.98)  # Trailing 2%
```

### Position Limits

**Per-position**:
```python
MAX_POSITION_SIZE = 0.20  # Max 20% of capital per position
```

**Per-sector**:
```python
MAX_SECTOR_EXPOSURE = 0.40  # Max 40% in any sector
```

**Total leverage**:
```python
MAX_LEVERAGE = 1.5  # Max 150% gross exposure
```

### Drawdown Limits

**Reduce risk after losses**:

```python
if current_drawdown > 0.10:  # -10% drawdown
    position_size *= 0.5  # Cut position sizes in half

if current_drawdown > 0.15:  # -15% drawdown
    stop_trading = True  # Stop trading, review strategy
```

---

## Additional Resources

### Books
- "Quantitative Trading" by Ernest Chan
- "Algorithmic Trading" by Ernest Chan
- "Advances in Financial Machine Learning" by Marcos Lopez de Prado

### Papers
- ["Pairs Trading: Performance of a Relative-Value Arbitrage Rule"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615)
- ["Time Series Momentum"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463)
- ["The Predictive Power of the Sentiment of news"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2193128)

### Courses
- Quantopian Lectures (free)
- Coursera: Machine Learning for Trading
- QuantStart: Algorithmic Trading Course
