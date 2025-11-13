# High-Frequency Trading Techniques in the Platform

This document explains the HFT-inspired techniques implemented in the trading platform and how they improve execution quality.

## Overview

While we use LLMs (which are relatively slow), we've incorporated many techniques from high-frequency trading to optimize execution:

1. **Market Microstructure Analysis**
2. **Statistical Arbitrage**
3. **Smart Order Routing**
4. **Latency Optimization**
5. **Execution Quality Metrics**

## 1. Market Microstructure Analysis

### What is Market Microstructure?

Market microstructure studies how trades are executed and how prices form. HFT firms use this to:
- Predict short-term price movements
- Minimize transaction costs
- Detect market manipulation

### Techniques Implemented

#### Order Book Analysis

```python
from src.trading_engine.hft_techniques import OrderBookSnapshot

# Create order book snapshot
order_book = OrderBookSnapshot(
    symbol="AAPL",
    bid_price=150.45,
    bid_size=100,
    ask_price=150.55,
    ask_size=100,
    ...
)

# Calculate key metrics
spread_bps = order_book.spread_bps  # Spread in basis points
imbalance = order_book.order_imbalance  # Buy/sell pressure (0-1)
microprice = order_book.microprice  # Volume-weighted fair price
```

**Key Metrics:**

1. **Bid-Ask Spread**: Tighter spread = more liquid = better for trading
   - Typical liquid stock: 1-5 basis points
   - Illiquid stock: 10-50+ basis points

2. **Order Imbalance**: Predicts short-term direction
   - Imbalance > 0.6: Strong buy pressure → Price likely to rise
   - Imbalance < 0.4: Strong sell pressure → Price likely to fall

3. **Microprice**: Better estimate than mid-price
   - Formula: `(bid_size * ask + ask_size * bid) / (bid_size + ask_size)`
   - Used by HFT for more accurate valuations

#### Liquidity Scoring

```python
from src.trading_engine.hft_techniques import MarketMicrostructure

mm = MarketMicrostructure()
liquidity_score = mm.calculate_liquidity_score("AAPL", order_book)

# Score interpretation:
# 0.9-1.0: Excellent liquidity (trade freely)
# 0.7-0.9: Good liquidity
# 0.5-0.7: Moderate liquidity (be cautious)
# <0.5: Poor liquidity (avoid or use limit orders)
```

**Components:**
- Spread tightness (60% weight)
- Order book depth (40% weight)

#### Price Impact Estimation

```python
# Estimate how much our order will move the price
impact_bps = mm.estimate_price_impact(
    symbol="AAPL",
    order_size=1000,  # shares
    order_book=order_book
)

if impact_bps > 10:
    print("High impact! Consider splitting order")
```

**Kyle's Lambda Model:**
- Impact ∝ √(order_size / average_daily_volume)
- Larger orders have disproportionate impact
- HFT firms split large orders to minimize this

### Practical Application

```python
# Before executing trade, check market quality
if liquidity_score < 0.5:
    print("❌ Low liquidity, skipping trade")
    return

if impact_bps > 10:
    print("⚠️  High impact, using smart routing")
    use_vwap_execution()
else:
    print("✅ Good conditions, executing immediately")
    execute_market_order()
```

## 2. Statistical Arbitrage

### What is Statistical Arbitrage?

Stat arb exploits statistical patterns in price movements:
- Mean reversion: Prices return to average
- Pairs trading: Correlated stocks diverge/converge
- Momentum: Trends continue

### Techniques Implemented

#### Mean Reversion

```python
from src.trading_engine.hft_techniques import StatisticalArbitrage

stat_arb = StatisticalArbitrage(lookback_window=20)

# Calculate z-score
price_history = [100.0, 101.0, 99.5, ..., 105.0]
zscore = stat_arb.calculate_zscore(price_history)

# Interpret z-score:
# Z > +2: Overbought → SELL signal
# Z < -2: Oversold → BUY signal
# |Z| < 0.5: Near mean → Close position
```

**Trading Rules:**

| Z-Score Range | Signal | Action |
|--------------|--------|--------|
| Z < -2.0 | Oversold | Buy (expect reversion up) |
| -2.0 < Z < -0.5 | Slightly oversold | Watch |
| -0.5 < Z < +0.5 | At mean | Hold/Close |
| +0.5 < Z < +2.0 | Slightly overbought | Watch |
| Z > +2.0 | Overbought | Sell (expect reversion down) |

#### Half-Life Calculation

```python
# How long does it take to revert to mean?
half_life = stat_arb.calculate_half_life(price_history)

# Interpretation:
# Half-life < 1 day: Fast mean reversion (good for HFT)
# Half-life 1-5 days: Moderate (good for our platform)
# Half-life > 10 days: Slow (not suitable for stat arb)
```

#### Pairs Trading

```python
# Find correlated stocks
prices_aapl = [150, 151, 149, ...]
prices_msft = [300, 302, 298, ...]

correlation = stat_arb.calculate_correlation(prices_aapl, prices_msft)

# If correlation > 0.8 and spread diverges:
# - Buy underperformer
# - Sell outperformer
# - Profit when they converge
```

**Common Pairs:**
- SPY / IWM (S&P 500 / Russell 2000)
- XLE / OIH (Energy sector / Oil services)
- GLD / GDX (Gold / Gold miners)

### Practical Application

```python
# Combined with LLM signals
llm_signal = generate_llm_signal("AAPL")
zscore = stat_arb.calculate_zscore(price_history)

# Only trade if both agree
if llm_signal == "BUY" and zscore < -1.5:
    print("✅ LLM + Stat Arb agree: Strong BUY")
    execute_trade()
elif llm_signal == "BUY" but zscore > 2.0:
    print("⚠️  Conflict: LLM says BUY but price overbought")
    wait_for_pullback()
```

## 3. Smart Order Routing

### What is Smart Order Routing?

SOR algorithms minimize market impact by:
- Splitting large orders into smaller slices
- Timing execution optimally
- Routing to best venues

### Techniques Implemented

#### VWAP (Volume-Weighted Average Price)

```python
from src.trading_engine.hft_techniques import SmartOrderRouting

sor = SmartOrderRouting()

# Calculate benchmark VWAP
prices = [100.0, 100.5, 101.0]
volumes = [1000, 2000, 1000]
vwap = sor.calculate_vwap(prices, volumes)

# Goal: Execute close to VWAP
# Good execution: Within 5-10 bps of VWAP
```

**VWAP Strategy:**
```
For a 1000-share buy order:
1. Estimate volume distribution throughout day
2. Split order proportionally to expected volume
3. Execute slices when volume peaks
4. Result: Minimal market impact
```

#### TWAP (Time-Weighted Average Price)

```python
# Simpler than VWAP - equal slices over time
twap = sor.calculate_twap(prices)

# Best for:
# - Illiquid stocks (volume unreliable)
# - After hours
# - Consistent execution pace
```

**TWAP Strategy:**
```
For a 1000-share order over 10 minutes:
1. Split into 10 equal slices of 100 shares
2. Execute one slice per minute
3. Predictable, consistent execution
```

#### Order Splitting

```python
# Split large order to minimize impact
total_qty = 1000
num_slices = 10

slices = sor.split_order(total_qty, num_slices, strategy="TWAP")
# Result: [100, 100, 100, ..., 100]

# Execute slices over time
for slice_qty in slices:
    execute_slice(slice_qty)
    time.sleep(60)  # Wait 1 minute between slices
```

#### Implementation Shortfall

```python
# Measure execution quality
decision_price = 100.0  # Price when we decided to trade
execution_prices = [100.05, 100.10, 100.08]  # Actual execution prices
quantities = [300, 400, 300]

shortfall = sor.calculate_implementation_shortfall(
    decision_price,
    execution_prices,
    quantities
)

# Shortfall = cost of trading vs. instant execution
# Lower is better
# Goal: < 10 bps for liquid stocks
```

### Practical Application

```python
from src.trading_engine.advanced_executor import AdvancedTradingExecutor

executor = AdvancedTradingExecutor(use_smart_routing=True)

# Executor automatically:
# 1. Estimates price impact
# 2. Chooses execution strategy (VWAP/TWAP/IMMEDIATE)
# 3. Splits order if needed
# 4. Tracks execution quality

trade = executor.execute_signal(
    signal,
    execution_strategy="VWAP"  # or "TWAP" or "IMMEDIATE"
)
```

## 4. Latency Optimization

### HFT Latency Targets

| Component | HFT Target | Our Platform | Notes |
|-----------|-----------|--------------|-------|
| Signal Generation | <1ms | 50-100ms | LLMs are slower but smarter |
| Order Routing | <100μs | 10-20ms | Good enough for our timeframe |
| Exchange Latency | <50μs | 5-10ms | Depends on broker |
| **Total** | **<1ms** | **<100ms** | Still very fast for LLM-based |

### Optimization Techniques

#### Caching

```python
from src.utils.cache import cache

# Cache LLM analysis for 24 hours
@cached(ttl=3600, key_prefix="llm_analysis")
def analyze_stock(symbol):
    return expensive_llm_call(symbol)

# First call: 2000ms (LLM API)
# Subsequent calls: <5ms (Redis cache)
# 400x speedup!
```

**Cache Strategy:**
| Data Type | TTL | Reason |
|-----------|-----|--------|
| LLM Analysis | 24 hours | Fundamentals change slowly |
| Market Quote | 15 seconds | Real-time during hours |
| News Articles | 1 hour | News doesn't change |
| Historical Data | 24 hours | Static once day closes |

#### Latency Budget

```python
from src.trading_engine.hft_techniques import LatencyOptimization

# Estimate total latency
budget = LatencyOptimization.estimate_latency_budget(
    signal_generation_ms=75,
    order_routing_ms=15,
    exchange_latency_ms=8
)

print(f"Total latency: {budget['total_latency_ms']}ms")
print(f"HFT competitive: {budget['can_compete_hft']}")
print(f"Fast for LLM: {budget['is_fast']}")
```

### Practical Application

```python
# Measure and optimize
import time

start = time.time()
signal = generate_signal("AAPL")
elapsed_ms = (time.time() - start) * 1000

if elapsed_ms > 200:
    print("⚠️  Slow! Check:")
    print("  1. Cache hit rate")
    print("  2. LLM provider (use Groq for speed)")
    print("  3. Network latency")
```

## 5. Execution Quality Metrics

### Key Metrics

#### Fill Rate

```python
# What % of orders get filled?
fill_rate = filled_orders / total_orders

# Target: >95% for market orders
# Target: >70% for limit orders
```

#### Slippage

```python
# How much did price move against us?
slippage_bps = ((execution_price - quote_price) / quote_price) * 10000

# Good: <5 bps
# Acceptable: 5-15 bps
# Poor: >15 bps (need better execution)
```

#### Effective Spread

```python
# What was our actual cost?
effective_spread = mm.calculate_effective_spread(
    execution_price=100.05,
    mid_price=100.00,
    side="buy"
)

# Compare to quoted spread
# Goal: Effective spread ≈ Quoted spread
```

### Tracking Metrics

```python
from src.trading_engine.advanced_executor import AdvancedTradingExecutor

executor = AdvancedTradingExecutor()

# After some trades...
metrics = executor.get_execution_metrics()

print(f"Avg latency: {metrics['avg_execution_time_ms']:.2f}ms")
print(f"P95 latency: {metrics['p95_execution_time_ms']:.2f}ms")
print(f"Total trades: {metrics['total_executions']}")
```

## Best Practices

### When to Use Each Technique

#### Market Microstructure
- ✅ Always check before trading
- ✅ Avoid illiquid stocks (liquidity_score < 0.5)
- ✅ Use limit orders when spread > 10 bps

#### Statistical Arbitrage
- ✅ Confirm LLM signals with z-score
- ✅ Exit when z-score returns to mean
- ✅ Use for mean-reverting stocks (short half-life)
- ❌ Don't use for trending stocks

#### Smart Order Routing
- ✅ Always use for orders > $10,000
- ✅ Use VWAP during normal hours
- ✅ Use TWAP after hours or for illiquid stocks
- ❌ Don't use for urgent trades

### Cost-Benefit Analysis

| Technique | Implementation Cost | Latency Impact | Benefit |
|-----------|-------------------|----------------|---------|
| Order Book Analysis | Low | <1ms | High (prevents bad trades) |
| Liquidity Scoring | Low | <1ms | High (improves fill rate) |
| Mean Reversion | Medium | 1-5ms | Medium (extra signals) |
| VWAP/TWAP | High | +50-200ms | High (reduces slippage) |
| Price Impact Est. | Low | <1ms | Medium (cost awareness) |

## Example: Complete Execution Flow

```python
# 1. Generate LLM signal
signal = ensemble_strategy.generate_signal("AAPL")
# Latency: ~75ms

# 2. Get market data
quote = market_data.get_quote("AAPL")
order_book = create_order_book_snapshot("AAPL", quote)
# Latency: ~10ms (cached)

# 3. Check market quality
liquidity_score = mm.calculate_liquidity_score("AAPL", order_book)
if liquidity_score < 0.5:
    print("❌ Skip: Poor liquidity")
    return
# Latency: <1ms

# 4. Validate with stat arb
zscore = stat_arb.calculate_zscore(price_history)
if signal == "BUY" and zscore > 2.0:
    print("⚠️  Warning: Price overbought, reducing size")
    reduce_position_size(0.5)
# Latency: ~2ms

# 5. Estimate price impact
impact_bps = mm.estimate_price_impact("AAPL", quantity, order_book)
execution_strategy = "VWAP" if impact_bps > 10 else "IMMEDIATE"
# Latency: <1ms

# 6. Execute trade
trade = advanced_executor.execute_signal(
    signal,
    execution_strategy=execution_strategy
)
# Latency: 15-50ms

# Total latency: ~100-150ms
# Result: High-quality execution with minimal impact
```

## Further Reading

### Academic Papers
1. **O'Hara (1995)**: "Market Microstructure Theory" - Classic text
2. **Hasbrouck (2007)**: "Empirical Market Microstructure" - Practical guide
3. **Chan (2013)**: "Algorithmic Trading" - Stat arb strategies

### Industry Resources
1. **QuantStart**: Tutorials on execution algorithms
2. **Hudson River Trading**: Blog on market making
3. **Jane Street**: OCaml and low-latency trading

### Tools
1. **Arctic** (Man AHL): High-performance time-series database
2. **PyAlgoTrade**: Backtesting framework
3. **Zipline**: Algorithmic trading library

---

**Remember**: These techniques work best on liquid stocks during market hours. Always start with paper trading!
