# Cost Optimization Guide

This guide explains how the platform minimizes API costs and how you can optimize further.

## Cost Breakdown

### Expected Monthly Costs (Moderate Usage)

| Service | Free Tier | Expected Cost |
|---------|-----------|---------------|
| Groq API | 15 RPM free | $1-5/month |
| Anthropic Claude | $5 credit | $5-15/month |
| Alpaca (Paper) | Free | $0 |
| Alpha Vantage | 25 calls/day | $0 |
| **Total** | - | **$6-20/month** |

### Cost Per Trading Signal

- With caching: **$0.002 - $0.005**
- Without caching: **$0.01 - $0.05**

**Savings: 80-90% through caching!**

## Optimization Strategies

### 1. Smart LLM Routing

The platform automatically chooses the cheapest model for each task:

```python
# Simple tasks → Groq (very cheap)
sentiment_analysis → Groq Llama 3.1 (~$0.0001 per 1M tokens)

# Complex tasks → Claude (more expensive but better)
fundamental_analysis → Claude Sonnet (~$3 per 1M tokens)
```

**Configuration:**
```bash
# .env
USE_GROQ_FOR_SPEED=true          # Use Groq for simple tasks
USE_ANTHROPIC_FOR_COMPLEX=true    # Use Claude only when needed
```

### 2. Aggressive Caching

Every LLM response is cached in Redis:

```bash
# .env - Increase cache duration to save more
CACHE_ANALYSIS_HOURS=24  # Default: 24 hours
```

**Example savings:**
- Without cache: 100 signals/day × $0.01 = $1/day = **$30/month**
- With cache (24h): 10 unique symbols × $0.01 = $0.10/day = **$3/month**
- **Savings: $27/month (90%)**

### 3. Token Limits

Limit response length to control costs:

```bash
# .env
MAX_TOKENS_PER_ANALYSIS=2000  # Maximum tokens per request
```

**Impact:**
- 2000 tokens: ~$0.006 per Claude call
- 4000 tokens: ~$0.012 per Claude call
- **Recommendation: 1500-2000 tokens is optimal**

### 4. Batch Processing

Process multiple symbols together to amortize overhead:

```python
# Instead of:
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    generate_signal(symbol)  # 3 separate API calls

# Do this:
generate_batch_signals(["AAPL", "MSFT", "GOOGL"])  # More efficient
```

### 5. Schedule-Based Analysis

Only analyze during market hours or key times:

```python
# Analyze at market open and mid-day only
ANALYSIS_SCHEDULE = ["09:30", "12:00", "15:30"]  # EST

# Avoid:
# - Analyzing every minute (expensive!)
# - Analyzing after hours (no point)
```

### 6. Watchlist Curation

Focus on liquid, high-volume stocks:

```bash
# .env
DEFAULT_WATCHLIST=AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META

# Avoid:
# - Penny stocks (low liquidity)
# - Too many symbols (>20 increases costs)
```

## Advanced Cost Controls

### 1. Use Groq for Everything (Cheapest)

```bash
# .env
USE_GROQ_FOR_SPEED=true
USE_ANTHROPIC_FOR_COMPLEX=false  # Disable Claude

# In code: src/llm_agents/base_agent.py
# Force Groq for all tasks
def _call_llm(self, prompt, complexity="simple"):
    return self._call_groq(prompt)  # Always use Groq
```

**Cost reduction: 95%**
**Trade-off: Lower quality fundamental analysis**

### 2. Increase Cache Duration

```bash
# Cache for 48 hours instead of 24
CACHE_ANALYSIS_HOURS=48

# Or cache for a week for stable stocks
CACHE_ANALYSIS_HOURS=168
```

**When to use:**
- Stable, large-cap stocks (fundamentals change slowly)
- Long-term holding strategies

**When NOT to use:**
- Earnings season (fundamental changes fast)
- Breaking news events

### 3. Implement Analysis Thresholds

Only analyze when conditions warrant:

```python
# Only analyze if:
# 1. Volume spike (>150% of average)
# 2. Price change (>2%)
# 3. Major news event

def should_analyze(symbol):
    quote = get_quote(symbol)
    if quote['volume'] < 1.5 * quote['avg_volume']:
        return False  # Skip analysis, save money
    if abs(quote['price_change_pct']) < 2:
        return False  # Not interesting enough
    return True
```

### 4. Optimize Prompt Length

Shorter prompts = lower cost:

```python
# Bad (expensive):
prompt = f"""
Please analyze the following company...
{10000 words of context}
"""

# Good (cheap):
prompt = f"""
Analyze {symbol}:
Metrics: PE={pe}, Revenue={rev}
Task: Score 0-100, 2 sentences why
"""
```

## Monitoring Costs

### Track in MLflow

Every signal logs its cost:

```python
# View in MLflow UI
http://localhost:5000

# Metrics tracked:
# - tokens_used
# - api_cost
# - cache_hit (true/false)
```

### Dashboard Queries

```sql
-- PostgreSQL query for monthly costs
SELECT
    DATE_TRUNC('month', created_at) as month,
    SUM(api_cost) as total_cost,
    AVG(api_cost) as avg_cost_per_signal,
    COUNT(*) as num_signals,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cached_signals
FROM llm_analyses
GROUP BY month
ORDER BY month DESC;
```

### Cost Alerts

Set up alerts when costs exceed thresholds:

```python
# In your monitoring system
if daily_cost > 2.00:  # $2/day threshold
    send_alert("LLM costs high today: $" + str(daily_cost))
```

## Cost Comparison: Different Strategies

### Strategy A: Maximum Quality (Expensive)
- Use Claude Sonnet for everything
- No caching
- Analyze every hour
- **Cost: $150-300/month**

### Strategy B: Balanced (Recommended)
- Groq for sentiment, Claude for fundamentals
- 24-hour caching
- Analyze 2-3 times per day
- **Cost: $6-20/month** ✅

### Strategy C: Ultra-Cheap
- Groq for everything
- 48-hour caching
- Analyze once per day
- **Cost: $1-5/month**
- Trade-off: Lower signal quality

## Real-World Cost Examples

### Example 1: Day Trading (High Frequency)
- 10 symbols
- Analyze every 30 minutes during market hours
- 6.5 hours × 2 = 13 signals/day/symbol
- 10 × 13 = 130 signals/day

**Without optimization:**
- 130 × $0.01 = $1.30/day
- **$39/month**

**With caching (80% hit rate):**
- 26 × $0.01 = $0.26/day
- **$7.80/month** ✅

### Example 2: Swing Trading (Low Frequency)
- 20 symbols
- Analyze once per day
- 20 signals/day

**With caching:**
- 20 × $0.005 = $0.10/day
- **$3/month** ✅

### Example 3: Portfolio Monitoring
- 50 symbols
- Analyze weekly
- ~7 signals/day

**With caching:**
- 7 × $0.002 = $0.014/day
- **$0.42/month** ✅

## Best Practices Summary

1. ✅ **Always enable caching** (saves 80-90%)
2. ✅ **Use Groq by default** (100x cheaper than Claude)
3. ✅ **Only use Claude for complex tasks**
4. ✅ **Limit tokens** (1500-2000 is plenty)
5. ✅ **Batch process** when possible
6. ✅ **Curate watchlist** (quality over quantity)
7. ✅ **Monitor costs** in MLflow
8. ✅ **Set cost alerts** ($2/day threshold)

## Emergency Cost Controls

If costs spike unexpectedly:

```bash
# 1. Disable all LLM calls temporarily
docker-compose stop trading-api

# 2. Check what's happening
docker-compose logs trading-api | grep "api_cost"

# 3. Clear cache if needed
docker-compose exec redis redis-cli FLUSHDB

# 4. Reduce watchlist
# Edit .env, reduce DEFAULT_WATCHLIST

# 5. Restart
docker-compose up -d trading-api
```

## Questions?

**Q: Can I use this with $0 budget?**
A: Almost! Use only the free tiers:
- Groq free tier: 15 RPM
- Anthropic $5 credit (lasts 1-2 months)
- Total: ~$2-3/month after credits run out

**Q: What if I run out of Anthropic credits?**
A: Switch to Groq-only mode (see Advanced Cost Controls #1)

**Q: How much for 100 signals/day?**
A: With caching: $3-6/month

**Q: How much for 1000 signals/day?**
A: With caching: $20-40/month
A: Without caching: $200-400/month ⚠️

---

**Remember:** The platform is designed for cost-efficiency. Follow these guidelines and you'll keep costs under $20/month even with active trading!
