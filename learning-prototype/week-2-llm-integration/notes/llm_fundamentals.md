# LLM Fundamentals for Trading Applications

**Time to Read**: 30-40 minutes | **Difficulty**: Beginner

---

## Table of Contents

1. [What are LLMs?](#what-are-llms)
2. [Groq vs Claude: The Right Tool for Trading](#groq-vs-claude)
3. [API Authentication](#api-authentication)
4. [Request and Response Format](#request-and-response-format)
5. [Tokens and Pricing](#tokens-and-pricing)
6. [Rate Limits and Quotas](#rate-limits-and-quotas)
7. [Best Practices for Trading Systems](#best-practices)

---

## What are LLMs?

**Large Language Models (LLMs)** are AI systems trained on vast amounts of text data to understand and generate human-like text. In trading applications, LLMs excel at:

- **Analyzing financial reports** and extracting insights
- **Summarizing market news** from multiple sources
- **Generating trading signals** with reasoning
- **Explaining complex financial concepts** to users
- **Classifying sentiment** in earnings calls and news
- **Comparing stocks** based on fundamentals

### How LLMs Work (Simplified)

LLMs predict the next token (word/subword) based on previous context. For trading:

```
Input: "AAPL's PE ratio is 28, sector average is 22. This suggests..."
LLM predicts: "...the stock is overvalued relative to peers."
```

**Key Concepts**:
- **Tokens**: Text chunks (roughly 4 chars or 0.75 words)
- **Context Window**: How much text the LLM can "remember" (4K-200K tokens)
- **Temperature**: Randomness in responses (0=deterministic, 1=creative)
- **System Prompt**: Instructions that guide the LLM's behavior

### Why Use LLMs in Trading?

Traditional algorithms struggle with:
- **Unstructured data** (news articles, social media, reports)
- **Nuanced analysis** (understanding context, sarcasm, implications)
- **Natural language reasoning** (explaining WHY a trade makes sense)

LLMs bridge this gap by turning text into actionable insights.

---

## Groq vs Claude: The Right Tool for Trading

Not all LLMs are created equal. Here's how to choose between Groq and Claude for different trading tasks.

### Groq (Llama 3.1 on Groq Infrastructure)

**What It Is**: Llama 3.1 running on Groq's ultra-fast inference chips.

**Strengths**:
- **Speed**: 500+ tokens/second (10x faster than most APIs)
- **Cost**: ~$0.0001 per 1K tokens (nearly free)
- **Simplicity**: Great for straightforward tasks

**Best For Trading**:
- Quick sentiment classification (bullish/bearish/neutral)
- Simple price alerts and summaries
- High-frequency signal generation
- Real-time analysis where speed matters
- Batch processing thousands of stocks

**Example Use Case**:
```python
# Fast sentiment analysis of 1000 tweets
for tweet in tweets:
    sentiment = groq.analyze(f"Is this bullish or bearish? {tweet}")
    # Response in 50ms vs 500ms with Claude
```

### Claude (Anthropic's Claude 3.5 Sonnet)

**What It Is**: Anthropic's most advanced reasoning model.

**Strengths**:
- **Reasoning**: Deep, nuanced analysis
- **Accuracy**: Better understanding of complex financial concepts
- **Context**: 200K token window (can analyze entire 10-K reports)
- **Safety**: Less likely to hallucinate or make risky suggestions

**Best For Trading**:
- Complex fundamental analysis
- Multi-factor decision making
- Regulatory compliance checks
- Risk assessment requiring nuance
- Analyzing long financial documents

**Example Use Case**:
```python
# Deep analysis of earnings report
analysis = claude.analyze(f"""
Analyze this earnings report and provide:
1. Key revenue drivers
2. Management concerns
3. Competitive positioning
4. 3-month outlook
5. Risk factors

{earnings_report_text}
""")
```

### Decision Matrix

| Task | Use Groq? | Use Claude? | Why |
|------|-----------|-------------|-----|
| Classify news sentiment | ✓ | | Speed + cost |
| Generate daily signals for 500 stocks | ✓ | | Batch efficiency |
| Analyze earnings call transcript | | ✓ | Nuance required |
| Explain trade reasoning to user | | ✓ | Quality matters |
| Quick price alert | ✓ | | Sub-second response |
| Risk assessment for portfolio | | ✓ | Accuracy critical |

### Cost Example (100 Analyses/Day)

**Groq**:
- 100 analyses × 500 tokens each = 50K tokens
- 50K / 1000 × $0.0001 = $0.005/day = $0.15/month

**Claude**:
- 100 analyses × 2000 tokens each = 200K tokens
- 200K / 1M × $3 = $0.60/day = $18/month

**Smart Strategy**: Use Groq for 80% of tasks, Claude for 20% critical analyses = ~$4/month.

---

## API Authentication

Both Groq and Claude use API keys for authentication. Here's how to set them up securely.

### Getting API Keys

**Groq**:
1. Visit https://console.groq.com/
2. Sign up (free tier available)
3. Navigate to API Keys
4. Generate key (starts with `gsk_`)

**Claude (Anthropic)**:
1. Visit https://console.anthropic.com/
2. Sign up (pay-as-you-go)
3. Navigate to API Keys
4. Generate key (starts with `sk-ant-`)

### Secure Storage

**Never hardcode API keys in your code!** Use environment variables:

```python
# .env file (add to .gitignore!)
GROQ_API_KEY=gsk_your_key_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

```python
# Python code
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")

if not GROQ_KEY or not CLAUDE_KEY:
    raise ValueError("Missing API keys in environment!")
```

### Making Authenticated Requests

**Groq**:
```python
import httpx

headers = {
    "Authorization": f"Bearer {GROQ_KEY}",
    "Content-Type": "application/json"
}

response = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers=headers,
    json={...}
)
```

**Claude**:
```python
import anthropic

client = anthropic.Anthropic(api_key=CLAUDE_KEY)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[...]
)
```

### Common Authentication Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `401 Unauthorized` | Invalid API key | Check key is correct and active |
| `403 Forbidden` | API access not enabled | Verify billing/account status |
| `429 Too Many Requests` | Rate limit exceeded | Implement backoff/retry |

---

## Request and Response Format

Understanding the API structure is crucial for reliable trading systems.

### Groq Request Format

Groq uses the OpenAI-compatible format:

```python
request = {
    "model": "llama-3.1-70b-versatile",
    "messages": [
        {
            "role": "system",
            "content": "You are a financial analyst focused on value investing."
        },
        {
            "role": "user",
            "content": "Analyze AAPL with PE ratio 28.5 and revenue growth 8%"
        }
    ],
    "temperature": 0.7,
    "max_tokens": 500
}
```

**Key Parameters**:
- `model`: Which LLM to use (`llama-3.1-70b-versatile`, `llama-3.1-8b-instant`)
- `messages`: Conversation history (system + user + assistant)
- `temperature`: 0=deterministic, 1=creative (use 0-0.3 for trading)
- `max_tokens`: Response length limit

### Groq Response Format

```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1699999999,
    "model": "llama-3.1-70b-versatile",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Based on the PE ratio of 28.5..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 45,
        "completion_tokens": 120,
        "total_tokens": 165
    }
}
```

### Claude Request Format

Claude uses a different structure:

```python
request = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "system": "You are a financial analyst focused on value investing.",
    "messages": [
        {
            "role": "user",
            "content": "Analyze AAPL with PE ratio 28.5 and revenue growth 8%"
        }
    ],
    "temperature": 0.5
}
```

**Differences from Groq**:
- `system` is a separate parameter (not in messages)
- `max_tokens` is required
- More advanced parameters available (thinking, citations)

### Claude Response Format

```json
{
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Based on the PE ratio of 28.5..."
        }
    ],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 42,
        "output_tokens": 156
    }
}
```

### Extracting the Response

```python
# Groq
text = response.json()["choices"][0]["message"]["content"]

# Claude
text = response.content[0].text
```

---

## Tokens and Pricing

Understanding tokens is essential for cost optimization in trading systems.

### What is a Token?

A token is a chunk of text. Rules of thumb:
- 1 token ≈ 4 characters
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words

**Examples**:
- "AAPL" = 2 tokens (`AA`, `PL`)
- "P/E ratio" = 4 tokens (`P`, `/`, `E`, ` ratio`)
- "The stock is overvalued" = 6 tokens

### Counting Tokens

**For Groq (OpenAI tokenizer)**:
```python
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = encoder.encode("Analyze AAPL stock")
print(f"Token count: {len(tokens)}")  # 4
```

**For Claude**:
```python
import anthropic

client = anthropic.Anthropic(api_key=CLAUDE_KEY)
tokens = client.count_tokens("Analyze AAPL stock")
```

### Pricing Breakdown

**Groq Pricing** (as of 2024):
- Input: ~$0.05 / 1M tokens
- Output: ~$0.10 / 1M tokens
- Effectively free for most trading use cases

**Claude 3.5 Sonnet Pricing**:
- Input: $3 / 1M tokens
- Output: $15 / 1M tokens
- Caching: $0.30 / 1M tokens (90% discount!)

### Real Trading Cost Examples

**Scenario 1**: Daily sentiment analysis of 100 news articles

```python
# Each article: 200 tokens input + 50 tokens output
input_tokens = 100 * 200 = 20K
output_tokens = 100 * 50 = 5K

# Groq cost
groq_cost = (20K/1M * 0.05) + (5K/1M * 0.10)
# = $0.001 + $0.0005 = $0.0015/day = $0.045/month

# Claude cost
claude_cost = (20K/1M * 3) + (5K/1M * 15)
# = $0.06 + $0.075 = $0.135/day = $4.05/month
```

**Scenario 2**: Weekly deep analysis of 10 stocks

```python
# Each analysis: 5000 tokens input + 2000 tokens output
# Only 10 stocks/week

input_tokens = 10 * 5000 = 50K/week
output_tokens = 10 * 2000 = 20K/week

# Claude cost (where quality matters)
weekly_cost = (50K/1M * 3) + (20K/1M * 15)
# = $0.15 + $0.30 = $0.45/week = $1.80/month
```

### Cost Optimization Tips

1. **Use Groq for simple tasks** (80% of use cases)
2. **Cache system prompts** (90% cost reduction on Claude)
3. **Batch requests** where possible
4. **Limit max_tokens** to what you need
5. **Use shorter prompts** without sacrificing clarity

---

## Rate Limits and Quotas

All APIs have limits to prevent abuse. Hitting rate limits can break your trading system mid-session.

### Groq Rate Limits

**Free Tier**:
- 30 requests/minute
- 6,000 tokens/minute
- 14,400 requests/day

**Paid Tier**:
- 100 requests/minute
- 100,000 tokens/minute
- Higher daily limits

### Claude Rate Limits

**Tier 1** (New users):
- 50 requests/minute
- 40K tokens/minute
- $100/month spend limit

**Tier 2** ($5+ spent):
- 1,000 requests/minute
- 80K tokens/minute
- $500/month spend limit

### Handling Rate Limits

**Strategy 1: Exponential Backoff**
```python
import time
import random

def call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited. Retrying in {delay:.1f}s...")
            time.sleep(delay)
```

**Strategy 2: Request Queue**
```python
import asyncio
from collections import deque

class RateLimiter:
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.requests = deque()

    async def acquire(self):
        now = time.time()

        # Remove requests older than 1 minute
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()

        # Wait if at limit
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0])
            await asyncio.sleep(sleep_time)

        self.requests.append(time.time())

# Usage
limiter = RateLimiter(max_requests_per_minute=30)

async def analyze_stock(symbol):
    await limiter.acquire()
    response = await groq.chat(...)
    return response
```

**Strategy 3: Caching**

The best rate limit is the one you never hit:

```python
import redis
import hashlib

cache = redis.Redis()

def cached_analysis(stock_data, ttl=3600):
    # Generate cache key
    key = hashlib.md5(str(stock_data).encode()).hexdigest()

    # Check cache
    cached = cache.get(key)
    if cached:
        return json.loads(cached)

    # Call LLM
    result = llm.analyze(stock_data)

    # Store in cache
    cache.setex(key, ttl, json.dumps(result))
    return result
```

---

## Best Practices for Trading Systems

Building reliable LLM-powered trading systems requires following proven patterns.

### 1. Always Set Temperature Low

For trading decisions, you want consistency, not creativity.

```python
# Bad: Unpredictable results
response = llm.chat(temperature=1.0, ...)

# Good: Consistent analysis
response = llm.chat(temperature=0.2, ...)
```

### 2. Validate All Responses

LLMs can hallucinate or return malformed data.

```python
from pydantic import BaseModel, Field, validator

class StockAnalysis(BaseModel):
    symbol: str
    recommendation: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str

    @validator('recommendation')
    def validate_recommendation(cls, v):
        if v not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError('Invalid recommendation')
        return v

# Parse and validate
try:
    analysis = StockAnalysis(**response_json)
except ValidationError as e:
    # Handle error, use fallback
    logger.error(f"Invalid LLM response: {e}")
```

### 3. Always Have Fallbacks

Never let your trading system crash because an API is down.

```python
async def analyze_stock(symbol, data):
    try:
        # Try Claude first (best quality)
        return await claude_analyze(symbol, data)
    except Exception as e:
        logger.warning(f"Claude failed: {e}")

        try:
            # Fallback to Groq
            return await groq_analyze(symbol, data)
        except Exception as e:
            logger.error(f"Groq failed: {e}")

            # Final fallback: rule-based
            return rule_based_analysis(symbol, data)
```

### 4. Log Everything

You need to audit LLM decisions for debugging and compliance.

```python
import logging
import json

logger = logging.getLogger(__name__)

def log_llm_call(symbol, input_data, response, cost):
    logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "model": "claude-3-5-sonnet",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": cost,
        "recommendation": response.recommendation,
        "confidence": response.confidence
    }))
```

### 5. Test with Mocks

Don't spend money testing. Mock LLM calls in development.

```python
from unittest.mock import Mock, patch

def test_stock_analysis():
    mock_response = {
        "symbol": "AAPL",
        "recommendation": "BUY",
        "confidence": 0.85,
        "reasoning": "Strong fundamentals"
    }

    with patch('llm.analyze') as mock_llm:
        mock_llm.return_value = mock_response

        result = analyze_stock("AAPL", {...})
        assert result["recommendation"] == "BUY"
```

### 6. Monitor Costs in Production

Set up alerts before costs spiral.

```python
class CostTracker:
    def __init__(self, daily_limit=10.0):
        self.daily_limit = daily_limit
        self.today_cost = 0.0
        self.today_date = datetime.now().date()

    def add_cost(self, cost):
        # Reset if new day
        if datetime.now().date() != self.today_date:
            self.today_cost = 0.0
            self.today_date = datetime.now().date()

        self.today_cost += cost

        if self.today_cost > self.daily_limit:
            raise Exception(f"Daily cost limit exceeded: ${self.today_cost:.2f}")
```

### 7. Use Structured Outputs

Force the LLM to return valid JSON.

```python
prompt = """
Analyze AAPL stock and return JSON in this exact format:
{
    "recommendation": "BUY|SELL|HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "risks": ["risk1", "risk2"]
}

Data: {stock_data}

Return ONLY the JSON, no other text.
"""
```

---

## Summary

### Key Takeaways

1. **LLMs are powerful** for analyzing unstructured financial data
2. **Choose wisely**: Groq for speed/cost, Claude for quality/reasoning
3. **Secure API keys** with environment variables
4. **Understand tokens** to optimize costs (target <$5/month)
5. **Implement rate limiting** to avoid API failures
6. **Follow best practices**: low temperature, validation, fallbacks, logging, testing

### Quick Reference

```python
# Groq for fast sentiment analysis
groq_response = groq.chat(
    model="llama-3.1-70b-versatile",
    temperature=0.2,
    max_tokens=100,
    messages=[...]
)

# Claude for deep analysis
claude_response = claude.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    temperature=0.3,
    system="You are a financial analyst...",
    messages=[...]
)
```

### Next Steps

1. Get your API keys (Groq + Anthropic)
2. Complete `starter-code/llm_client.py`
3. Read `prompt_engineering.md`
4. Start building your financial agent!

**Remember**: Start simple, test thoroughly, and optimize costs early. LLMs are tools, not magic. Use them where they add real value to your trading system.
