# Week 2: LLM Integration 🤖

**Goal:** Integrate AI models to analyze stocks and generate intelligent trading signals.

**Time Estimate:** 10-12 hours

## 📚 What You'll Learn

- How to call LLM APIs (Claude, GPT-4)
- Prompt engineering for financial analysis
- Parsing structured data from LLM responses
- Combining AI signals with technical indicators
- Caching strategies to reduce API costs

## 🎯 Learning Objectives

- [ ] Set up API clients for Claude and GPT-4
- [ ] Write prompts for fundamental analysis
- [ ] Parse JSON responses from LLMs
- [ ] Create an ensemble strategy (combine multiple signals)
- [ ] Implement intelligent caching
- [ ] Generate your first AI-powered trading signal

## 📝 Daily Plan

### Day 1: LLM API Setup
- Create API clients for Claude/GPT
- Test basic prompts
- Handle rate limits and errors
- **File:** `llm_client.py`

### Day 2: Prompt Engineering
- Write prompts for stock analysis
- Extract financial metrics
- Sentiment analysis
- **File:** `prompts.py`

### Day 3: Response Parsing
- Parse JSON from LLM responses
- Validate structured data
- Handle parsing errors
- **File:** `response_parser.py`

### Day 4: AI Trading Agent
- Create FinancialAnalystAgent
- Combine with technical indicators
- Generate composite signals
- **File:** `ai_agent.py`

### Day 5: Caching & Optimization
- Implement Redis caching
- Reduce API costs
- Performance optimization
- **File:** `cache_manager.py`

## 🛠️ Key Concepts

### LLM Prompting for Finance

**Good Prompt:**
```
Analyze Apple Inc (AAPL) for trading:

Financial Data:
- P/E Ratio: 28.5
- Revenue Growth: 12% YoY
- Profit Margin: 26%
- Debt/Equity: 1.8

Recent News:
- Q4 earnings beat estimates
- New iPhone launch successful
- Services revenue up 15%

Provide:
1. Fundamental score (0-100)
2. Buy/Sell/Hold recommendation
3. 2-3 sentence reasoning

Response in JSON format.
```

**Bad Prompt:**
```
Is AAPL a good buy?
```

### Response Parsing

```python
# LLM returns JSON:
{
    "symbol": "AAPL",
    "recommendation": "BUY",
    "fundamental_score": 85,
    "reasoning": "Strong earnings growth..."
}

# Parse into Pydantic model:
signal = AISignal(**llm_response)
```

## 📚 Starter Code

See `starter.py` for implementation templates with TODOs.

## ⏭️ Next Week

**Week 3: Risk Management** - Add Kelly Criterion and CVaR to protect your capital.
