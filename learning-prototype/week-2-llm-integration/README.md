# Week 2: LLM Integration - AI-Powered Trading Signals

**Time**: 12-14 hours | **Difficulty**: Intermediate | **Prerequisites**: Week 1 completed

---

## 🎯 Learning Goals

By the end of this week, you will:

1. ✅ Integrate Groq and Claude LLM APIs
2. ✅ Write effective prompts for financial analysis
3. ✅ Parse LLM responses into structured data
4. ✅ Implement caching to reduce API costs
5. ✅ Build an LLM agent for trading decisions
6. ✅ Handle LLM errors and fallbacks
7. ✅ Optimize costs with smart model selection
8. ✅ Test LLM integrations with mocks

---

## 📚 What You'll Build

An **AI-Powered Trading System** that:
- Analyzes stocks using Claude and Groq
- Generates buy/sell signals with reasoning
- Caches responses to minimize API costs
- Handles errors gracefully with fallbacks
- Returns structured, validated responses
- Costs < $1/week to run

**Example Usage**:
```python
# Analyze a stock with AI
agent = FinancialAgent()
analysis = await agent.analyze_stock("AAPL", {
    "price": 150.25,
    "pe_ratio": 28.5,
    "revenue_growth": 0.08
})

# Returns structured response:
{
    "symbol": "AAPL",
    "recommendation": "BUY",
    "confidence": 0.82,
    "reasoning": "Strong fundamentals with PE ratio below sector average...",
    "risks": ["High valuation", "Market volatility"],
    "score": 82
}
```

---

## 🗺️ Learning Path

### Day 1-2: LLM Basics & Integration (4-5 hours)

1. Read `notes/llm_fundamentals.md`
2. Get API keys (Groq, Anthropic)
3. Complete `starter-code/llm_client.py` TODOs
4. Test basic LLM calls
5. Do Exercise 1 & 2

**Key Concepts**:
- API authentication
- Prompt structure
- Temperature parameter
- Tokens and pricing
- Rate limits

### Day 2-3: Prompt Engineering (3-4 hours)

1. Read `notes/prompt_engineering.md`
2. Complete `starter-code/prompts.py` TODOs
3. Test different prompt styles
4. Do Exercise 3

**Key Concepts**:
- System vs user prompts
- Few-shot examples
- Output formatting
- Prompt templates
- Iteration and refinement

### Day 3-4: Financial Agent (4-5 hours)

1. Read `notes/financial_agents.md`
2. Complete `starter-code/financial_agent.py` TODOs
3. Build complete analysis pipeline
4. Do Exercise 4 & 5

**Key Concepts**:
- Agent architecture
- Response parsing
- Error handling
- Fallback strategies
- Testing with mocks

### Day 4-5: Caching & Optimization (2-3 hours)

1. Read `notes/caching_strategies.md`
2. Implement Redis caching
3. Optimize API costs
4. Run `scripts/test_week2.sh`
5. Complete final validation

**Key Concepts**:
- Cache invalidation
- TTL strategies
- Cost optimization
- Model selection
- Rate limit handling

---

## 📋 Starter Code Overview

### llm_client.py
```python
"""
TODO #1: Create base LLM client class
TODO #2: Implement Groq client
TODO #3: Implement Claude client
TODO #4: Add retry logic
TODO #5: Add rate limit handling
TODO #6: Add response validation
"""
```

**What You'll Learn**:
- Abstract base classes
- API client patterns
- Error handling
- Retry with backoff

### prompts.py
```python
"""
TODO #1: Create prompt templates
TODO #2: Add system prompts
TODO #3: Add few-shot examples
TODO #4: Create prompt builder
TODO #5: Add validation
"""
```

**What You'll Learn**:
- Template design
- Dynamic prompts
- Context injection
- Output formatting

### financial_agent.py
```python
"""
TODO #1: Create agent class
TODO #2: Implement analyze_stock method
TODO #3: Parse LLM responses
TODO #4: Add error handling
TODO #5: Add caching
TODO #6: Integrate with trading API
"""
```

**What You'll Learn**:
- Agent design patterns
- JSON parsing
- Fallback strategies
- Integration testing

### cache.py
```python
"""
TODO #1: Setup Redis connection
TODO #2: Implement cache decorator
TODO #3: Add cache invalidation
TODO #4: Add TTL strategies
"""
```

**What You'll Learn**:
- Redis basics
- Decorators
- Cache patterns
- TTL management

---

## 🎓 Exercises

### Exercise 1: Basic LLM Call
**File**: `exercises/exercise_1_llm_basics.py`

**Objective**: Make your first LLM API call

```python
# TODO: Call Groq API
# TODO: Call Claude API
# TODO: Compare responses
# TODO: Measure latency
```

### Exercise 2: Prompt Engineering
**File**: `exercises/exercise_2_prompts.py`

**Objective**: Write effective financial prompts

```python
# TODO: Create stock analysis prompt
# TODO: Add few-shot examples
# TODO: Test different temperatures
# TODO: Format output as JSON
```

### Exercise 3: Response Parsing
**File**: `exercises/exercise_3_parsing.py`

**Objective**: Parse LLM responses reliably

```python
# TODO: Extract JSON from markdown
# TODO: Handle malformed responses
# TODO: Validate with Pydantic
# TODO: Add error recovery
```

### Exercise 4: Financial Agent
**File**: `exercises/exercise_4_agent.py`

**Objective**: Build complete analysis agent

```python
# TODO: Combine prompts + LLM + parsing
# TODO: Add caching
# TODO: Add fallbacks
# TODO: Test end-to-end
```

### Exercise 5: Cost Optimization
**File**: `exercises/exercise_5_optimization.py`

**Objective**: Minimize API costs

```python
# TODO: Implement smart caching
# TODO: Use Groq for simple tasks
# TODO: Use Claude for complex analysis
# TODO: Measure cost per signal
```

---

## 📖 Detailed Notes

### 1. llm_fundamentals.md (2,500 words)
- What are LLMs?
- Groq vs Claude comparison
- API authentication
- Request/response format
- Tokens and pricing
- Rate limits
- Best practices

### 2. prompt_engineering.md (2,000 words)
- Prompt structure
- System vs user messages
- Few-shot learning
- Output formatting
- Temperature and top_p
- Iteration techniques
- Common pitfalls

### 3. financial_agents.md (1,800 words)
- Agent architecture
- Analysis pipeline
- Response validation
- Error handling
- Testing strategies
- Real-world examples

### 4. caching_strategies.md (1,500 words)
- Why cache?
- Redis setup
- Cache keys
- TTL strategies
- Invalidation
- Cost savings
- Monitoring

### 5. llm_testing.md (1,200 words)
- Mocking LLM calls
- Deterministic testing
- Response fixtures
- Integration tests
- Cost-free testing

---

## 🛠️ Scripts

### test_week2.sh
Comprehensive testing for Week 2:
```bash
#!/bin/bash
# Test LLM integration
# Test caching
# Test financial agent
# Cost analysis
```

### compare_models.sh
Compare Groq vs Claude:
```bash
#!/bin/bash
# Latency comparison
# Cost comparison
# Quality comparison
# Recommendations
```

### validate.py
Final Week 2 validation:
```python
# All TODOs complete?
# Exercises passing?
# Cost < $1/week?
# Ready for Week 3?
```

---

## ✅ Completion Checklist

- [ ] Read all notes
- [ ] Obtained API keys (Groq, Claude)
- [ ] Completed all starter code TODOs
- [ ] All 5 exercises passing
- [ ] `test_week2.sh` passes
- [ ] Cost analysis < $1/week
- [ ] Can explain:
  - [ ] Prompt engineering basics
  - [ ] Token pricing
  - [ ] Caching strategies
  - [ ] When to use Groq vs Claude
  - [ ] Agent architecture

**Time Spent**: _____ hours (target: 12-14)

---

## 💡 Key Takeaways

### API Costs
**Groq** (for simple tasks):
- Speed: ~500 tokens/sec
- Cost: ~$0.0001 per 1K tokens
- Use for: Quick analysis, simple classifications

**Claude** (for complex reasoning):
- Speed: ~50 tokens/sec
- Cost: ~$3 per 1M tokens
- Use for: Deep analysis, complex reasoning

**Cost Example**:
- 100 signals/day with Groq = $0.30/month
- 100 signals/day with Claude = $9/month
- Smart caching = 90% reduction

### Prompt Quality
Good prompt = Better results + Lower costs
- Be specific
- Provide context
- Use examples
- Specify format
- Iterate and test

### Caching Impact
- Without cache: $10/week
- With cache (1 hour TTL): $1/week
- With cache (1 day TTL): $0.10/week

---

## 🚀 Next Week Preview

**Week 3: Data & Risk Management**

You'll learn to:
- Fetch real-time market data
- Calculate risk metrics (VaR, Sharpe)
- Implement position sizing
- Build portfolio tracker
- Store data in PostgreSQL
- Create data pipelines

Get ready to combine AI analysis with solid risk management! 📊

---

**Ready? Start with `notes/llm_fundamentals.md` and dive into `starter-code/llm_client.py`!**

Let's make your trading system intelligent! 🧠✨
