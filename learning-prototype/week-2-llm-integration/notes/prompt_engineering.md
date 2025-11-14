# Prompt Engineering for Financial Analysis

**Time to Read**: 25-35 minutes | **Difficulty**: Intermediate

---

## Table of Contents

1. [Introduction to Prompt Engineering](#introduction)
2. [Prompt Structure](#prompt-structure)
3. [System vs User Messages](#system-vs-user-messages)
4. [Few-Shot Learning](#few-shot-learning)
5. [Output Formatting](#output-formatting)
6. [Temperature and Top-P](#temperature-and-top-p)
7. [Iteration Techniques](#iteration-techniques)
8. [Common Pitfalls](#common-pitfalls)

---

## Introduction to Prompt Engineering

**Prompt engineering** is the art and science of crafting inputs that get the best outputs from LLMs. In trading, a well-engineered prompt can mean the difference between:

- A vague "stock looks good" vs. a structured analysis with confidence scores
- Hallucinated data vs. honest "I don't know" responses
- $50/month in API costs vs. $5/month

### Why It Matters for Trading

Poor prompt:
```
"Tell me about AAPL"
```

Good prompt:
```
"Analyze AAPL stock with PE ratio 28.5, revenue growth 8%, and debt ratio 1.2.
Compare to sector average PE of 22. Provide: 1) BUY/SELL/HOLD, 2) confidence 0-1,
3) three key reasons, 4) top risk. Format as JSON."
```

The second prompt:
- Provides specific data (no hallucination)
- Sets clear expectations (no rambling)
- Specifies format (easy parsing)
- Limits scope (lower cost)

**Result**: 5x faster, 3x cheaper, 10x more useful.

---

## Prompt Structure

A well-structured prompt has clear sections. Use this template for trading analysis:

### The CRISP Framework

**C**ontext - **R**ole - **I**nstruction - **S**pecifics - **P**arameters

```python
prompt = f"""
CONTEXT:
You are analyzing {symbol} for a value investing strategy.
Current market: bullish, tech sector outperforming.

ROLE:
You are a senior equity analyst with 15 years experience in tech stocks.

INSTRUCTION:
Analyze whether {symbol} is a good value investment.

SPECIFICS:
Company data:
- PE Ratio: {pe_ratio}
- Revenue Growth: {revenue_growth}%
- Debt/Equity: {debt_ratio}
- Sector PE Average: {sector_pe}

PARAMETERS:
Provide your analysis in JSON format:
{{
    "recommendation": "BUY|SELL|HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "max 100 words",
    "risks": ["risk1", "risk2", "risk3"]
}}
"""
```

### Why This Works

1. **Context** prevents generic responses
2. **Role** sets expertise level and perspective
3. **Instruction** is clear and actionable
4. **Specifics** provide concrete data (no hallucination)
5. **Parameters** ensure parseable output

### Bad vs Good Examples

**Bad Prompt**:
```
"Is TSLA a good buy?"
```

Problems:
- No context (timeframe? strategy?)
- No data (LLM will hallucinate)
- No format (parsing nightmare)
- Too vague (unhelpful response)

**Good Prompt**:
```
"As a growth investor with 5-year horizon, analyze TSLA.

Data (Q4 2024):
- Price: $242
- PE: 65 (sector avg: 25)
- Revenue growth: 18% YoY
- Profit margin: 9.2%
- Delivery growth: 12%

Question: Should I buy TSLA at current valuation?

Format:
RECOMMENDATION: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]
REASON: [one sentence]
RISKS: [bullet list]"
```

Response will be structured, specific, and parseable.

---

## System vs User Messages

Understanding the difference between system and user messages is crucial for consistent results.

### System Messages

The **system message** sets the AI's behavior, personality, and constraints. It persists across the conversation.

**Characteristics**:
- Sets the "rules" of engagement
- Defines output format
- Establishes expertise and tone
- Applies to ALL subsequent user messages

**Best for**:
- Trading strategy philosophy
- Output format requirements
- Risk tolerance settings
- Compliance constraints

**Example**:
```python
system_message = """
You are a conservative financial analyst specializing in value investing.

Rules:
1. Never recommend stocks with PE > 25
2. Always consider debt/equity ratio
3. Flag any missing data as "INSUFFICIENT_DATA"
4. Output MUST be valid JSON
5. Confidence scores must reflect uncertainty

Risk tolerance: Low
Investment horizon: 5+ years
Strategy: Value investing (Buffett-style)
"""
```

### User Messages

The **user message** contains the specific request or data for this analysis.

**Characteristics**:
- Contains the actual query
- Provides specific data points
- Changes with each request
- References system context

**Example**:
```python
user_message = f"""
Analyze this stock:

Symbol: {symbol}
Price: ${price}
PE Ratio: {pe_ratio}
Debt/Equity: {debt_equity}
Revenue Growth: {revenue_growth}%
Sector: {sector}

Should I buy this stock today?
"""
```

### Combining System and User

```python
messages = [
    {
        "role": "system",
        "content": system_message  # Set once, applies to all
    },
    {
        "role": "user",
        "content": user_message  # Specific to this stock
    }
]

# For Claude
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system=system_message,  # Separate parameter
    messages=[
        {"role": "user", "content": user_message}
    ]
)
```

### System Message Best Practices

1. **Be specific about output format**:
```python
system = """
ALWAYS respond with valid JSON in this exact format:
{
    "recommendation": "BUY" | "SELL" | "HOLD",
    "confidence": float between 0 and 1,
    "reasoning": string max 100 words,
    "risks": array of strings
}
"""
```

2. **Set clear boundaries**:
```python
system = """
Constraints:
- Only analyze US stocks
- Never recommend penny stocks (price < $5)
- Flag any data over 1 week old as "STALE"
- Admit when you lack information
"""
```

3. **Define personality**:
```python
# Conservative
system = "You are risk-averse and prefer blue-chip stocks."

# Aggressive
system = "You seek high-growth opportunities and accept volatility."

# Balanced
system = "You balance growth potential with downside protection."
```

---

## Few-Shot Learning

**Few-shot learning** means providing examples in your prompt. This dramatically improves output quality.

### Zero-Shot (No Examples)

```python
prompt = "Analyze AAPL stock and give a recommendation."
```

Result: Unpredictable format, may hallucinate data.

### One-Shot (One Example)

```python
prompt = """
Analyze this stock and format like the example:

EXAMPLE:
Input: MSFT, PE=32, Growth=15%, Debt=0.5
Output: {"rec": "BUY", "conf": 0.75, "reason": "Solid growth, low debt"}

NOW ANALYZE:
Input: AAPL, PE=28, Growth=8%, Debt=1.2
Output:
"""
```

Result: Better formatting, but still variable.

### Few-Shot (Multiple Examples)

```python
prompt = """
Analyze the stock following these examples:

Example 1:
Input: MSFT | PE: 32 | Growth: 15% | Debt: 0.5 | Sector PE: 28
Output: {
    "recommendation": "BUY",
    "confidence": 0.80,
    "reasoning": "PE slightly above sector average but strong growth justifies premium. Low debt provides safety.",
    "risks": ["Valuation risk if growth slows", "Regulatory scrutiny"]
}

Example 2:
Input: NFLX | PE: 45 | Growth: 6% | Debt: 15.0 | Sector PE: 22
Output: {
    "recommendation": "SELL",
    "confidence": 0.72,
    "reasoning": "PE double sector average while growth decelerating. High debt increases risk.",
    "risks": ["Subscriber churn", "Debt servicing costs", "Competition"]
}

Example 3:
Input: JNJ | PE: 18 | Growth: 4% | Debt: 0.3 | Sector PE: 20
Output: {
    "recommendation": "HOLD",
    "confidence": 0.65,
    "reasoning": "Fair valuation with steady growth. Low risk but limited upside.",
    "risks": ["Litigation exposure", "Slow growth"]
}

NOW ANALYZE:
Input: {symbol} | PE: {pe} | Growth: {growth}% | Debt: {debt} | Sector PE: {sector_pe}
Output:
"""
```

Result: Consistent format, better reasoning, accurate confidence scores.

### Few-Shot Best Practices

1. **Use 2-5 examples** (diminishing returns after 5)
2. **Cover diverse cases** (BUY, SELL, HOLD)
3. **Show edge cases** (missing data, unclear signals)
4. **Match your actual data format** exactly
5. **Include the reasoning you want** to see

### Few-Shot for Sentiment Analysis

```python
prompt = """
Classify financial news sentiment as: BULLISH, BEARISH, or NEUTRAL.

Examples:

Text: "Apple reports record iPhone sales, beating estimates by 15%"
Sentiment: BULLISH
Confidence: 0.95

Text: "Tesla recalls 2M vehicles due to safety concerns"
Sentiment: BEARISH
Confidence: 0.85

Text: "Microsoft announces new office location in Austin"
Sentiment: NEUTRAL
Confidence: 0.90

Text: "Google revenue meets expectations, stock unchanged"
Sentiment: NEUTRAL
Confidence: 0.75

Now classify:
Text: "{news_text}"
Sentiment:
"""
```

---

## Output Formatting

Getting structured, parseable output is critical for trading systems.

### Strategy 1: JSON Output

**Technique**: Explicitly request JSON and show the schema.

```python
prompt = f"""
Analyze {symbol} and return ONLY valid JSON (no other text):

{{
    "symbol": "string",
    "recommendation": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "target_price": number,
    "reasoning": "string (max 100 words)",
    "key_metrics": {{
        "pe_ratio": number,
        "growth_rate": number,
        "debt_ratio": number
    }},
    "risks": ["string", "string"],
    "catalysts": ["string", "string"]
}}

Stock data:
{json.dumps(stock_data, indent=2)}
"""
```

**Parsing**:
```python
import json

response_text = llm_response.content[0].text

# Remove markdown code blocks if present
if "```json" in response_text:
    response_text = response_text.split("```json")[1].split("```")[0]

# Parse JSON
try:
    result = json.loads(response_text.strip())
except json.JSONDecodeError as e:
    # Fallback: extract JSON with regex
    import re
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
    else:
        raise ValueError("No valid JSON found in response")
```

### Strategy 2: Delimited Output

**Technique**: Use delimiters for easy parsing.

```python
prompt = f"""
Analyze {symbol} and format exactly as shown:

RECOMMENDATION: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]
TARGET: [price]
REASONING: [one sentence]
RISKS: [risk1] | [risk2] | [risk3]
CATALYSTS: [catalyst1] | [catalyst2]

Stock data: {stock_data}
"""
```

**Parsing**:
```python
def parse_delimited_response(text):
    result = {}
    for line in text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'confidence':
                result[key] = float(value) / 100
            elif key in ['risks', 'catalysts']:
                result[key] = [v.strip() for v in value.split('|')]
            else:
                result[key] = value

    return result
```

### Strategy 3: Pydantic Validation

**Technique**: Define a schema and validate.

```python
from pydantic import BaseModel, Field, validator
from typing import List, Literal

class StockAnalysis(BaseModel):
    symbol: str
    recommendation: Literal['BUY', 'SELL', 'HOLD']
    confidence: float = Field(ge=0, le=1)
    target_price: float = Field(gt=0)
    reasoning: str = Field(max_length=200)
    risks: List[str] = Field(min_items=1, max_items=5)
    catalysts: List[str] = Field(min_items=1, max_items=5)

    @validator('reasoning')
    def reasoning_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Reasoning cannot be empty')
        return v

# Parse and validate
response_json = json.loads(llm_response_text)
analysis = StockAnalysis(**response_json)  # Raises ValidationError if invalid
```

### Strategy 4: Force JSON Mode (Groq)

Some APIs support forced JSON output:

```python
response = groq.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=messages,
    response_format={"type": "json_object"}  # Forces valid JSON
)
```

---

## Temperature and Top-P

These parameters control randomness. Getting them right is crucial for trading.

### Temperature

**Controls randomness**: 0 = deterministic, 1 = creative, 2 = chaotic.

**For Trading**:
- **Analysis & Recommendations**: 0.0 - 0.3 (need consistency)
- **Explanations**: 0.3 - 0.5 (some variety is okay)
- **Creative tasks**: 0.7 - 1.0 (don't use for trading)

**Example**:
```python
# Temperature 0.0 - Same input, same output (good for trading)
for i in range(3):
    response = llm.chat(temperature=0.0, messages=[...])
    # All 3 responses will be identical

# Temperature 1.0 - Same input, different outputs (bad for trading)
for i in range(3):
    response = llm.chat(temperature=1.0, messages=[...])
    # All 3 responses will vary significantly
```

### Top-P (Nucleus Sampling)

**Controls diversity**: 0.1 = very focused, 1.0 = consider all options.

**Recommendation**: Use 0.9-0.95 for trading to allow slight variation while staying focused.

```python
# Conservative (recommended for trading)
response = llm.chat(
    temperature=0.2,
    top_p=0.9,
    messages=[...]
)
```

### Temperature Impact Example

Same prompt with different temperatures:

**Temperature 0.0**:
```
Recommendation: HOLD
Confidence: 0.68
Reasoning: PE ratio of 28.5 is 30% above sector average of 22, suggesting overvaluation.
```

**Temperature 0.5**:
```
Recommendation: HOLD
Confidence: 0.72
Reasoning: While PE ratio is elevated, strong revenue growth of 8% provides some justification.
```

**Temperature 1.0**:
```
Recommendation: BUY
Confidence: 0.81
Reasoning: Despite high PE, the company's market position and innovation pipeline justify a premium.
```

Notice how higher temperature changes both the recommendation and reasoning. **For trading, stick to 0.0-0.3**.

---

## Iteration Techniques

Prompts are rarely perfect on the first try. Here's how to improve them systematically.

### The 5-Iteration Method

**Iteration 1: Baseline**
```python
prompt = "Analyze AAPL stock"
```
Result: Too vague, unhelpful.

**Iteration 2: Add Structure**
```python
prompt = """
Analyze AAPL stock:
- Recommendation: BUY/SELL/HOLD
- Confidence: 0-1
- Reasoning: brief
"""
```
Result: Better, but still generic.

**Iteration 3: Add Data**
```python
prompt = """
Analyze AAPL stock with this data:
- PE: 28.5
- Growth: 8%
- Debt: 1.2

Recommendation:
Confidence:
Reasoning:
"""
```
Result: More specific, but hallucinated sector comparison.

**Iteration 4: Add Context**
```python
prompt = """
You are a value investor. Analyze AAPL:

Data:
- PE: 28.5 (Sector avg: 22)
- Growth: 8%
- Debt: 1.2

Rules:
- Only use provided data
- Admit if data is insufficient
- Compare to sector average

Format:
Recommendation: BUY/SELL/HOLD
Confidence: 0.0-1.0
Reasoning: [one sentence]
"""
```
Result: Much better! But format still varies.

**Iteration 5: Add Examples (Few-Shot)**
```python
prompt = """
You are a value investor. Analyze stocks using provided data only.

Example:
Data: MSFT | PE: 32 | Sector: 28 | Growth: 15% | Debt: 0.5
Output: {"rec": "BUY", "conf": 0.75, "reason": "PE premium justified by growth"}

Now analyze:
Data: AAPL | PE: 28.5 | Sector: 22 | Growth: 8% | Debt: 1.2
Output:
"""
```
Result: Consistent, structured, no hallucination.

### A/B Testing Prompts

Test prompts systematically:

```python
import asyncio

async def test_prompts(stock_data, prompt_variants):
    results = []

    for variant in prompt_variants:
        prompt = variant.format(**stock_data)
        response = await llm.chat(prompt)

        results.append({
            "variant": variant_name,
            "response": response,
            "tokens": count_tokens(response),
            "parseable": is_valid_json(response),
            "cost": calculate_cost(prompt, response)
        })

    return results

# Compare
variants = {
    "v1_simple": "Analyze {symbol}",
    "v2_structured": "Analyze {symbol}\nRec:\nConf:\nReason:",
    "v3_fewshot": few_shot_template
}

results = await test_prompts(stock_data, variants)
best = min(results, key=lambda x: x['cost'] if x['parseable'] else float('inf'))
```

### Logging and Analysis

Track what works:

```python
import logging
import json

def log_prompt_performance(prompt, response, metadata):
    logging.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "prompt_hash": hash(prompt),
        "prompt_length": len(prompt),
        "response_length": len(response),
        "parseable": metadata.get("parseable"),
        "confidence": metadata.get("confidence"),
        "cost": metadata.get("cost"),
        "latency_ms": metadata.get("latency_ms")
    }))

# Analyze logs later to find best prompts
```

---

## Common Pitfalls

Avoid these mistakes that cost time and money.

### Pitfall 1: Asking for Real-Time Data

**Bad**:
```python
prompt = "What is the current price of AAPL?"
```

LLMs don't have real-time data. They'll hallucinate or use outdated training data.

**Good**:
```python
prompt = f"AAPL is currently ${current_price}. Given this price and PE of {pe_ratio}, should I buy?"
```

Always provide data to the LLM, never ask it to retrieve data.

### Pitfall 2: Vague Instructions

**Bad**:
```python
prompt = "Analyze this stock and tell me if it's good"
```

"Good" is subjective. Good for what? Day trading? Long-term hold?

**Good**:
```python
prompt = """
As a value investor with 5-year horizon, determine if this stock is
undervalued based on PE ratio, debt levels, and revenue growth.
Recommendation: BUY/SELL/HOLD
"""
```

### Pitfall 3: No Output Constraints

**Bad**:
```python
prompt = "Analyze AAPL and explain your reasoning"
```

Response could be 10 words or 1000 words. Unpredictable cost and parsing.

**Good**:
```python
prompt = "Analyze AAPL. Reasoning must be exactly 2-3 sentences, max 50 words."
```

### Pitfall 4: Not Handling Uncertainty

**Bad**:
```python
# Prompt doesn't allow for "I don't know"
prompt = "What will AAPL stock price be in 6 months?"
```

LLM will hallucinate a confident answer.

**Good**:
```python
prompt = """
Based on current PE of 28.5 and growth of 8%, estimate AAPL price in 6 months.

If insufficient data or too uncertain, respond:
{"recommendation": "INSUFFICIENT_DATA", "confidence": 0.0}
"""
```

### Pitfall 5: Ignoring Token Costs

**Bad**:
```python
# Unnecessarily long prompt
prompt = f"""
{10_000_word_company_history}

Based on all this, should I buy the stock? Yes or no?
"""
```

You just spent $0.30 for a yes/no answer.

**Good**:
```python
# Concise, focused prompt
prompt = f"""
Company: {symbol}
PE: {pe} | Sector: {sector_pe}
Growth: {growth}%

Recommendation: BUY/SELL/HOLD
"""
```

### Pitfall 6: Not Validating Outputs

**Bad**:
```python
response = llm.chat(prompt)
recommendation = response["recommendation"]  # Could crash or be malformed
place_trade(recommendation)
```

**Good**:
```python
response = llm.chat(prompt)

try:
    data = json.loads(response)
    analysis = StockAnalysis(**data)  # Pydantic validation

    if analysis.confidence < 0.6:
        logging.warning("Low confidence, skipping trade")
        return None

    return analysis.recommendation
except (json.JSONDecodeError, ValidationError) as e:
    logging.error(f"Invalid LLM response: {e}")
    return None  # Don't trade on bad data
```

### Pitfall 7: Over-Relying on LLMs

**Bad**:
```python
# Let LLM make final trading decision
if llm.analyze(stock) == "BUY":
    buy_stock(symbol, quantity=1000)
```

LLMs are tools, not oracles.

**Good**:
```python
# LLM provides insight, you make decision
llm_analysis = llm.analyze(stock)
technical_signal = calculate_rsi(stock)
fundamental_score = calculate_piotroski_score(stock)

if (llm_analysis == "BUY" and
    technical_signal == "BULLISH" and
    fundamental_score >= 7):
    buy_stock(symbol, quantity=100)  # Start small
```

---

## Summary and Best Practices

### Quick Reference

```python
# Template for stock analysis prompts
PROMPT_TEMPLATE = """
CONTEXT: {strategy} investor analyzing {sector} stocks
ROLE: Senior analyst with {years} years experience
INSTRUCTION: Determine if {symbol} is {undervalued/overvalued}

DATA:
- Symbol: {symbol}
- PE Ratio: {pe_ratio} (Sector: {sector_pe})
- Revenue Growth: {growth}%
- Debt/Equity: {debt_ratio}
- Price: ${price}

EXAMPLES:
[2-3 few-shot examples showing desired output]

FORMAT:
{{
    "recommendation": "BUY|SELL|HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "max 100 words",
    "risks": ["risk1", "risk2"]
}}

CONSTRAINTS:
- Use only provided data
- If uncertain, set confidence < 0.5
- Explain reasoning briefly
"""
```

### Golden Rules

1. **Be specific**: Vague in = vague out
2. **Provide data**: Never let LLMs hallucinate facts
3. **Use examples**: Few-shot learning is powerful
4. **Set format**: JSON or delimited for easy parsing
5. **Constrain length**: Control costs and focus
6. **Low temperature**: 0.0-0.3 for consistent trading decisions
7. **Validate outputs**: Always use Pydantic or similar
8. **Iterate**: Test and refine prompts systematically
9. **Log everything**: Track what works
10. **Have fallbacks**: LLMs are not 100% reliable

### Cost Optimization

- **Shorter prompts**: Every token costs money
- **Cache system prompts**: Reuse across requests
- **Limit max_tokens**: Don't generate more than you need
- **Use Groq for simple tasks**: 100x cheaper than Claude
- **Batch requests**: Reduce overhead

### Next Steps

1. Complete `starter-code/prompts.py`
2. Test different prompt structures with A/B testing
3. Read `financial_agents.md` to build complete system
4. Practice exercises in `exercises/exercise_2_prompts.py`

**Remember**: Great prompts are the difference between a $50/month API bill and a $5/month bill. Between accurate signals and hallucinated nonsense. Between a robust trading system and a fragile one.

Invest time in prompt engineering—it pays dividends.
