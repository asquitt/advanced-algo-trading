# Building Financial Agents with LLMs

**Time to Read**: 20-30 minutes | **Difficulty**: Intermediate-Advanced

---

## Table of Contents

1. [What is a Financial Agent?](#what-is-a-financial-agent)
2. [Agent Architecture](#agent-architecture)
3. [Analysis Pipeline](#analysis-pipeline)
4. [Response Validation](#response-validation)
5. [Error Handling](#error-handling)
6. [Testing Strategies](#testing-strategies)
7. [Real-World Examples](#real-world-examples)

---

## What is a Financial Agent?

A **financial agent** is an autonomous system that uses LLMs to analyze data, make decisions, and execute actions with minimal human intervention.

### Traditional vs LLM-Powered Analysis

**Traditional Approach**:
```python
def analyze_stock(symbol, data):
    if data['pe_ratio'] < 15 and data['growth'] > 10:
        return "BUY"
    elif data['pe_ratio'] > 30 or data['growth'] < 0:
        return "SELL"
    else:
        return "HOLD"
```

Limitations:
- Rigid rules don't adapt to market conditions
- Can't process unstructured data (news, sentiment)
- No nuanced reasoning
- Doesn't explain decisions

**LLM Agent Approach**:
```python
class FinancialAgent:
    async def analyze_stock(self, symbol, data):
        # Gather data from multiple sources
        fundamentals = await self.get_fundamentals(symbol)
        news = await self.get_recent_news(symbol)
        sentiment = await self.analyze_sentiment(news)

        # Build context-aware prompt
        prompt = self.build_prompt(symbol, fundamentals, sentiment)

        # Get LLM analysis
        analysis = await self.llm.analyze(prompt)

        # Validate and parse
        validated = self.validate_response(analysis)

        # Log for auditing
        self.log_decision(symbol, validated)

        return validated
```

Benefits:
- Processes structured + unstructured data
- Adapts reasoning to specific situations
- Provides explainable decisions
- Handles edge cases gracefully

### Key Components of a Financial Agent

1. **Data Gathering**: Fetch relevant information
2. **Prompt Engineering**: Build context-aware prompts
3. **LLM Integration**: Call appropriate models
4. **Response Parsing**: Extract structured data
5. **Validation**: Ensure output quality
6. **Error Handling**: Graceful fallbacks
7. **Logging**: Audit trail for compliance
8. **Caching**: Reduce costs and latency

---

## Agent Architecture

A well-designed agent separates concerns and is easy to test and extend.

### Layered Architecture

```
┌─────────────────────────────────────┐
│      API Layer (FastAPI)            │
│  - Endpoints for analysis           │
│  - Authentication                   │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      Agent Layer                    │
│  - FinancialAgent class             │
│  - Orchestration logic              │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      Service Layer                  │
│  - LLMClient (Groq/Claude)         │
│  - DataFetcher (APIs)              │
│  - CacheService (Redis)            │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      Data Layer                     │
│  - Database (PostgreSQL)            │
│  - Cache (Redis)                    │
└─────────────────────────────────────┘
```

### Code Structure

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class StockAnalysis(BaseModel):
    """Structured output from agent"""
    symbol: str
    recommendation: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: str
    risks: list[str]
    catalysts: list[str]
    target_price: Optional[float] = None
    metadata: Dict[str, Any] = {}


class BaseLLMClient(ABC):
    """Abstract base for LLM clients"""

    @abstractmethod
    async def chat(self, messages: list, **kwargs) -> str:
        pass


class FinancialAgent:
    """Main agent orchestrating analysis"""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        cache_service: Optional['CacheService'] = None,
        data_fetcher: Optional['DataFetcher'] = None
    ):
        self.llm = llm_client
        self.cache = cache_service
        self.data_fetcher = data_fetcher
        self.prompt_builder = PromptBuilder()

    async def analyze_stock(
        self,
        symbol: str,
        user_data: Optional[Dict] = None
    ) -> StockAnalysis:
        """
        Main entry point for stock analysis.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            user_data: Optional user-provided data

        Returns:
            StockAnalysis object with recommendation
        """
        try:
            # Check cache first
            cached = await self._get_cached_analysis(symbol)
            if cached:
                logger.info(f"Cache hit for {symbol}")
                return cached

            # Gather data
            data = await self._gather_data(symbol, user_data)

            # Build prompt
            prompt = self.prompt_builder.build_analysis_prompt(
                symbol, data
            )

            # Get LLM analysis
            response = await self.llm.chat(prompt)

            # Parse and validate
            analysis = self._parse_response(response, symbol)

            # Cache result
            await self._cache_analysis(symbol, analysis)

            # Log for auditing
            self._log_analysis(symbol, analysis)

            return analysis

        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return await self._fallback_analysis(symbol, user_data)

    async def _gather_data(
        self,
        symbol: str,
        user_data: Optional[Dict]
    ) -> Dict:
        """Gather all relevant data for analysis"""
        # Use user data if provided, else fetch from APIs
        if user_data:
            return user_data

        if not self.data_fetcher:
            raise ValueError("No data provided and no data fetcher configured")

        # Fetch from multiple sources
        fundamentals = await self.data_fetcher.get_fundamentals(symbol)
        news = await self.data_fetcher.get_recent_news(symbol)
        price_data = await self.data_fetcher.get_price_data(symbol)

        return {
            "fundamentals": fundamentals,
            "news": news,
            "price_data": price_data
        }

    def _parse_response(self, response: str, symbol: str) -> StockAnalysis:
        """Parse LLM response into structured format"""
        import json

        # Extract JSON from response
        try:
            # Try direct JSON parse
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            import re
            json_match = re.search(
                r'```json\s*(\{.*?\})\s*```',
                response,
                re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                raise ValueError("Could not extract JSON from response")

        # Validate with Pydantic
        data['symbol'] = symbol
        return StockAnalysis(**data)

    async def _get_cached_analysis(
        self,
        symbol: str
    ) -> Optional[StockAnalysis]:
        """Get cached analysis if available"""
        if not self.cache:
            return None

        cached = await self.cache.get(f"analysis:{symbol}")
        if cached:
            return StockAnalysis(**cached)
        return None

    async def _cache_analysis(
        self,
        symbol: str,
        analysis: StockAnalysis
    ):
        """Cache analysis result"""
        if self.cache:
            await self.cache.set(
                f"analysis:{symbol}",
                analysis.model_dump(),
                ttl=3600  # 1 hour
            )

    def _log_analysis(self, symbol: str, analysis: StockAnalysis):
        """Log analysis for auditing"""
        logger.info({
            "event": "stock_analysis",
            "symbol": symbol,
            "recommendation": analysis.recommendation,
            "confidence": analysis.confidence,
            "timestamp": datetime.now().isoformat()
        })

    async def _fallback_analysis(
        self,
        symbol: str,
        user_data: Optional[Dict]
    ) -> StockAnalysis:
        """Fallback when LLM fails"""
        logger.warning(f"Using fallback analysis for {symbol}")

        # Simple rule-based fallback
        return StockAnalysis(
            symbol=symbol,
            recommendation="HOLD",
            confidence=0.3,
            reasoning="LLM analysis unavailable, defaulting to HOLD",
            risks=["Analysis system temporarily unavailable"],
            catalysts=[],
            metadata={"fallback": True}
        )
```

---

## Analysis Pipeline

A robust pipeline handles data flow from input to actionable output.

### Pipeline Stages

```python
class AnalysisPipeline:
    """Multi-stage analysis pipeline"""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def run(self, symbol: str, data: Dict) -> StockAnalysis:
        """Execute full pipeline"""

        # Stage 1: Data validation
        validated_data = self._validate_input_data(data)

        # Stage 2: Fundamental analysis
        fundamental_score = await self._analyze_fundamentals(
            symbol, validated_data
        )

        # Stage 3: Sentiment analysis
        sentiment_score = await self._analyze_sentiment(
            symbol, validated_data.get('news', [])
        )

        # Stage 4: Technical analysis (optional)
        technical_score = await self._analyze_technical(
            symbol, validated_data.get('price_data')
        )

        # Stage 5: Combine signals
        final_analysis = await self._synthesize_analysis(
            symbol,
            fundamental_score,
            sentiment_score,
            technical_score
        )

        return final_analysis

    def _validate_input_data(self, data: Dict) -> Dict:
        """Ensure data quality"""
        required_fields = ['pe_ratio', 'revenue_growth', 'debt_ratio']

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

            if data[field] is None:
                raise ValueError(f"Field {field} cannot be None")

        return data

    async def _analyze_fundamentals(
        self,
        symbol: str,
        data: Dict
    ) -> Dict:
        """Analyze fundamental metrics"""
        prompt = f"""
        Analyze fundamental metrics for {symbol}:

        - PE Ratio: {data['pe_ratio']}
        - Revenue Growth: {data['revenue_growth']}%
        - Debt/Equity: {data['debt_ratio']}
        - Profit Margin: {data.get('profit_margin', 'N/A')}

        Return JSON:
        {{
            "score": 0-100,
            "strengths": ["..."],
            "weaknesses": ["..."]
        }}
        """

        response = await self.llm.chat(prompt, temperature=0.2)
        return json.loads(response)

    async def _analyze_sentiment(
        self,
        symbol: str,
        news_items: list
    ) -> Dict:
        """Analyze news sentiment"""
        if not news_items:
            return {"score": 50, "sentiment": "NEUTRAL"}

        # Use fast Groq for sentiment
        sentiments = []
        for article in news_items[:5]:  # Limit to recent 5
            prompt = f"""
            Classify sentiment (BULLISH/BEARISH/NEUTRAL):
            {article['headline']}

            Return JSON: {{"sentiment": "...", "confidence": 0.0-1.0}}
            """
            response = await self.llm.chat(prompt, temperature=0.0)
            sentiments.append(json.loads(response))

        # Aggregate
        bullish_count = sum(1 for s in sentiments if s['sentiment'] == 'BULLISH')
        bearish_count = sum(1 for s in sentiments if s['sentiment'] == 'BEARISH')

        if bullish_count > bearish_count:
            sentiment = "BULLISH"
            score = 50 + (bullish_count / len(sentiments)) * 50
        elif bearish_count > bullish_count:
            sentiment = "BEARISH"
            score = 50 - (bearish_count / len(sentiments)) * 50
        else:
            sentiment = "NEUTRAL"
            score = 50

        return {"score": score, "sentiment": sentiment}

    async def _analyze_technical(
        self,
        symbol: str,
        price_data: Optional[Dict]
    ) -> Dict:
        """Analyze technical indicators"""
        if not price_data:
            return {"score": 50, "signal": "NEUTRAL"}

        # Calculate basic indicators
        rsi = self._calculate_rsi(price_data['prices'])
        sma_50 = self._calculate_sma(price_data['prices'], 50)
        sma_200 = self._calculate_sma(price_data['prices'], 200)

        # Simple scoring
        score = 50
        if rsi < 30:
            score += 20  # Oversold
        elif rsi > 70:
            score -= 20  # Overbought

        if sma_50 > sma_200:
            score += 10  # Golden cross territory
        else:
            score -= 10  # Death cross territory

        return {"score": score, "rsi": rsi}

    async def _synthesize_analysis(
        self,
        symbol: str,
        fundamental: Dict,
        sentiment: Dict,
        technical: Dict
    ) -> StockAnalysis:
        """Combine all signals into final recommendation"""

        # Weighted average (customize weights as needed)
        total_score = (
            fundamental['score'] * 0.5 +
            sentiment['score'] * 0.3 +
            technical['score'] * 0.2
        )

        # Determine recommendation
        if total_score >= 70:
            recommendation = "BUY"
            confidence = min(total_score / 100, 0.95)
        elif total_score <= 40:
            recommendation = "SELL"
            confidence = min((100 - total_score) / 100, 0.95)
        else:
            recommendation = "HOLD"
            confidence = 0.6

        # Build reasoning
        reasoning = f"""
        Fundamental score: {fundamental['score']}/100.
        Sentiment: {sentiment['sentiment']}.
        Technical score: {technical['score']}/100.
        Overall: {recommendation} with {confidence:.0%} confidence.
        """

        return StockAnalysis(
            symbol=symbol,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning.strip(),
            risks=fundamental.get('weaknesses', []),
            catalysts=fundamental.get('strengths', []),
            metadata={
                "fundamental_score": fundamental['score'],
                "sentiment_score": sentiment['score'],
                "technical_score": technical['score']
            }
        )

    def _calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate RSI indicator"""
        # Simplified RSI calculation
        if len(prices) < period:
            return 50

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_sma(self, prices: list, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        return sum(prices[-period:]) / period
```

---

## Response Validation

Never trust LLM outputs blindly. Validate everything.

### Validation Layers

```python
from pydantic import BaseModel, validator, Field
from typing import Literal

class StockAnalysis(BaseModel):
    """Validated stock analysis"""

    symbol: str = Field(..., regex=r'^[A-Z]{1,5}$')
    recommendation: Literal['BUY', 'SELL', 'HOLD']
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10, max_length=500)
    risks: list[str] = Field(min_items=1, max_items=10)
    catalysts: list[str] = Field(min_items=0, max_items=10)
    target_price: Optional[float] = Field(default=None, gt=0)

    @validator('symbol')
    def symbol_uppercase(cls, v):
        return v.upper()

    @validator('reasoning')
    def reasoning_not_generic(cls, v):
        # Prevent generic/lazy responses
        forbidden = [
            'based on the data',
            'according to analysis',
            'it depends'
        ]
        v_lower = v.lower()
        for phrase in forbidden:
            if phrase in v_lower:
                raise ValueError(f'Reasoning too generic: contains "{phrase}"')
        return v

    @validator('risks')
    def risks_not_empty(cls, v):
        if not v or all(not r.strip() for r in v):
            raise ValueError('At least one meaningful risk required')
        return [r.strip() for r in v if r.strip()]

    @validator('confidence')
    def confidence_reasonable(cls, v, values):
        # Lower confidence if recommendation is HOLD
        if values.get('recommendation') == 'HOLD' and v > 0.8:
            raise ValueError('HOLD recommendation should have lower confidence')
        return v


class ResponseValidator:
    """Additional validation logic"""

    @staticmethod
    def validate_response(response: str, symbol: str) -> StockAnalysis:
        """Parse and validate LLM response"""

        # Extract JSON
        data = ResponseValidator._extract_json(response)

        # Add symbol if missing
        if 'symbol' not in data:
            data['symbol'] = symbol

        # Validate with Pydantic
        try:
            analysis = StockAnalysis(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid LLM response: {e}")

        # Additional business logic validation
        ResponseValidator._validate_business_rules(analysis)

        return analysis

    @staticmethod
    def _extract_json(response: str) -> Dict:
        """Extract JSON from various response formats"""
        import json
        import re

        # Try direct parse
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try markdown code block
        json_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```',
            response,
            re.DOTALL
        )
        if json_match:
            return json.loads(json_match.group(1))

        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        raise ValueError("No valid JSON found in response")

    @staticmethod
    def _validate_business_rules(analysis: StockAnalysis):
        """Validate business logic"""

        # Rule: BUY recommendation should have target price
        if analysis.recommendation == "BUY" and not analysis.target_price:
            logger.warning(f"BUY recommendation for {analysis.symbol} without target price")

        # Rule: High confidence requires multiple risks identified
        if analysis.confidence > 0.8 and len(analysis.risks) < 2:
            raise ValueError("High confidence requires identifying multiple risks")

        # Rule: Confidence should match reasoning strength
        if analysis.confidence > 0.9 and len(analysis.reasoning) < 50:
            raise ValueError("Very high confidence requires detailed reasoning")
```

---

## Error Handling

Robust error handling prevents system failures and provides graceful degradation.

### Error Categories

```python
class LLMError(Exception):
    """Base exception for LLM errors"""
    pass

class RateLimitError(LLMError):
    """API rate limit exceeded"""
    pass

class InvalidResponseError(LLMError):
    """LLM returned invalid/unparseable response"""
    pass

class InsufficientDataError(LLMError):
    """Not enough data to perform analysis"""
    pass


class ErrorHandler:
    """Centralized error handling"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def with_retry(self, func, *args, **kwargs):
        """Retry logic with exponential backoff"""
        import asyncio
        import random

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)

            except RateLimitError as e:
                last_exception = e
                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)

            except InvalidResponseError as e:
                last_exception = e
                # Try with different temperature
                kwargs['temperature'] = kwargs.get('temperature', 0.3) - 0.1
                logger.warning(f"Invalid response, retrying with lower temperature")

            except Exception as e:
                # Don't retry on unexpected errors
                logger.error(f"Unexpected error: {e}")
                raise

        raise last_exception

    @staticmethod
    def handle_llm_error(e: Exception, symbol: str) -> StockAnalysis:
        """Provide fallback analysis on error"""

        if isinstance(e, InsufficientDataError):
            return StockAnalysis(
                symbol=symbol,
                recommendation="HOLD",
                confidence=0.0,
                reasoning="Insufficient data for analysis",
                risks=["Incomplete data"],
                catalysts=[],
                metadata={"error": "insufficient_data"}
            )

        elif isinstance(e, RateLimitError):
            logger.error(f"Rate limit exceeded for {symbol}")
            return StockAnalysis(
                symbol=symbol,
                recommendation="HOLD",
                confidence=0.0,
                reasoning="Analysis temporarily unavailable (rate limit)",
                risks=["System temporarily unavailable"],
                catalysts=[],
                metadata={"error": "rate_limit"}
            )

        else:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return StockAnalysis(
                symbol=symbol,
                recommendation="HOLD",
                confidence=0.0,
                reasoning="Analysis failed due to technical error",
                risks=["System error"],
                catalysts=[],
                metadata={"error": str(e)}
            )
```

---

## Testing Strategies

Comprehensive testing ensures reliability without breaking the bank.

### Unit Tests with Mocks

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    client = AsyncMock()
    client.chat.return_value = {
        "recommendation": "BUY",
        "confidence": 0.75,
        "reasoning": "Strong fundamentals with reasonable valuation",
        "risks": ["Market volatility", "Competition"],
        "catalysts": ["Product launch", "Market expansion"]
    }
    return client

@pytest.mark.asyncio
async def test_analyze_stock_success(mock_llm_client):
    """Test successful stock analysis"""
    agent = FinancialAgent(llm_client=mock_llm_client)

    result = await agent.analyze_stock("AAPL", {
        "pe_ratio": 28.5,
        "revenue_growth": 8.0,
        "debt_ratio": 1.2
    })

    assert result.symbol == "AAPL"
    assert result.recommendation == "BUY"
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.risks) >= 1

@pytest.mark.asyncio
async def test_analyze_stock_invalid_response(mock_llm_client):
    """Test handling of invalid LLM response"""
    mock_llm_client.chat.return_value = "This is not JSON"

    agent = FinancialAgent(llm_client=mock_llm_client)

    # Should use fallback
    result = await agent.analyze_stock("AAPL", {...})
    assert result.recommendation == "HOLD"
    assert result.confidence < 0.5

@pytest.mark.asyncio
async def test_analyze_stock_with_caching(mock_llm_client):
    """Test caching works correctly"""
    cache = InMemoryCache()
    agent = FinancialAgent(llm_client=mock_llm_client, cache_service=cache)

    # First call - should hit LLM
    result1 = await agent.analyze_stock("AAPL", {...})
    assert mock_llm_client.chat.call_count == 1

    # Second call - should hit cache
    result2 = await agent.analyze_stock("AAPL", {...})
    assert mock_llm_client.chat.call_count == 1  # Still 1, not 2

    assert result1.model_dump() == result2.model_dump()
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_llm_analysis():
    """Test with real LLM (costs money, run sparingly)"""
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        pytest.skip("No API key available")

    client = GroqClient(api_key=api_key)
    agent = FinancialAgent(llm_client=client)

    result = await agent.analyze_stock("AAPL", {
        "pe_ratio": 28.5,
        "revenue_growth": 8.0,
        "debt_ratio": 1.2
    })

    # Verify result structure
    assert result.symbol == "AAPL"
    assert result.recommendation in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 0
```

### Response Fixtures

```python
# tests/fixtures/llm_responses.py

BUY_RESPONSE = """
{
    "recommendation": "BUY",
    "confidence": 0.82,
    "reasoning": "PE ratio of 28.5 is reasonable given 8% revenue growth. Debt level is manageable.",
    "risks": ["Market downturn", "Competition from rivals", "Regulatory changes"],
    "catalysts": ["New product launch", "Market expansion", "Cost optimization"],
    "target_price": 185.50
}
"""

SELL_RESPONSE = """
{
    "recommendation": "SELL",
    "confidence": 0.75,
    "reasoning": "PE ratio too high relative to slowing growth. Elevated debt raises concerns.",
    "risks": ["Earnings miss", "Debt refinancing", "Market volatility"],
    "catalysts": [],
    "target_price": 120.00
}
"""

HOLD_RESPONSE = """
{
    "recommendation": "HOLD",
    "confidence": 0.60,
    "reasoning": "Mixed signals. Valuation fair but growth uncertain. Wait for more clarity.",
    "risks": ["Unclear growth trajectory", "Competitive pressure"],
    "catalysts": ["Potential acquisition", "Market recovery"]
}
"""
```

---

## Real-World Examples

### Example 1: Multi-Model Agent

```python
class MultiModelAgent:
    """Use Groq for speed, Claude for quality"""

    def __init__(self):
        self.groq = GroqClient()
        self.claude = ClaudeClient()

    async def quick_analysis(self, symbol: str, data: Dict) -> str:
        """Fast sentiment check with Groq"""
        prompt = f"Quick take on {symbol}: {data}. BULLISH/BEARISH/NEUTRAL?"
        return await self.groq.chat(prompt, max_tokens=10)

    async def deep_analysis(self, symbol: str, data: Dict) -> StockAnalysis:
        """Thorough analysis with Claude"""
        prompt = build_detailed_prompt(symbol, data)
        response = await self.claude.chat(prompt)
        return parse_response(response)

    async def analyze(self, symbol: str, data: Dict) -> StockAnalysis:
        """Smart routing based on complexity"""

        # Quick check first (Groq - $0.0001)
        quick_take = await self.quick_analysis(symbol, data)

        if quick_take == "NEUTRAL" or data.get("complex"):
            # Needs deeper analysis (Claude - $0.01)
            return await self.deep_analysis(symbol, data)
        else:
            # Simple case, Groq is enough
            return await self.simple_analysis(symbol, data)
```

### Example 2: Compliance-Aware Agent

```python
class ComplianceAgent(FinancialAgent):
    """Agent with compliance checks"""

    RESTRICTED_SYMBOLS = ["GME", "AMC"]  # Example

    async def analyze_stock(self, symbol: str, data: Dict) -> StockAnalysis:
        # Pre-flight compliance check
        if symbol in self.RESTRICTED_SYMBOLS:
            return StockAnalysis(
                symbol=symbol,
                recommendation="HOLD",
                confidence=0.0,
                reasoning="Restricted symbol - compliance restriction",
                risks=["Compliance restriction"],
                catalysts=[],
                metadata={"restricted": True}
            )

        # Regular analysis
        analysis = await super().analyze_stock(symbol, data)

        # Post-analysis compliance check
        if analysis.recommendation == "BUY":
            if await self._check_position_limits(symbol):
                logger.warning(f"Position limit reached for {symbol}")
                analysis.recommendation = "HOLD"
                analysis.reasoning += " [Position limit reached]"

        return analysis
```

---

## Summary

### Key Takeaways

1. **Agents orchestrate** multiple components (data, LLM, validation, caching)
2. **Layered architecture** separates concerns and improves testability
3. **Always validate** LLM responses with Pydantic and business rules
4. **Error handling** with retries and fallbacks prevents system failures
5. **Test with mocks** to avoid API costs during development
6. **Use multiple models** strategically (Groq for speed, Claude for quality)

### Agent Checklist

- [ ] Clear separation of concerns (layers)
- [ ] Pydantic validation for all outputs
- [ ] Retry logic with exponential backoff
- [ ] Fallback analysis when LLM fails
- [ ] Caching to reduce costs
- [ ] Comprehensive logging for auditing
- [ ] Unit tests with mocks
- [ ] Integration tests (run sparingly)
- [ ] Error handling for all failure modes
- [ ] Compliance checks if needed

### Next Steps

1. Complete `starter-code/financial_agent.py`
2. Implement your own validation rules
3. Add caching (read `caching_strategies.md`)
4. Write comprehensive tests
5. Deploy and monitor in production

A well-built agent is reliable, cost-effective, and maintainable. Invest time in architecture now, reap benefits later.
