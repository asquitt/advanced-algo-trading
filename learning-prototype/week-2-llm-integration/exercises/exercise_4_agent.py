"""
Exercise 4: Build a Complete Financial Analysis Agent
======================================================

Objective: Combine prompts, LLM calls, parsing, caching, and fallbacks

Time: 2 hours
Difficulty: Advanced

What You'll Learn:
- Design an end-to-end AI agent
- Combine all previous exercises
- Implement caching to reduce costs
- Add fallback strategies for reliability
- Test complex systems
- Handle edge cases

Prerequisites:
- Completed Exercises 1, 2, and 3
- Understanding of async/await (optional but recommended)

Setup:
------
pip install pydantic requests anthropic redis
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timedelta


# ============================================================================
# Schema Definitions (from Exercise 3)
# ============================================================================

class Recommendation(str, Enum):
    """Valid recommendation types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StockAnalysis(BaseModel):
    """Validated stock analysis response."""
    symbol: str
    recommendation: Recommendation
    confidence: int = Field(ge=0, le=100)
    reasoning: str
    risks: List[str]
    price_target: Optional[float] = None
    timestamp: Optional[str] = None  # When analysis was created


# ============================================================================
# TODO #1: Build the Financial Agent
# ============================================================================

class FinancialAgent:
    """
    Complete financial analysis agent that:
    1. Takes stock data
    2. Creates optimized prompt
    3. Calls appropriate LLM (Groq or Claude)
    4. Parses and validates response
    5. Caches results
    6. Handles errors with fallbacks
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hour
        default_model: str = "groq"  # "groq" or "claude"
    ):
        """
        Initialize the financial agent.

        Args:
            groq_api_key: Groq API key (or from env)
            claude_api_key: Claude API key (or from env)
            use_cache: Enable response caching
            cache_ttl: Cache time-to-live in seconds
            default_model: Which model to use by default
        """
        # TODO: Initialize API keys
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")

        # TODO: Initialize configuration
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.default_model = default_model

        # TODO: Initialize cache (simple dict for now)
        # In production, use Redis
        self.cache: Dict[str, tuple[StockAnalysis, float]] = {}

        # TODO: Initialize statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "errors": 0,
            "fallbacks": 0
        }

    def _create_cache_key(self, symbol: str, metrics: Dict[str, float]) -> str:
        """
        Create a unique cache key for this analysis request.

        Args:
            symbol: Stock symbol
            metrics: Financial metrics

        Returns:
            Hash string as cache key

        Hints:
        - Combine symbol and sorted metrics into string
        - Use hashlib.md5 or sha256
        - Ensure deterministic ordering (sort dict keys)
        """
        # TODO: Create deterministic string from inputs
        # cache_input = f"{symbol}:{sorted(metrics.items())}"

        # TODO: Hash it
        # cache_key = hashlib.md5(cache_input.encode()).hexdigest()

        return ""  # Replace with actual implementation

    def _get_from_cache(self, cache_key: str) -> Optional[StockAnalysis]:
        """
        Retrieve analysis from cache if valid.

        Args:
            cache_key: Cache key to lookup

        Returns:
            StockAnalysis if in cache and not expired, None otherwise
        """
        if not self.use_cache:
            return None

        # TODO: Check if key exists in cache
        if cache_key not in self.cache:
            return None

        # TODO: Check if cache entry is still valid (not expired)
        analysis, cached_at = self.cache[cache_key]
        age = time.time() - cached_at

        if age > self.cache_ttl:
            # Expired, remove from cache
            del self.cache[cache_key]
            return None

        # TODO: Update stats and return cached analysis
        self.stats["cache_hits"] += 1
        return analysis

    def _save_to_cache(self, cache_key: str, analysis: StockAnalysis) -> None:
        """Save analysis to cache."""
        if not self.use_cache:
            return

        # TODO: Save to cache with timestamp
        self.cache[cache_key] = (analysis, time.time())

    def _create_prompt(
        self,
        symbol: str,
        metrics: Dict[str, float]
    ) -> tuple[str, str]:
        """
        Create optimized prompt for stock analysis.

        Args:
            symbol: Stock ticker
            metrics: Financial metrics

        Returns:
            Tuple of (system_prompt, user_prompt)

        Hints:
        - Use techniques from Exercise 2
        - Include few-shot examples
        - Request JSON output
        - Specify exact fields needed
        """
        # TODO: Create system prompt
        system_prompt = """You are a professional financial analyst. Analyze stocks and provide recommendations.

You must respond with valid JSON only, in this exact format:
{
    "symbol": "TICKER",
    "recommendation": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "reasoning": "Clear explanation of your recommendation",
    "risks": ["risk1", "risk2", "risk3"],
    "price_target": number or null
}

Example analyses:

Input: MSFT with PE=30, Growth=15%, Margin=35%, Debt/Equity=0.5
Output: {
    "symbol": "MSFT",
    "recommendation": "BUY",
    "confidence": 85,
    "reasoning": "Strong fundamentals with PE below tech average, healthy growth, excellent margins, and low debt",
    "risks": ["Cloud competition", "Regulatory pressure", "Market volatility"],
    "price_target": 420.0
}

Input: XYZ with PE=65, Growth=-5%, Margin=10%, Debt/Equity=3.5
Output: {
    "symbol": "XYZ",
    "recommendation": "SELL",
    "confidence": 90,
    "reasoning": "Severely overvalued with high PE, negative growth, weak margins, and excessive debt burden",
    "risks": ["Bankruptcy risk", "Debt default", "Continued losses"],
    "price_target": null
}
"""

        # TODO: Create user prompt with actual data
        user_prompt = f"""Analyze this stock:

Symbol: {symbol}
Metrics:
{json.dumps(metrics, indent=2)}

Provide your analysis in the exact JSON format specified above.
"""

        return system_prompt, user_prompt

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Groq API.

        Returns:
            Raw response text

        Hints:
        - Similar to Exercise 1
        - Use llama3-8b-8192 model
        - Set temperature to 0.1 for consistency
        """
        # TODO: Implement Groq API call
        # Similar to exercise_1_llm_basics.py
        # Return the text response

        raise NotImplementedError("TODO: Implement Groq API call")

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Claude API.

        Returns:
            Raw response text

        Hints:
        - Use claude-3-haiku-20240307 for cost-effectiveness
        - Set temperature to 0.1
        - Use anthropic library
        """
        # TODO: Implement Claude API call
        # Similar to exercise_1_llm_basics.py
        # Return the text response

        raise NotImplementedError("TODO: Implement Claude API call")

    def _parse_response(self, raw_response: str, symbol: str) -> StockAnalysis:
        """
        Parse and validate LLM response.

        Args:
            raw_response: Raw text from LLM
            symbol: Expected stock symbol

        Returns:
            Validated StockAnalysis object

        Raises:
            ValueError: If parsing fails

        Hints:
        - Use techniques from Exercise 3
        - Extract JSON from markdown
        - Validate with Pydantic
        - Add timestamp
        """
        # TODO: Import and use functions from Exercise 3
        # 1. Extract JSON from markdown
        # 2. Parse JSON
        # 3. Validate with StockAnalysis model
        # 4. Add timestamp

        raise NotImplementedError("TODO: Implement response parsing")

    def analyze_stock(
        self,
        symbol: str,
        metrics: Dict[str, float],
        use_model: Optional[str] = None
    ) -> StockAnalysis:
        """
        Analyze a stock with full error handling and caching.

        This is the main public method that brings everything together!

        Args:
            symbol: Stock ticker symbol
            metrics: Dictionary of financial metrics
            use_model: Override default model ("groq" or "claude")

        Returns:
            StockAnalysis object

        Process:
        1. Check cache
        2. Create prompt
        3. Call LLM
        4. Parse response
        5. Save to cache
        6. Return analysis

        Hints:
        - Update self.stats throughout
        - Use try/except for error handling
        - Implement fallback from Groq to Claude if Groq fails
        """
        self.stats["total_requests"] += 1

        # TODO: Step 1 - Check cache
        cache_key = self._create_cache_key(symbol, metrics)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        self.stats["cache_misses"] += 1

        # TODO: Step 2 - Create prompt
        system_prompt, user_prompt = self._create_prompt(symbol, metrics)

        # TODO: Step 3 - Call LLM (with fallback)
        model = use_model or self.default_model
        raw_response = None

        try:
            if model == "groq":
                # Try Groq first
                raw_response = self._call_groq(system_prompt, user_prompt)
            else:
                # Use Claude
                raw_response = self._call_claude(system_prompt, user_prompt)

            self.stats["api_calls"] += 1

        except Exception as e:
            # TODO: Implement fallback
            # If Groq fails, try Claude
            # If Claude fails, raise error
            self.stats["errors"] += 1

            if model == "groq" and self.claude_api_key:
                print(f"⚠️  Groq failed, falling back to Claude: {e}")
                self.stats["fallbacks"] += 1
                try:
                    raw_response = self._call_claude(system_prompt, user_prompt)
                    self.stats["api_calls"] += 1
                except Exception as e2:
                    raise ValueError(f"Both Groq and Claude failed: {e}, {e2}")
            else:
                raise ValueError(f"API call failed: {e}")

        # TODO: Step 4 - Parse response
        try:
            analysis = self._parse_response(raw_response, symbol)
        except Exception as e:
            self.stats["errors"] += 1
            raise ValueError(f"Failed to parse response: {e}")

        # TODO: Step 5 - Save to cache
        self._save_to_cache(cache_key, analysis)

        # TODO: Step 6 - Return analysis
        return analysis

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary with performance metrics
        """
        cache_hit_rate = (
            self.stats["cache_hits"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0
        )

        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache)
        }


# ============================================================================
# TODO #2: Add batch analysis capability
# ============================================================================

class BatchFinancialAgent(FinancialAgent):
    """
    Enhanced agent that can analyze multiple stocks efficiently.

    Improvements over base agent:
    - Batch processing
    - Parallel API calls (if using async)
    - Cost tracking
    - Detailed reporting
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_per_token = {
            "groq": 0.0000001,  # ~$0.0001 per 1K tokens
            "claude": 0.0000025  # ~$2.50 per 1M tokens
        }
        self.total_tokens = 0
        self.total_cost = 0.0

    def analyze_portfolio(
        self,
        stocks: List[Dict[str, Any]]
    ) -> List[StockAnalysis]:
        """
        Analyze multiple stocks.

        Args:
            stocks: List of dicts with 'symbol' and 'metrics' keys

        Returns:
            List of StockAnalysis objects

        Example:
            stocks = [
                {"symbol": "AAPL", "metrics": {...}},
                {"symbol": "MSFT", "metrics": {...}}
            ]
        """
        # TODO: Implement batch analysis
        # For each stock:
        # 1. Call analyze_stock
        # 2. Track tokens and cost
        # 3. Handle errors gracefully (continue on failure)
        # 4. Return list of successful analyses

        results = []

        for stock_data in stocks:
            try:
                symbol = stock_data["symbol"]
                metrics = stock_data["metrics"]

                analysis = self.analyze_stock(symbol, metrics)
                results.append(analysis)

                # TODO: Update token count and cost
                # Estimate ~500 tokens per analysis
                # self.total_tokens += 500
                # self.total_cost += 500 * self.cost_per_token[self.default_model]

            except Exception as e:
                print(f"❌ Failed to analyze {stock_data.get('symbol', 'unknown')}: {e}")

        return results

    def get_cost_report(self) -> Dict[str, Any]:
        """Get detailed cost analysis."""
        stats = self.get_stats()

        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "api_calls": stats["api_calls"],
            "cache_hits": stats["cache_hits"],
            "cost_saved_by_cache": round(
                stats["cache_hits"] * 500 * self.cost_per_token[self.default_model],
                4
            ),
            "avg_cost_per_request": round(
                self.total_cost / stats["total_requests"] if stats["total_requests"] > 0 else 0,
                6
            )
        }


# ============================================================================
# Test Cases
# ============================================================================

def test_basic_analysis():
    """Test basic stock analysis."""
    print("\n" + "=" * 80)
    print("TEST: Basic Stock Analysis")
    print("=" * 80)

    agent = FinancialAgent(
        use_cache=True,
        cache_ttl=3600,
        default_model="groq"
    )

    metrics = {
        "price": 150.25,
        "pe_ratio": 28.5,
        "revenue_growth": 0.08,
        "profit_margin": 0.25,
        "debt_to_equity": 1.2
    }

    try:
        print(f"Analyzing AAPL...")
        analysis = agent.analyze_stock("AAPL", metrics)

        print(f"\n✅ Analysis complete!")
        print(f"   Symbol: {analysis.symbol}")
        print(f"   Recommendation: {analysis.recommendation}")
        print(f"   Confidence: {analysis.confidence}%")
        print(f"   Reasoning: {analysis.reasoning[:100]}...")
        print(f"   Risks: {analysis.risks}")
        if analysis.price_target:
            print(f"   Price Target: ${analysis.price_target:.2f}")

    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_caching():
    """Test that caching works correctly."""
    print("\n" + "=" * 80)
    print("TEST: Caching")
    print("=" * 80)

    agent = FinancialAgent(use_cache=True, cache_ttl=60)

    metrics = {
        "price": 100.0,
        "pe_ratio": 20.0,
        "revenue_growth": 0.10,
        "profit_margin": 0.20,
        "debt_to_equity": 1.0
    }

    try:
        # First call - should miss cache
        print("First call (cache miss expected)...")
        start = time.time()
        analysis1 = agent.analyze_stock("MSFT", metrics)
        time1 = time.time() - start

        # Second call - should hit cache
        print("Second call (cache hit expected)...")
        start = time.time()
        analysis2 = agent.analyze_stock("MSFT", metrics)
        time2 = time.time() - start

        stats = agent.get_stats()

        print(f"\n✅ Cache test complete!")
        print(f"   First call time: {time1*1000:.2f}ms")
        print(f"   Second call time: {time2*1000:.2f}ms")
        print(f"   Speedup: {time1/time2:.1f}x")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")

        assert stats['cache_hits'] > 0, "Should have at least one cache hit"
        assert time2 < time1, "Cached call should be faster"

    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_batch_analysis():
    """Test batch portfolio analysis."""
    print("\n" + "=" * 80)
    print("TEST: Batch Portfolio Analysis")
    print("=" * 80)

    agent = BatchFinancialAgent(use_cache=True, default_model="groq")

    portfolio = [
        {
            "symbol": "AAPL",
            "metrics": {
                "price": 150.0,
                "pe_ratio": 28.5,
                "revenue_growth": 0.08,
                "profit_margin": 0.25,
                "debt_to_equity": 1.2
            }
        },
        {
            "symbol": "MSFT",
            "metrics": {
                "price": 380.0,
                "pe_ratio": 32.0,
                "revenue_growth": 0.12,
                "profit_margin": 0.35,
                "debt_to_equity": 0.5
            }
        },
        {
            "symbol": "GOOGL",
            "metrics": {
                "price": 140.0,
                "pe_ratio": 25.0,
                "revenue_growth": 0.10,
                "profit_margin": 0.28,
                "debt_to_equity": 0.3
            }
        }
    ]

    try:
        print(f"Analyzing portfolio of {len(portfolio)} stocks...")
        results = agent.analyze_portfolio(portfolio)

        print(f"\n✅ Batch analysis complete!")
        print(f"\nResults:")
        for analysis in results:
            print(f"   {analysis.symbol}: {analysis.recommendation} "
                  f"(confidence: {analysis.confidence}%)")

        cost_report = agent.get_cost_report()
        print(f"\nCost Report:")
        print(f"   Total API calls: {cost_report['api_calls']}")
        print(f"   Total tokens: {cost_report['total_tokens']}")
        print(f"   Total cost: ${cost_report['total_cost_usd']}")
        print(f"   Cost saved by cache: ${cost_report['cost_saved_by_cache']}")
        print(f"   Avg cost per request: ${cost_report['avg_cost_per_request']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_fallback():
    """Test fallback from Groq to Claude."""
    print("\n" + "=" * 80)
    print("TEST: Fallback Strategy")
    print("=" * 80)

    # Create agent with invalid Groq key to force fallback
    agent = FinancialAgent(
        groq_api_key="invalid_key",
        default_model="groq"
    )

    metrics = {
        "price": 75.0,
        "pe_ratio": 18.0,
        "revenue_growth": 0.15,
        "profit_margin": 0.22,
        "debt_to_equity": 0.9
    }

    try:
        print("Attempting analysis with invalid Groq key...")
        print("(Should fallback to Claude if Claude key is valid)")

        analysis = agent.analyze_stock("TSLA", metrics)

        stats = agent.get_stats()
        print(f"\n✅ Fallback test complete!")
        print(f"   Fallbacks triggered: {stats['fallbacks']}")
        print(f"   Final recommendation: {analysis.recommendation}")

    except Exception as e:
        print(f"Expected behavior - both APIs unavailable: {e}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║         Exercise 4: Complete Financial Analysis Agent             ║
    ║                                                                    ║
    ║  Build a production-ready agent with all the features!            ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Check API keys
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))

    print(f"\nAPI Keys Status:")
    print(f"   Groq: {'✅' if has_groq else '❌'}")
    print(f"   Claude: {'✅' if has_claude else '❌'}")

    if not has_groq and not has_claude:
        print("\n⚠️  Warning: No API keys found. Tests will fail.")
        print("   Set GROQ_API_KEY and/or ANTHROPIC_API_KEY environment variables")

    print("\nRunning tests...\n")

    # Run tests
    test_basic_analysis()
    test_caching()
    test_batch_analysis()
    test_fallback()

    print("\n" + "=" * 80)
    print("EXERCISE COMPLETE!")
    print("=" * 80)
    print("""
    Congratulations! You've built a production-ready financial agent!

    Your agent includes:
    ✅ Smart prompt engineering
    ✅ Multi-model support (Groq + Claude)
    ✅ Response parsing and validation
    ✅ Caching for cost reduction
    ✅ Fallback strategies for reliability
    ✅ Batch processing capabilities
    ✅ Cost tracking and reporting
    ✅ Comprehensive error handling

    Architecture Summary:
    ┌─────────────────────────────────────────────────────┐
    │  Stock Data → Prompt → LLM → Parse → Validate → ✓  │
    │                ↑                ↓                    │
    │              Cache          Fallback                │
    └─────────────────────────────────────────────────────┘

    Production Checklist:
    - [ ] All tests passing
    - [ ] Cache hit rate > 50%
    - [ ] Error handling for all failure modes
    - [ ] Cost tracking implemented
    - [ ] Monitoring and logging added
    - [ ] Rate limit handling
    - [ ] Async support for high volume

    Next: Exercise 5 - Cost Optimization
    """)
