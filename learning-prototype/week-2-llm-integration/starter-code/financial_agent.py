"""
Financial Analysis Agent

This module implements a complete AI agent for financial analysis that:
- Combines LLM clients with prompt engineering
- Parses LLM responses into structured data
- Implements caching to reduce costs
- Handles errors with fallback strategies
- Integrates with trading APIs

Learning Goals:
- Build production-ready AI agents
- Parse and validate LLM responses
- Implement robust error handling
- Optimize costs with caching
- Test AI components effectively
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import re

# TODO: Import from your other modules
# from llm_client import GroqClient, ClaudeClient, LLMProvider, LLMResponse
# from prompts import PromptBuilder, create_stock_analysis_prompt
# from cache import RedisCache, cache_response


# =============================================================================
# TODO #1: Create agent class
# =============================================================================
# Instructions:
# 1. Create FinancialAgent class
# 2. Initialize with LLM clients (Groq and Claude)
# 3. Add configuration for model selection
# 4. Implement health check method
# 5. Add request logging and metrics
#
# The agent should:
# - Support multiple LLM providers
# - Select appropriate model based on task complexity
# - Track usage and costs
# - Provide async interface
# =============================================================================


class TaskComplexity(Enum):
    """Complexity level for choosing appropriate LLM"""
    SIMPLE = "simple"      # Use Groq (fast, cheap)
    MODERATE = "moderate"  # Use Claude Haiku (balanced)
    COMPLEX = "complex"    # Use Claude Sonnet (best quality)


class Recommendation(Enum):
    """Stock recommendation types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class StockMetrics:
    """Input metrics for stock analysis"""
    symbol: str
    price: float
    pe_ratio: float
    revenue_growth: float  # Percentage
    profit_margin: float   # Percentage
    debt_equity: float
    market_cap: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AnalysisResult:
    """Structured output from stock analysis"""
    symbol: str
    recommendation: Recommendation
    confidence: float  # 0.0 to 1.0
    reasoning: str
    key_metrics: List[str]
    risks: List[str]
    score: int  # 0 to 100
    timestamp: datetime
    provider: str
    model: str
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['recommendation'] = self.recommendation.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class FinancialAgent:
    """
    AI-powered financial analysis agent.

    This agent uses LLMs to analyze stocks and provide trading recommendations.
    It intelligently selects between Groq (fast/cheap) and Claude (high-quality)
    based on task complexity.

    Example:
        agent = FinancialAgent()
        result = await agent.analyze_stock("AAPL", metrics)
        print(f"Recommendation: {result.recommendation}")
        print(f"Confidence: {result.confidence}")
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        default_complexity: TaskComplexity = TaskComplexity.SIMPLE,
        enable_cache: bool = True,
        debug: bool = False
    ):
        """
        Initialize the financial agent.

        TODO: Implement initialization
        Steps:
        1. Get API keys from parameters or environment
        2. Initialize Groq and Claude clients
        3. Store configuration
        4. Initialize metrics tracking
        5. Setup cache if enabled

        Args:
            groq_api_key: Groq API key (or use GROQ_API_KEY env var)
            claude_api_key: Claude API key (or use ANTHROPIC_API_KEY env var)
            default_complexity: Default task complexity for model selection
            enable_cache: Whether to cache responses
            debug: Enable debug logging
        """
        # TODO: Implement initialization
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # self.debug = debug
        # self.default_complexity = default_complexity
        # self.enable_cache = enable_cache
        #
        # # Initialize LLM clients
        # groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        # claude_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        #
        # if groq_key:
        #     self.groq_client = GroqClient(api_key=groq_key, debug=debug)
        # else:
        #     self.groq_client = None
        #     self._log("Groq client not initialized - no API key", "WARNING")
        #
        # if claude_key:
        #     self.claude_client = ClaudeClient(api_key=claude_key, debug=debug)
        # else:
        #     self.claude_client = None
        #     self._log("Claude client not initialized - no API key", "WARNING")
        #
        # # Initialize metrics
        # self.metrics = {
        #     "total_requests": 0,
        #     "cache_hits": 0,
        #     "groq_requests": 0,
        #     "claude_requests": 0,
        #     "errors": 0,
        #     "total_tokens": 0,
        # }
        #
        # # Initialize cache
        # if enable_cache:
        #     self.cache = RedisCache()
        # else:
        #     self.cache = None

    def _log(self, message: str, level: str = "INFO"):
        """Log a message if debug is enabled"""
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def _select_model(
        self,
        complexity: Optional[TaskComplexity] = None
    ) -> tuple[Any, str]:
        """
        Select appropriate LLM client based on task complexity.

        TODO: Implement model selection
        Steps:
        1. Use provided complexity or default
        2. For SIMPLE: return Groq if available
        3. For MODERATE: return Claude Haiku if available
        4. For COMPLEX: return Claude Sonnet if available
        5. Fallback to available client if first choice unavailable

        Args:
            complexity: Task complexity level

        Returns:
            Tuple of (client, model_name)

        Raises:
            ValueError: If no clients are available
        """
        # TODO: Implement model selection logic
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # complexity = complexity or self.default_complexity
        #
        # # Simple tasks: use Groq for speed and cost
        # if complexity == TaskComplexity.SIMPLE:
        #     if self.groq_client:
        #         return self.groq_client, "groq"
        #     elif self.claude_client:
        #         return self.claude_client, "claude-haiku"
        #
        # # Moderate tasks: use Claude Haiku for balance
        # elif complexity == TaskComplexity.MODERATE:
        #     if self.claude_client:
        #         return self.claude_client, "claude-haiku"
        #     elif self.groq_client:
        #         return self.groq_client, "groq"
        #
        # # Complex tasks: use Claude Sonnet for quality
        # else:  # COMPLEX
        #     if self.claude_client:
        #         return self.claude_client, "claude-sonnet"
        #     elif self.groq_client:
        #         return self.groq_client, "groq"
        #
        # raise ValueError("No LLM clients available")


# =============================================================================
# TODO #2: Implement analyze_stock method
# =============================================================================
# Instructions:
# 1. Create analyze_stock() method
# 2. Check cache first
# 3. Build prompt using PromptBuilder
# 4. Call appropriate LLM
# 5. Parse response into AnalysisResult
# 6. Cache the result
# 7. Update metrics
#
# This is the main method that ties everything together!
# =============================================================================

    async def analyze_stock(
        self,
        symbol: str,
        metrics: Union[StockMetrics, Dict[str, Any]],
        complexity: Optional[TaskComplexity] = None,
        force_refresh: bool = False
    ) -> AnalysisResult:
        """
        Analyze a stock and provide recommendation.

        TODO: Implement this method
        Steps:
        1. Convert metrics dict to StockMetrics if needed
        2. Generate cache key from symbol and metrics
        3. Check cache (if enabled and not force_refresh)
        4. If cache miss:
        5.   - Build prompt using create_stock_analysis_prompt()
        6.   - Select appropriate LLM client
        7.   - Call LLM with prompt
        8.   - Parse response into AnalysisResult
        9.   - Cache result
        10. Update metrics
        11. Return AnalysisResult

        Args:
            symbol: Stock ticker symbol
            metrics: Stock metrics (StockMetrics or dict)
            complexity: Task complexity (None = use default)
            force_refresh: Skip cache and get fresh analysis

        Returns:
            AnalysisResult with recommendation and reasoning

        Raises:
            AnalysisError: If analysis fails
        """
        # TODO: Implement stock analysis
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # self.metrics["total_requests"] += 1
        # self._log(f"Analyzing {symbol}")
        #
        # # Convert metrics if needed
        # if isinstance(metrics, dict):
        #     metrics = StockMetrics(**metrics)
        #
        # # Check cache
        # cache_key = f"analysis:{symbol}:{hash(str(metrics.to_dict()))}"
        # if self.cache and not force_refresh:
        #     cached = self.cache.get(cache_key)
        #     if cached:
        #         self.metrics["cache_hits"] += 1
        #         self._log(f"Cache hit for {symbol}")
        #         result = AnalysisResult(**cached)
        #         result.cache_hit = True
        #         return result
        #
        # # Select model
        # client, model_name = self._select_model(complexity)
        # self._log(f"Using model: {model_name}")
        #
        # # Build prompt
        # prompt = create_stock_analysis_prompt(
        #     symbol=metrics.symbol,
        #     price=metrics.price,
        #     pe_ratio=metrics.pe_ratio,
        #     revenue_growth=metrics.revenue_growth,
        #     profit_margin=metrics.profit_margin,
        #     debt_equity=metrics.debt_equity,
        #     include_examples=True
        # )
        #
        # # Call LLM
        # try:
        #     llm_response = await client.generate(
        #         prompt=prompt["user"],
        #         system_prompt=prompt["system"],
        #         temperature=0.7,
        #         max_tokens=1000
        #     )
        #
        #     # Update metrics
        #     self.metrics["total_tokens"] += llm_response.tokens_used
        #     if llm_response.provider == LLMProvider.GROQ:
        #         self.metrics["groq_requests"] += 1
        #     else:
        #         self.metrics["claude_requests"] += 1
        #
        #     # Parse response
        #     result = self._parse_response(
        #         llm_response.content,
        #         symbol,
        #         llm_response.provider.value,
        #         llm_response.model
        #     )
        #
        #     # Cache result
        #     if self.cache:
        #         self.cache.set(cache_key, result.to_dict(), ttl=3600)  # 1 hour
        #
        #     return result
        #
        # except Exception as e:
        #     self.metrics["errors"] += 1
        #     self._log(f"Analysis failed: {str(e)}", "ERROR")
        #     raise AnalysisError(f"Failed to analyze {symbol}: {str(e)}")


# =============================================================================
# TODO #3: Parse LLM responses
# =============================================================================
# Instructions:
# 1. Create _parse_response() method
# 2. Extract JSON from markdown code blocks
# 3. Validate required fields
# 4. Convert to AnalysisResult object
# 5. Handle malformed responses gracefully
#
# LLM responses can be messy! Your parser needs to handle:
# - JSON in markdown blocks (```json ... ```)
# - Plain JSON
# - Extra text before/after JSON
# - Missing fields
# - Invalid values
# =============================================================================

    def _parse_response(
        self,
        content: str,
        symbol: str,
        provider: str,
        model: str
    ) -> AnalysisResult:
        """
        Parse LLM response into structured AnalysisResult.

        TODO: Implement response parsing
        Steps:
        1. Extract JSON from content (handle markdown blocks)
        2. Parse JSON string to dict
        3. Validate required fields exist
        4. Convert recommendation string to Recommendation enum
        5. Create AnalysisResult object
        6. Add metadata (symbol, provider, model, timestamp)

        Args:
            content: Raw LLM response content
            symbol: Stock symbol
            provider: LLM provider name
            model: Model name

        Returns:
            Parsed AnalysisResult

        Raises:
            ParseError: If response cannot be parsed
        """
        # TODO: Implement response parsing
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # try:
        #     # Extract JSON from markdown
        #     json_data = self._extract_json(content)
        #
        #     # Validate required fields
        #     required = ["recommendation", "confidence", "reasoning", "score"]
        #     missing = [f for f in required if f not in json_data]
        #     if missing:
        #         raise ParseError(f"Missing required fields: {missing}")
        #
        #     # Convert recommendation to enum
        #     rec_str = json_data["recommendation"].upper().replace(" ", "_")
        #     try:
        #         recommendation = Recommendation[rec_str]
        #     except KeyError:
        #         # Try to map variations
        #         if "BUY" in rec_str:
        #             recommendation = Recommendation.BUY
        #         elif "SELL" in rec_str:
        #             recommendation = Recommendation.SELL
        #         else:
        #             recommendation = Recommendation.HOLD
        #
        #     # Create result
        #     return AnalysisResult(
        #         symbol=symbol,
        #         recommendation=recommendation,
        #         confidence=float(json_data["confidence"]),
        #         reasoning=json_data["reasoning"],
        #         key_metrics=json_data.get("key_metrics", []),
        #         risks=json_data.get("risks", []),
        #         score=int(json_data["score"]),
        #         timestamp=datetime.now(),
        #         provider=provider,
        #         model=model
        #     )
        #
        # except Exception as e:
        #     raise ParseError(f"Failed to parse response: {str(e)}\nContent: {content}")

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response.

        TODO: Implement JSON extraction
        Steps:
        1. Check for markdown code blocks (```json ... ```)
        2. If found, extract content between markers
        3. Otherwise use full content
        4. Parse JSON
        5. Return parsed dict

        Args:
            content: LLM response content

        Returns:
            Parsed JSON dict

        Raises:
            json.JSONDecodeError: If JSON is invalid
        """
        # TODO: Implement JSON extraction
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # # Try to find JSON in markdown code blocks
        # json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
        # if json_match:
        #     json_str = json_match.group(1)
        # else:
        #     # Try to find JSON object
        #     json_match = re.search(r'\{.*\}', content, re.DOTALL)
        #     if json_match:
        #         json_str = json_match.group(0)
        #     else:
        #         json_str = content
        #
        # return json.loads(json_str.strip())


# =============================================================================
# TODO #4: Add error handling
# =============================================================================
# Instructions:
# 1. Create custom exception classes
# 2. Implement retry logic with fallback
# 3. Add graceful degradation
# 4. Log errors for debugging
# 5. Return default/safe responses on critical failure
#
# Error scenarios to handle:
# - API failures (rate limits, timeouts)
# - Parse errors (malformed JSON)
# - Invalid data (missing metrics)
# - Cache failures
# =============================================================================


class FinancialAgentError(Exception):
    """Base exception for FinancialAgent errors"""
    pass


class AnalysisError(FinancialAgentError):
    """Raised when analysis fails"""
    pass


class ParseError(FinancialAgentError):
    """Raised when response parsing fails"""
    pass


class FinancialAgent:  # Continued from above
    """Additional methods for error handling"""

    async def analyze_with_fallback(
        self,
        symbol: str,
        metrics: Union[StockMetrics, Dict[str, Any]],
        max_retries: int = 2
    ) -> AnalysisResult:
        """
        Analyze with automatic fallback strategies.

        TODO: Implement fallback logic
        Steps:
        1. Try analyze_stock() with SIMPLE complexity
        2. If fails, retry with MODERATE
        3. If still fails, retry with COMPLEX
        4. If all fail, return conservative default
        5. Log all failures for debugging

        Args:
            symbol: Stock symbol
            metrics: Stock metrics
            max_retries: Max retry attempts per complexity level

        Returns:
            AnalysisResult (may be default on failure)
        """
        # TODO: Implement fallback strategy
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # complexities = [
        #     TaskComplexity.SIMPLE,
        #     TaskComplexity.MODERATE,
        #     TaskComplexity.COMPLEX
        # ]
        #
        # last_error = None
        # for complexity in complexities:
        #     for attempt in range(max_retries):
        #         try:
        #             self._log(f"Attempt {attempt + 1} with {complexity.value}")
        #             return await self.analyze_stock(symbol, metrics, complexity)
        #         except Exception as e:
        #             last_error = e
        #             self._log(f"Failed: {str(e)}", "WARNING")
        #             await asyncio.sleep(1 * (attempt + 1))  # Backoff
        #
        # # All attempts failed - return conservative default
        # self._log(f"All attempts failed, returning default", "ERROR")
        # return self._get_default_analysis(symbol, last_error)

    def _get_default_analysis(
        self,
        symbol: str,
        error: Optional[Exception] = None
    ) -> AnalysisResult:
        """
        Return a conservative default analysis.

        TODO: Implement default response
        - Recommendation: HOLD (safest)
        - Confidence: 0.0 (no confidence)
        - Reasoning: Explain the error
        - Score: 50 (neutral)

        Args:
            symbol: Stock symbol
            error: The error that caused fallback

        Returns:
            Safe default AnalysisResult
        """
        # TODO: Implement default analysis
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # error_msg = str(error) if error else "Analysis unavailable"
        # return AnalysisResult(
        #     symbol=symbol,
        #     recommendation=Recommendation.HOLD,
        #     confidence=0.0,
        #     reasoning=f"Unable to analyze due to error: {error_msg}. Defaulting to HOLD as conservative approach.",
        #     key_metrics=[],
        #     risks=["Analysis failed - insufficient data"],
        #     score=50,
        #     timestamp=datetime.now(),
        #     provider="default",
        #     model="fallback"
        # )


# =============================================================================
# TODO #5: Add caching
# =============================================================================
# Instructions:
# 1. Implement cache key generation
# 2. Add TTL strategies (different durations for different scenarios)
# 3. Implement cache warming for popular symbols
# 4. Add cache invalidation on data updates
# 5. Track cache hit rate
#
# Caching strategies:
# - Short TTL (1 hour) for volatile stocks
# - Longer TTL (24 hours) for stable stocks
# - Cache popular symbols proactively
# - Invalidate on major news/events
# =============================================================================

    def _generate_cache_key(
        self,
        symbol: str,
        metrics: StockMetrics
    ) -> str:
        """
        Generate a unique cache key.

        TODO: Implement cache key generation
        Steps:
        1. Include symbol
        2. Hash the metrics to detect changes
        3. Optionally include date for daily refresh
        4. Format as "analysis:{symbol}:{hash}"

        Args:
            symbol: Stock symbol
            metrics: Stock metrics

        Returns:
            Cache key string
        """
        # TODO: Implement cache key generation
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # import hashlib
        # metrics_str = json.dumps(metrics.to_dict(), sort_keys=True)
        # metrics_hash = hashlib.md5(metrics_str.encode()).hexdigest()[:8]
        # return f"analysis:{symbol}:{metrics_hash}"

    async def warm_cache(self, symbols: List[str], metrics_dict: Dict[str, StockMetrics]):
        """
        Pre-populate cache for commonly requested symbols.

        TODO: Implement cache warming
        Steps:
        1. Iterate through symbols
        2. Check if already cached
        3. If not, analyze and cache
        4. Run analyses in parallel for speed
        5. Log cache warming results

        Args:
            symbols: List of symbols to warm
            metrics_dict: Dict mapping symbol to metrics
        """
        # TODO: Implement cache warming
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # self._log(f"Warming cache for {len(symbols)} symbols")
        # tasks = []
        # for symbol in symbols:
        #     if symbol in metrics_dict:
        #         task = self.analyze_stock(symbol, metrics_dict[symbol])
        #         tasks.append(task)
        #
        # results = await asyncio.gather(*tasks, return_exceptions=True)
        # success = sum(1 for r in results if not isinstance(r, Exception))
        # self._log(f"Cache warmed: {success}/{len(symbols)} successful")


# =============================================================================
# TODO #6: Integrate with trading API
# =============================================================================
# Instructions:
# 1. Create method to fetch real-time metrics
# 2. Add method to execute trades based on analysis
# 3. Implement position sizing logic
# 4. Add risk checks before trading
# 5. Log all trading decisions
#
# This connects your AI agent to actual trading!
# Be very careful with real money - start with paper trading.
# =============================================================================

    async def analyze_and_trade(
        self,
        symbol: str,
        trading_client: Any,  # Your trading API client
        portfolio_value: float,
        max_position_size: float = 0.1,  # 10% max
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze a stock and execute trade if conditions are met.

        TODO: Implement trading integration
        Steps:
        1. Fetch current metrics from trading API
        2. Analyze stock with agent
        3. Check if confidence meets minimum
        4. Calculate position size based on confidence and risk
        5. Execute trade through trading API
        6. Log decision and outcome
        7. Return trade details

        Args:
            symbol: Stock to analyze and trade
            trading_client: Trading API client
            portfolio_value: Total portfolio value
            max_position_size: Maximum position as fraction of portfolio
            min_confidence: Minimum confidence to trade (0.0-1.0)

        Returns:
            Dict with trade details
        """
        # TODO: Implement trading integration
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # self._log(f"Analyze and trade: {symbol}")
        #
        # # 1. Fetch metrics (pseudo-code - adjust for your API)
        # # metrics = await trading_client.get_stock_metrics(symbol)
        #
        # # 2. Analyze
        # analysis = await self.analyze_stock(symbol, metrics)
        #
        # # 3. Check confidence
        # if analysis.confidence < min_confidence:
        #     return {
        #         "action": "no_trade",
        #         "reason": f"Confidence {analysis.confidence} below minimum {min_confidence}",
        #         "analysis": analysis.to_dict()
        #     }
        #
        # # 4. Calculate position size
        # base_size = portfolio_value * max_position_size
        # adjusted_size = base_size * analysis.confidence
        #
        # # 5. Execute trade (pseudo-code)
        # if analysis.recommendation in [Recommendation.BUY, Recommendation.STRONG_BUY]:
        #     # order = await trading_client.place_order(
        #     #     symbol=symbol,
        #     #     side="buy",
        #     #     amount=adjusted_size
        #     # )
        #     action = "buy"
        # elif analysis.recommendation in [Recommendation.SELL, Recommendation.STRONG_SELL]:
        #     # order = await trading_client.place_order(
        #     #     symbol=symbol,
        #     #     side="sell",
        #     #     amount=adjusted_size
        #     # )
        #     action = "sell"
        # else:
        #     action = "no_trade"
        #
        # # 6. Log and return
        # self._log(f"Trade executed: {action} {symbol} @ {adjusted_size}")
        # return {
        #     "action": action,
        #     "symbol": symbol,
        #     "amount": adjusted_size,
        #     "analysis": analysis.to_dict(),
        #     # "order": order
        # }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.

        Returns:
            Dict with usage statistics
        """
        metrics = self.metrics.copy()
        if metrics["total_requests"] > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["total_requests"]
        else:
            metrics["cache_hit_rate"] = 0.0

        return metrics


# =============================================================================
# Testing Utilities
# =============================================================================

class MockLLMClient:
    """
    Mock LLM client for testing without API calls.

    Use this to test your agent without spending money on API calls.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Return a mock response"""
        # TODO: Implement mock response
        # Return a canned response that matches expected format
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # from llm_client import LLMResponse, LLMProvider
        #
        # mock_content = """{
        #   "recommendation": "BUY",
        #   "confidence": 0.75,
        #   "reasoning": "This is a mock analysis for testing purposes.",
        #   "key_metrics": ["mock_metric_1", "mock_metric_2"],
        #   "risks": ["mock_risk_1"],
        #   "score": 75
        # }"""
        #
        # return LLMResponse(
        #     content=mock_content,
        #     provider=LLMProvider.GROQ,
        #     model="mock-model",
        #     tokens_used=100,
        #     latency_ms=50.0,
        #     timestamp=datetime.now(),
        #     raw_response={}
        # )


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """
    Example of how to use the FinancialAgent.

    Run this after implementing all TODOs to test your code.
    """
    print("=== Initializing Financial Agent ===")
    agent = FinancialAgent(debug=True)

    # Example stock metrics
    aapl_metrics = StockMetrics(
        symbol="AAPL",
        price=175.50,
        pe_ratio=28.5,
        revenue_growth=8.2,
        profit_margin=25.3,
        debt_equity=1.5,
        market_cap=2.8e12,
        dividend_yield=0.5,
        beta=1.2
    )

    print("\n=== Analyzing AAPL ===")
    result = await agent.analyze_stock("AAPL", aapl_metrics)

    print(f"Recommendation: {result.recommendation.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Score: {result.score}/100")
    print(f"Reasoning: {result.reasoning}")
    print(f"Key Metrics: {', '.join(result.key_metrics)}")
    print(f"Risks: {', '.join(result.risks)}")
    print(f"Provider: {result.provider}")
    print(f"Cache Hit: {result.cache_hit}")

    print("\n=== Second Request (should hit cache) ===")
    result2 = await agent.analyze_stock("AAPL", aapl_metrics)
    print(f"Cache Hit: {result2.cache_hit}")

    print("\n=== Agent Metrics ===")
    metrics = agent.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")


async def test_with_mock():
    """Test the agent with mock LLM client (no API costs)"""
    print("=== Testing with Mock Client ===")

    # Create agent with mock client
    # You'll need to modify the agent to accept a client parameter
    # agent = FinancialAgent(llm_client=MockLLMClient(), debug=True)

    metrics = StockMetrics(
        symbol="TEST",
        price=100.0,
        pe_ratio=20.0,
        revenue_growth=10.0,
        profit_margin=15.0,
        debt_equity=1.0
    )

    # result = await agent.analyze_stock("TEST", metrics)
    # print(f"Mock Result: {result.to_dict()}")


if __name__ == "__main__":
    # Uncomment to test your implementation
    # asyncio.run(example_usage())

    # Or test with mock (no API costs)
    # asyncio.run(test_with_mock())
    pass
