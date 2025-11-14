"""
Exercise 5: Cost Optimization - Minimize API Costs
===================================================

Objective: Reduce LLM API costs by 90%+ through smart strategies

Time: 1.5 hours
Difficulty: Advanced

What You'll Learn:
- Calculate and track API costs
- Implement intelligent caching strategies
- Choose the right model for each task
- Use Groq for simple tasks (10x cheaper)
- Use Claude for complex analysis (better quality)
- Optimize prompt length to reduce tokens
- Batch requests efficiently
- Monitor and report cost metrics

Prerequisites:
- Completed Exercises 1-4
- Understanding of token-based pricing

Setup:
------
pip install pydantic requests anthropic redis
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


# ============================================================================
# Cost Constants
# ============================================================================

@dataclass
class ModelPricing:
    """Pricing information for LLM models."""
    name: str
    input_cost_per_1m: float  # Cost per 1M input tokens
    output_cost_per_1m: float  # Cost per 1M output tokens
    avg_latency_ms: float  # Average latency
    quality_score: float  # Subjective quality 0-10


# Pricing as of 2024 (approximate)
MODELS = {
    "groq-llama3-8b": ModelPricing(
        name="groq-llama3-8b-8192",
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.08,
        avg_latency_ms=250,
        quality_score=7.0
    ),
    "groq-llama3-70b": ModelPricing(
        name="groq-llama3-70b-8192",
        input_cost_per_1m=0.59,
        output_cost_per_1m=0.79,
        avg_latency_ms=350,
        quality_score=8.5
    ),
    "claude-haiku": ModelPricing(
        name="claude-3-haiku-20240307",
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
        avg_latency_ms=1500,
        quality_score=8.0
    ),
    "claude-sonnet": ModelPricing(
        name="claude-3-5-sonnet-20241022",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        avg_latency_ms=2000,
        quality_score=9.5
    )
}


# ============================================================================
# TODO #1: Build a cost tracker
# ============================================================================

class CostTracker:
    """
    Track API costs in real-time.

    Features:
    - Track costs per request
    - Track costs per model
    - Track costs per day/week/month
    - Calculate savings from caching
    - Generate cost reports
    """

    def __init__(self):
        """Initialize cost tracker."""
        self.requests: List[Dict[str, Any]] = []
        self.total_cost = 0.0
        self.costs_by_model = defaultdict(float)
        self.costs_by_day = defaultdict(float)
        self.tokens_used = 0
        self.cache_saves = 0

    def record_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False
    ) -> float:
        """
        Record a request and calculate its cost.

        Args:
            model: Model key (e.g., "groq-llama3-8b")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached: Whether this was served from cache

        Returns:
            Cost of this request in USD

        Hints:
        - Get pricing from MODELS dict
        - Calculate: (input_tokens/1M * input_cost) + (output_tokens/1M * output_cost)
        - If cached, cost is $0 but track potential cost saved
        """
        # TODO: Get model pricing
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")

        pricing = MODELS[model]

        # TODO: Calculate cost
        if cached:
            cost = 0.0
            # Track what we saved
            potential_cost = (
                (input_tokens / 1_000_000) * pricing.input_cost_per_1m +
                (output_tokens / 1_000_000) * pricing.output_cost_per_1m
            )
            self.cache_saves += potential_cost
        else:
            cost = (
                (input_tokens / 1_000_000) * pricing.input_cost_per_1m +
                (output_tokens / 1_000_000) * pricing.output_cost_per_1m
            )

        # TODO: Record the request
        today = datetime.now().strftime("%Y-%m-%d")
        request_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cached": cached
        }

        self.requests.append(request_record)
        self.total_cost += cost
        self.costs_by_model[model] += cost
        self.costs_by_day[today] += cost
        self.tokens_used += input_tokens + output_tokens

        return cost

    def get_report(self, period: str = "all") -> Dict[str, Any]:
        """
        Generate cost report.

        Args:
            period: "today", "week", "month", or "all"

        Returns:
            Dictionary with cost metrics
        """
        # TODO: Calculate metrics based on period
        # For simplicity, return all-time metrics

        avg_cost_per_request = (
            self.total_cost / len(self.requests)
            if self.requests else 0
        )

        return {
            "total_requests": len(self.requests),
            "total_cost_usd": round(self.total_cost, 4),
            "total_tokens": self.tokens_used,
            "cache_savings_usd": round(self.cache_saves, 4),
            "avg_cost_per_request": round(avg_cost_per_request, 6),
            "costs_by_model": {
                model: round(cost, 4)
                for model, cost in self.costs_by_model.items()
            },
            "projected_monthly_cost": round(self.total_cost * 30, 2)  # Simple projection
        }

    def print_report(self) -> None:
        """Print a formatted cost report."""
        report = self.get_report()

        print("\n" + "=" * 80)
        print("COST REPORT")
        print("=" * 80)
        print(f"Total Requests: {report['total_requests']}")
        print(f"Total Cost: ${report['total_cost_usd']}")
        print(f"Total Tokens: {report['total_tokens']:,}")
        print(f"Cache Savings: ${report['cache_savings_usd']}")
        print(f"Avg Cost/Request: ${report['avg_cost_per_request']}")
        print(f"\nProjected Monthly Cost: ${report['projected_monthly_cost']}")
        print(f"\nCosts by Model:")
        for model, cost in report['costs_by_model'].items():
            print(f"  {model}: ${cost}")
        print("=" * 80)


# ============================================================================
# TODO #2: Implement smart caching with Redis
# ============================================================================

class SmartCache:
    """
    Smart caching with multiple strategies.

    Features:
    - TTL based on query complexity
    - Different TTLs for different query types
    - Automatic cache warming
    - Cache statistics
    """

    def __init__(self, use_redis: bool = False):
        """
        Initialize cache.

        Args:
            use_redis: Use Redis if available, otherwise use in-memory dict
        """
        self.use_redis = use_redis
        self.cache: Dict[str, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, ttl)
        self.hits = 0
        self.misses = 0

    def _get_ttl_for_query(self, query_type: str, complexity: str) -> int:
        """
        Determine TTL based on query characteristics.

        Args:
            query_type: "analysis", "price_check", "news", etc.
            complexity: "simple", "medium", "complex"

        Returns:
            TTL in seconds

        Strategy:
        - Simple queries: cache longer (they're cheaper to cache)
        - Complex queries: cache medium time (they're expensive but change)
        - Real-time data: cache very short
        - Historical analysis: cache very long

        Examples:
        - Simple stock metrics: 1 hour
        - Complex analysis: 15 minutes
        - Price data: 1 minute
        - Historical data: 1 day
        """
        # TODO: Implement smart TTL logic
        ttl_matrix = {
            ("analysis", "simple"): 3600,      # 1 hour
            ("analysis", "medium"): 1800,      # 30 minutes
            ("analysis", "complex"): 900,      # 15 minutes
            ("price_check", "simple"): 60,     # 1 minute
            ("news", "simple"): 300,           # 5 minutes
            ("historical", "simple"): 86400,   # 1 day
        }

        return ttl_matrix.get((query_type, complexity), 900)  # Default 15 min

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.

        Returns:
            Cached value or None
        """
        # TODO: Check if key exists
        if key not in self.cache:
            self.misses += 1
            return None

        value, cached_at, ttl = self.cache[key]

        # TODO: Check if expired
        age = time.time() - cached_at
        if age > ttl:
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return value

    def set(
        self,
        key: str,
        value: Any,
        query_type: str = "analysis",
        complexity: str = "medium"
    ) -> None:
        """
        Save value to cache with smart TTL.

        Args:
            key: Cache key
            value: Value to cache
            query_type: Type of query
            complexity: Complexity level
        """
        # TODO: Determine TTL
        ttl = self._get_ttl_for_query(query_type, complexity)

        # TODO: Save to cache
        self.cache[key] = (value, time.time(), ttl)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }


# ============================================================================
# TODO #3: Implement model selection strategy
# ============================================================================

class ModelSelector:
    """
    Choose the optimal model for each task.

    Strategy:
    - Use Groq (cheap, fast) for simple tasks
    - Use Claude (expensive, smart) for complex reasoning
    - Balance cost vs quality based on importance
    """

    @staticmethod
    def estimate_complexity(
        prompt: str,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Estimate query complexity.

        Args:
            prompt: The prompt text
            metrics: Input metrics

        Returns:
            "simple", "medium", or "complex"

        Hints:
        - Simple: Basic metrics lookup, single number analysis
        - Medium: Standard stock analysis with multiple metrics
        - Complex: Multi-factor analysis, comparisons, predictions
        """
        # TODO: Implement complexity estimation
        # Factors to consider:
        # - Prompt length
        # - Number of metrics
        # - Keywords like "compare", "predict", "analyze deeply"

        prompt_lower = prompt.lower()

        # Simple indicators
        if len(prompt) < 100 and len(metrics) <= 3:
            return "simple"

        # Complex indicators
        complex_keywords = ["compare", "predict", "forecast", "deep", "comprehensive"]
        if any(keyword in prompt_lower for keyword in complex_keywords):
            return "complex"

        # Default to medium
        return "medium"

    @staticmethod
    def select_model(
        complexity: str,
        budget_mode: bool = False,
        quality_mode: bool = False
    ) -> str:
        """
        Select the best model for the task.

        Args:
            complexity: Task complexity ("simple", "medium", "complex")
            budget_mode: Prioritize cost savings
            quality_mode: Prioritize quality over cost

        Returns:
            Model key (e.g., "groq-llama3-8b")

        Strategy Matrix:
        ┌───────────┬──────────────┬──────────────┬──────────────┐
        │           │ Budget Mode  │   Balanced   │ Quality Mode │
        ├───────────┼──────────────┼──────────────┼──────────────┤
        │ Simple    │ groq-8b      │ groq-8b      │ groq-70b     │
        │ Medium    │ groq-8b      │ groq-70b     │ claude-haiku │
        │ Complex   │ groq-70b     │ claude-haiku │ claude-sonnet│
        └───────────┴──────────────┴──────────────┴──────────────┘
        """
        # TODO: Implement model selection logic

        if quality_mode:
            return {
                "simple": "groq-llama3-70b",
                "medium": "claude-haiku",
                "complex": "claude-sonnet"
            }[complexity]

        elif budget_mode:
            return {
                "simple": "groq-llama3-8b",
                "medium": "groq-llama3-8b",
                "complex": "groq-llama3-70b"
            }[complexity]

        else:  # Balanced
            return {
                "simple": "groq-llama3-8b",
                "medium": "groq-llama3-70b",
                "complex": "claude-haiku"
            }[complexity]


# ============================================================================
# TODO #4: Optimize prompt length
# ============================================================================

class PromptOptimizer:
    """
    Optimize prompts to reduce token usage.

    Strategies:
    - Remove unnecessary words
    - Use abbreviations
    - Compress examples
    - Remove redundancy
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count.

        Rule of thumb: ~4 characters per token for English.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # TODO: Implement token estimation
        # Simple estimation: chars / 4
        return len(text) // 4

    @staticmethod
    def compress_prompt(prompt: str, target_reduction: float = 0.3) -> str:
        """
        Compress prompt to reduce tokens.

        Args:
            prompt: Original prompt
            target_reduction: Target reduction (0.3 = 30% shorter)

        Returns:
            Compressed prompt

        Strategies:
        - Remove filler words (very, really, just, etc.)
        - Use abbreviations (P/E instead of Price-to-Earnings)
        - Remove redundant instructions
        - Shorten examples

        Warning: Don't over-compress! Quality matters more than tokens.
        """
        # TODO: Implement compression
        compressed = prompt

        # Remove common filler words
        fillers = ["very", "really", "just", "quite", "rather", "actually"]
        for filler in fillers:
            compressed = compressed.replace(f" {filler} ", " ")

        # Remove extra whitespace
        compressed = " ".join(compressed.split())

        # Remove redundant phrases (example)
        compressed = compressed.replace("Please provide", "Provide")
        compressed = compressed.replace("I would like you to", "")

        return compressed

    @staticmethod
    def compare_prompts(original: str, optimized: str) -> Dict[str, Any]:
        """
        Compare original and optimized prompts.

        Returns:
            Dictionary with comparison metrics
        """
        orig_tokens = PromptOptimizer.estimate_tokens(original)
        opt_tokens = PromptOptimizer.estimate_tokens(optimized)

        reduction = (orig_tokens - opt_tokens) / orig_tokens if orig_tokens > 0 else 0

        # Estimate cost savings (using Groq pricing as example)
        cost_per_1m = 0.05
        orig_cost = (orig_tokens / 1_000_000) * cost_per_1m
        opt_cost = (opt_tokens / 1_000_000) * cost_per_1m
        cost_savings = orig_cost - opt_cost

        return {
            "original_tokens": orig_tokens,
            "optimized_tokens": opt_tokens,
            "tokens_saved": orig_tokens - opt_tokens,
            "reduction_pct": reduction,
            "original_cost": orig_cost,
            "optimized_cost": opt_cost,
            "cost_savings": cost_savings
        }


# ============================================================================
# TODO #5: Build cost-optimized agent
# ============================================================================

class CostOptimizedAgent:
    """
    Financial agent optimized for minimal cost.

    Features:
    - Smart caching with variable TTLs
    - Intelligent model selection
    - Prompt optimization
    - Cost tracking
    - Budget limits
    """

    def __init__(
        self,
        daily_budget_usd: float = 1.0,
        mode: str = "balanced"  # "budget", "balanced", "quality"
    ):
        """
        Initialize cost-optimized agent.

        Args:
            daily_budget_usd: Maximum spend per day
            mode: Operation mode
        """
        self.daily_budget = daily_budget_usd
        self.mode = mode

        self.cache = SmartCache()
        self.cost_tracker = CostTracker()
        self.model_selector = ModelSelector()
        self.prompt_optimizer = PromptOptimizer()

    def analyze_stock(
        self,
        symbol: str,
        metrics: Dict[str, float],
        importance: str = "medium"  # "low", "medium", "high"
    ) -> Dict[str, Any]:
        """
        Analyze stock with cost optimization.

        Args:
            symbol: Stock ticker
            metrics: Financial metrics
            importance: How important is this analysis?

        Returns:
            Analysis result with cost info

        Process:
        1. Check budget
        2. Check cache
        3. Optimize prompt
        4. Select model
        5. Make API call
        6. Track cost
        7. Cache result
        """
        # TODO: Step 1 - Check daily budget
        today_cost = self.cost_tracker.costs_by_day.get(
            datetime.now().strftime("%Y-%m-%d"),
            0.0
        )

        if today_cost >= self.daily_budget:
            raise ValueError(f"Daily budget of ${self.daily_budget} exceeded")

        # TODO: Step 2 - Check cache
        cache_key = f"{symbol}:{sorted(metrics.items())}"
        cached = self.cache.get(cache_key)

        if cached:
            # Record as cached (zero cost)
            self.cost_tracker.record_request(
                model="groq-llama3-8b",  # Placeholder
                input_tokens=500,
                output_tokens=200,
                cached=True
            )
            return cached

        # TODO: Step 3 - Create and optimize prompt
        prompt = f"Analyze {symbol} with metrics: {json.dumps(metrics)}"

        # Estimate complexity
        complexity = self.model_selector.estimate_complexity(prompt, metrics)

        # TODO: Step 4 - Select model
        budget_mode = (self.mode == "budget")
        quality_mode = (self.mode == "quality")

        model = self.model_selector.select_model(
            complexity,
            budget_mode=budget_mode,
            quality_mode=quality_mode
        )

        # TODO: Step 5 - Make API call (mocked for this exercise)
        # In real implementation, call actual API
        result = {
            "symbol": symbol,
            "recommendation": "BUY",
            "confidence": 75,
            "model_used": model,
            "complexity": complexity
        }

        # TODO: Step 6 - Track cost
        # Estimate tokens (in real implementation, get from API response)
        input_tokens = self.prompt_optimizer.estimate_tokens(prompt)
        output_tokens = 200  # Estimate

        cost = self.cost_tracker.record_request(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False
        )

        result["cost_usd"] = cost

        # TODO: Step 7 - Cache result
        self.cache.set(
            cache_key,
            result,
            query_type="analysis",
            complexity=complexity
        )

        return result

    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Generate optimization report.

        Returns:
            Report with cost savings and efficiency metrics
        """
        cost_report = self.cost_tracker.get_report()
        cache_stats = self.cache.get_stats()

        total_potential_cost = cost_report["total_cost_usd"] + cost_report["cache_savings_usd"]
        savings_pct = (
            cost_report["cache_savings_usd"] / total_potential_cost
            if total_potential_cost > 0 else 0
        )

        return {
            "cost_report": cost_report,
            "cache_stats": cache_stats,
            "total_savings_pct": savings_pct,
            "optimization_score": cache_stats["hit_rate"] * 100,
            "budget_status": {
                "daily_budget": self.daily_budget,
                "spent_today": cost_report["total_cost_usd"],
                "remaining": self.daily_budget - cost_report["total_cost_usd"],
                "budget_used_pct": (cost_report["total_cost_usd"] / self.daily_budget) * 100
            }
        }


# ============================================================================
# Test Cases
# ============================================================================

def test_cost_tracker():
    """Test cost tracking."""
    print("\n" + "=" * 80)
    print("TEST: Cost Tracker")
    print("=" * 80)

    tracker = CostTracker()

    # Record some requests
    tracker.record_request("groq-llama3-8b", input_tokens=500, output_tokens=200)
    tracker.record_request("claude-haiku", input_tokens=500, output_tokens=200)
    tracker.record_request("groq-llama3-8b", input_tokens=500, output_tokens=200, cached=True)

    tracker.print_report()


def test_smart_cache():
    """Test smart caching with variable TTLs."""
    print("\n" + "=" * 80)
    print("TEST: Smart Cache")
    print("=" * 80)

    cache = SmartCache()

    # Set values with different query types
    cache.set("key1", "value1", query_type="analysis", complexity="simple")
    cache.set("key2", "value2", query_type="price_check", complexity="simple")
    cache.set("key3", "value3", query_type="historical", complexity="simple")

    # Get values
    print(f"Get key1: {cache.get('key1')}")
    print(f"Get key2: {cache.get('key2')}")
    print(f"Get key3: {cache.get('key3')}")

    stats = cache.get_stats()
    print(f"\nCache Stats:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1%}")
    print(f"  Cache Size: {stats['cache_size']}")


def test_model_selection():
    """Test intelligent model selection."""
    print("\n" + "=" * 80)
    print("TEST: Model Selection")
    print("=" * 80)

    selector = ModelSelector()

    test_cases = [
        ("Simple lookup", {}, "simple"),
        ("Standard analysis with multiple metrics", {"pe": 20, "growth": 0.1}, "medium"),
        ("Compare multiple stocks and predict future performance", {"many": "metrics"}, "complex")
    ]

    for prompt, metrics, expected_complexity in test_cases:
        complexity = selector.estimate_complexity(prompt, metrics)
        budget_model = selector.select_model(complexity, budget_mode=True)
        balanced_model = selector.select_model(complexity)
        quality_model = selector.select_model(complexity, quality_mode=True)

        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Complexity: {complexity}")
        print(f"  Budget model: {budget_model}")
        print(f"  Balanced model: {balanced_model}")
        print(f"  Quality model: {quality_model}")


def test_prompt_optimization():
    """Test prompt optimization."""
    print("\n" + "=" * 80)
    print("TEST: Prompt Optimization")
    print("=" * 80)

    original = """
    I would like you to please provide a very detailed analysis of this stock.
    Really analyze it thoroughly and just make sure you consider all the factors.
    """

    optimizer = PromptOptimizer()
    optimized = optimizer.compress_prompt(original)

    comparison = optimizer.compare_prompts(original, optimized)

    print(f"Original: {original}")
    print(f"\nOptimized: {optimized}")
    print(f"\nComparison:")
    print(f"  Original tokens: {comparison['original_tokens']}")
    print(f"  Optimized tokens: {comparison['optimized_tokens']}")
    print(f"  Tokens saved: {comparison['tokens_saved']}")
    print(f"  Reduction: {comparison['reduction_pct']:.1%}")


def test_cost_optimized_agent():
    """Test complete cost-optimized agent."""
    print("\n" + "=" * 80)
    print("TEST: Cost-Optimized Agent")
    print("=" * 80)

    agent = CostOptimizedAgent(daily_budget_usd=1.0, mode="balanced")

    # Analyze some stocks
    stocks = [
        ("AAPL", {"pe_ratio": 28.5, "growth": 0.08}),
        ("MSFT", {"pe_ratio": 32.0, "growth": 0.12}),
        ("AAPL", {"pe_ratio": 28.5, "growth": 0.08}),  # Duplicate - should hit cache
    ]

    print("\nAnalyzing stocks...")
    for symbol, metrics in stocks:
        result = agent.analyze_stock(symbol, metrics)
        print(f"\n{symbol}: {result['recommendation']} "
              f"(model: {result['model_used']}, cost: ${result.get('cost_usd', 0):.6f})")

    # Print optimization report
    print("\n" + "=" * 80)
    print("OPTIMIZATION REPORT")
    print("=" * 80)

    report = agent.get_optimization_report()

    print(f"\nCost Summary:")
    print(f"  Total Cost: ${report['cost_report']['total_cost_usd']}")
    print(f"  Cache Savings: ${report['cost_report']['cache_savings_usd']}")
    print(f"  Total Savings: {report['total_savings_pct']:.1%}")

    print(f"\nCache Performance:")
    print(f"  Hit Rate: {report['cache_stats']['hit_rate']:.1%}")
    print(f"  Optimization Score: {report['optimization_score']:.1f}/100")

    print(f"\nBudget Status:")
    print(f"  Daily Budget: ${report['budget_status']['daily_budget']}")
    print(f"  Spent Today: ${report['budget_status']['spent_today']:.4f}")
    print(f"  Remaining: ${report['budget_status']['remaining']:.4f}")
    print(f"  Budget Used: {report['budget_status']['budget_used_pct']:.1f}%")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║          Exercise 5: Cost Optimization                             ║
    ║                                                                    ║
    ║  Reduce API costs by 90%+ through smart strategies                ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    print("\nCost Optimization Strategies:")
    print("""
    1. 📊 Smart Caching (50-90% cost reduction)
       - Cache analysis results
       - Variable TTLs based on query type
       - Simple queries: cache longer
       - Real-time data: cache shorter

    2. 🎯 Model Selection (10-50x cost difference)
       - Groq for simple tasks ($0.05/1M tokens)
       - Claude for complex reasoning ($3-15/1M tokens)
       - Match model to task complexity

    3. ✂️ Prompt Optimization (20-40% token reduction)
       - Remove filler words
       - Use abbreviations
       - Compress examples
       - Stay concise

    4. 📦 Batch Processing (reduce overhead)
       - Group similar requests
       - Reuse system prompts
       - Share context

    5. 💰 Budget Limits (prevent overspend)
       - Set daily/monthly budgets
       - Track costs in real-time
       - Fail gracefully when limit reached
    """)

    # Run tests
    test_cost_tracker()
    test_smart_cache()
    test_model_selection()
    test_prompt_optimization()
    test_cost_optimized_agent()

    print("\n" + "=" * 80)
    print("COST COMPARISON: Before vs After Optimization")
    print("=" * 80)
    print("""
    Scenario: 100 stock analyses per day for 30 days

    ❌ BEFORE OPTIMIZATION:
       Model: Claude Sonnet (high quality, high cost)
       Caching: None
       Prompt: Verbose (1000 tokens avg)

       Cost per analysis: ~$0.015
       Daily cost: $1.50
       Monthly cost: $45.00 💸

    ✅ AFTER OPTIMIZATION:
       Model: Smart selection (Groq for simple, Claude for complex)
       Caching: 70% hit rate
       Prompt: Optimized (600 tokens avg)

       Cost per analysis: ~$0.0015
       Daily cost: $0.15
       Monthly cost: $4.50 💰

    SAVINGS: 90% reduction! ($40.50/month saved)
    """)

    print("\n" + "=" * 80)
    print("EXERCISE COMPLETE!")
    print("=" * 80)
    print("""
    Congratulations! You've mastered LLM cost optimization!

    Key Learnings:
    ✅ Smart caching is the #1 cost saver (50-90% reduction)
    ✅ Model selection matters (10-50x cost difference)
    ✅ Prompt optimization reduces token usage (20-40%)
    ✅ Budget limits prevent overspend
    ✅ Real-time cost tracking enables monitoring

    Production Best Practices:
    - [ ] Implement Redis caching for persistence
    - [ ] Monitor cache hit rates (target >50%)
    - [ ] Set up cost alerts
    - [ ] Use Groq for 80% of tasks
    - [ ] Reserve Claude for critical analyses
    - [ ] Review and optimize top-cost queries
    - [ ] A/B test model selection strategies

    Cost Optimization Checklist:
    - [ ] Caching implemented with smart TTLs
    - [ ] Model selection based on complexity
    - [ ] Prompts optimized for token efficiency
    - [ ] Budget limits enforced
    - [ ] Cost tracking and reporting
    - [ ] Monthly cost < $5 for 3000 requests

    Next Steps:
    - Integrate with real trading system
    - Add monitoring and alerting
    - Implement async processing
    - Scale to production workloads

    You're ready for Week 3! 🚀
    """)
