"""
Base class for LLM agents.

All specialized agents (Financial, Sentiment, Earnings) inherit from this
base class which provides:
- LLM API integration (Groq and Anthropic)
- Cost tracking
- Response caching
- Error handling and retries
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import time
from groq import Groq
from anthropic import Anthropic
from src.utils.config import settings
from src.utils.logger import app_logger
from src.utils.cache import cache
from src.data_layer.models import LLMAnalysis


class BaseLLMAgent(ABC):
    """
    Abstract base class for all LLM agents.

    Each agent implements its own analysis logic but shares common
    infrastructure for API calls, caching, and cost tracking.
    """

    def __init__(self, agent_name: str):
        """
        Initialize the agent.

        Args:
            agent_name: Unique name for this agent
        """
        self.agent_name = agent_name
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)

        # Cost tracking (approximate costs per 1M tokens)
        self.cost_per_token = {
            "groq": 0.0001,  # Very cheap, fast
            "anthropic_haiku": 0.25,  # Cheap, fast
            "anthropic_sonnet": 3.0,  # Expensive, best quality
        }

        app_logger.info(f"Initialized {agent_name} agent")

    def _get_cached_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Check if we have a recent cached analysis.

        This is critical for cost optimization - we don't want to
        re-analyze the same stock multiple times per day.

        Args:
            symbol: Stock ticker

        Returns:
            Cached analysis or None
        """
        cache_key = f"llm_analysis:{self.agent_name}:{symbol}"
        cached = cache.get(cache_key)

        if cached:
            app_logger.info(
                f"Cache HIT for {symbol} ({self.agent_name}) - "
                f"Saved ${self._estimate_cost(settings.max_tokens_per_analysis):.4f}"
            )
            return cached

        return None

    def _cache_analysis(
        self,
        symbol: str,
        analysis: Dict[str, Any]
    ):
        """
        Cache analysis result.

        Args:
            symbol: Stock ticker
            analysis: Analysis result to cache
        """
        cache_key = f"llm_analysis:{self.agent_name}:{symbol}"
        ttl = settings.cache_analysis_hours * 3600
        cache.set(cache_key, analysis, ttl=ttl)
        app_logger.debug(
            f"Cached analysis for {symbol} ({self.agent_name}) "
            f"for {settings.cache_analysis_hours} hours"
        )

    def _estimate_cost(
        self,
        tokens: int,
        provider: str = "groq"
    ) -> float:
        """
        Estimate API cost for a number of tokens.

        Args:
            tokens: Number of tokens
            provider: API provider

        Returns:
            Estimated cost in USD
        """
        cost_per_million = self.cost_per_token.get(provider, 0.0001)
        return (tokens / 1_000_000) * cost_per_million

    def _call_groq(
        self,
        prompt: str,
        model: str = "llama-3.1-70b-versatile",
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> tuple[str, int, float]:
        """
        Call Groq API for fast, cheap inference.

        Groq is excellent for:
        - Quick sentiment analysis
        - Extracting structured data
        - Classification tasks

        Args:
            prompt: User prompt
            model: Groq model name
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            Tuple of (response_text, tokens_used, cost)
        """
        start_time = time.time()

        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analysis expert. Provide concise, factual analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            cost = self._estimate_cost(tokens_used, "groq")
            latency_ms = int((time.time() - start_time) * 1000)

            app_logger.info(
                f"Groq API call: {tokens_used} tokens, "
                f"${cost:.6f}, {latency_ms}ms"
            )

            return content, tokens_used, cost

        except Exception as e:
            app_logger.error(f"Groq API error: {e}")
            raise

    def _call_anthropic(
        self,
        prompt: str,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> tuple[str, int, float]:
        """
        Call Anthropic API for complex reasoning.

        Use Claude for:
        - Deep fundamental analysis
        - Complex reasoning over financial statements
        - Nuanced sentiment analysis

        Args:
            prompt: User prompt
            model: Claude model (haiku, sonnet, or opus)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, tokens_used, cost)
        """
        start_time = time.time()

        try:
            # Determine cost tier
            if "sonnet" in model:
                cost_key = "anthropic_sonnet"
            else:
                cost_key = "anthropic_haiku"

            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = self._estimate_cost(tokens_used, cost_key)
            latency_ms = int((time.time() - start_time) * 1000)

            app_logger.info(
                f"Anthropic API call ({model}): {tokens_used} tokens, "
                f"${cost:.6f}, {latency_ms}ms"
            )

            return content, tokens_used, cost

        except Exception as e:
            app_logger.error(f"Anthropic API error: {e}")
            raise

    def _call_llm(
        self,
        prompt: str,
        complexity: Literal["simple", "complex"] = "simple",
        max_tokens: int = None
    ) -> tuple[str, int, float]:
        """
        Smart LLM router that chooses the best provider for the task.

        Cost optimization strategy:
        - Use Groq for simple tasks (90% of cases) - very cheap and fast
        - Use Claude Haiku for moderate complexity - good balance
        - Use Claude Sonnet for complex reasoning - expensive but best

        Args:
            prompt: User prompt
            complexity: Task complexity level
            max_tokens: Max response tokens (default from settings)

        Returns:
            Tuple of (response_text, tokens_used, cost)
        """
        if max_tokens is None:
            max_tokens = settings.max_tokens_per_analysis

        # Route based on complexity and settings
        if complexity == "simple" and settings.use_groq_for_speed:
            # Use Groq for simple tasks (default)
            return self._call_groq(prompt, max_tokens=max_tokens)
        elif complexity == "complex" and settings.use_anthropic_for_complex:
            # Use Claude Sonnet for complex reasoning
            return self._call_anthropic(
                prompt,
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens
            )
        else:
            # Use Claude Haiku as middle ground
            return self._call_anthropic(
                prompt,
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens
            )

    @abstractmethod
    def analyze(self, symbol: str, **kwargs) -> LLMAnalysis:
        """
        Analyze a stock and return structured results.

        Each subclass must implement this method with its own logic.

        Args:
            symbol: Stock ticker
            **kwargs: Additional parameters specific to the agent

        Returns:
            LLMAnalysis object with results
        """
        pass

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this agent.

        Returns:
            Dict with metrics like total cost, average latency, etc.
        """
        # This would query the database for historical performance
        # Simplified for now
        return {
            "agent_name": self.agent_name,
            "total_analyses": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0,
            "cache_hit_rate": 0.0,
        }
