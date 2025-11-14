"""
LLM Client Implementation for Trading Analysis

This module provides a unified interface for interacting with different LLM providers
(Groq and Claude) for financial analysis. It includes retry logic, rate limiting,
and response validation.

Learning Goals:
- Understand API client design patterns
- Implement retry logic with exponential backoff
- Handle rate limits gracefully
- Validate API responses
- Work with async/await patterns
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

# TODO: Install required packages
# pip install anthropic groq tenacity


# =============================================================================
# TODO #1: Create base LLM client class
# =============================================================================
# Instructions:
# 1. Create an abstract base class called BaseLLMClient
# 2. Define abstract methods: generate(), validate_response()
# 3. Add common attributes: model_name, api_key, max_retries
# 4. Implement a _log() method for debugging
#
# Example usage after implementation:
# client = GroqClient(api_key="your-key")
# response = await client.generate("Analyze AAPL stock")
# =============================================================================


class LLMProvider(Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    CLAUDE = "claude"


@dataclass
class LLMResponse:
    """Standardized response format from any LLM provider"""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: int
    latency_ms: float
    timestamp: datetime
    raw_response: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/caching"""
        return {
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    This class defines the interface that all LLM clients must implement.
    It provides common functionality like logging and configuration.

    Attributes:
        api_key: API key for the LLM provider
        model_name: Name of the model to use
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        debug: Enable debug logging
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_retries: int = 3,
        timeout: int = 30,
        debug: bool = False
    ):
        # TODO: Initialize attributes
        # - Store api_key, model_name, max_retries, timeout, debug
        # - Validate that api_key is not empty
        # - Initialize request counter for rate limiting
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # self.api_key = api_key
        # self.model_name = model_name
        # self.max_retries = max_retries
        # self.timeout = timeout
        # self.debug = debug
        # self.request_count = 0
        #
        # if not api_key:
        #     raise ValueError("API key cannot be empty")

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse object with standardized format

        Raises:
            LLMError: If the API call fails after retries
        """
        pass

    @abstractmethod
    def validate_response(self, response: Any) -> bool:
        """
        Validate the API response.

        Args:
            response: Raw response from the API

        Returns:
            True if valid, False otherwise
        """
        pass

    def _log(self, message: str, level: str = "INFO"):
        """
        Log a message if debug is enabled.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")


# =============================================================================
# TODO #2: Implement Groq client
# =============================================================================
# Instructions:
# 1. Create GroqClient class that inherits from BaseLLMClient
# 2. Implement the generate() method using Groq API
# 3. Implement validate_response() for Groq responses
# 4. Handle Groq-specific errors
# 5. Track token usage and latency
#
# Groq Documentation: https://console.groq.com/docs
# Recommended models:
# - mixtral-8x7b-32768 (fast, cheap)
# - llama2-70b-4096 (balanced)
# =============================================================================


class GroqClient(BaseLLMClient):
    """
    Groq LLM client for fast, cost-effective inference.

    Groq is optimized for speed (~500 tokens/sec) and low cost.
    Use for simple tasks like quick analysis and classification.

    Example:
        client = GroqClient(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )
        response = await client.generate("Analyze AAPL")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "mixtral-8x7b-32768",
        **kwargs
    ):
        # TODO: Initialize the Groq client
        # - Get API key from parameter or environment variable
        # - Call parent __init__
        # - Import and initialize groq.AsyncGroq client
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # api_key = api_key or os.getenv("GROQ_API_KEY")
        # if not api_key:
        #     raise ValueError("Groq API key required")
        #
        # super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        #
        # from groq import AsyncGroq
        # self.client = AsyncGroq(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using Groq.

        TODO: Implement this method
        Steps:
        1. Build messages array with system_prompt and prompt
        2. Start timer for latency measurement
        3. Call self.client.chat.completions.create()
        4. Calculate latency
        5. Extract content and token usage
        6. Return LLMResponse object
        """
        # TODO: Implement Groq API call
        # Hint: Use self.client.chat.completions.create()
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # start_time = time.time()
        #
        # messages = []
        # if system_prompt:
        #     messages.append({"role": "system", "content": system_prompt})
        # messages.append({"role": "user", "content": prompt})
        #
        # try:
        #     response = await self.client.chat.completions.create(
        #         model=self.model_name,
        #         messages=messages,
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #         **kwargs
        #     )
        #
        #     latency = (time.time() - start_time) * 1000  # Convert to ms
        #
        #     return LLMResponse(
        #         content=response.choices[0].message.content,
        #         provider=LLMProvider.GROQ,
        #         model=self.model_name,
        #         tokens_used=response.usage.total_tokens,
        #         latency_ms=latency,
        #         timestamp=datetime.now(),
        #         raw_response=response.model_dump()
        #     )
        # except Exception as e:
        #     self._log(f"Groq API error: {str(e)}", "ERROR")
        #     raise LLMError(f"Groq generation failed: {str(e)}")

    def validate_response(self, response: Any) -> bool:
        """
        Validate Groq API response.

        TODO: Implement validation
        - Check if response has 'choices' attribute
        - Check if choices is not empty
        - Check if content is not None
        """
        # TODO: Implement validation logic
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # return (
        #     hasattr(response, 'choices') and
        #     len(response.choices) > 0 and
        #     response.choices[0].message.content is not None
        # )


# =============================================================================
# TODO #3: Implement Claude client
# =============================================================================
# Instructions:
# 1. Create ClaudeClient class that inherits from BaseLLMClient
# 2. Implement the generate() method using Anthropic API
# 3. Implement validate_response() for Claude responses
# 4. Handle Claude-specific errors
# 5. Track token usage and latency
#
# Claude Documentation: https://docs.anthropic.com/claude/reference
# Recommended models:
# - claude-3-haiku-20240307 (fast, cheap)
# - claude-3-sonnet-20240229 (balanced)
# - claude-3-opus-20240229 (best quality)
# =============================================================================


class ClaudeClient(BaseLLMClient):
    """
    Claude LLM client for high-quality reasoning.

    Claude excels at complex analysis and structured reasoning.
    Use for deep financial analysis and complex decision-making.

    Example:
        client = ClaudeClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name="claude-3-sonnet-20240229"
        )
        response = await client.generate("Analyze AAPL fundamentals")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-sonnet-20240229",
        **kwargs
    ):
        # TODO: Initialize the Claude client
        # - Get API key from parameter or environment variable
        # - Call parent __init__
        # - Import and initialize anthropic.AsyncAnthropic client
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # if not api_key:
        #     raise ValueError("Anthropic API key required")
        #
        # super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        #
        # from anthropic import AsyncAnthropic
        # self.client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using Claude.

        TODO: Implement this method
        Steps:
        1. Build messages array (Claude uses different format than Groq)
        2. Start timer for latency measurement
        3. Call self.client.messages.create()
        4. Calculate latency
        5. Extract content and token usage
        6. Return LLMResponse object
        """
        # TODO: Implement Claude API call
        # Hint: Use self.client.messages.create()
        # Note: Claude uses system parameter separately from messages
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # start_time = time.time()
        #
        # messages = [{"role": "user", "content": prompt}]
        #
        # try:
        #     response = await self.client.messages.create(
        #         model=self.model_name,
        #         messages=messages,
        #         system=system_prompt,
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #         **kwargs
        #     )
        #
        #     latency = (time.time() - start_time) * 1000  # Convert to ms
        #
        #     return LLMResponse(
        #         content=response.content[0].text,
        #         provider=LLMProvider.CLAUDE,
        #         model=self.model_name,
        #         tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        #         latency_ms=latency,
        #         timestamp=datetime.now(),
        #         raw_response=response.model_dump()
        #     )
        # except Exception as e:
        #     self._log(f"Claude API error: {str(e)}", "ERROR")
        #     raise LLMError(f"Claude generation failed: {str(e)}")

    def validate_response(self, response: Any) -> bool:
        """
        Validate Claude API response.

        TODO: Implement validation
        - Check if response has 'content' attribute
        - Check if content is not empty
        - Check if content[0].text is not None
        """
        # TODO: Implement validation logic
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # return (
        #     hasattr(response, 'content') and
        #     len(response.content) > 0 and
        #     hasattr(response.content[0], 'text') and
        #     response.content[0].text is not None
        # )


# =============================================================================
# TODO #4: Add retry logic
# =============================================================================
# Instructions:
# 1. Install tenacity: pip install tenacity
# 2. Import retry, stop_after_attempt, wait_exponential
# 3. Create a decorator that retries on specific exceptions
# 4. Apply decorator to generate() methods
# 5. Log retry attempts
#
# Example retry behavior:
# - Attempt 1: Immediate
# - Attempt 2: Wait 1s
# - Attempt 3: Wait 2s
# - Attempt 4: Wait 4s
# - Fail after 4 attempts
# =============================================================================


# TODO: Import tenacity and create retry decorator
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# EXAMPLE RETRY DECORATOR (uncomment to use):
# def with_retry(max_attempts: int = 3):
#     """
#     Decorator to add retry logic with exponential backoff.
#
#     Args:
#         max_attempts: Maximum number of retry attempts
#     """
#     return retry(
#         stop=stop_after_attempt(max_attempts),
#         wait=wait_exponential(multiplier=1, min=1, max=10),
#         retry=retry_if_exception_type((ConnectionError, TimeoutError)),
#         reraise=True
#     )


# =============================================================================
# TODO #5: Add rate limit handling
# =============================================================================
# Instructions:
# 1. Create a RateLimiter class
# 2. Track requests per minute
# 3. Add async sleep when rate limit is reached
# 4. Integrate with LLM clients
# 5. Make limits configurable per provider
#
# Rate limits to handle:
# - Groq: 30 requests/minute (free tier)
# - Claude: 50 requests/minute (varies by tier)
# =============================================================================


class RateLimiter:
    """
    Rate limiter to prevent API quota exhaustion.

    Tracks requests and adds delays when limits are approached.
    """

    def __init__(self, requests_per_minute: int = 30):
        # TODO: Initialize rate limiter
        # - Store requests_per_minute
        # - Create a list to track request timestamps
        # - Add lock for thread safety (asyncio.Lock)
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # self.requests_per_minute = requests_per_minute
        # self.request_times: List[float] = []
        # self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire permission to make a request.

        TODO: Implement rate limiting logic
        Steps:
        1. Acquire lock
        2. Remove request times older than 1 minute
        3. If at limit, calculate wait time
        4. Sleep if necessary
        5. Add current time to request_times
        6. Release lock
        """
        # TODO: Implement acquire logic
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # async with self.lock:
        #     now = time.time()
        #
        #     # Remove requests older than 1 minute
        #     self.request_times = [
        #         t for t in self.request_times
        #         if now - t < 60
        #     ]
        #
        #     # If at limit, wait
        #     if len(self.request_times) >= self.requests_per_minute:
        #         oldest = self.request_times[0]
        #         wait_time = 60 - (now - oldest)
        #         if wait_time > 0:
        #             await asyncio.sleep(wait_time)
        #
        #     self.request_times.append(now)


# =============================================================================
# TODO #6: Add response validation
# =============================================================================
# Instructions:
# 1. Create validation functions for different response types
# 2. Add JSON extraction from markdown code blocks
# 3. Create Pydantic models for structured responses
# 4. Add error recovery for malformed responses
# 5. Log validation failures
# =============================================================================


def extract_json_from_response(content: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response, handling markdown code blocks.

    TODO: Implement JSON extraction
    Steps:
    1. Check if content contains ```json markers
    2. Extract content between markers
    3. Try parsing as JSON
    4. Return parsed JSON or None if invalid

    Args:
        content: LLM response content

    Returns:
        Parsed JSON dict or None
    """
    # TODO: Implement JSON extraction
    pass

    # EXAMPLE IMPLEMENTATION (uncomment to use):
    # import re
    #
    # # Try to find JSON in markdown code blocks
    # json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
    # if json_match:
    #     json_str = json_match.group(1)
    # else:
    #     json_str = content
    #
    # try:
    #     return json.loads(json_str)
    # except json.JSONDecodeError:
    #     return None


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded"""
    pass


class ValidationError(LLMError):
    """Raised when response validation fails"""
    pass


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """
    Example of how to use the LLM clients.

    Run this after implementing all TODOs to test your code.
    """
    # Initialize clients
    groq_client = GroqClient(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768",
        debug=True
    )

    claude_client = ClaudeClient(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model_name="claude-3-haiku-20240307",
        debug=True
    )

    # Simple prompt
    prompt = "Is AAPL a good buy right now? Answer in one sentence."

    # Get responses from both
    print("\n=== Groq Response ===")
    groq_response = await groq_client.generate(prompt)
    print(f"Content: {groq_response.content}")
    print(f"Tokens: {groq_response.tokens_used}")
    print(f"Latency: {groq_response.latency_ms:.2f}ms")

    print("\n=== Claude Response ===")
    claude_response = await claude_client.generate(prompt)
    print(f"Content: {claude_response.content}")
    print(f"Tokens: {claude_response.tokens_used}")
    print(f"Latency: {claude_response.latency_ms:.2f}ms")

    # Compare
    print("\n=== Comparison ===")
    print(f"Groq: {groq_response.latency_ms:.2f}ms, {groq_response.tokens_used} tokens")
    print(f"Claude: {claude_response.latency_ms:.2f}ms, {claude_response.tokens_used} tokens")


if __name__ == "__main__":
    # Uncomment to test your implementation
    # asyncio.run(example_usage())
    pass
