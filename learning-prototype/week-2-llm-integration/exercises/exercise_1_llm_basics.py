"""
Exercise 1: Basic LLM API Calls
================================

Objective: Make your first LLM API call and compare Groq vs Claude

Time: 1 hour
Difficulty: Beginner

What You'll Learn:
- How to call LLM APIs (Groq and Claude)
- Compare response quality and latency
- Understand API authentication
- Handle basic errors
- Measure performance

Prerequisites:
- Python 3.8+
- Groq API key (get from https://console.groq.com)
- Anthropic API key (get from https://console.anthropic.com)
- Required packages: requests, anthropic

Setup:
------
pip install requests anthropic

Set environment variables:
export GROQ_API_KEY="your_groq_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"
"""

import os
import time
import json
from typing import Dict, Any


# ============================================================================
# TODO #1: Implement Groq API call
# ============================================================================

def call_groq_api(prompt: str, model: str = "llama3-8b-8192") -> Dict[str, Any]:
    """
    Call Groq API with a prompt and return response with metadata.

    Args:
        prompt: The text prompt to send to the model
        model: Groq model to use (default: llama3-8b-8192)

    Returns:
        Dictionary with:
        - response: The model's text response
        - latency_ms: Time taken in milliseconds
        - tokens: Number of tokens used
        - model: Model name used

    Hints:
    - Groq API endpoint: https://api.groq.com/openai/v1/chat/completions
    - Use Bearer token authentication
    - Format: {"Authorization": f"Bearer {api_key}"}
    - Request body should include "model" and "messages"
    - Messages format: [{"role": "user", "content": prompt}]

    Example response structure:
    {
        "choices": [{"message": {"content": "..."}}],
        "usage": {"total_tokens": 100}
    }
    """
    # TODO: Get GROQ_API_KEY from environment
    api_key = None  # Replace with: os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    # TODO: Set up API endpoint and headers
    url = ""  # Fill in: "https://api.groq.com/openai/v1/chat/completions"
    headers = {}  # Fill in: authorization and content-type

    # TODO: Create request body
    body = {
        # Add "model" field
        # Add "messages" field with user message
    }

    # TODO: Make API request and measure latency
    start_time = time.time()
    # response = requests.post(url, headers=headers, json=body)
    # latency_ms = (time.time() - start_time) * 1000

    # TODO: Parse response
    # data = response.json()
    # text_response = data["choices"][0]["message"]["content"]
    # tokens = data["usage"]["total_tokens"]

    # TODO: Return structured result
    return {
        "response": "",  # Fill in
        "latency_ms": 0,  # Fill in
        "tokens": 0,  # Fill in
        "model": model
    }


# ============================================================================
# TODO #2: Implement Claude API call
# ============================================================================

def call_claude_api(prompt: str, model: str = "claude-3-haiku-20240307") -> Dict[str, Any]:
    """
    Call Claude API with a prompt and return response with metadata.

    Args:
        prompt: The text prompt to send to the model
        model: Claude model to use (default: claude-3-haiku-20240307)

    Returns:
        Dictionary with same structure as call_groq_api

    Hints:
    - Use the official anthropic Python library
    - Import: from anthropic import Anthropic
    - Client initialization: client = Anthropic(api_key=api_key)
    - Method: client.messages.create(model=..., max_tokens=..., messages=...)
    - Response structure is different from Groq!
    - Access text via: response.content[0].text
    - Access tokens via: response.usage.input_tokens + response.usage.output_tokens
    """
    # TODO: Get ANTHROPIC_API_KEY from environment
    api_key = None  # Replace with: os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    # TODO: Initialize Anthropic client
    # from anthropic import Anthropic
    # client = Anthropic(api_key=api_key)

    # TODO: Make API request and measure latency
    start_time = time.time()
    # response = client.messages.create(
    #     model=model,
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # latency_ms = (time.time() - start_time) * 1000

    # TODO: Parse response
    # text_response = response.content[0].text
    # tokens = response.usage.input_tokens + response.usage.output_tokens

    # TODO: Return structured result
    return {
        "response": "",  # Fill in
        "latency_ms": 0,  # Fill in
        "tokens": 0,  # Fill in
        "model": model
    }


# ============================================================================
# TODO #3: Compare model responses
# ============================================================================

def compare_models(prompt: str) -> None:
    """
    Call both Groq and Claude with the same prompt and compare results.

    This function should:
    1. Call both APIs with the same prompt
    2. Print side-by-side comparison
    3. Show latency difference
    4. Show token usage
    5. Highlight which is faster/cheaper

    Hints:
    - Use try/except to handle API errors gracefully
    - Format output nicely with headers and separators
    - Calculate cost: Groq ~$0.0001/1K tokens, Claude ~$0.25/1M input + $1.25/1M output
    """
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"Prompt: {prompt[:100]}...\n")

    # TODO: Call Groq API
    try:
        # groq_result = call_groq_api(prompt)
        # Print Groq results
        pass
    except Exception as e:
        print(f"❌ Groq API error: {e}")

    # TODO: Call Claude API
    try:
        # claude_result = call_claude_api(prompt)
        # Print Claude results
        pass
    except Exception as e:
        print(f"❌ Claude API error: {e}")

    # TODO: Compare and print summary
    # Show which is faster
    # Show which is cheaper
    # Show response length differences


# ============================================================================
# TODO #4: Measure latency across multiple calls
# ============================================================================

def benchmark_latency(prompt: str, num_calls: int = 5) -> Dict[str, Any]:
    """
    Measure average latency for both models over multiple calls.

    Args:
        prompt: Test prompt to use
        num_calls: Number of calls to average over

    Returns:
        Dictionary with benchmark results for both models

    Expected output structure:
    {
        "groq": {
            "avg_latency_ms": 250.5,
            "min_latency_ms": 200.0,
            "max_latency_ms": 300.0,
            "avg_tokens": 150
        },
        "claude": {
            ...same structure...
        }
    }
    """
    # TODO: Run multiple calls for Groq
    groq_latencies = []
    groq_tokens = []

    # for i in range(num_calls):
    #     try:
    #         result = call_groq_api(prompt)
    #         groq_latencies.append(result["latency_ms"])
    #         groq_tokens.append(result["tokens"])
    #     except Exception as e:
    #         print(f"Groq call {i+1} failed: {e}")

    # TODO: Run multiple calls for Claude
    claude_latencies = []
    claude_tokens = []

    # for i in range(num_calls):
    #     try:
    #         result = call_claude_api(prompt)
    #         claude_latencies.append(result["latency_ms"])
    #         claude_tokens.append(result["tokens"])
    #     except Exception as e:
    #         print(f"Claude call {i+1} failed: {e}")

    # TODO: Calculate statistics
    # Use min(), max(), sum()/len() for calculations

    return {
        "groq": {
            "avg_latency_ms": 0,  # Calculate
            "min_latency_ms": 0,  # Calculate
            "max_latency_ms": 0,  # Calculate
            "avg_tokens": 0  # Calculate
        },
        "claude": {
            "avg_latency_ms": 0,  # Calculate
            "min_latency_ms": 0,  # Calculate
            "max_latency_ms": 0,  # Calculate
            "avg_tokens": 0  # Calculate
        }
    }


# ============================================================================
# Test Cases - Run these to verify your implementation
# ============================================================================

def test_groq_api():
    """Test basic Groq API call."""
    print("\n" + "=" * 80)
    print("TEST: Groq API Call")
    print("=" * 80)

    prompt = "What is 2+2? Answer in one sentence."

    try:
        result = call_groq_api(prompt)
        print(f"✅ Success!")
        print(f"   Response: {result['response'][:100]}")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        print(f"   Tokens: {result['tokens']}")
        assert result['response'], "Response should not be empty"
        assert result['latency_ms'] > 0, "Latency should be positive"
        assert result['tokens'] > 0, "Tokens should be positive"
    except Exception as e:
        print(f"❌ Failed: {e}")


def test_claude_api():
    """Test basic Claude API call."""
    print("\n" + "=" * 80)
    print("TEST: Claude API Call")
    print("=" * 80)

    prompt = "What is 2+2? Answer in one sentence."

    try:
        result = call_claude_api(prompt)
        print(f"✅ Success!")
        print(f"   Response: {result['response'][:100]}")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        print(f"   Tokens: {result['tokens']}")
        assert result['response'], "Response should not be empty"
        assert result['latency_ms'] > 0, "Latency should be positive"
        assert result['tokens'] > 0, "Tokens should be positive"
    except Exception as e:
        print(f"❌ Failed: {e}")


def test_comparison():
    """Test model comparison."""
    print("\n" + "=" * 80)
    print("TEST: Model Comparison")
    print("=" * 80)

    prompt = "Explain what a P/E ratio is in one sentence."
    compare_models(prompt)


def test_benchmark():
    """Test latency benchmarking."""
    print("\n" + "=" * 80)
    print("TEST: Latency Benchmark")
    print("=" * 80)

    prompt = "What is algorithmic trading?"
    results = benchmark_latency(prompt, num_calls=3)

    print("\nBenchmark Results:")
    print(f"Groq avg latency: {results['groq']['avg_latency_ms']:.2f}ms")
    print(f"Claude avg latency: {results['claude']['avg_latency_ms']:.2f}ms")

    if results['groq']['avg_latency_ms'] < results['claude']['avg_latency_ms']:
        speedup = results['claude']['avg_latency_ms'] / results['groq']['avg_latency_ms']
        print(f"\n🚀 Groq is {speedup:.1f}x faster!")
    else:
        speedup = results['groq']['avg_latency_ms'] / results['claude']['avg_latency_ms']
        print(f"\n🚀 Claude is {speedup:.1f}x faster!")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           Exercise 1: Basic LLM API Calls                          ║
    ║                                                                    ║
    ║  Complete the TODOs above and run this file to test!              ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Check environment variables
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not set")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set")

    print("\nRunning tests...\n")

    # Run all tests
    test_groq_api()
    test_claude_api()
    test_comparison()
    test_benchmark()

    print("\n" + "=" * 80)
    print("EXERCISE COMPLETE!")
    print("=" * 80)
    print("""
    Next Steps:
    1. Review your implementation
    2. Experiment with different prompts
    3. Try different models (e.g., llama3-70b-8192, claude-3-opus-20240229)
    4. Move on to Exercise 2: Prompt Engineering

    Key Learnings:
    ✅ How to authenticate with LLM APIs
    ✅ How to make API requests
    ✅ How to parse responses
    ✅ Groq is typically faster (5-10x)
    ✅ Claude often provides more detailed responses
    ✅ Both have different pricing models
    """)
