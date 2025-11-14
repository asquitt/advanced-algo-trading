# Testing LLM Integrations Without Breaking the Bank

**Time to Read**: 15-20 minutes | **Difficulty**: Intermediate

---

## Table of Contents

1. [Why Test LLM Integrations?](#why-test-llm-integrations)
2. [Mocking LLM Calls](#mocking-llm-calls)
3. [Deterministic Testing](#deterministic-testing)
4. [Response Fixtures](#response-fixtures)
5. [Integration Tests](#integration-tests)
6. [Cost-Free Testing](#cost-free-testing)

---

## Why Test LLM Integrations?

Testing LLM-powered systems presents unique challenges:

1. **Non-deterministic outputs** - Same input ≠ same output
2. **API costs** - Each test costs real money
3. **Rate limits** - Can't run hundreds of tests quickly
4. **Latency** - Tests are slow (500ms+ per LLM call)
5. **Flaky tests** - Random variations cause test failures

**Solution**: Mock LLM calls during development, use real APIs sparingly for integration tests.

### Testing Strategy Overview

```
Unit Tests (95%)              Integration Tests (5%)
-----------------             ----------------------
✓ Mock all LLM calls          ✓ Real LLM API calls
✓ Fast (<100ms per test)      ✓ Slow (1-5s per test)
✓ Free                        ✓ Costs money
✓ Deterministic               ✓ May vary
✓ Run on every commit         ✓ Run before deploy

Goal: Comprehensive coverage without breaking the bank
```

---

## Mocking LLM Calls

Mocking replaces real LLM calls with predefined responses. This is the foundation of cost-free testing.

### Using unittest.mock

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

# Your agent class
class FinancialAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def analyze_stock(self, symbol, data):
        prompt = self._build_prompt(symbol, data)
        response = await self.llm.chat(prompt)
        return self._parse_response(response)

# Test with mock
@pytest.mark.asyncio
async def test_analyze_stock_returns_buy_recommendation():
    """Test BUY recommendation path"""

    # Create mock LLM client
    mock_llm = AsyncMock()

    # Define mock response
    mock_llm.chat.return_value = json.dumps({
        "recommendation": "BUY",
        "confidence": 0.85,
        "reasoning": "Strong fundamentals with reasonable valuation",
        "risks": ["Market volatility", "Competition"],
        "catalysts": ["Product launch", "Expansion"]
    })

    # Create agent with mock
    agent = FinancialAgent(llm_client=mock_llm)

    # Test
    result = await agent.analyze_stock("AAPL", {
        "pe_ratio": 25,
        "revenue_growth": 10,
        "debt_ratio": 0.8
    })

    # Assertions
    assert result["recommendation"] == "BUY"
    assert result["confidence"] == 0.85
    assert "Strong fundamentals" in result["reasoning"]

    # Verify mock was called
    mock_llm.chat.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_stock_handles_sell_recommendation():
    """Test SELL recommendation path"""

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = json.dumps({
        "recommendation": "SELL",
        "confidence": 0.75,
        "reasoning": "Overvalued with declining growth",
        "risks": ["Earnings miss", "High debt"],
        "catalysts": []
    })

    agent = FinancialAgent(llm_client=mock_llm)

    result = await agent.analyze_stock("XYZ", {
        "pe_ratio": 50,
        "revenue_growth": -2,
        "debt_ratio": 3.0
    })

    assert result["recommendation"] == "SELL"
    assert result["confidence"] == 0.75
```

### Mocking with Side Effects

Test error handling by simulating failures:

```python
@pytest.mark.asyncio
async def test_analyze_stock_handles_llm_failure():
    """Test error handling when LLM fails"""

    mock_llm = AsyncMock()

    # Simulate API error
    mock_llm.chat.side_effect = Exception("API rate limit exceeded")

    agent = FinancialAgent(llm_client=mock_llm)

    result = await agent.analyze_stock("AAPL", {...})

    # Should return fallback
    assert result["recommendation"] == "HOLD"
    assert result["confidence"] < 0.5
    assert "unavailable" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_analyze_stock_retries_on_failure():
    """Test retry logic"""

    mock_llm = AsyncMock()

    # Fail twice, succeed third time
    mock_llm.chat.side_effect = [
        Exception("Timeout"),
        Exception("Timeout"),
        json.dumps({"recommendation": "BUY", ...})
    ]

    agent = FinancialAgent(llm_client=mock_llm)

    result = await agent.analyze_stock("AAPL", {...})

    # Should succeed after retries
    assert result["recommendation"] == "BUY"
    assert mock_llm.chat.call_count == 3
```

### Mocking with Context Managers

For patching at import time:

```python
@pytest.mark.asyncio
@patch('myapp.llm_client.ClaudeClient')
async def test_analyze_with_patched_client(mock_claude_class):
    """Test with patched Claude client"""

    # Setup mock
    mock_instance = AsyncMock()
    mock_instance.chat.return_value = json.dumps({
        "recommendation": "BUY",
        "confidence": 0.8,
        "reasoning": "Test reasoning",
        "risks": ["Test risk"],
        "catalysts": ["Test catalyst"]
    })

    mock_claude_class.return_value = mock_instance

    # Import and use (will use mock)
    from myapp.agent import FinancialAgent

    agent = FinancialAgent()
    result = await agent.analyze_stock("AAPL", {...})

    assert result["recommendation"] == "BUY"
```

---

## Deterministic Testing

Make non-deterministic LLMs behave deterministically in tests.

### Force Deterministic Responses

```python
@pytest.fixture
def deterministic_llm():
    """LLM client with temperature=0"""

    async def mock_chat(messages, temperature=0.0, **kwargs):
        # Always use temperature 0 in tests
        assert temperature == 0.0, "Tests must use temperature=0"

        # Return deterministic mock response
        return json.dumps({
            "recommendation": "BUY",
            "confidence": 0.75,
            "reasoning": "Deterministic test response",
            "risks": ["Risk 1"],
            "catalysts": ["Catalyst 1"]
        })

    mock = AsyncMock()
    mock.chat = mock_chat
    return mock


@pytest.mark.asyncio
async def test_with_deterministic_llm(deterministic_llm):
    """Test always produces same result"""

    agent = FinancialAgent(llm_client=deterministic_llm)

    # Run test 10 times
    results = []
    for _ in range(10):
        result = await agent.analyze_stock("AAPL", {...})
        results.append(result)

    # All results should be identical
    for result in results[1:]:
        assert result == results[0]
```

### Snapshot Testing

Capture LLM output once, verify it doesn't change:

```python
import pytest
from syrupy import snapshot

@pytest.mark.asyncio
async def test_prompt_output_snapshot(mock_llm, snapshot):
    """Ensure prompt changes are intentional"""

    mock_llm.chat.return_value = json.dumps({
        "recommendation": "BUY",
        "confidence": 0.8,
        "reasoning": "Expected reasoning",
        "risks": ["Risk"],
        "catalysts": ["Catalyst"]
    })

    agent = FinancialAgent(llm_client=mock_llm)

    result = await agent.analyze_stock("AAPL", {
        "pe_ratio": 28,
        "revenue_growth": 8,
        "debt_ratio": 1.2
    })

    # Compare to snapshot
    assert result == snapshot

    # On first run, creates snapshot file
    # On subsequent runs, compares to snapshot
    # If changed, test fails (update with --snapshot-update)
```

### Testing Edge Cases

Test boundary conditions:

```python
@pytest.mark.parametrize("confidence,expected_action", [
    (0.95, "execute_trade"),
    (0.75, "execute_trade"),
    (0.59, "skip_trade"),  # Below threshold
    (0.30, "skip_trade"),
    (0.0, "skip_trade")
])
@pytest.mark.asyncio
async def test_confidence_threshold(mock_llm, confidence, expected_action):
    """Test confidence threshold logic"""

    mock_llm.chat.return_value = json.dumps({
        "recommendation": "BUY",
        "confidence": confidence,
        "reasoning": "Test",
        "risks": ["Risk"],
        "catalysts": ["Catalyst"]
    })

    agent = FinancialAgent(llm_client=mock_llm)
    result = await agent.analyze_stock("AAPL", {...})

    if expected_action == "execute_trade":
        assert result["should_trade"] is True
    else:
        assert result["should_trade"] is False
```

---

## Response Fixtures

Create reusable test data for consistent testing.

### Fixture Files

```python
# tests/fixtures/llm_responses.py

"""Standardized LLM response fixtures"""

BUY_STRONG = {
    "recommendation": "BUY",
    "confidence": 0.90,
    "reasoning": "Exceptional fundamentals. PE of 22 below sector average of 28. Revenue growth of 15% with improving margins. Low debt provides safety.",
    "risks": ["Market downturn", "Sector rotation"],
    "catalysts": ["Product launch Q3", "Market expansion", "Cost optimization"],
    "target_price": 185.50
}

BUY_WEAK = {
    "recommendation": "BUY",
    "confidence": 0.62,
    "reasoning": "Modest upside. PE reasonable at 25. Growth slowing but stable. Some debt concerns.",
    "risks": ["Slowing growth", "High debt", "Competition"],
    "catalysts": ["New management", "Restructuring"],
    "target_price": 145.00
}

SELL_STRONG = {
    "recommendation": "SELL",
    "confidence": 0.85,
    "reasoning": "Severely overvalued. PE of 65 more than double sector average. Decelerating growth. High debt raises concerns.",
    "risks": ["Earnings miss imminent", "Debt covenant breach", "Management exodus"],
    "catalysts": [],
    "target_price": 95.00
}

HOLD_NEUTRAL = {
    "recommendation": "HOLD",
    "confidence": 0.55,
    "reasoning": "Mixed signals. Valuation fair but growth uncertain. Wait for clarity.",
    "risks": ["Unclear direction", "Competitive pressure"],
    "catalysts": ["Potential acquisition", "Market recovery"]
}

INVALID_RECOMMENDATION = {
    "recommendation": "MAYBE",  # Invalid value
    "confidence": 0.5,
    "reasoning": "Unclear",
    "risks": [],
    "catalysts": []
}

MALFORMED_JSON = "{invalid json content"

MISSING_REQUIRED_FIELDS = {
    "recommendation": "BUY"
    # Missing confidence, reasoning, risks, catalysts
}
```

### Using Fixtures in Tests

```python
import pytest
from tests.fixtures.llm_responses import *

@pytest.fixture
def mock_llm_with_fixture():
    """Mock LLM that returns fixture responses"""

    class MockLLM:
        def __init__(self):
            self._responses = []

        def set_response(self, response):
            """Set next response"""
            self._responses.append(response)

        async def chat(self, *args, **kwargs):
            """Return next queued response"""
            if not self._responses:
                raise Exception("No response queued")
            response = self._responses.pop(0)
            return json.dumps(response) if isinstance(response, dict) else response

    return MockLLM()


@pytest.mark.asyncio
async def test_strong_buy_signal(mock_llm_with_fixture):
    """Test strong BUY recommendation"""

    mock_llm_with_fixture.set_response(BUY_STRONG)
    agent = FinancialAgent(llm_client=mock_llm_with_fixture)

    result = await agent.analyze_stock("AAPL", {...})

    assert result["recommendation"] == "BUY"
    assert result["confidence"] >= 0.8
    assert result["target_price"] is not None


@pytest.mark.asyncio
async def test_handles_invalid_recommendation(mock_llm_with_fixture):
    """Test handling of invalid recommendation"""

    mock_llm_with_fixture.set_response(INVALID_RECOMMENDATION)
    agent = FinancialAgent(llm_client=mock_llm_with_fixture)

    with pytest.raises(ValueError, match="Invalid recommendation"):
        await agent.analyze_stock("AAPL", {...})


@pytest.mark.asyncio
async def test_handles_malformed_json(mock_llm_with_fixture):
    """Test handling of malformed JSON"""

    mock_llm_with_fixture.set_response(MALFORMED_JSON)
    agent = FinancialAgent(llm_client=mock_llm_with_fixture)

    # Should fall back gracefully
    result = await agent.analyze_stock("AAPL", {...})
    assert result["recommendation"] == "HOLD"
    assert "error" in result.get("metadata", {})


@pytest.mark.parametrize("fixture", [
    BUY_STRONG,
    BUY_WEAK,
    SELL_STRONG,
    HOLD_NEUTRAL
])
@pytest.mark.asyncio
async def test_all_recommendation_types(mock_llm_with_fixture, fixture):
    """Test all recommendation types parse correctly"""

    mock_llm_with_fixture.set_response(fixture)
    agent = FinancialAgent(llm_client=mock_llm_with_fixture)

    result = await agent.analyze_stock("TEST", {...})

    # Should parse without errors
    assert result["recommendation"] in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= result["confidence"] <= 1.0
```

### Fixture Generators

Create fixtures dynamically:

```python
def create_buy_response(confidence=0.8, **overrides):
    """Generate BUY response with custom values"""
    response = {
        "recommendation": "BUY",
        "confidence": confidence,
        "reasoning": f"Test BUY with confidence {confidence}",
        "risks": ["Default risk"],
        "catalysts": ["Default catalyst"]
    }
    response.update(overrides)
    return response


@pytest.mark.asyncio
async def test_confidence_levels():
    """Test different confidence levels"""

    for confidence in [0.95, 0.80, 0.65, 0.50]:
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = json.dumps(
            create_buy_response(confidence=confidence)
        )

        agent = FinancialAgent(llm_client=mock_llm)
        result = await agent.analyze_stock("TEST", {...})

        assert result["confidence"] == confidence
```

---

## Integration Tests

Test with real LLMs occasionally to ensure mocks match reality.

### Marking Integration Tests

```python
import pytest
import os

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Integration tests disabled (set RUN_INTEGRATION_TESTS=1)"
)
@pytest.mark.asyncio
async def test_real_groq_analysis():
    """Test with real Groq API (costs ~$0.0001)"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    # Use real client
    from myapp.llm_client import GroqClient
    client = GroqClient(api_key=api_key)

    agent = FinancialAgent(llm_client=client)

    result = await agent.analyze_stock("AAPL", {
        "pe_ratio": 28.5,
        "revenue_growth": 8.0,
        "debt_ratio": 1.2
    })

    # Basic sanity checks
    assert result["recommendation"] in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["reasoning"]) > 10
    assert len(result["risks"]) >= 1


@pytest.mark.integration
@pytest.mark.expensive  # Custom marker for costly tests
@pytest.mark.asyncio
async def test_real_claude_analysis():
    """Test with real Claude API (costs ~$0.01)"""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from myapp.llm_client import ClaudeClient
    client = ClaudeClient(api_key=api_key)

    agent = FinancialAgent(llm_client=client)

    result = await agent.analyze_stock("AAPL", {
        "pe_ratio": 28.5,
        "revenue_growth": 8.0,
        "debt_ratio": 1.2,
        "profit_margin": 25.0,
        "sector_pe": 22.0
    })

    # Verify quality of Claude response
    assert result["recommendation"] in ["BUY", "SELL", "HOLD"]
    assert len(result["reasoning"]) > 50  # Claude gives detailed reasoning
    assert len(result["risks"]) >= 2  # Should identify multiple risks
```

### Running Integration Tests Selectively

```bash
# Skip integration tests (default)
pytest

# Run only integration tests
pytest -m integration

# Run all tests including integration
RUN_INTEGRATION_TESTS=1 pytest

# Run integration tests but skip expensive ones
pytest -m "integration and not expensive"
```

### Cost-Controlled Integration Testing

```python
class CostTracker:
    """Track integration test costs"""

    MAX_DAILY_COST = 1.00  # $1/day limit

    def __init__(self):
        self.total_cost = 0.0

    def can_run_test(self, estimated_cost: float) -> bool:
        """Check if test is within budget"""
        return (self.total_cost + estimated_cost) <= self.MAX_DAILY_COST

    def record_cost(self, actual_cost: float):
        """Record test cost"""
        self.total_cost += actual_cost

# Global tracker
cost_tracker = CostTracker()


@pytest.fixture
def check_budget():
    """Skip test if over budget"""
    if not cost_tracker.can_run_test(0.01):
        pytest.skip(f"Daily budget exceeded: ${cost_tracker.total_cost:.2f}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expensive_analysis(check_budget):
    """Test that respects budget"""

    # Run test...
    result = await real_llm.analyze(...)

    # Record cost
    cost_tracker.record_cost(0.01)
```

---

## Cost-Free Testing

Techniques to test thoroughly without spending money.

### Local LLM for Testing

Use a local model for tests:

```python
@pytest.fixture
def local_llm():
    """Use Ollama for local testing (free)"""

    try:
        import ollama
    except ImportError:
        pytest.skip("Ollama not installed")

    class LocalLLM:
        async def chat(self, messages, **kwargs):
            response = ollama.chat(
                model='llama2',
                messages=messages
            )
            return response['message']['content']

    return LocalLLM()


@pytest.mark.asyncio
async def test_with_local_llm(local_llm):
    """Test with free local LLM"""

    agent = FinancialAgent(llm_client=local_llm)
    result = await agent.analyze_stock("AAPL", {...})

    # Won't be as good as Claude, but free for testing
    assert result["recommendation"] in ["BUY", "SELL", "HOLD"]
```

### Record and Replay

Record real responses, replay in tests:

```python
import json
from pathlib import Path

class RecordingLLM:
    """Record LLM responses for replay"""

    def __init__(self, real_llm, recordings_dir="tests/recordings"):
        self.real_llm = real_llm
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(exist_ok=True)

    async def chat(self, messages, **kwargs):
        # Generate key from messages
        key = hashlib.md5(str(messages).encode()).hexdigest()
        recording_file = self.recordings_dir / f"{key}.json"

        # Try to load recording
        if recording_file.exists():
            print(f"Replaying recording: {key}")
            with open(recording_file) as f:
                return json.load(f)["response"]

        # Make real call
        print(f"Recording new response: {key}")
        response = await self.real_llm.chat(messages, **kwargs)

        # Save recording
        with open(recording_file, 'w') as f:
            json.dump({
                "messages": messages,
                "kwargs": kwargs,
                "response": response
            }, f, indent=2)

        return response


# Usage in tests
@pytest.fixture
def llm_client(request):
    """Use recording LLM in tests"""

    if request.config.getoption("--record"):
        # Recording mode: use real LLM
        return RecordingLLM(real_llm=RealGroqClient())
    else:
        # Replay mode: use recordings only
        return RecordingLLM(real_llm=None)
```

### Parameterized Testing

Test multiple scenarios with one test:

```python
@pytest.mark.parametrize("scenario", [
    {
        "name": "undervalued_growth",
        "data": {"pe_ratio": 18, "revenue_growth": 15, "debt_ratio": 0.5},
        "expected": "BUY",
        "min_confidence": 0.7
    },
    {
        "name": "overvalued_declining",
        "data": {"pe_ratio": 45, "revenue_growth": -3, "debt_ratio": 2.5},
        "expected": "SELL",
        "min_confidence": 0.7
    },
    {
        "name": "fair_value_stable",
        "data": {"pe_ratio": 22, "revenue_growth": 5, "debt_ratio": 1.0},
        "expected": "HOLD",
        "min_confidence": 0.5
    }
])
@pytest.mark.asyncio
async def test_analysis_scenarios(mock_llm_with_fixture, scenario):
    """Test various market scenarios"""

    # Setup mock based on expected outcome
    mock_llm_with_fixture.set_response({
        "recommendation": scenario["expected"],
        "confidence": scenario["min_confidence"],
        "reasoning": f"Test for {scenario['name']}",
        "risks": ["Risk 1"],
        "catalysts": ["Catalyst 1"]
    })

    agent = FinancialAgent(llm_client=mock_llm_with_fixture)
    result = await agent.analyze_stock("TEST", scenario["data"])

    assert result["recommendation"] == scenario["expected"]
    assert result["confidence"] >= scenario["min_confidence"]
```

---

## Summary

### Testing Pyramid

```
       /\
      /  \     Integration Tests (5%)
     /    \    - Real API calls
    /------\   - Expensive
   /        \  - Slow
  /          \
 /    Unit    \ Unit Tests (95%)
/    Tests     \- Mocked LLMs
----------------\- Free
                 - Fast
```

### Best Practices Checklist

- [ ] Mock LLM calls in unit tests
- [ ] Use fixtures for consistent test data
- [ ] Test with temperature=0 for determinism
- [ ] Test error handling (API failures, malformed responses)
- [ ] Parameterize tests for multiple scenarios
- [ ] Mark integration tests with `@pytest.mark.integration`
- [ ] Set budget limits for integration tests
- [ ] Use record/replay for expensive tests
- [ ] Verify Pydantic validation with invalid data
- [ ] Test retry logic with side effects

### Quick Reference

```python
# Mock LLM in test
@pytest.mark.asyncio
async def test_analysis(mock_llm):
    mock_llm.chat.return_value = json.dumps(BUY_STRONG)
    agent = FinancialAgent(llm_client=mock_llm)
    result = await agent.analyze_stock("AAPL", {...})
    assert result["recommendation"] == "BUY"

# Run tests
pytest                              # Unit tests only
pytest -m integration              # Integration tests
RUN_INTEGRATION_TESTS=1 pytest     # All tests
```

### Cost Comparison

| Testing Approach | Cost per Test Run | Feedback Time |
|------------------|-------------------|---------------|
| Mocked (recommended) | $0 | <1 second |
| Groq integration | ~$0.01 | ~5 seconds |
| Claude integration | ~$0.10 | ~10 seconds |
| No tests | $1000s+ | Production failures |

### Next Steps

1. Set up pytest with fixtures
2. Create mock responses for common scenarios
3. Write unit tests for all agent methods
4. Add integration tests for critical paths
5. Set up CI/CD to run unit tests on every commit
6. Run integration tests before major releases

**Remember**: Testing LLM integrations is not optional. The cost of bugs in production far exceeds the cost of proper testing. But you can test thoroughly without spending money—mock everything, test with real APIs sparingly.

Good testing gives you confidence to deploy. Start testing today!
