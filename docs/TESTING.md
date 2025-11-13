# Testing Guide

Comprehensive testing guide for the LLM Trading Platform.

## Overview

The platform has extensive test coverage across multiple layers:

| Test Type | Location | Purpose | Speed |
|-----------|----------|---------|-------|
| Unit Tests | `tests/test_*.py` | Test individual components | Fast (<1s) |
| Integration Tests | `tests/test_integration.py` | Test component interactions | Medium (1-5s) |
| Performance Tests | `tests/test_performance.py` | Measure latency and throughput | Slow (10-30s) |

**Current Coverage:** ~70-80% of code

## Quick Start

### Run All Tests

```bash
./scripts/run_tests.sh
```

### Run Specific Test Suites

```bash
# Unit tests only (fast)
./scripts/run_tests.sh unit

# Integration tests
./scripts/run_tests.sh integration

# Performance tests
./scripts/run_tests.sh performance

# HFT technique tests
./scripts/run_tests.sh hft

# Generate coverage report
./scripts/run_tests.sh coverage
```

### Run Individual Test Files

```bash
# Test utilities
pytest tests/test_utils.py -v

# Test data models
pytest tests/test_data_models.py -v

# Test HFT techniques
pytest tests/test_hft_techniques.py -v

# Test LLM agents
pytest tests/test_llm_agents.py -v

# Test trading engine
pytest tests/test_trading_engine.py -v
```

### Run Specific Tests

```bash
# Run a specific test class
pytest tests/test_utils.py::TestCache -v

# Run a specific test function
pytest tests/test_utils.py::TestCache::test_cache_set_and_get -v

# Run tests matching a pattern
pytest -k "cache" -v

# Run tests with a marker
pytest -m unit -v
```

## Test Structure

### Directory Layout

```
tests/
├── __init__.py                # Test package
├── conftest.py                # Shared fixtures
├── test_utils.py              # Utility tests
├── test_data_models.py        # Data model tests
├── test_hft_techniques.py     # HFT technique tests
├── test_llm_agents.py         # LLM agent tests
├── test_trading_engine.py     # Trading engine tests
├── test_integration.py        # Integration tests
└── test_performance.py        # Performance tests
```

### Test Organization

Each test file follows this pattern:

```python
"""
Module docstring explaining what is tested.
"""

import pytest
from unittest.mock import Mock, patch

class TestComponentName:
    """Test a specific component."""

    def test_basic_functionality(self):
        """Test description."""
        # Arrange
        input_data = {...}

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_output

    @patch('module.external_dependency')
    def test_with_mocking(self, mock_dependency):
        """Test with external dependencies mocked."""
        # Setup mock
        mock_dependency.return_value = "mocked value"

        # Test
        result = function_that_uses_dependency()

        # Verify
        assert result == "expected"
        mock_dependency.assert_called_once()
```

## Fixtures

Shared fixtures are defined in `conftest.py`:

### Configuration Fixtures

```python
def test_with_settings(mock_settings):
    """Use mock settings."""
    assert mock_settings.paper_trading is True
```

### Data Fixtures

```python
def test_signal(sample_trading_signal):
    """Use sample trading signal."""
    assert sample_trading_signal.symbol == "AAPL"

def test_quote(sample_quote):
    """Use sample market quote."""
    assert sample_quote["price"] > 0
```

### Mock Fixtures

```python
def test_broker(mock_broker):
    """Use mock broker."""
    mock_broker.get_account.return_value = {...}

def test_cache(mock_cache):
    """Use mock cache."""
    mock_cache.set("key", "value")
```

## Writing Tests

### Unit Test Example

```python
class TestStatisticalArbitrage:
    """Test statistical arbitrage calculations."""

    def test_zscore_calculation(self):
        """Test z-score calculation for mean reversion."""
        from src.trading_engine.hft_techniques import StatisticalArbitrage

        stat_arb = StatisticalArbitrage(lookback_window=20)

        # Price series with clear mean
        prices = [100.0] * 18 + [110.0, 115.0]

        zscore = stat_arb.calculate_zscore(prices)

        # Z-score should be positive (price above mean)
        assert zscore > 0
```

### Integration Test Example

```python
@patch('src.llm_agents.financial_agent.market_data')
@patch.object('src.llm_agents.base_agent.BaseLLMAgent', '_call_llm')
def test_full_signal_pipeline(mock_llm, mock_data, sample_company_info):
    """Test generating a signal through the full pipeline."""
    from src.llm_agents.ensemble_strategy import EnsembleStrategy

    # Mock data sources
    mock_data.get_company_info.return_value = sample_company_info

    # Mock LLM response
    mock_llm.return_value = ('{"score": 80, ...}', 1500, 0.0045)

    # Generate signal
    strategy = EnsembleStrategy()
    signal = strategy.generate_signal("AAPL", use_cache=False)

    # Verify
    assert signal is not None
    assert signal.symbol == "AAPL"
```

### Performance Test Example

```python
def test_signal_generation_latency(mock_llm, mock_data):
    """Test signal generation completes within acceptable time."""
    import time
    from src.llm_agents.ensemble_strategy import EnsembleStrategy

    # Setup mocks
    mock_data.get_company_info.return_value = {...}
    mock_llm.return_value = ('{"score": 75, ...}', 1000, 0.003)

    # Measure time
    strategy = EnsembleStrategy()
    start_time = time.time()
    signal = strategy.generate_signal("AAPL", use_cache=False)
    elapsed_ms = (time.time() - start_time) * 1000

    # Should complete within 500ms
    assert elapsed_ms < 500
```

## Coverage

### Viewing Coverage

```bash
# Generate HTML coverage report
./scripts/run_tests.sh coverage

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Goals

| Component | Current | Target |
|-----------|---------|--------|
| Utils | 85% | 90% |
| Data Models | 90% | 95% |
| HFT Techniques | 80% | 85% |
| LLM Agents | 75% | 80% |
| Trading Engine | 70% | 80% |
| **Overall** | **75%** | **85%** |

### Improving Coverage

1. **Identify uncovered code:**
   ```bash
   pytest --cov=src --cov-report=term-missing
   ```

2. **Focus on critical paths:**
   - Trade execution logic
   - Risk management
   - Signal generation

3. **Add edge case tests:**
   - Error conditions
   - Boundary values
   - Invalid inputs

## Mocking

### Mocking External APIs

```python
@patch('src.llm_agents.base_agent.Groq')
def test_groq_api(mock_groq):
    """Test Groq API integration."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "LLM response"
    mock_response.usage.total_tokens = 1000

    mock_groq.return_value.chat.completions.create.return_value = mock_response

    # Test
    agent = FinancialAnalyzerAgent()
    response, tokens, cost = agent._call_groq("test prompt")

    # Verify
    assert tokens == 1000
```

### Mocking Database

```python
@patch('src.utils.database.SessionLocal')
def test_database_operations(mock_session):
    """Test database operations."""
    # Mock session
    mock_db = MagicMock()
    mock_session.return_value = mock_db

    # Test
    from src.utils.database import get_db
    with get_db() as db:
        pass

    # Verify
    mock_db.commit.assert_called_once()
```

### Mocking Broker

```python
@patch('src.trading_engine.executor.broker')
def test_trade_execution(mock_broker, sample_signal):
    """Test trade execution."""
    # Mock broker
    mock_broker.get_account.return_value = {
        "portfolio_value": 100000.0,
        "buying_power": 50000.0,
    }
    mock_broker.place_market_order.return_value = Trade(...)

    # Test
    executor = TradingExecutor()
    trade = executor.execute_signal(sample_signal)

    # Verify
    assert trade is not None
```

## Best Practices

### 1. Arrange-Act-Assert Pattern

```python
def test_example():
    # Arrange: Setup test data
    input_data = {...}

    # Act: Execute the function
    result = function_under_test(input_data)

    # Assert: Verify the output
    assert result == expected
```

### 2. Test One Thing

```python
# Good: Tests one specific behavior
def test_cache_hit_returns_value():
    cache.set("key", "value")
    assert cache.get("key") == "value"

# Bad: Tests multiple things
def test_cache():
    cache.set("key", "value")
    assert cache.get("key") == "value"
    cache.delete("key")
    assert cache.get("key") is None
    cache.set("key2", "value2", ttl=60)
    # Too much in one test!
```

### 3. Use Descriptive Names

```python
# Good: Clear what's being tested
def test_zscore_is_positive_when_price_above_mean():
    ...

# Bad: Unclear
def test_zscore():
    ...
```

### 4. Test Edge Cases

```python
def test_position_sizing():
    # Normal case
    assert calculate_size(100000, 0.02) == 2000

    # Edge cases
    assert calculate_size(0, 0.02) == 0  # Zero portfolio
    assert calculate_size(100000, 0) == 0  # Zero risk
    assert calculate_size(100000, 1.0) == 100000  # Max risk
```

### 5. Mock External Dependencies

```python
# Good: Mock external API
@patch('src.data_layer.market_data.requests.get')
def test_quote_fetching(mock_get):
    mock_get.return_value.json.return_value = {"price": 150.0}
    quote = get_quote("AAPL")
    assert quote["price"] == 150.0

# Bad: Call real API (slow, unreliable, costs money)
def test_quote_fetching():
    quote = get_quote("AAPL")  # Real API call!
    assert quote["price"] > 0
```

## Continuous Integration

### GitHub Actions (Example)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Debugging Tests

### Run with verbose output

```bash
pytest -vv tests/test_utils.py
```

### Print debug information

```python
def test_with_debug():
    result = function()
    print(f"Debug: result = {result}")  # Will show in output
    assert result == expected
```

### Run with pdb (debugger)

```bash
pytest --pdb tests/test_utils.py
```

### Run failed tests only

```bash
pytest --lf  # Last failed
pytest --ff  # Failed first
```

## Performance Testing

### Benchmarking

```python
def test_performance():
    import time

    start = time.time()
    for _ in range(1000):
        function()
    elapsed = time.time() - start

    # Should complete in < 1 second
    assert elapsed < 1.0
```

### Profiling

```bash
# Profile tests
pytest --profile tests/

# Profile with cProfile
python -m cProfile -s cumtime -m pytest tests/
```

## Troubleshooting

### Tests Failing Locally

1. **Check dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Clear cache:**
   ```bash
   pytest --cache-clear
   ```

3. **Run in verbose mode:**
   ```bash
   pytest -vv
   ```

### Import Errors

```bash
# Make sure PYTHONPATH includes src/
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Flaky Tests

If tests pass sometimes and fail others:
- Check for race conditions
- Ensure proper mocking
- Add delays if needed (use `time.sleep()`)
- Use `pytest-repeat` to run multiple times

## Further Reading

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Mock Library](https://docs.python.org/3/library/unittest.mock.html)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**Remember**: Good tests are fast, isolated, repeatable, and self-validating!
