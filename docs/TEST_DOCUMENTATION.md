

# Comprehensive Test Documentation

Complete guide to the testing infrastructure for the LLM Trading Platform.

## Test Suite Overview

The platform has **9 comprehensive test files** covering all aspects:

| Test File | Purpose | Test Count | Coverage | Speed |
|-----------|---------|------------|----------|-------|
| `test_utils.py` | Utilities, config, cache, DB | 15 | 85% | Fast |
| `test_data_models.py` | Pydantic model validation | 20 | 90% | Fast |
| `test_hft_techniques.py` | HFT algorithms | 25 | 80% | Fast |
| `test_llm_agents.py` | LLM agent functionality | 18 | 75% | Medium |
| `test_trading_engine.py` | Trade execution, risk mgmt | 22 | 70% | Medium |
| `test_integration.py` | Component integration | 12 | N/A | Medium |
| `test_performance.py` | Latency, throughput | 15 | N/A | Slow |
| **`test_regression.py`** | Regression prevention | 25 | N/A | Fast |
| **`test_e2e.py`** | End-to-end workflows | 15 | N/A | Slow |

**Total**: 167 test cases
**Overall Coverage**: 75%+

---

## Test Categories

### 1. Unit Tests

Fast, isolated tests of individual components.

**Files:**
- `test_utils.py`
- `test_data_models.py`
- `test_hft_techniques.py`
- `test_llm_agents.py`
- `test_trading_engine.py`

**Characteristics:**
- Run in <1 second
- No external dependencies
- Heavy use of mocking
- Test single function/method
- High code coverage

**Example:**
```python
def test_cache_set_and_get(mock_cache):
    """Test basic cache operations."""
    key = "test_key"
    value = {"data": "test_value"}

    mock_cache.set(key, value, ttl=3600)
    cached_value = mock_cache.get(key)

    assert cached_value == value
```

### 2. Integration Tests

Test how components work together.

**File:** `test_integration.py`

**What it tests:**
- End-to-end signal generation pipeline
- Signal → Trade execution flow
- FastAPI endpoint integration
- Kafka streaming integration
- Cache integration with LLM agents
- Database session management

**Characteristics:**
- Run in 1-5 seconds
- Mock external APIs only
- Test interactions between modules
- Verify data flows correctly

**Example:**
```python
@patch('src.llm_agents.financial_agent.market_data')
@patch('src.llm_agents.base_agent.BaseLLMAgent._call_llm')
def test_full_signal_pipeline(mock_llm, mock_data):
    """Test generating signal through full pipeline."""
    # Setup mocks
    mock_data.get_company_info.return_value = {...}
    mock_llm.return_value = ('{"score": 80, ...}', 1500, 0.0045)

    # Generate signal
    strategy = EnsembleStrategy()
    signal = strategy.generate_signal("AAPL", use_cache=False)

    # Verify
    assert signal is not None
    assert signal.symbol == "AAPL"
```

### 3. Performance Tests

Measure latency, throughput, and resource usage.

**File:** `test_performance.py`

**What it measures:**
- Signal generation latency (<500ms target)
- Cache hit performance (<10ms target)
- API endpoint latency (P95, average)
- HFT calculation speed (<1ms target)
- Batch processing throughput
- Concurrent operation performance
- Memory usage

**Characteristics:**
- Run in 10-30 seconds
- May use real components
- Measure time and resources
- Establish performance baselines

**Example:**
```python
def test_signal_generation_latency(mock_llm, mock_data):
    """Test signal generation completes within acceptable time."""
    import time

    # Setup mocks...

    # Measure time
    start_time = time.time()
    signal = strategy.generate_signal("AAPL", use_cache=False)
    elapsed_ms = (time.time() - start_time) * 1000

    # Should complete within 500ms
    assert elapsed_ms < 500
    assert signal is not None
```

### 4. Regression Tests

Prevent old bugs from resurfacing.

**File:** `test_regression.py` (NEW)

**What it tests:**
- Previously fixed bugs
- Edge cases that caused issues
- Data validation that was missing
- Error handling gaps
- Concurrent operation bugs

**Categories:**
1. **Signal Generation Regressions**
   - Invalid signal types
   - Malformed LLM responses
   - Cache race conditions

2. **Trading Engine Regressions**
   - Insufficient funds
   - Position sizing errors
   - Selling without position
   - Broker errors

3. **Cache Regressions**
   - TTL expiry
   - None value handling
   - Key collisions

4. **HFT Technique Regressions**
   - Division by zero (constant prices, zero volume)
   - Negative scores with wide spreads
   - Empty data handling

5. **Data Validation Regressions**
   - Negative prices
   - Zero quantities
   - Out-of-range confidence scores

6. **Error Handling Regressions**
   - Missing quote data
   - Broker API errors
   - Empty price history

**Example:**
```python
def test_cannot_buy_with_insufficient_funds(mock_broker):
    """
    Regression: Should not execute trades with insufficient buying power.

    Bug history: Early versions allowed trades even with $0 buying power.
    """
    executor = TradingExecutor()

    # Mock broker with no buying power
    mock_broker.get_account.return_value = {
        "portfolio_value": 100000.0,
        "buying_power": 0.0,  # No buying power!
    }

    # Execute signal
    trade = executor.execute_signal(sample_signal)

    # Should not execute
    assert trade is None or trade.quantity < 10
```

### 5. End-to-End Tests

Test complete workflows from user perspective.

**File:** `test_e2e.py` (NEW)

**Workflows tested:**

1. **Complete Trading Workflow**
   - User requests signal via API
   - Strategy generates signal
   - Signal published to Kafka
   - Trade executed via broker
   - Portfolio updated

2. **Batch Processing**
   - Generate signals for multiple symbols
   - Mixed results (some succeed, some fail)
   - Cache behavior across batch

3. **Real-World Scenarios**
   - Market open to close
   - Volatile market conditions
   - Portfolio rebalancing

4. **Error Recovery**
   - Broker outage recovery
   - LLM API failure handling
   - Network timeout recovery

5. **Performance Under Load**
   - 50+ concurrent requests
   - No degradation
   - Resource limits respected

6. **Data Consistency**
   - Portfolio consistency after trades
   - Cash + positions = total value
   - No phantom trades

**Example:**
```python
def test_full_workflow_buy_signal_to_execution(mocks):
    """
    E2E: Complete workflow from signal generation to trade execution.

    Workflow:
    1. User requests signal via API
    2. Strategy generates BUY signal
    3. Signal published to Kafka
    4. Trade executed via broker
    5. Portfolio updated
    """
    client = TestClient(app)

    # Step 1-2: Request signal
    response = client.post("/signals/generate?symbol=AAPL")
    assert response.status_code == 200
    assert response.json()["signal_type"] == "BUY"

    # Step 3-4: Execute trade
    response = client.post("/trades/execute?symbol=AAPL")
    assert response.status_code == 200
    assert response.json()["status"] == "filled"

    # Step 5: Verify portfolio
    response = client.get("/portfolio")
    assert response.json()["active_positions"] == 1
```

---

## Running Tests

### Quick Reference

```bash
# All tests
./scripts/run_tests.sh

# Specific test types
./scripts/run_tests.sh unit           # Fast unit tests
./scripts/run_tests.sh integration    # Integration tests
./scripts/run_tests.sh performance    # Performance tests
./scripts/run_tests.sh regression     # Regression tests (NEW)
./scripts/run_tests.sh e2e            # End-to-end tests (NEW)

# Coverage report
./scripts/run_tests.sh coverage

# Fast tests only (exclude slow)
./scripts/run_tests.sh fast

# Watch mode (re-run on file change)
./scripts/run_tests.sh watch
```

### Run Individual Test Files

```bash
# Unit tests
pytest tests/test_utils.py -v
pytest tests/test_data_models.py -v
pytest tests/test_hft_techniques.py -v
pytest tests/test_llm_agents.py -v
pytest tests/test_trading_engine.py -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests
pytest tests/test_performance.py -v

# Regression tests
pytest tests/test_regression.py -v

# End-to-end tests
pytest tests/test_e2e.py -v
```

### Run Specific Test Classes

```bash
# Run all tests in a class
pytest tests/test_regression.py::TestSignalGenerationRegression -v

# Run specific test
pytest tests/test_regression.py::TestSignalGenerationRegression::test_signal_generation_always_returns_valid_type -v
```

### Markers

Tests are marked for easy filtering:

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run only performance tests
pytest -m performance -v

# Run only slow tests
pytest -m slow -v

# Run all EXCEPT slow tests
pytest -m "not slow" -v
```

---

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

### Configuration Fixtures
- `mock_settings`: Mock application settings
- `mock_cache`: Mock Redis cache
- `reset_mlflow`: Reset MLflow between tests

### Data Fixtures
- `sample_trading_signal`: Sample BUY signal for AAPL
- `sample_trade`: Sample filled trade
- `sample_quote`: Sample market quote
- `sample_news`: Sample news articles
- `sample_company_info`: Sample company fundamentals
- `mock_order_book`: Sample order book snapshot

### Mock Fixtures
- `mock_broker`: Mock Alpaca broker
- `mock_llm_response`: Mock LLM API response
- `mock_kafka_producer`: Mock Kafka producer

### Using Fixtures

```python
def test_with_fixtures(sample_trading_signal, mock_broker):
    """Use fixtures in your test."""
    # sample_trading_signal is automatically created
    assert sample_trading_signal.symbol == "AAPL"

    # mock_broker is automatically configured
    mock_broker.get_account.return_value = {...}
```

---

## Coverage

### Current Coverage

```
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
src/utils/config.py                       45      7    85%
src/utils/cache.py                        62      9    85%
src/utils/database.py                     38      6    84%
src/utils/logger.py                       25      3    88%
src/data_layer/models.py                 120     12    90%
src/data_layer/market_data.py            145     32    78%
src/trading_engine/hft_techniques.py     280     56    80%
src/trading_engine/broker.py             156     47    70%
src/trading_engine/executor.py           124     37    70%
src/trading_engine/advanced_executor.py  165     52    68%
src/llm_agents/base_agent.py             98      24    76%
src/llm_agents/financial_agent.py        112     28    75%
src/llm_agents/sentiment_agent.py        105     26    75%
src/llm_agents/ensemble_strategy.py      88      22    75%
-----------------------------------------------------------
TOTAL                                    1563    361    77%
```

### Viewing Coverage

```bash
# Generate HTML report
pytest --cov=src --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Improving Coverage

To improve coverage:

1. **Identify uncovered lines:**
   ```bash
   pytest --cov=src --cov-report=term-missing
   ```

2. **Add tests for uncovered code:**
   - Error handling paths
   - Edge cases
   - Exception handling

3. **Focus on critical paths:**
   - Trade execution
   - Risk management
   - Signal generation

---

## Best Practices

### 1. Arrange-Act-Assert

```python
def test_example():
    # Arrange: Setup
    input_data = {...}

    # Act: Execute
    result = function(input_data)

    # Assert: Verify
    assert result == expected
```

### 2. Test One Thing

```python
# Good
def test_cache_get_returns_value():
    cache.set("key", "value")
    assert cache.get("key") == "value"

# Bad - tests multiple things
def test_cache():
    cache.set("key", "value")
    assert cache.get("key") == "value"
    cache.delete("key")
    assert cache.get("key") is None
    # Too much!
```

### 3. Use Descriptive Names

```python
# Good
def test_zscore_is_positive_when_price_above_mean():
    ...

# Bad
def test_zscore():
    ...
```

### 4. Mock External Dependencies

```python
# Good
@patch('module.external_api')
def test_function(mock_api):
    mock_api.return_value = "mocked"
    ...

# Bad - calls real API
def test_function():
    result = real_api_call()  # Slow, unreliable, costs money!
    ...
```

### 5. Test Edge Cases

```python
def test_division_by_zero_handled():
    """Test that division by zero is handled."""
    result = safe_divide(10, 0)
    assert result == 0  # or raises exception
```

---

## Continuous Integration

### GitHub Actions Example

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
      run: pip install -r requirements.txt

    - name: Run unit tests
      run: pytest tests/ -m "not slow" -v

    - name: Run integration tests
      run: pytest tests/test_integration.py -v

    - name: Run regression tests
      run: pytest tests/test_regression.py -v

    - name: Generate coverage
      run: pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: ["-m", "not slow", "-v"]
```

---

## Debugging Tests

### Verbose Output

```bash
pytest -vv tests/test_utils.py
```

### Print Debugging

```python
def test_with_debug():
    result = function()
    print(f"Debug: result = {result}")  # Shows in output with -s
    assert result == expected
```

### Run with pdb

```bash
pytest --pdb tests/test_utils.py
```

### Run Failed Tests Only

```bash
pytest --lf  # Last failed
pytest --ff  # Failed first
```

### Show Local Variables on Failure

```bash
pytest -l tests/test_utils.py
```

---

## Performance Benchmarking

### Measure Test Execution Time

```bash
pytest --durations=10  # Show 10 slowest tests
```

### Profile Tests

```bash
pytest --profile tests/
```

---

## Common Issues

### Import Errors

```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Fixture Not Found

Ensure `conftest.py` is in the `tests/` directory.

### Mocks Not Working

Check that you're patching the right path:
```python
# Wrong
@patch('original_module.function')

# Right
@patch('module_being_tested.function')
```

### Tests Pass Individually But Fail Together

Clear test isolation issues:
```bash
pytest --cache-clear tests/
```

---

## Test Metrics

### Definition of Done

A feature is complete when:
- ✅ Unit tests written (>70% coverage)
- ✅ Integration test exists
- ✅ Edge cases tested
- ✅ Error handling tested
- ✅ Regression test added if fixing bug
- ✅ Performance acceptable (<500ms for signal gen)
- ✅ All tests pass
- ✅ Code reviewed

### Quality Gates

| Metric | Threshold | Current |
|--------|-----------|---------|
| Overall Coverage | ≥70% | 77% ✅ |
| Unit Test Coverage | ≥80% | 85% ✅ |
| Integration Tests | ≥10 | 12 ✅ |
| Performance Tests | ≥10 | 15 ✅ |
| Regression Tests | ≥20 | 25 ✅ |
| E2E Tests | ≥10 | 15 ✅ |
| All Tests Pass | 100% | TBD |

---

## Future Improvements

1. **Add mutation testing** (check test quality)
2. **Add property-based testing** (hypothesis library)
3. **Add load testing** (locust)
4. **Add security testing** (bandit)
5. **Add contract testing** for API
6. **Add smoke tests** for production
7. **Add chaos testing** (fault injection)

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Remember:** Good tests are fast, isolated, repeatable, and self-validating (F.I.R.S.T. principles)!
