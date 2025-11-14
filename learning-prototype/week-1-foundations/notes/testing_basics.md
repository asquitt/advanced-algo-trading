# Testing Basics with Pytest

## Table of Contents
1. [Why Testing Matters](#why-testing-matters)
2. [Introduction to Pytest](#introduction-to-pytest)
3. [Writing Good Tests](#writing-good-tests)
4. [Test Fixtures](#test-fixtures)
5. [Mocking External APIs](#mocking-external-apis)
6. [Test Coverage](#test-coverage)
7. [Test-Driven Development (TDD)](#test-driven-development-tdd)
8. [Best Practices](#best-practices)
9. [Common Mistakes](#common-mistakes)

---

## Why Testing Matters

In algorithmic trading, bugs can cost real money. A single mistake in your code could:

- Execute trades at wrong prices
- Buy when you meant to sell
- Risk more capital than intended
- Miss trading opportunities
- Crash during market hours

**Testing prevents these disasters.**

### The Cost of Bugs

```python
# Bug: Missing negative sign
def calculate_stop_loss(entry_price, stop_pct=2):
    return entry_price * (1 + stop_pct / 100)  # Should be minus!

# Result: Stop loss at $153 instead of $147 (wrong direction!)
stop = calculate_stop_loss(150, 2)  # Returns 153.0, not 147.0
```

Without tests, you might not catch this until it's too late. With tests:

```python
def test_calculate_stop_loss():
    """Test stop loss calculation"""
    stop = calculate_stop_loss(150, 2)
    assert stop == 147.0  # Test fails! Bug caught before trading.
```

### Benefits of Testing

1. **Catch Bugs Early**: Find issues before they affect trading
2. **Confidence**: Deploy changes knowing they work
3. **Documentation**: Tests show how code should be used
4. **Refactoring**: Change code safely without breaking functionality
5. **Sleep Better**: Know your trading bot won't do something crazy

---

## Introduction to Pytest

**Pytest** is Python's most popular testing framework. It's simple, powerful, and widely used in production systems.

### Installation

```bash
pip install pytest pytest-cov pytest-asyncio
```

### Your First Test

```python
# test_example.py
def add(a, b):
    """Add two numbers"""
    return a + b

def test_add():
    """Test the add function"""
    result = add(2, 3)
    assert result == 5

def test_add_negative():
    """Test adding negative numbers"""
    result = add(-1, 1)
    assert result == 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test_example.py

# Run specific test function
pytest test_example.py::test_add

# Run with coverage report
pytest --cov=.
```

### Test Output

```
======================= test session starts =======================
collected 2 items

test_example.py::test_add PASSED                            [ 50%]
test_example.py::test_add_negative PASSED                   [100%]

======================= 2 passed in 0.01s ========================
```

---

## Writing Good Tests

### The AAA Pattern

Structure tests using **Arrange, Act, Assert**:

```python
def test_calculate_position_size():
    # ARRANGE - Set up test data
    account_value = 100000
    risk_percent = 1
    entry_price = 150
    stop_price = 145

    # ACT - Execute the function being tested
    result = calculate_position_size(
        account_value, risk_percent, entry_price, stop_price
    )

    # ASSERT - Verify the result
    assert result == 200  # $1,000 risk / $5 per share = 200 shares
```

### Test One Thing at a Time

```python
# BAD - Testing multiple things
def test_trading_strategy():
    signal = generate_signal("AAPL")
    order = execute_trade(signal)
    position = get_position("AAPL")
    assert signal.action == "buy"
    assert order.status == "filled"
    assert position.qty == 100

# GOOD - Separate tests
def test_generate_signal():
    signal = generate_signal("AAPL")
    assert signal.action == "buy"

def test_execute_trade():
    signal = Signal(action="buy", symbol="AAPL", qty=100)
    order = execute_trade(signal)
    assert order.status == "filled"

def test_get_position():
    position = get_position("AAPL")
    assert position.qty == 100
```

### Use Descriptive Names

```python
# BAD
def test_1():
    assert calculate_stop_loss(100, 2) == 98

# GOOD
def test_stop_loss_calculates_correct_price():
    entry_price = 100
    stop_percent = 2
    expected_stop = 98

    result = calculate_stop_loss(entry_price, stop_percent)

    assert result == expected_stop
```

### Test Edge Cases

```python
def test_position_size_edge_cases():
    """Test position sizing with edge cases"""

    # Zero risk (should return 0)
    assert calculate_position_size(100000, 0, 150, 145) == 0

    # Very small account
    assert calculate_position_size(100, 1, 150, 145) == 0

    # Entry equals stop (divide by zero)
    with pytest.raises(ValueError):
        calculate_position_size(100000, 1, 150, 150)

    # Negative values
    with pytest.raises(ValueError):
        calculate_position_size(-100000, 1, 150, 145)
```

### Test Exceptions

```python
import pytest

def test_invalid_symbol_raises_error():
    """Test that invalid symbols raise ValueError"""
    with pytest.raises(ValueError, match="Invalid symbol"):
        validate_symbol("invalid123")

def test_insufficient_funds_raises_error():
    """Test that insufficient funds raise exception"""
    with pytest.raises(InsufficientFundsError):
        execute_trade(symbol="AAPL", qty=1000000)
```

---

## Test Fixtures

**Fixtures** provide reusable test data and setup code.

### Basic Fixture

```python
import pytest

@pytest.fixture
def sample_account():
    """Provide sample account data"""
    return {
        "account_id": "ACC123",
        "cash": 100000.00,
        "equity": 100000.00,
        "buying_power": 100000.00
    }

def test_account_has_cash(sample_account):
    """Test that fixture provides cash"""
    assert sample_account["cash"] == 100000.00

def test_account_has_buying_power(sample_account):
    """Test that fixture provides buying power"""
    assert sample_account["buying_power"] > 0
```

### Fixture with Setup/Teardown

```python
@pytest.fixture
def trading_api():
    """Set up and tear down API connection"""
    # SETUP
    api = TradingAPI(
        api_key="test_key",
        base_url="https://paper-api.alpaca.markets"
    )

    yield api  # Provide to test

    # TEARDOWN
    api.cancel_all_orders()
    api.close()

def test_submit_order(trading_api):
    """Test order submission"""
    order = trading_api.submit_order("AAPL", 100, "buy")
    assert order.status in ["pending", "filled"]
```

### Fixture Scope

```python
# Function scope (default) - runs for each test
@pytest.fixture
def temp_data():
    return {"value": 42}

# Module scope - runs once per module
@pytest.fixture(scope="module")
def database_connection():
    conn = create_connection()
    yield conn
    conn.close()

# Session scope - runs once per test session
@pytest.fixture(scope="session")
def app_config():
    return load_config("test_config.yaml")
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("price,stop_pct,expected", [
    (100, 2, 98),
    (150, 5, 142.5),
    (200, 1, 198),
    (50.50, 3, 48.99),
])
def test_stop_loss_calculations(price, stop_pct, expected):
    """Test stop loss with multiple inputs"""
    result = calculate_stop_loss(price, stop_pct)
    assert result == pytest.approx(expected, rel=0.01)
```

---

## Mocking External APIs

Never call real APIs in tests! Use **mocking** to simulate external dependencies.

### Why Mock?

1. **Speed**: Tests run instantly, no network delays
2. **Reliability**: Tests don't fail due to API downtime
3. **Cost**: Avoid API rate limits and charges
4. **Control**: Test error conditions easily

### Mocking with unittest.mock

```python
from unittest.mock import Mock, patch

def test_fetch_stock_price_with_mock():
    """Test fetching stock price with mocked API"""

    # Create a mock API client
    mock_api = Mock()
    mock_api.get_latest_trade.return_value = Mock(price=150.25)

    # Use the mock in your code
    price = fetch_stock_price(mock_api, "AAPL")

    # Verify
    assert price == 150.25
    mock_api.get_latest_trade.assert_called_once_with("AAPL")
```

### Patching Functions

```python
@patch('alpaca_trade_api.REST')
def test_execute_trade(mock_rest):
    """Test trade execution with patched API"""

    # Configure mock
    mock_api = mock_rest.return_value
    mock_api.submit_order.return_value = Mock(
        id="order123",
        status="filled"
    )

    # Execute function
    result = execute_trade("AAPL", 100, "buy")

    # Verify
    assert result.id == "order123"
    assert result.status == "filled"
```

### Mocking HTTP Requests

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_fetch_market_data():
    """Test fetching market data with mocked HTTP client"""

    # Create mock HTTP client
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "symbol": "AAPL",
        "price": 150.25
    }
    mock_client.get.return_value = mock_response

    # Test the function
    data = await fetch_market_data(mock_client, "AAPL")

    # Verify
    assert data["symbol"] == "AAPL"
    assert data["price"] == 150.25
```

### Realistic Mock Data

```python
@pytest.fixture
def mock_alpaca_account():
    """Provide realistic mock account data"""
    return Mock(
        status="ACTIVE",
        cash="100000.00",
        portfolio_value="105000.00",
        buying_power="100000.00",
        equity="105000.00",
        last_equity="100000.00"
    )

@pytest.fixture
def mock_alpaca_position():
    """Provide realistic mock position data"""
    return Mock(
        symbol="AAPL",
        qty="100",
        avg_entry_price="150.00",
        current_price="152.50",
        market_value="15250.00",
        unrealized_pl="250.00",
        unrealized_plpc="1.67"
    )

def test_calculate_portfolio_value(mock_alpaca_account):
    """Test portfolio value calculation"""
    value = calculate_portfolio_value(mock_alpaca_account)
    assert value == 105000.00
```

---

## Test Coverage

**Test coverage** measures what percentage of your code is tested.

### Running Coverage

```bash
# Run tests with coverage
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Coverage Report Example

```
Name                    Stmts   Miss  Cover
-------------------------------------------
trading_api.py            100     10    90%
models.py                  50      5    90%
broker.py                  75     15    80%
-------------------------------------------
TOTAL                     225     30    87%
```

### What to Aim For

- **Critical Code**: 100% coverage (trade execution, risk management)
- **Business Logic**: 90%+ coverage (strategies, signals)
- **Utilities**: 80%+ coverage (helpers, formatters)
- **Overall**: 80%+ is good, 90%+ is excellent

### Coverage is Not Everything

```python
# 100% coverage but bad test!
def calculate_stop_loss(price, pct):
    return price * (1 - pct / 100)

def test_calculate_stop_loss():
    result = calculate_stop_loss(100, 2)
    # Bad test - doesn't verify result!
    assert result is not None  # Covers code but doesn't test logic
```

---

## Test-Driven Development (TDD)

**TDD** is writing tests *before* writing code.

### The TDD Cycle

```
1. RED: Write a failing test
2. GREEN: Write minimal code to pass the test
3. REFACTOR: Improve the code while keeping tests passing
4. REPEAT: Move to the next feature
```

### TDD Example

```python
# STEP 1: Write the test first (RED)
def test_validate_symbol():
    assert validate_symbol("AAPL") == True
    assert validate_symbol("123") == False
    assert validate_symbol("") == False

# Test fails because validate_symbol doesn't exist

# STEP 2: Write minimal code to pass (GREEN)
def validate_symbol(symbol):
    if not symbol:
        return False
    if symbol.isdigit():
        return False
    return True

# Test passes!

# STEP 3: Refactor (if needed)
def validate_symbol(symbol):
    """Validate stock symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    if not symbol.isalpha():
        return False
    return len(symbol) <= 5

# Tests still pass, code is better!
```

### Benefits of TDD

1. **Better Design**: Forces you to think about API before implementation
2. **Complete Tests**: Every line of code has a test
3. **Less Debugging**: Catch bugs immediately
4. **Confidence**: Know code works before deploying

### When to Use TDD

- **New Features**: Perfect for greenfield development
- **Bug Fixes**: Write test that reproduces bug, then fix it
- **Refactoring**: Tests ensure you don't break functionality
- **Critical Code**: Trading logic, risk management

---

## Best Practices

### 1. Organize Tests Properly

```
project/
├── trading_api/
│   ├── __init__.py
│   ├── broker.py
│   ├── models.py
│   └── strategies.py
└── tests/
    ├── __init__.py
    ├── test_broker.py
    ├── test_models.py
    └── test_strategies.py
```

### 2. Name Tests Clearly

```python
# BAD
def test_1():
    ...

def test_function():
    ...

# GOOD
def test_calculate_position_size_returns_correct_value():
    ...

def test_execute_trade_raises_error_with_invalid_symbol():
    ...
```

### 3. Keep Tests Independent

```python
# BAD - Tests depend on each other
order_id = None

def test_submit_order():
    global order_id
    order = api.submit_order(...)
    order_id = order.id

def test_cancel_order():
    api.cancel_order(order_id)  # Depends on previous test!

# GOOD - Each test is independent
def test_submit_order():
    order = api.submit_order(...)
    assert order.id is not None

def test_cancel_order():
    order = api.submit_order(...)
    result = api.cancel_order(order.id)
    assert result.status == "cancelled"
```

### 4. Use Meaningful Assertions

```python
# BAD
assert result

# GOOD
assert result is not None
assert result.status == "filled"
assert result.filled_qty == 100
assert result.filled_avg_price > 0
```

### 5. Test Async Code Properly

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    result = await async_fetch_price("AAPL")
    assert result > 0
```

---

## Common Mistakes

### 1. Not Testing Edge Cases

```python
# Only testing happy path
def test_calculate_position_size():
    result = calculate_position_size(100000, 1, 150, 145)
    assert result == 200

# Also test edge cases!
def test_calculate_position_size_edge_cases():
    # Zero risk
    assert calculate_position_size(100000, 0, 150, 145) == 0

    # Negative values
    with pytest.raises(ValueError):
        calculate_position_size(-100000, 1, 150, 145)

    # Entry equals stop
    with pytest.raises(ValueError):
        calculate_position_size(100000, 1, 150, 150)
```

### 2. Testing Implementation, Not Behavior

```python
# BAD - Testing internal implementation
def test_uses_correct_formula():
    assert "* 0.98" in inspect.getsource(calculate_stop_loss)

# GOOD - Testing behavior
def test_stop_loss_is_2_percent_below_entry():
    result = calculate_stop_loss(100, 2)
    assert result == 98
```

### 3. Slow Tests

```python
# BAD - Actual API call in test (slow!)
def test_fetch_price():
    price = requests.get("https://api.example.com/price")  # Slow!
    assert price > 0

# GOOD - Mocked API call (fast!)
@patch('requests.get')
def test_fetch_price(mock_get):
    mock_get.return_value.json.return_value = {"price": 150.25}
    price = fetch_price("AAPL")
    assert price == 150.25
```

### 4. Not Running Tests Regularly

```bash
# Set up pre-commit hook to run tests
# .git/hooks/pre-commit
#!/bin/bash
pytest
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

### 5. Ignoring Failing Tests

```python
# NEVER do this!
@pytest.mark.skip(reason="Broken, will fix later")
def test_important_feature():
    ...

# Fix the test or remove it!
```

---

## Summary

Testing is essential for building reliable trading systems. Key takeaways:

1. **Write Tests**: Every feature needs tests
2. **Use Pytest**: Simple, powerful, industry standard
3. **Mock External APIs**: Never call real APIs in tests
4. **Aim for High Coverage**: 80%+ overall, 100% for critical code
5. **Test Edge Cases**: Don't just test the happy path
6. **TDD When Possible**: Write tests first for better design
7. **Run Tests Often**: Before every commit

In algorithmic trading, untested code is a liability. Tests give you confidence that your trading bot will behave correctly, even in unexpected market conditions.

---

## Next Steps

1. Install pytest: `pip install pytest pytest-cov`
2. Write tests for your existing code
3. Practice TDD with new features
4. Set up continuous integration (CI) to run tests automatically
5. Read your test coverage report and fill gaps
6. Complete the testing exercises in `/exercises`
7. Review `fastapi_fundamentals.md` for testing FastAPI endpoints

Remember: **If it's not tested, it's broken.** Happy testing!
