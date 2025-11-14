"""
Exercise 5: Integration Testing

Objective: Test complete workflow end-to-end with pytest.

Time: 60 minutes
Difficulty: Hard

What you'll learn:
- pytest basics
- TestClient for FastAPI
- Mocking external APIs
- Test fixtures
- Assertions
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

# ============================================================================
# YOUR CODE HERE
# ============================================================================

# TODO #1: Import your FastAPI app
# HINT: from starter_code.main import app


# TODO #2: Create TestClient fixture
# @pytest.fixture
# def client():
#     """Create test client for API."""
#     return TestClient(app)


# TODO #3: Test signal generation endpoint
# def test_generate_signal(client):
#     """Test GET /signal/{symbol} endpoint."""
#     # TODO: Test successful signal generation
#     # HINT:
#     # response = client.get("/signal/AAPL")
#     # assert response.status_code == 200
#     # data = response.json()
#     # assert data["symbol"] == "AAPL"
#     # assert "signal_type" in data
#     # assert "confidence_score" in data
#     # assert 0 <= data["confidence_score"] <= 1
#     pass


# TODO #4: Test trade execution with mocked broker
# @patch('starter_code.broker.AlpacaBroker.execute_trade')
# def test_execute_trade(mock_execute, client):
#     """Test POST /trade endpoint."""
#     # TODO: Mock broker response
#     # HINT:
#     # mock_execute.return_value = {
#     #     "id": "test-order-123",
#     #     "symbol": "AAPL",
#     #     "side": "buy",
#     #     "qty": "10",
#     #     "status": "filled"
#     # }
#     #
#     # # Execute trade
#     # response = client.post("/trade", json={
#     #     "symbol": "AAPL",
#     #     "side": "buy",
#     #     "quantity": 10
#     # })
#     #
#     # # Verify response
#     # assert response.status_code == 200
#     # data = response.json()
#     # assert data["success"] == True
#     # assert data["order_id"] == "test-order-123"
#     #
#     # # Verify broker was called
#     # mock_execute.assert_called_once()
#     pass


# TODO #5: Test portfolio endpoint
# @patch('starter_code.broker.AlpacaBroker.get_account')
# @patch('starter_code.broker.AlpacaBroker.get_positions')
# def test_get_portfolio(mock_positions, mock_account, client):
#     """Test GET /portfolio endpoint."""
#     # TODO: Mock broker responses
#     # HINT:
#     # mock_account.return_value = {
#     #     "equity": "105000.00",
#     #     "cash": "100000.00"
#     # }
#     #
#     # mock_positions.return_value = [
#     #     {
#     #         "symbol": "AAPL",
#     #         "qty": "10",
#     #         "current_price": "150.25",
#     #         "market_value": "1502.50"
#     #     }
#     # ]
#     #
#     # # Get portfolio
#     # response = client.get("/portfolio")
#     #
#     # # Verify response
#     # assert response.status_code == 200
#     # data = response.json()
#     # assert float(data["account_value"]) == 105000.00
#     # assert len(data["positions"]) == 1
#     # assert data["positions"][0]["symbol"] == "AAPL"
#     pass


# TODO #6: Test complete workflow
# @patch('starter_code.broker.AlpacaBroker')
# def test_complete_workflow(mock_broker_class, client):
#     """Test complete trading workflow: signal → trade → portfolio."""
#     # TODO: Setup mock broker
#     # HINT:
#     # mock_broker = Mock()
#     # mock_broker_class.return_value = mock_broker
#     #
#     # # 1. Generate signal
#     # signal_response = client.get("/signal/AAPL?confidence_threshold=0.5")
#     # assert signal_response.status_code == 200
#     # signal = signal_response.json()
#     #
#     # # 2. If BUY signal, execute trade
#     # if signal["signal_type"] == "BUY":
#     #     mock_broker.execute_trade.return_value = {
#     #         "id": "order-123",
#     #         "symbol": "AAPL",
#     #         "status": "filled"
#     #     }
#     #
#     #     trade_response = client.post("/trade", json={
#     #         "symbol": "AAPL",
#     #         "side": "buy",
#     #         "quantity": 10
#     #     })
#     #     assert trade_response.status_code == 200
#     #
#     # # 3. Check portfolio updated
#     # mock_broker.get_account.return_value = {"equity": "105000.00", "cash": "98000.00"}
#     # mock_broker.get_positions.return_value = [
#     #     {"symbol": "AAPL", "qty": "10", "current_price": "150.0", "market_value": "1500.0"}
#     # ]
#     #
#     # portfolio_response = client.get("/portfolio")
#     # assert portfolio_response.status_code == 200
#     # portfolio = portfolio_response.json()
#     # assert len(portfolio["positions"]) > 0
#     pass


# TODO #7: Test error cases
# def test_invalid_symbol(client):
#     """Test handling of invalid stock symbol."""
#     # TODO: Test error handling
#     # HINT:
#     # response = client.get("/signal/INVALID123456")
#     # # Should either return HOLD signal or 404, depending on implementation
#     # assert response.status_code in [200, 404]
#     pass


# def test_invalid_trade_quantity(client):
#     """Test validation of trade quantity."""
#     # TODO: Test validation
#     # HINT:
#     # response = client.post("/trade", json={
#     #     "symbol": "AAPL",
#     #     "side": "buy",
#     #     "quantity": -10  # Invalid!
#     # })
#     # assert response.status_code == 422  # Validation error
#     pass


# ============================================================================
# FIXTURES FOR TESTING
# ============================================================================

# TODO #8: Create fixture for mock account
# @pytest.fixture
# def mock_account():
#     """Mock Alpaca account."""
#     return {
#         "account_number": "PA123456",
#         "cash": "100000.00",
#         "equity": "100000.00",
#         "buying_power": "100000.00"
#     }


# TODO #9: Create fixture for mock positions
# @pytest.fixture
# def mock_positions():
#     """Mock portfolio positions."""
#     return [
#         {
#             "symbol": "AAPL",
#             "qty": "10",
#             "current_price": "150.0",
#             "market_value": "1500.0",
#             "avg_entry_price": "145.0",
#             "unrealized_pl": "50.0"
#         },
#         {
#             "symbol": "GOOGL",
#             "qty": "5",
#             "current_price": "2800.0",
#             "market_value": "14000.0",
#             "avg_entry_price": "2750.0",
#             "unrealized_pl": "250.0"
#         }
#     ]


# ============================================================================
# RUNNING TESTS
# ============================================================================

"""
Run your tests:

1. Install pytest:
   pip install pytest pytest-asyncio

2. Run all tests:
   pytest exercise_5_integration.py -v

3. Run specific test:
   pytest exercise_5_integration.py::test_generate_signal -v

4. Run with coverage:
   pytest exercise_5_integration.py --cov=starter_code --cov-report=html

5. View coverage report:
   open htmlcov/index.html

Expected output:
  test_generate_signal PASSED
  test_execute_trade PASSED
  test_get_portfolio PASSED
  test_complete_workflow PASSED
  test_invalid_symbol PASSED
  test_invalid_trade_quantity PASSED

  ============= 6 passed in 0.15s =============
"""


# ============================================================================
# PYTEST BASICS
# ============================================================================

"""
Key pytest concepts:

1. Test Discovery:
   - Files must start with test_
   - Functions must start with test_
   - Classes must start with Test

2. Assertions:
   assert value == expected
   assert "text" in response
   assert response.status_code == 200

3. Fixtures:
   @pytest.fixture
   def my_fixture():
       return "test data"

   def test_something(my_fixture):
       assert my_fixture == "test data"

4. Mocking:
   from unittest.mock import Mock, patch

   @patch('module.function')
   def test_with_mock(mock_function):
       mock_function.return_value = "mocked"
       assert module.function() == "mocked"

5. Parametrize:
   @pytest.mark.parametrize("input,expected", [
       ("AAPL", "BUY"),
       ("GOOGL", "SELL"),
   ])
   def test_signals(input, expected):
       signal = generate_signal(input)
       assert signal.signal_type == expected
"""


# ============================================================================
# BONUS CHALLENGES
# ============================================================================

"""
If you finish early, try these:

1. Add parametrized tests:
   @pytest.mark.parametrize("symbol,expected_type", [
       ("AAPL", ["BUY", "SELL", "HOLD"]),
       ("GOOGL", ["BUY", "SELL", "HOLD"]),
       ("MSFT", ["BUY", "SELL", "HOLD"]),
   ])
   def test_signal_types(client, symbol, expected_type):
       response = client.get(f"/signal/{symbol}")
       assert response.json()["signal_type"] in expected_type

2. Add async tests:
   @pytest.mark.asyncio
   async def test_async_endpoint():
       async with AsyncClient(app=app, base_url="http://test") as ac:
           response = await ac.get("/signal/AAPL")
           assert response.status_code == 200

3. Add test for database (if using):
   @pytest.fixture(scope="function")
   def db():
       # Setup test database
       db = create_test_database()
       yield db
       # Teardown
       db.drop_all()

4. Add performance tests:
   def test_signal_performance(client):
       import time
       start = time.time()
       for i in range(100):
           client.get("/signal/AAPL")
       elapsed = time.time() - start
       assert elapsed < 1.0  # Should complete in <1 second
"""
