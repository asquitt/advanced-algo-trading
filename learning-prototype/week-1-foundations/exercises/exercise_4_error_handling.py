"""
Exercise 4: Error Handling

Objective: Handle errors gracefully with proper HTTP status codes.

Time: 45 minutes
Difficulty: Medium

What you'll learn:
- HTTP status codes
- HTTPException
- Error responses
- Logging errors
- Custom exception handlers
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Error Handling Exercise")

# ============================================================================
# MODELS
# ============================================================================

class StockPrice(BaseModel):
    """Stock price data."""
    symbol: str
    price: float = Field(gt=0, description="Price must be positive")
    currency: str = "USD"

# Mock database of stock prices
STOCK_PRICES = {
    "AAPL": 150.25,
    "GOOGL": 2800.50,
    "MSFT": 380.75
}

# ============================================================================
# YOUR CODE HERE
# ============================================================================

# TODO #1: Create endpoint that returns 200 for valid stock, 404 for unknown
@app.get("/price/{symbol}", status_code=status.HTTP_200_OK)
async def get_price(symbol: str):
    """
    Get stock price.

    Returns:
        200: Stock found
        404: Stock not found
    """
    # TODO: Implement this
    # HINT:
    # symbol = symbol.upper()
    # if symbol not in STOCK_PRICES:
    #     logger.warning(f"Stock not found: {symbol}")
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail=f"Stock {symbol} not found"
    #     )
    #
    # price = STOCK_PRICES[symbol]
    # logger.info(f"Retrieved price for {symbol}: ${price}")
    # return StockPrice(symbol=symbol, price=price)
    pass


# TODO #2: Create endpoint that handles validation errors
@app.post("/update-price", status_code=status.HTTP_200_OK)
async def update_price(stock: StockPrice):
    """
    Update stock price.

    Returns:
        200: Price updated
        400: Invalid data (handled by Pydantic automatically)
        422: Validation error (handled by Pydantic automatically)
    """
    # TODO: Implement this
    # HINT:
    # symbol = stock.symbol.upper()
    # STOCK_PRICES[symbol] = stock.price
    # logger.info(f"Updated {symbol} to ${stock.price}")
    # return {"message": f"Updated {symbol} to ${stock.price}"}
    pass


# TODO #3: Create endpoint that might have server errors
@app.get("/risky-operation/{symbol}")
async def risky_operation(symbol: str, fail: bool = False):
    """
    Simulates an operation that might fail.

    Args:
        fail: If True, simulates a server error

    Returns:
        200: Success
        500: Server error
    """
    # TODO: Implement this
    # HINT:
    # try:
    #     if fail:
    #         # Simulate unexpected error
    #         raise Exception("Something went wrong!")
    #
    #     price = STOCK_PRICES.get(symbol.upper())
    #     if not price:
    #         raise HTTPException(status_code=404, detail="Stock not found")
    #
    #     # Simulate some processing
    #     result = price * 1.05  # 5% increase
    #     return {"symbol": symbol, "new_price": result}
    #
    # except HTTPException:
    #     # Re-raise HTTP exceptions
    #     raise
    # except Exception as e:
    #     # Log the error but don't expose details to user
    #     logger.error(f"Error in risky_operation: {e}", exc_info=True)
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail="Internal server error occurred"
    #     )
    pass


# TODO #4: Create endpoint with business logic validation
@app.post("/execute-trade")
async def execute_trade(symbol: str, quantity: int, price: float):
    """
    Execute a trade with business logic validation.

    Returns:
        200: Trade executed
        400: Invalid trade parameters
        409: Conflict (e.g., insufficient funds)
    """
    # TODO: Implement this
    # HINT:
    # # Validate inputs
    # if quantity <= 0:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Quantity must be positive"
    #     )
    #
    # if price <= 0:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Price must be positive"
    #     )
    #
    # # Check if symbol exists
    # if symbol.upper() not in STOCK_PRICES:
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail=f"Stock {symbol} not found"
    #     )
    #
    # # Simulate insufficient funds check
    # total_cost = quantity * price
    # available_cash = 10000.0  # Mock account balance
    #
    # if total_cost > available_cash:
    #     raise HTTPException(
    #         status_code=status.HTTP_409_CONFLICT,
    #         detail=f"Insufficient funds. Need ${total_cost}, have ${available_cash}"
    #     )
    #
    # logger.info(f"Executed trade: {quantity} shares of {symbol} at ${price}")
    # return {
    #     "message": "Trade executed successfully",
    #     "symbol": symbol,
    #     "quantity": quantity,
    #     "price": price,
    #     "total": total_cost
    # }
    pass


# ============================================================================
# CUSTOM EXCEPTION HANDLERS
# ============================================================================

# TODO #5: Add custom exception handler for validation errors
# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc):
#     """Custom handler for validation errors."""
#     logger.warning(f"Validation error: {exc}")
#     return JSONResponse(
#         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#         content={
#             "message": "Validation error",
#             "details": str(exc)
#         }
#     )


# TODO #6: Add global exception handler
# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     """Catch-all exception handler."""
#     logger.error(f"Unhandled exception: {exc}", exc_info=True)
#     return JSONResponse(
#         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         content={
#             "message": "An unexpected error occurred",
#             "type": type(exc).__name__
#         }
#     )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================================================
# TESTING YOUR WORK
# ============================================================================

"""
Test your error handling:

1. Start server:
   python exercise_4_error_handling.py

2. Test successful request (200):
   curl http://localhost:8000/price/AAPL
   → Should return: {"symbol": "AAPL", "price": 150.25, "currency": "USD"}

3. Test not found (404):
   curl http://localhost:8000/price/INVALID
   → Should return: {"detail": "Stock INVALID not found"}

4. Test validation error (422):
   curl -X POST http://localhost:8000/update-price \\
        -H "Content-Type: application/json" \\
        -d '{"symbol": "AAPL", "price": -10.0}'
   → Should return validation error (price must be positive)

5. Test server error (500):
   curl http://localhost:8000/risky-operation/AAPL?fail=true
   → Should return: {"detail": "Internal server error occurred"}

6. Test business logic error (409):
   curl -X POST "http://localhost:8000/execute-trade?symbol=AAPL&quantity=1000&price=150"
   → Should return: {"detail": "Insufficient funds..."}

7. Check logs:
   Look for INFO, WARNING, and ERROR messages in console
"""


# ============================================================================
# HTTP STATUS CODE REFERENCE
# ============================================================================

"""
Common HTTP status codes:

Success (2xx):
- 200 OK: Request successful
- 201 Created: Resource created
- 204 No Content: Successful, no response body

Client Errors (4xx):
- 400 Bad Request: Invalid request
- 401 Unauthorized: Authentication required
- 403 Forbidden: Authenticated but not allowed
- 404 Not Found: Resource doesn't exist
- 409 Conflict: Business logic conflict
- 422 Unprocessable Entity: Validation error

Server Errors (5xx):
- 500 Internal Server Error: Unexpected error
- 502 Bad Gateway: Upstream service error
- 503 Service Unavailable: Service down
- 504 Gateway Timeout: Upstream timeout
"""


# ============================================================================
# BONUS CHALLENGES
# ============================================================================

"""
If you finish early, try these:

1. Add rate limiting with custom error:
   class RateLimitExceeded(Exception):
       pass

   @app.exception_handler(RateLimitExceeded)
   async def rate_limit_handler(request, exc):
       return JSONResponse(
           status_code=429,
           content={"detail": "Too many requests"},
           headers={"Retry-After": "60"}
       )

2. Add request ID tracking:
   from uuid import uuid4

   @app.middleware("http")
   async def add_request_id(request, call_next):
       request_id = str(uuid4())
       request.state.request_id = request_id
       response = await call_next(request)
       response.headers["X-Request-ID"] = request_id
       return response

3. Add structured error responses:
   class ErrorResponse(BaseModel):
       error: str
       message: str
       request_id: Optional[str] = None
       timestamp: datetime = Field(default_factory=datetime.utcnow)
"""
