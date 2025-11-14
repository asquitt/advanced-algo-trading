"""
Week 1 - Starter Code: FastAPI Trading API

Your task: Complete all TODOs to build a working trading API.

Run with: uvicorn main:app --reload
Test at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import uvicorn

# Import your models (you'll create these)
# TODO #1: Import TradingSignal, TradeRequest, TradeResponse from models.py
# from models import TradingSignal, TradeRequest, TradeResponse

# Import your broker (you'll create this)
# TODO #2: Import AlpacaBroker from broker.py
# from broker import AlpacaBroker

# Import config (you'll create this)
# TODO #3: Import settings from config.py
# from config import settings


# TODO #4: Create FastAPI app instance
# HINT: app = FastAPI(title="...", description="...", version="1.0.0")
app = None  # Replace with your code


# TODO #5: Add CORS middleware to allow frontend access
# HINT: app.add_middleware(CORSMiddleware, ...)
# Allow all origins for development (restrict in production!)


# TODO #6: Create broker instance
# HINT: broker = AlpacaBroker()
broker = None  # Replace with your code


# ============================================================================
# HEALTH CHECK ENDPOINT (Example - Already Complete)
# ============================================================================

@app.get("/")
async def root():
    """Welcome endpoint - shows API is running."""
    return {
        "message": "🤖 LLM Trading API v1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}


# ============================================================================
# YOUR ENDPOINTS (Complete These)
# ============================================================================

# TODO #7: Implement /signal/{symbol} endpoint
@app.get("/signal/{symbol}")
async def generate_signal(symbol: str, confidence_threshold: float = 0.6):
    """
    Generate a trading signal for a given symbol.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        confidence_threshold: Minimum confidence to generate signal (0-1)

    Returns:
        TradingSignal: BUY, SELL, or HOLD signal with confidence score

    Steps to implement:
    1. Validate symbol (uppercase, length check)
    2. Generate a simple signal (for now, just random or basic logic)
    3. Check if confidence meets threshold
    4. Return TradingSignal model

    HINT: For now, you can use simple logic like:
    - If symbol starts with 'A'-'M': BUY with 0.7 confidence
    - If symbol starts with 'N'-'Z': SELL with 0.6 confidence
    - Otherwise: HOLD

    Later weeks will add LLM-powered analysis!
    """
    # TODO: Your code here
    pass


# TODO #8: Implement /trade endpoint
@app.post("/trade")
async def execute_trade(trade_request: None):  # TODO: Replace None with TradeRequest
    """
    Execute a paper trade via Alpaca.

    Args:
        trade_request: TradeRequest with symbol, side, quantity

    Returns:
        TradeResponse: Confirmation with order details

    Steps to implement:
    1. Validate trade request (quantity > 0, valid side)
    2. Check if market is open (optional for now)
    3. Execute trade via broker.execute_trade()
    4. Return TradeResponse with order details

    Error handling:
    - If broker fails, raise HTTPException with 500
    - If invalid parameters, raise HTTPException with 400
    """
    # TODO: Your code here
    pass


# TODO #9: Implement /portfolio endpoint
@app.get("/portfolio")
async def get_portfolio():
    """
    Get current portfolio status.

    Returns:
        Dict with:
        - account_value: Total portfolio value
        - cash: Available cash
        - positions: List of current positions

    Steps to implement:
    1. Get account info from broker.get_account()
    2. Get positions from broker.get_positions()
    3. Format and return data

    Error handling:
    - If broker fails, raise HTTPException with 503
    """
    # TODO: Your code here
    pass


# TODO #10: Add error handling
# Create an exception handler for general errors
# HINT: Use @app.exception_handler(Exception)


# ============================================================================
# MAIN (For Development)
# ============================================================================

if __name__ == "__main__":
    # TODO #11: Run the app with uvicorn
    # HINT: uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    pass


# ============================================================================
# TESTING YOUR WORK
# ============================================================================

"""
Once you've completed all TODOs, test your API:

1. Start the server:
   python main.py

2. Open browser:
   http://localhost:8000/docs

3. Try the endpoints:
   - GET /signal/AAPL
   - POST /trade with body:
     {
       "symbol": "AAPL",
       "side": "buy",
       "quantity": 10
     }
   - GET /portfolio

4. Run the test script:
   cd ../scripts
   ./test_week1.sh

If all tests pass, you've completed Week 1 starter code! 🎉
"""

# ============================================================================
# HELPFUL HINTS
# ============================================================================

"""
Stuck? Here are some hints:

HINT #1 - Creating TradingSignal:
    return TradingSignal(
        symbol=symbol.upper(),
        signal_type="BUY",  # or "SELL" or "HOLD"
        confidence_score=0.75,
        reasoning="Your logic here"
    )

HINT #2 - Executing Trade:
    order = await broker.execute_trade(
        symbol=trade_request.symbol,
        side=trade_request.side,
        quantity=trade_request.quantity
    )

    return TradeResponse(
        success=True,
        order_id=order["id"],
        message=f"Executed {trade_request.side} order for {trade_request.quantity} shares"
    )

HINT #3 - Getting Portfolio:
    account = await broker.get_account()
    positions = await broker.get_positions()

    return {
        "account_value": float(account.get("equity", 0)),
        "cash": float(account.get("cash", 0)),
        "positions": [
            {
                "symbol": pos["symbol"],
                "quantity": int(pos["qty"]),
                "current_price": float(pos["current_price"]),
                "market_value": float(pos["market_value"])
            }
            for pos in positions
        ]
    }

HINT #4 - Error Handling:
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "detail": str(exc)}
        )

Need more help? Check:
- ../solutions/main_solution.py (after trying yourself!)
- ../notes/fastapi_fundamentals.md
- FastAPI docs: https://fastapi.tiangolo.com
"""
