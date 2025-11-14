"""
Week 1 - Starter Code: Pydantic Models

Your task: Create validated data models for the trading API.

These models ensure type safety and automatic validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# EXAMPLE: Complete Model (Learn from this!)
# ============================================================================

class HealthStatus(BaseModel):
    """Example model - already complete. Study this structure!"""

    status: str = Field(..., description="Health status", example="healthy")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-11-13T12:00:00"
            }
        }


# ============================================================================
# YOUR MODELS (Complete These)
# ============================================================================

# TODO #1: Create SignalType Enum
class SignalType(str, Enum):
    """
    Enum for signal types.

    Your task: Define three signal types: BUY, SELL, HOLD

    HINT:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    """
    # TODO: Add signal types here
    pass


# TODO #2: Create TradingSignal Model
class TradingSignal(BaseModel):
    """
    Model for trading signals.

    Fields to include:
    - symbol: str (the stock ticker, required)
    - signal_type: SignalType (BUY/SELL/HOLD, required)
    - confidence_score: float (0-1, required)
    - reasoning: str (explanation, optional)
    - generated_at: datetime (auto-generated timestamp)

    Add validators:
    - symbol should be uppercase
    - confidence_score should be between 0 and 1
    """

    # TODO: Add fields here
    # symbol: str = Field(...)
    # signal_type: SignalType = Field(...)
    # etc.

    # TODO #3: Add validator for symbol
    @validator("symbol")
    def symbol_must_be_uppercase(cls, v):
        """Ensure symbol is uppercase."""
        # HINT: return v.upper()
        pass

    # TODO #4: Add validator for confidence_score
    @validator("confidence_score")
    def confidence_must_be_valid(cls, v):
        """Ensure confidence is between 0 and 1."""
        # HINT:
        # if not 0 <= v <= 1:
        #     raise ValueError("Confidence must be between 0 and 1")
        # return v
        pass

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "signal_type": "BUY",
                "confidence_score": 0.85,
                "reasoning": "Strong fundamentals and positive sentiment"
            }
        }


# TODO #5: Create OrderSide Enum
class OrderSide(str, Enum):
    """
    Enum for order sides.

    Your task: Define BUY and SELL
    """
    # TODO: Add order sides here
    pass


# TODO #6: Create TradeRequest Model
class TradeRequest(BaseModel):
    """
    Model for trade requests.

    Fields to include:
    - symbol: str (stock ticker, required)
    - side: OrderSide (buy or sell, required)
    - quantity: int (number of shares, required, must be > 0)
    - order_type: Literal["market", "limit"] (default: "market")
    - limit_price: Optional[float] (only if order_type is "limit")

    Add validators:
    - quantity must be positive
    - if order_type is "limit", limit_price must be provided
    """

    # TODO: Add fields here

    # TODO #7: Add validator for quantity
    @validator("quantity")
    def quantity_must_be_positive(cls, v):
        """Ensure quantity is positive."""
        # HINT:
        # if v <= 0:
        #     raise ValueError("Quantity must be positive")
        # return v
        pass

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "order_type": "market"
            }
        }


# TODO #8: Create TradeResponse Model
class TradeResponse(BaseModel):
    """
    Model for trade responses.

    Fields to include:
    - success: bool (whether trade succeeded)
    - order_id: Optional[str] (Alpaca order ID)
    - message: str (confirmation or error message)
    - executed_at: datetime (timestamp)
    """

    # TODO: Add fields here

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "order_id": "abc123",
                "message": "Executed BUY order for 10 shares of AAPL",
                "executed_at": "2024-11-13T12:00:00"
            }
        }


# TODO #9: Create Position Model
class Position(BaseModel):
    """
    Model for portfolio positions.

    Fields to include:
    - symbol: str
    - quantity: int
    - current_price: float
    - market_value: float
    - unrealized_pnl: float (profit/loss)
    """

    # TODO: Add fields here

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "quantity": 10,
                "current_price": 150.25,
                "market_value": 1502.50,
                "unrealized_pnl": 25.50
            }
        }


# TODO #10: Create Portfolio Model
class Portfolio(BaseModel):
    """
    Model for complete portfolio.

    Fields to include:
    - account_value: float (total portfolio value)
    - cash: float (available cash)
    - positions: List[Position] (list of current positions)
    - total_pnl: float (total unrealized P&L)
    """

    # TODO: Add fields here

    class Config:
        json_schema_extra = {
            "example": {
                "account_value": 105000.00,
                "cash": 100000.00,
                "positions": [
                    {
                        "symbol": "AAPL",
                        "quantity": 10,
                        "current_price": 150.25,
                        "market_value": 1502.50,
                        "unrealized_pnl": 25.50
                    }
                ],
                "total_pnl": 25.50
            }
        }


# ============================================================================
# TESTING YOUR MODELS
# ============================================================================

"""
Test your models with this code:

# Test TradingSignal
signal = TradingSignal(
    symbol="aapl",  # Should auto-uppercase
    signal_type=SignalType.BUY,
    confidence_score=0.85,
    reasoning="Test"
)
print(signal.model_dump_json(indent=2))

# Test TradeRequest
trade = TradeRequest(
    symbol="AAPL",
    side=OrderSide.BUY,
    quantity=10,
    order_type="market"
)
print(trade.model_dump_json(indent=2))

# This should fail validation
try:
    bad_signal = TradingSignal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence_score=1.5  # Invalid!
    )
except Exception as e:
    print(f"Validation error (expected): {e}")
"""

# ============================================================================
# HELPFUL HINTS
# ============================================================================

"""
HINT #1 - Basic Field Definition:
    symbol: str = Field(
        ...,  # Required field (... means required)
        description="Stock ticker symbol",
        example="AAPL"
    )

HINT #2 - Optional Field:
    limit_price: Optional[float] = Field(
        None,  # Default value is None (optional)
        description="Limit price for limit orders",
        example=150.50
    )

HINT #3 - Field with Validation:
    confidence_score: float = Field(
        ...,
        ge=0.0,  # Greater than or equal to 0
        le=1.0,  # Less than or equal to 1
        description="Confidence in signal (0-1)"
    )

HINT #4 - Auto-generated Field:
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When signal was generated"
    )

HINT #5 - Validator Example:
    @validator("symbol")
    def symbol_must_be_uppercase(cls, v):
        if not v.isupper():
            return v.upper()
        return v

Still stuck? Check:
- ../solutions/models_solution.py
- ../notes/pydantic_explained.md
- Pydantic docs: https://docs.pydantic.dev
"""
