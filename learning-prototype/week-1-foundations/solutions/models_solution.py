"""
Week 1 - SOLUTION: Pydantic Models

This is the complete solution for models.py starter code.
Try to complete the starter code yourself before looking at this!
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class SignalType(str, Enum):
    """Enum for signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    """Enum for order sides."""
    BUY = "buy"
    SELL = "sell"


# ============================================================================
# TRADING SIGNAL MODEL
# ============================================================================

class TradingSignal(BaseModel):
    """Model for trading signals."""

    symbol: str = Field(
        ...,
        description="Stock ticker symbol",
        example="AAPL"
    )
    signal_type: SignalType = Field(
        ...,
        description="Type of signal (BUY, SELL, HOLD)",
        example=SignalType.BUY
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in signal (0-1)",
        example=0.85
    )
    reasoning: Optional[str] = Field(
        None,
        description="Explanation for the signal",
        example="Strong fundamentals and positive sentiment"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when signal was generated"
    )

    @validator("symbol")
    def symbol_must_be_uppercase(cls, v):
        """Ensure symbol is uppercase."""
        return v.upper()

    @validator("confidence_score")
    def confidence_must_be_valid(cls, v):
        """Ensure confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "signal_type": "BUY",
                "confidence_score": 0.85,
                "reasoning": "Strong fundamentals and positive sentiment"
            }
        }


# ============================================================================
# TRADE REQUEST MODEL
# ============================================================================

class TradeRequest(BaseModel):
    """Model for trade requests."""

    symbol: str = Field(
        ...,
        description="Stock ticker symbol",
        example="AAPL"
    )
    side: OrderSide = Field(
        ...,
        description="Order side (buy or sell)",
        example=OrderSide.BUY
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="Number of shares to trade",
        example=10
    )
    order_type: Literal["market", "limit"] = Field(
        "market",
        description="Type of order",
        example="market"
    )
    limit_price: Optional[float] = Field(
        None,
        gt=0,
        description="Limit price (required for limit orders)",
        example=150.50
    )

    @validator("quantity")
    def quantity_must_be_positive(cls, v):
        """Ensure quantity is positive."""
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @validator("limit_price")
    def limit_price_required_for_limit_orders(cls, v, values):
        """Ensure limit_price is provided for limit orders."""
        if values.get("order_type") == "limit" and v is None:
            raise ValueError("limit_price required for limit orders")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "order_type": "market"
            }
        }


# ============================================================================
# TRADE RESPONSE MODEL
# ============================================================================

class TradeResponse(BaseModel):
    """Model for trade responses."""

    success: bool = Field(
        ...,
        description="Whether trade was successful",
        example=True
    )
    order_id: Optional[str] = Field(
        None,
        description="Order ID from broker",
        example="abc-123-def"
    )
    message: str = Field(
        ...,
        description="Confirmation or error message",
        example="Executed BUY order for 10 shares of AAPL"
    )
    executed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of execution"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "order_id": "abc123",
                "message": "Executed BUY order for 10 shares of AAPL",
                "executed_at": "2024-11-13T12:00:00"
            }
        }


# ============================================================================
# POSITION MODEL
# ============================================================================

class Position(BaseModel):
    """Model for portfolio positions."""

    symbol: str = Field(
        ...,
        description="Stock ticker symbol",
        example="AAPL"
    )
    quantity: int = Field(
        ...,
        description="Number of shares held",
        example=10
    )
    current_price: float = Field(
        ...,
        gt=0,
        description="Current market price",
        example=150.25
    )
    market_value: float = Field(
        ...,
        ge=0,
        description="Total market value of position",
        example=1502.50
    )
    unrealized_pnl: float = Field(
        ...,
        description="Unrealized profit/loss",
        example=25.50
    )

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


# ============================================================================
# PORTFOLIO MODEL
# ============================================================================

class Portfolio(BaseModel):
    """Model for complete portfolio."""

    account_value: float = Field(
        ...,
        ge=0,
        description="Total portfolio value",
        example=105000.00
    )
    cash: float = Field(
        ...,
        ge=0,
        description="Available cash",
        example=100000.00
    )
    positions: List[Position] = Field(
        default_factory=list,
        description="List of current positions"
    )
    total_pnl: float = Field(
        ...,
        description="Total unrealized P&L",
        example=25.50
    )

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
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test TradingSignal
    signal = TradingSignal(
        symbol="aapl",  # Should auto-uppercase
        signal_type=SignalType.BUY,
        confidence_score=0.85,
        reasoning="Test signal"
    )
    print("TradingSignal:")
    print(signal.model_dump_json(indent=2))
    print()

    # Test TradeRequest
    trade = TradeRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type="market"
    )
    print("TradeRequest:")
    print(trade.model_dump_json(indent=2))
    print()

    # Test validation error
    try:
        bad_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence_score=1.5  # Invalid!
        )
    except Exception as e:
        print(f"Validation error (expected): {e}")
        print()

    # Test Portfolio
    portfolio = Portfolio(
        account_value=105000.00,
        cash=100000.00,
        positions=[
            Position(
                symbol="AAPL",
                quantity=10,
                current_price=150.25,
                market_value=1502.50,
                unrealized_pnl=25.50
            )
        ],
        total_pnl=25.50
    )
    print("Portfolio:")
    print(portfolio.model_dump_json(indent=2))
