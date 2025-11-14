"""
Exercise 2: Pydantic Validation

Objective: Create validated data models with Pydantic.

Time: 45 minutes
Difficulty: Medium

What you'll learn:
- Defining Pydantic models
- Field validation
- Type hints
- Validators
- JSON serialization
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

# ============================================================================
# YOUR CODE HERE
# ============================================================================

# TODO #1: Create Stock model
class Stock(BaseModel):
    """
    Represents a stock with price and volume.

    Fields to add:
    - symbol: str (required, must be uppercase)
    - price: float (required, must be > 0)
    - volume: int (required, must be >= 0)
    - timestamp: datetime (auto-generated)
    """

    # HINT:
    # symbol: str = Field(..., description="Stock ticker symbol")
    # price: float = Field(..., gt=0, description="Current price")
    # volume: int = Field(..., ge=0, description="Trading volume")
    # timestamp: datetime = Field(default_factory=datetime.utcnow)

    # TODO: Add fields here
    pass

    # TODO #2: Add validator to ensure symbol is uppercase
    # HINT:
    # @validator("symbol")
    # def symbol_uppercase(cls, v):
    #     return v.upper()


# TODO #3: Create Portfolio model
class Portfolio(BaseModel):
    """
    Represents a portfolio of stocks.

    Fields to add:
    - name: str (portfolio name)
    - stocks: List[Stock] (list of stocks)
    - total_value: float (computed from stocks)
    - created_at: datetime (auto-generated)
    """

    # TODO: Add fields here
    pass

    # TODO #4: Add method to calculate total value
    # HINT:
    # def calculate_total_value(self) -> float:
    #     return sum(stock.price * stock.volume for stock in self.stocks)


# TODO #5: Create Trade model
class Trade(BaseModel):
    """
    Represents a trade order.

    Fields to add:
    - symbol: str (required)
    - side: str (must be "buy" or "sell")
    - quantity: int (must be positive)
    - price: Optional[float] (limit price, optional)
    """

    # TODO: Add fields here
    pass

    # TODO #6: Add validator for side
    # HINT:
    # @validator("side")
    # def validate_side(cls, v):
    #     if v.lower() not in ["buy", "sell"]:
    #         raise ValueError("Side must be 'buy' or 'sell'")
    #     return v.lower()

    # TODO #7: Add validator for quantity
    # HINT:
    # @validator("quantity")
    # def validate_quantity(cls, v):
    #     if v <= 0:
    #         raise ValueError("Quantity must be positive")
    #     return v


# ============================================================================
# TESTING YOUR WORK
# ============================================================================

"""
Test your models:

# Test Stock
stock = Stock(
    symbol="aapl",  # Should auto-uppercase
    price=150.0,
    volume=1000000
)
print(stock.model_dump_json(indent=2))
# Should work!

# Test validation error
try:
    bad_stock = Stock(
        symbol="AAPL",
        price=-10.0,  # Invalid! Should fail
        volume=1000000
    )
except Exception as e:
    print(f"Validation error (expected): {e}")

# Test Portfolio
portfolio = Portfolio(
    name="My Portfolio",
    stocks=[
        Stock(symbol="AAPL", price=150.0, volume=100),
        Stock(symbol="GOOGL", price=2800.0, volume=50)
    ],
    total_value=0.0  # Will be calculated
)
print(f"Portfolio value: ${portfolio.calculate_total_value()}")

# Test Trade
trade = Trade(
    symbol="AAPL",
    side="BUY",  # Should auto-lowercase
    quantity=10,
    price=150.0
)
print(trade.model_dump_json(indent=2))

# This should fail
try:
    bad_trade = Trade(
        symbol="AAPL",
        side="hold",  # Invalid! Not buy or sell
        quantity=10
    )
except Exception as e:
    print(f"Validation error (expected): {e}")
"""

if __name__ == "__main__":
    print("Run the test code above to verify your models!")
    print("Uncomment the test code and run:")
    print("  python exercise_2_pydantic_models.py")


# ============================================================================
# BONUS CHALLENGES
# ============================================================================

"""
If you finish early, try these:

1. Add a WatchList model:
   class WatchList(BaseModel):
       name: str
       symbols: List[str]
       created_at: datetime = Field(default_factory=datetime.utcnow)

       @validator("symbols", each_item=True)
       def validate_symbols(cls, v):
           # Ensure each symbol is uppercase and 1-5 chars
           if not 1 <= len(v) <= 5:
               raise ValueError("Invalid symbol length")
           return v.upper()

2. Add computed fields:
   from pydantic import computed_field

   class Stock(BaseModel):
       # ... existing fields ...

       @computed_field
       @property
       def market_cap(self) -> float:
           return self.price * self.volume

3. Add model validators:
   class Portfolio(BaseModel):
       # ... existing fields ...

       @validator("stocks")
       def validate_stocks(cls, v):
           if len(v) == 0:
               raise ValueError("Portfolio must have at least one stock")
           return v
"""
