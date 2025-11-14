# Pydantic Explained

## Table of Contents
1. [What is Pydantic?](#what-is-pydantic)
2. [Why Use Pydantic?](#why-use-pydantic)
3. [Type Hints and Validation](#type-hints-and-validation)
4. [Field Types and Constraints](#field-types-and-constraints)
5. [Validators](#validators)
6. [Model Configuration](#model-configuration)
7. [Serialization and Parsing](#serialization-and-parsing)
8. [Best Practices](#best-practices)
9. [Common Mistakes](#common-mistakes)

---

## What is Pydantic?

Pydantic is a data validation library for Python that uses Python type hints to validate, serialize, and deserialize data. Think of it as a powerful combination of type checking, data validation, and serialization all in one package.

Created to solve the problem of validating complex data structures in Python, Pydantic has become the de facto standard for data validation in modern Python applications, especially in web frameworks like FastAPI.

### Core Concepts

- **Type Safety**: Uses Python's type hints to enforce data types
- **Validation**: Automatically validates data against defined schemas
- **Parsing**: Converts raw data (JSON, dict, etc.) into Python objects
- **Serialization**: Converts Python objects back to JSON/dict format

---

## Why Use Pydantic?

### 1. Runtime Type Checking

Python's type hints are usually only checked by static analysis tools. Pydantic enforces them at runtime:

```python
from pydantic import BaseModel

class Stock(BaseModel):
    symbol: str
    price: float
    volume: int

# This works
stock = Stock(symbol="AAPL", price=150.25, volume=1000000)

# This fails - price must be a float
stock = Stock(symbol="AAPL", price="expensive", volume=1000000)
# ValidationError: price must be a valid number
```

### 2. Automatic Data Conversion

Pydantic intelligently converts compatible types:

```python
class TradeOrder(BaseModel):
    symbol: str
    quantity: int
    price: float

# String numbers are automatically converted
order = TradeOrder(symbol="AAPL", quantity="100", price="150.25")
print(order.quantity)  # 100 (int, not string)
print(order.price)     # 150.25 (float, not string)
```

### 3. Clear Error Messages

When validation fails, you get detailed, helpful error messages:

```python
try:
    stock = Stock(symbol="AAPL", price=-10, volume="lots")
except ValidationError as e:
    print(e)
    # Shows exactly which fields failed and why
    # - volume: value is not a valid integer
```

### 4. IDE Support

Full autocomplete and type checking in your IDE:

```python
stock = Stock(symbol="AAPL", price=150.25, volume=1000000)
stock.  # IDE shows: symbol, price, volume with correct types
```

---

## Type Hints and Validation

### Basic Types

```python
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class Portfolio(BaseModel):
    # Required fields
    account_id: str
    created_at: datetime

    # Optional field (can be None)
    nickname: Optional[str] = None

    # Field with default value
    is_active: bool = True

    # Collections
    positions: List[str] = []
    metadata: Dict[str, str] = {}
```

### Advanced Types

```python
from typing import Union, Literal, Annotated
from pydantic import Field

class TradeOrder(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]  # Only "buy" or "sell" allowed
    quantity: int

    # Union type - can be multiple types
    price: Union[float, None] = None  # Market order if None

    # Annotated type with metadata
    order_type: Annotated[str, Field(pattern="^(market|limit)$")]
```

### Nested Models

```python
class Position(BaseModel):
    symbol: str
    quantity: int
    avg_price: float

class Portfolio(BaseModel):
    account_id: str
    cash: float
    positions: List[Position]  # List of Position objects

# Usage
portfolio = Portfolio(
    account_id="ACC123",
    cash=10000.00,
    positions=[
        {"symbol": "AAPL", "quantity": 100, "avg_price": 150.25},
        {"symbol": "GOOGL", "quantity": 50, "avg_price": 2800.50}
    ]
)

# Access nested data
print(portfolio.positions[0].symbol)  # "AAPL"
```

---

## Field Types and Constraints

### Field Validation with Constraints

```python
from pydantic import BaseModel, Field

class TradeOrder(BaseModel):
    # String constraints
    symbol: str = Field(
        ...,  # Required field (no default)
        min_length=1,
        max_length=5,
        pattern="^[A-Z]+$"  # Uppercase letters only
    )

    # Numeric constraints
    quantity: int = Field(..., gt=0, le=10000)  # Greater than 0, less than or equal to 10000
    price: float = Field(..., gt=0.0)  # Must be positive

    # With description for API docs
    notes: str = Field(
        default="",
        description="Optional notes about the trade",
        max_length=500
    )

# Constraints:
# gt: greater than
# ge: greater than or equal to
# lt: less than
# le: less than or equal to
# multiple_of: must be a multiple of this value
```

### Constrained Types

```python
from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    NonNegativeFloat,
    constr,
    conint,
    confloat
)

class Stock(BaseModel):
    # Constrained types
    symbol: constr(min_length=1, max_length=5, to_upper=True)
    price: PositiveFloat  # Must be > 0
    volume: PositiveInt   # Must be > 0
    change_percent: confloat(ge=-100, le=100)  # Between -100 and 100

# Usage
stock = Stock(
    symbol="aapl",  # Automatically converted to "AAPL"
    price=150.25,
    volume=1000000,
    change_percent=2.5
)
```

---

## Validators

### Field Validators

Custom validation logic for specific fields:

```python
from pydantic import BaseModel, field_validator, ValidationError

class TradeOrder(BaseModel):
    symbol: str
    quantity: int
    side: str

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Ensure symbol is uppercase"""
        if not v.isupper():
            raise ValueError('Symbol must be uppercase')
        return v

    @field_validator('side')
    @classmethod
    def validate_side(cls, v):
        """Ensure side is buy or sell"""
        if v.lower() not in ['buy', 'sell']:
            raise ValueError('Side must be "buy" or "sell"')
        return v.lower()

    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        """Ensure quantity is positive and reasonable"""
        if v <= 0:
            raise ValueError('Quantity must be positive')
        if v > 100000:
            raise ValueError('Quantity too large (max 100,000)')
        return v
```

### Model Validators

Validate relationships between multiple fields:

```python
from pydantic import BaseModel, model_validator

class LimitOrder(BaseModel):
    symbol: str
    quantity: int
    side: str
    limit_price: float
    market_price: float

    @model_validator(mode='after')
    def check_price_makes_sense(self):
        """Ensure limit price makes sense given side"""
        if self.side == 'buy' and self.limit_price > self.market_price * 1.1:
            raise ValueError('Buy limit price too high (>10% above market)')
        if self.side == 'sell' and self.limit_price < self.market_price * 0.9:
            raise ValueError('Sell limit price too low (<10% below market)')
        return self

# Usage
order = LimitOrder(
    symbol="AAPL",
    quantity=100,
    side="buy",
    limit_price=150.00,
    market_price=140.00  # OK - within 10%
)
```

### Pre and Post Validation

```python
from pydantic import BaseModel, field_validator

class Stock(BaseModel):
    symbol: str
    price: float

    @field_validator('symbol', mode='before')
    @classmethod
    def uppercase_symbol(cls, v):
        """Pre-validator: runs before type conversion"""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator('price', mode='after')
    @classmethod
    def round_price(cls, v):
        """Post-validator: runs after type conversion"""
        return round(v, 2)  # Round to 2 decimal places
```

---

## Model Configuration

### Config Options

```python
from pydantic import BaseModel, ConfigDict

class Stock(BaseModel):
    model_config = ConfigDict(
        # Validation settings
        str_strip_whitespace=True,  # Strip whitespace from strings
        str_to_lower=False,          # Don't convert strings to lowercase
        validate_assignment=True,    # Validate on attribute assignment

        # Serialization settings
        use_enum_values=True,        # Use enum values instead of names
        populate_by_name=True,       # Allow population by field name

        # Extra fields
        extra='forbid',              # Forbid extra fields (strict)
        # extra='allow',              # Allow extra fields
        # extra='ignore',             # Ignore extra fields

        # JSON schema
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "price": 150.25
            }
        }
    )

    symbol: str
    price: float
```

### Field Aliases

```python
from pydantic import BaseModel, Field

class StockData(BaseModel):
    symbol: str
    current_price: float = Field(..., alias='price')
    daily_volume: int = Field(..., alias='vol')

# Can use either the field name or alias
stock1 = StockData(symbol="AAPL", price=150.25, vol=1000000)
stock2 = StockData(symbol="AAPL", current_price=150.25, daily_volume=1000000)

# Both work!
```

### Computed Fields

```python
from pydantic import BaseModel, computed_field

class Position(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    current_price: float

    @computed_field
    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        return self.quantity * self.current_price

    @computed_field
    @property
    def unrealized_pl(self) -> float:
        """Calculate unrealized profit/loss"""
        return (self.current_price - self.avg_price) * self.quantity

# Usage
position = Position(
    symbol="AAPL",
    quantity=100,
    avg_price=145.00,
    current_price=150.25
)

print(position.market_value)    # 15025.0
print(position.unrealized_pl)   # 525.0 (profit)
```

---

## Serialization and Parsing

### Converting to Dict/JSON

```python
from pydantic import BaseModel

class Stock(BaseModel):
    symbol: str
    price: float
    volume: int

stock = Stock(symbol="AAPL", price=150.25, volume=1000000)

# Convert to dictionary
stock_dict = stock.model_dump()
print(stock_dict)  # {'symbol': 'AAPL', 'price': 150.25, 'volume': 1000000}

# Convert to JSON string
stock_json = stock.model_dump_json()
print(stock_json)  # '{"symbol":"AAPL","price":150.25,"volume":1000000}'

# Exclude certain fields
stock_dict = stock.model_dump(exclude={'volume'})
print(stock_dict)  # {'symbol': 'AAPL', 'price': 150.25}
```

### Parsing from JSON/Dict

```python
# From dictionary
data = {"symbol": "AAPL", "price": 150.25, "volume": 1000000}
stock = Stock(**data)

# From JSON string
json_str = '{"symbol": "AAPL", "price": 150.25, "volume": 1000000}'
stock = Stock.model_validate_json(json_str)

# From file
with open('stock.json', 'r') as f:
    json_data = f.read()
    stock = Stock.model_validate_json(json_data)
```

### Updating Models

```python
stock = Stock(symbol="AAPL", price=150.25, volume=1000000)

# Update with dict
updated_data = {"price": 155.00, "volume": 1200000}
updated_stock = stock.model_copy(update=updated_data)

print(stock.price)          # 150.25 (original unchanged)
print(updated_stock.price)  # 155.00 (new instance)
```

---

## Best Practices

### 1. Use Descriptive Field Names

```python
# BAD
class Order(BaseModel):
    s: str  # What is this?
    q: int  # Quantity?
    p: float  # Price?

# GOOD
class Order(BaseModel):
    symbol: str
    quantity: int
    price: float
```

### 2. Provide Examples for Documentation

```python
class TradeOrder(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "quantity": 100,
                "side": "buy",
                "price": 150.25
            }
        }
    )

    symbol: str
    quantity: int
    side: str
    price: float
```

### 3. Use Enums for Fixed Choices

```python
from enum import Enum

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class TradeOrder(BaseModel):
    symbol: str
    side: OrderSide  # Only valid enum values accepted
    order_type: OrderType
```

### 4. Validate Business Logic

```python
class Portfolio(BaseModel):
    cash: float
    positions: List[Position]

    @field_validator('cash')
    @classmethod
    def validate_cash(cls, v):
        if v < 0:
            raise ValueError('Cash cannot be negative')
        return v

    @model_validator(mode='after')
    def validate_portfolio(self):
        """Ensure portfolio is valid"""
        total_value = self.cash + sum(p.market_value for p in self.positions)
        if total_value < 0:
            raise ValueError('Total portfolio value cannot be negative')
        return self
```

### 5. Use Type Aliases for Complex Types

```python
from typing import List, Dict, TypeAlias

PositionList: TypeAlias = List[Position]
MarketData: TypeAlias = Dict[str, float]

class Portfolio(BaseModel):
    positions: PositionList
    prices: MarketData
```

---

## Common Mistakes

### 1. Forgetting to Subclass BaseModel

```python
# WRONG
class Stock:
    symbol: str
    price: float

# RIGHT
from pydantic import BaseModel

class Stock(BaseModel):
    symbol: str
    price: float
```

### 2. Mutating Models Directly

```python
# WRONG - Models are immutable by default
stock = Stock(symbol="AAPL", price=150.25)
stock.price = 155.00  # Error if validate_assignment not enabled

# RIGHT - Create new instance
updated_stock = stock.model_copy(update={"price": 155.00})
```

### 3. Not Handling ValidationError

```python
# WRONG
order = TradeOrder(**data)  # May crash if data is invalid

# RIGHT
from pydantic import ValidationError

try:
    order = TradeOrder(**data)
except ValidationError as e:
    print(f"Invalid order data: {e}")
    # Handle the error appropriately
```

### 4. Over-Using Validators

```python
# WRONG - Field constraints are simpler
class Stock(BaseModel):
    price: float

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

# RIGHT - Use Field constraints
class Stock(BaseModel):
    price: float = Field(gt=0)
```

### 5. Incorrect Validator Signatures

```python
# WRONG - Missing @classmethod decorator
class Stock(BaseModel):
    symbol: str

    @field_validator('symbol')
    def validate_symbol(cls, v):  # Will fail!
        return v.upper()

# RIGHT - Include @classmethod
class Stock(BaseModel):
    symbol: str

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        return v.upper()
```

---

## Summary

Pydantic is an essential tool for building robust Python applications. Key takeaways:

1. **Type Safety**: Enforces types at runtime, not just static analysis
2. **Validation**: Automatic validation with clear error messages
3. **Conversion**: Smart type conversion from compatible types
4. **Constraints**: Easy field-level constraints with `Field()`
5. **Validators**: Custom validation logic for complex rules
6. **Serialization**: Easy conversion to/from JSON and dicts
7. **Documentation**: Automatic schema generation for API docs

In algorithmic trading, Pydantic ensures that all your data is valid before you execute trades, preventing costly mistakes from bad data. Combined with FastAPI, it creates a type-safe, validated trading system.

---

## Next Steps

1. Read `fastapi_fundamentals.md` to see Pydantic in action with FastAPI
2. Complete the Pydantic exercises in `/exercises`
3. Practice writing validators for business logic
4. Build models for your trading data structures
5. Study `async_python.md` to understand async programming

Happy validating!
