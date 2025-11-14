"""
Week 1 - Starter Code: Data Models & Basic Trading

Fill in the TODOs to create trading data structures.
Run: python starter.py
Test: pytest tests/test_day1.py
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


# ============================================================================
# TODO 1: Complete the TradingSignal model
# ============================================================================

class TradingSignal(BaseModel):
    """
    Represents a trading recommendation from an indicator or AI.

    Example:
        signal = TradingSignal(
            symbol="AAPL",
            signal_type="BUY",
            confidence=0.85,
            price=150.0,
            reasoning="RSI is oversold at 25"
        )
    """
    symbol: str  # Stock ticker (e.g., "AAPL", "GOOGL")

    # TODO: Add field for signal type (BUY, SELL, HOLD)
    # Hint: Use str type, valid values: "BUY", "SELL", "HOLD"
    signal_type: str

    # TODO: Add field for confidence score (0.0 to 1.0)
    # Hint: Use float type, add validation to ensure 0 <= confidence <= 1
    confidence: float = Field(ge=0.0, le=1.0)

    # TODO: Add field for target price
    price: float = Field(gt=0.0)  # Must be positive

    # TODO: Add field for reasoning (why this signal?)
    reasoning: str = ""

    # TODO: Add field for timestamp (when signal was generated)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator("signal_type")
    def validate_signal_type(cls, v):
        """Validate that signal type is one of: BUY, SELL, HOLD"""
        # TODO: Implement validation
        # Hint: Check if v is in ["BUY", "SELL", "HOLD"]
        # Hint: Raise ValueError if invalid
        allowed = ["BUY", "SELL", "HOLD"]
        if v not in allowed:
            raise ValueError(f"signal_type must be one of {allowed}")
        return v


# ============================================================================
# TODO 2: Complete the Trade model
# ============================================================================

class Trade(BaseModel):
    """
    Represents an executed trade.

    Example:
        trade = Trade(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0
        )
        print(f"P&L: ${trade.calculate_pnl()}")
    """
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int = Field(gt=0)  # Must be positive
    entry_price: float = Field(gt=0.0)
    exit_price: Optional[float] = None  # None if still open
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def calculate_pnl(self) -> float:
        """
        Calculate profit/loss for this trade.

        Returns:
            float: P&L in dollars
        """
        # TODO: Implement P&L calculation
        # Hint: If exit_price is None, return 0 (still open)
        # Hint: P&L = (exit_price - entry_price) * quantity for BUY
        # Hint: P&L = (entry_price - exit_price) * quantity for SELL (short)

        if self.exit_price is None:
            return 0.0  # Trade not closed yet

        if self.side == "BUY":
            # TODO: Calculate P&L for long position
            pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL (short)
            # TODO: Calculate P&L for short position
            pnl = (self.entry_price - self.exit_price) * self.quantity

        return pnl

    def return_pct(self) -> float:
        """Calculate percentage return."""
        # TODO: Implement percentage return calculation
        # Hint: return_pct = (pnl / cost_basis) * 100
        # Hint: cost_basis = entry_price * quantity

        if self.exit_price is None:
            return 0.0

        pnl = self.calculate_pnl()
        cost_basis = self.entry_price * self.quantity

        return (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0


# ============================================================================
# TODO 3: Complete the Position model
# ============================================================================

class Position(BaseModel):
    """
    Represents a current position (holdings).

    Example:
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.0,
            current_price=155.0
        )
        print(f"Unrealized P&L: ${position.unrealized_pnl()}")
    """
    symbol: str
    quantity: int  # Can be negative for short positions
    avg_entry_price: float = Field(gt=0.0)
    current_price: float = Field(gt=0.0)

    def market_value(self) -> float:
        """Calculate current market value of position."""
        # TODO: Implement market value calculation
        # Hint: market_value = current_price * quantity
        return self.current_price * self.quantity

    def unrealized_pnl(self) -> float:
        """Calculate unrealized (paper) profit/loss."""
        # TODO: Implement unrealized P&L calculation
        # Hint: Same as realized P&L, but using current_price instead of exit_price
        # Hint: P&L = (current_price - avg_entry_price) * quantity

        return (self.current_price - self.avg_entry_price) * self.quantity

    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L as percentage."""
        # TODO: Implement percentage calculation
        # Hint: pnl_pct = ((current_price - entry_price) / entry_price) * 100

        return ((self.current_price - self.avg_entry_price) / self.avg_entry_price) * 100


# ============================================================================
# TODO 4: Complete the Portfolio model
# ============================================================================

class Portfolio(BaseModel):
    """
    Represents entire portfolio state.

    Example:
        portfolio = Portfolio(
            cash=50000,
            positions=[
                Position(symbol="AAPL", quantity=100, avg_entry_price=150, current_price=155),
                Position(symbol="GOOGL", quantity=50, avg_entry_price=2800, current_price=2850),
            ]
        )
        print(f"Total value: ${portfolio.total_value()}")
    """
    cash: float = Field(ge=0.0)  # Available cash
    positions: list[Position] = Field(default_factory=list)

    def total_value(self) -> float:
        """Calculate total portfolio value."""
        # TODO: Implement total value calculation
        # Hint: total_value = cash + sum of all position market values

        positions_value = sum(pos.market_value() for pos in self.positions)
        return self.cash + positions_value

    def total_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        # TODO: Calculate sum of unrealized P&L for all positions

        return sum(pos.unrealized_pnl() for pos in self.positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        # TODO: Find and return position for given symbol
        # Hint: Loop through self.positions, return matching position
        # Hint: Return None if not found

        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def add_position(self, position: Position):
        """Add or update a position."""
        # TODO: Add position to portfolio
        # Hint: If position for symbol exists, update it
        # Hint: Otherwise, append new position

        existing = self.get_position(position.symbol)
        if existing:
            # Update existing position
            existing.quantity += position.quantity
            # Recalculate average entry price
            total_cost = (existing.avg_entry_price * existing.quantity +
                         position.avg_entry_price * position.quantity)
            total_quantity = existing.quantity + position.quantity
            existing.avg_entry_price = total_cost / total_quantity if total_quantity > 0 else 0
        else:
            # Add new position
            self.positions.append(position)


# ============================================================================
# Testing Your Implementation
# ============================================================================

def test_trading_signal():
    """Test TradingSignal creation and validation."""
    print("\n=== Testing TradingSignal ===")

    # Create a buy signal
    signal = TradingSignal(
        symbol="AAPL",
        signal_type="BUY",
        confidence=0.85,
        price=150.0,
        reasoning="RSI oversold at 25"
    )
    print(f"✓ Created signal: {signal.signal_type} {signal.symbol} at ${signal.price}")

    # Test validation
    try:
        bad_signal = TradingSignal(
            symbol="AAPL",
            signal_type="INVALID",  # Should fail
            confidence=0.85,
            price=150.0
        )
        print("✗ Validation failed to catch invalid signal type!")
    except ValueError:
        print("✓ Validation correctly rejected invalid signal type")


def test_trade_pnl():
    """Test Trade P&L calculation."""
    print("\n=== Testing Trade P&L ===")

    # Winning trade
    trade1 = Trade(
        symbol="AAPL",
        side="BUY",
        quantity=100,
        entry_price=150.0,
        exit_price=160.0
    )
    pnl1 = trade1.calculate_pnl()
    print(f"Trade 1: Buy 100 AAPL at $150, sell at $160")
    print(f"  P&L: ${pnl1:.2f} ({trade1.return_pct():.2f}%)")
    assert pnl1 == 1000, f"Expected $1000, got ${pnl1}"

    # Losing trade
    trade2 = Trade(
        symbol="GOOGL",
        side="BUY",
        quantity=50,
        entry_price=2800.0,
        exit_price=2750.0
    )
    pnl2 = trade2.calculate_pnl()
    print(f"\nTrade 2: Buy 50 GOOGL at $2800, sell at $2750")
    print(f"  P&L: ${pnl2:.2f} ({trade2.return_pct():.2f}%)")
    assert pnl2 == -2500, f"Expected -$2500, got ${pnl2}"

    print("✓ All P&L calculations correct!")


def test_portfolio():
    """Test Portfolio tracking."""
    print("\n=== Testing Portfolio ===")

    # Create portfolio
    portfolio = Portfolio(cash=50000)
    print(f"Starting cash: ${portfolio.cash:,.2f}")

    # Add positions
    pos1 = Position(symbol="AAPL", quantity=100, avg_entry_price=150.0, current_price=155.0)
    pos2 = Position(symbol="GOOGL", quantity=50, avg_entry_price=2800.0, current_price=2850.0)

    portfolio.add_position(pos1)
    portfolio.add_position(pos2)

    print(f"\nPositions:")
    for pos in portfolio.positions:
        print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_entry_price:.2f}")
        print(f"    Current: ${pos.current_price:.2f}, P&L: ${pos.unrealized_pnl():.2f}")

    total_value = portfolio.total_value()
    total_pnl = portfolio.total_pnl()

    print(f"\nPortfolio Summary:")
    print(f"  Total Value: ${total_value:,.2f}")
    print(f"  Total P&L: ${total_pnl:,.2f}")

    expected_value = 50000 + (100 * 155) + (50 * 2850)
    assert abs(total_value - expected_value) < 0.01, f"Expected ${expected_value}, got ${total_value}"
    print("✓ Portfolio calculations correct!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Week 1 - Day 1: Data Models")
    print("=" * 60)

    test_trading_signal()
    test_trade_pnl()
    test_portfolio()

    print("\n" + "=" * 60)
    print("✓ All manual tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run pytest: pytest tests/test_day1.py -v")
    print("2. Implement indicators.py (Day 2)")
    print("3. Compare with solutions/week1_solution.py")


if __name__ == "__main__":
    main()
