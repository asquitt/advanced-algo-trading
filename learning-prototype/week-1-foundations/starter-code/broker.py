"""
Week 1 - Starter Code: Alpaca Broker Integration

Your task: Connect to Alpaca paper trading API and execute trades.

This module handles all interactions with the Alpaca API.
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

# TODO #1: Import Alpaca SDK
# HINT: from alpaca.trading.client import TradingClient
# HINT: from alpaca.trading.requests import MarketOrderRequest
# HINT: from alpaca.trading.enums import OrderSide, TimeInForce


# ============================================================================
# ALPACA BROKER CLASS
# ============================================================================

class AlpacaBroker:
    """
    Handles paper trading via Alpaca API.

    Your task: Complete all TODO methods to enable trading functionality.

    Alpaca provides:
    - Paper trading (100% free, no real money)
    - Real-time market data
    - Order execution
    - Position tracking
    """

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key (or use env var ALPACA_API_KEY)
            secret_key: Alpaca secret key (or use env var ALPACA_SECRET_KEY)
        """
        # TODO #2: Get API keys from environment if not provided
        # HINT:
        # self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        # self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        self.api_key = None  # Replace with your code
        self.secret_key = None  # Replace with your code

        # TODO #3: Initialize Alpaca TradingClient
        # HINT:
        # self.client = TradingClient(
        #     api_key=self.api_key,
        #     secret_key=self.secret_key,
        #     paper=True  # ALWAYS use paper trading for learning!
        # )

        self.client = None  # Replace with your code

        # TODO #4: Verify connection on init
        # HINT: Call self._verify_connection()
        pass


    def _verify_connection(self) -> bool:
        """
        Verify connection to Alpaca API.

        Returns:
            bool: True if connected successfully

        Raises:
            Exception: If connection fails
        """
        # TODO #5: Test connection by getting account info
        # HINT:
        # try:
        #     account = self.client.get_account()
        #     print(f"✅ Connected to Alpaca. Account: {account.account_number}")
        #     return True
        # except Exception as e:
        #     print(f"❌ Failed to connect to Alpaca: {e}")
        #     raise
        pass


    async def get_account(self) -> Dict:
        """
        Get account information.

        Returns:
            Dict with:
            - account_number: Account ID
            - cash: Available cash
            - portfolio_value: Total account value
            - buying_power: Available buying power

        Example:
            {
                "account_number": "PA123456",
                "cash": "100000.00",
                "portfolio_value": "105000.00",
                "buying_power": "100000.00"
            }
        """
        # TODO #6: Get account info from Alpaca
        # HINT:
        # try:
        #     account = self.client.get_account()
        #     return {
        #         "account_number": account.account_number,
        #         "cash": account.cash,
        #         "portfolio_value": account.equity,
        #         "buying_power": account.buying_power
        #     }
        # except Exception as e:
        #     print(f"Error getting account: {e}")
        #     raise
        pass


    async def get_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of positions, each with:
            - symbol: Stock ticker
            - qty: Number of shares
            - current_price: Current market price
            - market_value: Total position value
            - avg_entry_price: Average purchase price
            - unrealized_pl: Unrealized profit/loss

        Example:
            [
                {
                    "symbol": "AAPL",
                    "qty": "10",
                    "current_price": "150.25",
                    "market_value": "1502.50",
                    "avg_entry_price": "148.00",
                    "unrealized_pl": "22.50"
                }
            ]
        """
        # TODO #7: Get positions from Alpaca
        # HINT:
        # try:
        #     positions = self.client.get_all_positions()
        #     return [
        #         {
        #             "symbol": pos.symbol,
        #             "qty": pos.qty,
        #             "current_price": pos.current_price,
        #             "market_value": pos.market_value,
        #             "avg_entry_price": pos.avg_entry_price,
        #             "unrealized_pl": pos.unrealized_pl
        #         }
        #         for pos in positions
        #     ]
        # except Exception as e:
        #     print(f"Error getting positions: {e}")
        #     return []
        pass


    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market"
    ) -> Dict:
        """
        Execute a trade.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            side: "buy" or "sell"
            quantity: Number of shares
            order_type: "market" or "limit" (default: "market")

        Returns:
            Dict with order details:
            - id: Order ID
            - symbol: Stock ticker
            - side: buy/sell
            - qty: Quantity
            - status: Order status
            - filled_at: Execution timestamp

        Example:
            {
                "id": "abc-123-def",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "10",
                "status": "filled",
                "filled_at": "2024-11-13T12:00:00Z"
            }
        """
        # TODO #8: Execute market order via Alpaca
        # HINT:
        # try:
        #     # Convert side to Alpaca enum
        #     order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        #
        #     # Create order request
        #     order_data = MarketOrderRequest(
        #         symbol=symbol.upper(),
        #         qty=quantity,
        #         side=order_side,
        #         time_in_force=TimeInForce.DAY  # Order valid for current day
        #     )
        #
        #     # Submit order
        #     order = self.client.submit_order(order_data)
        #
        #     return {
        #         "id": order.id,
        #         "symbol": order.symbol,
        #         "side": order.side.value,
        #         "qty": order.qty,
        #         "status": order.status.value,
        #         "filled_at": order.filled_at
        #     }
        # except Exception as e:
        #     print(f"Error executing trade: {e}")
        #     raise
        pass


    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with:
            - symbol: Stock ticker
            - bid: Best bid price
            - ask: Best ask price
            - last: Last trade price

        Example:
            {
                "symbol": "AAPL",
                "bid": 150.45,
                "ask": 150.55,
                "last": 150.50
            }
        """
        # TODO #9: Get quote from Alpaca
        # HINT:
        # try:
        #     from alpaca.data.historical import StockHistoricalDataClient
        #     from alpaca.data.requests import StockLatestQuoteRequest
        #
        #     # Create data client (doesn't need secret key)
        #     data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        #
        #     # Get latest quote
        #     request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        #     quote = data_client.get_stock_latest_quote(request_params)
        #
        #     return {
        #         "symbol": symbol,
        #         "bid": quote[symbol].bid_price,
        #         "ask": quote[symbol].ask_price,
        #         "last": (quote[symbol].bid_price + quote[symbol].ask_price) / 2
        #     }
        # except Exception as e:
        #     print(f"Error getting quote: {e}")
        #     return None
        pass


    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            bool: True if cancelled successfully
        """
        # TODO #10: Cancel order via Alpaca
        # HINT:
        # try:
        #     self.client.cancel_order_by_id(order_id)
        #     return True
        # except Exception as e:
        #     print(f"Error cancelling order: {e}")
        #     return False
        pass


# ============================================================================
# TESTING YOUR BROKER
# ============================================================================

"""
Test your broker with this code:

async def test_broker():
    # Initialize
    broker = AlpacaBroker()

    # Get account info
    account = await broker.get_account()
    print(f"Account Value: ${account['portfolio_value']}")
    print(f"Cash: ${account['cash']}")

    # Get quote
    quote = await broker.get_quote("AAPL")
    print(f"AAPL Price: ${quote['last']}")

    # Execute paper trade
    order = await broker.execute_trade(
        symbol="AAPL",
        side="buy",
        quantity=1
    )
    print(f"Order executed: {order['id']}")

    # Check positions
    positions = await broker.get_positions()
    print(f"Positions: {positions}")

# Run test
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_broker())
"""


# ============================================================================
# HELPFUL HINTS
# ============================================================================

"""
HINT #1 - Alpaca SDK Installation:
    pip install alpaca-py

HINT #2 - Getting API Keys:
    1. Sign up at https://alpaca.markets
    2. Choose "Paper Trading" (NOT live trading)
    3. Go to Dashboard → API Keys
    4. Copy API Key ID and Secret Key
    5. Add to .env file:
       ALPACA_API_KEY=PKXXXXXXXXXXXXX
       ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXX

HINT #3 - Paper Trading Safety:
    - paper=True ensures no real money
    - Test environment uses fake $100K
    - Trades are simulated but realistic
    - Perfect for learning!

HINT #4 - Error Handling:
    Always wrap API calls in try-except:

    try:
        account = self.client.get_account()
        return account
    except Exception as e:
        print(f"Error: {e}")
        raise  # Re-raise for caller to handle

HINT #5 - Async Functions:
    Mark functions as async if they might do I/O:

    async def get_account(self) -> Dict:
        # Even if not awaiting internally,
        # allows caller to await if needed
        return self.client.get_account()

Still stuck? Check:
- ../solutions/broker_solution.py
- ../notes/paper_trading.md
- Alpaca docs: https://alpaca.markets/docs/
"""
