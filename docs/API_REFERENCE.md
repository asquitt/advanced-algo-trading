

# API Reference

Complete API documentation for the LLM Trading Platform.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required (paper trading only). In production, implement:
- API keys via headers
- JWT tokens
- OAuth 2.0

---

## Endpoints

### Health & Status

#### `GET /`

Root endpoint with service information.

**Response:**
```json
{
  "status": "running",
  "service": "LLM Trading Platform",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### `GET /health`

Detailed health check.

**Response:**
```json
{
  "status": "healthy",
  "broker": "healthy",
  "paper_trading": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200`: Healthy
- `503`: Service unavailable

---

### Trading Signals

#### `POST /signals/generate`

Generate a trading signal for a symbol.

**Parameters:**
- `symbol` (string, required): Stock ticker (e.g., "AAPL")
- `use_cache` (boolean, optional): Use cached analysis (default: true)
- `execute` (boolean, optional): Auto-execute signal (default: false)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/signals/generate?symbol=AAPL&use_cache=true&execute=false"
```

**Response:**
```json
{
  "symbol": "AAPL",
  "signal_type": "BUY",
  "confidence_score": 0.85,
  "ai_conviction_score": 0.78,
  "fundamental_score": 0.82,
  "sentiment_score": 0.65,
  "technical_score": 0.75,
  "reasoning": "AI Conviction Score: 0.78/1.00\n\n📊 Fundamental Analysis:\n  - Score: 0.82\n  - Valuation: fairly_valued\n  ...",
  "source_agent": "ensemble_strategy",
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "financial_analysis": {...},
    "sentiment_analysis": {...},
    "total_cost_usd": 0.0046,
    "tokens_used": 2350
  }
}
```

**Status Codes:**
- `200`: Success
- `500`: Error generating signal

**Performance:**
- Cached: <10ms
- Uncached: 50-200ms

**Cost:**
- Cached: $0.00
- Uncached: $0.002-$0.01

---

#### `POST /signals/batch`

Generate signals for multiple symbols.

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "use_cache": true
}
```

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "signal_type": "BUY",
    ...
  },
  {
    "symbol": "MSFT",
    "signal_type": "HOLD",
    ...
  },
  {
    "symbol": "GOOGL",
    "signal_type": "SELL",
    ...
  }
]
```

**Performance:**
- 3 symbols: 100-500ms (depending on cache)

**Cost:**
- Up to $0.03 for 3 uncached symbols

---

### Trade Execution

#### `POST /trades/execute`

Execute a trade for a symbol.

**Parameters:**
- `symbol` (string, required): Stock ticker
- `generate_signal` (boolean, optional): Generate new signal first (default: true)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/trades/execute?symbol=AAPL&generate_signal=true"
```

**Response:**
```json
{
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 66,
  "entry_price": 150.50,
  "status": "filled",
  "order_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filled_at": "2024-01-15T10:30:00Z",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200`: Trade executed
- `400`: Trade not executed (see details)
- `500`: Internal error

**Notes:**
- Only executes during market hours (if `TRADING_HOURS_ONLY=true`)
- Respects risk management rules
- Paper trading only by default

---

### Portfolio Management

#### `GET /portfolio`

Get current portfolio state.

**Response:**
```json
{
  "cash_balance": 50000.00,
  "portfolio_value": 120000.00,
  "total_pnl": 5000.00,
  "total_pnl_percent": 0.0417,
  "active_positions": 5,
  "win_rate": 0.65,
  "sharpe_ratio": 1.2,
  "max_drawdown": -0.08,
  "snapshot_at": "2024-01-15T10:30:00Z"
}
```

---

#### `GET /portfolio/summary`

Get detailed portfolio summary with positions.

**Response:**
```json
{
  "account": {
    "cash": 50000.00,
    "portfolio_value": 120000.00,
    "buying_power": 100000.00,
    "equity": 120000.00,
    "last_equity": 115000.00,
    "pattern_day_trader": false,
    "trading_blocked": false,
    "transfers_blocked": false
  },
  "portfolio_state": {
    "cash_balance": 50000.00,
    "portfolio_value": 120000.00,
    "total_pnl": 5000.00,
    "total_pnl_percent": 0.0417,
    "active_positions": 5
  },
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 50,
      "avg_entry_price": 150.00,
      "current_price": 155.00,
      "market_value": 7750.00,
      "unrealized_pnl": 250.00,
      "unrealized_pnl_percent": 0.0333
    },
    ...
  ],
  "total_unrealized_pnl": 1250.00,
  "num_positions": 5
}
```

---

#### `GET /positions`

Get all open positions.

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "quantity": 50,
    "avg_entry_price": 150.00,
    "current_price": 155.00,
    "market_value": 7750.00,
    "unrealized_pnl": 250.00,
    "unrealized_pnl_percent": 0.0333,
    "updated_at": "2024-01-15T10:30:00Z"
  },
  ...
]
```

---

#### `GET /positions/{symbol}`

Get position for a specific symbol.

**Parameters:**
- `symbol` (string, required): Stock ticker

**Response:**
```json
{
  "symbol": "AAPL",
  "quantity": 50,
  "avg_entry_price": 150.00,
  "current_price": 155.00,
  "market_value": 7750.00,
  "unrealized_pnl": 250.00,
  "unrealized_pnl_percent": 0.0333,
  "updated_at": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200`: Position found
- `404`: No position for symbol

---

#### `DELETE /positions/{symbol}`

Close an entire position.

**Parameters:**
- `symbol` (string, required): Stock ticker

**Response:**
```json
{
  "message": "Position closed for AAPL"
}
```

**Status Codes:**
- `200`: Position closed
- `400`: Failed to close position

---

#### `GET /account`

Get account information.

**Response:**
```json
{
  "cash": 50000.00,
  "portfolio_value": 120000.00,
  "buying_power": 100000.00,
  "equity": 120000.00,
  "last_equity": 115000.00,
  "pattern_day_trader": false,
  "trading_blocked": false,
  "transfers_blocked": false
}
```

---

### Monitoring

#### `GET /metrics`

Get Prometheus metrics.

**Response:**
```
# HELP trading_signals_total Total number of trading signals generated
# TYPE trading_signals_total counter
trading_signals_total{signal_type="BUY",symbol="AAPL"} 15.0
trading_signals_total{signal_type="SELL",symbol="AAPL"} 8.0
trading_signals_total{signal_type="HOLD",symbol="AAPL"} 12.0

# HELP trades_executed_total Total number of trades executed
# TYPE trades_executed_total counter
trades_executed_total{side="buy",symbol="AAPL"} 10.0
trades_executed_total{side="sell",symbol="AAPL"} 5.0

# HELP api_request_duration_seconds API request latency
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{endpoint="/signals/generate",le="0.05"} 45.0
api_request_duration_seconds_bucket{endpoint="/signals/generate",le="0.1"} 82.0
...
```

**Format:** Prometheus text format

**Use:** Scrape with Prometheus, visualize in Grafana

---

## Data Models

### TradingSignal

```typescript
{
  symbol: string;              // Stock ticker
  signal_type: "BUY" | "SELL" | "HOLD";
  confidence_score: number;     // 0-1
  ai_conviction_score: number;  // 0-1
  fundamental_score?: number;   // 0-1
  sentiment_score?: number;     // -1 to 1
  technical_score?: number;     // 0-1
  reasoning: string;            // Human-readable explanation
  source_agent: string;         // Agent that generated signal
  metadata?: object;            // Additional context
  created_at: string;           // ISO 8601 timestamp
}
```

### Trade

```typescript
{
  symbol: string;
  side: "buy" | "sell";
  quantity: number;             // Positive integer
  entry_price?: number;         // Positive float
  exit_price?: number;          // Positive float
  pnl?: number;                 // Profit/loss in dollars
  pnl_percent?: number;         // Profit/loss percentage
  status: "pending" | "filled" | "partial" | "cancelled" | "rejected";
  order_id?: string;
  signal_id?: string;
  filled_at?: string;           // ISO 8601 timestamp
  closed_at?: string;           // ISO 8601 timestamp
  created_at: string;           // ISO 8601 timestamp
}
```

### Position

```typescript
{
  symbol: string;
  quantity: number;
  avg_entry_price: number;
  current_price?: number;
  market_value?: number;
  unrealized_pnl?: number;
  unrealized_pnl_percent?: number;
  updated_at: string;           // ISO 8601 timestamp
}
```

### PortfolioState

```typescript
{
  cash_balance: number;
  portfolio_value: number;
  total_pnl: number;
  total_pnl_percent: number;
  active_positions: number;
  win_rate?: number;            // 0-1
  sharpe_ratio?: number;
  max_drawdown?: number;
  snapshot_at: string;          // ISO 8601 timestamp
}
```

---

## Error Responses

All errors return the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Status Codes

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error

---

## Rate Limits

No rate limits for paper trading. In production:
- Signal generation: 100/minute
- Trade execution: 50/minute
- Portfolio queries: 1000/minute

---

## Code Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Generate signal
response = requests.post(
    f"{BASE_URL}/signals/generate",
    params={"symbol": "AAPL", "use_cache": True}
)
signal = response.json()

print(f"Signal: {signal['signal_type']}")
print(f"Conviction: {signal['ai_conviction_score']:.2%}")

# Execute trade
if signal['signal_type'] == 'BUY':
    response = requests.post(
        f"{BASE_URL}/trades/execute",
        params={"symbol": "AAPL"}
    )
    trade = response.json()
    print(f"Trade executed: {trade['quantity']} shares @ ${trade['entry_price']}")

# Check portfolio
response = requests.get(f"{BASE_URL}/portfolio")
portfolio = response.json()
print(f"Portfolio value: ${portfolio['portfolio_value']:,.2f}")
print(f"Total P&L: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_percent']:.2%})")
```

### JavaScript

```javascript
const BASE_URL = "http://localhost:8000";

// Generate signal
const generateSignal = async (symbol) => {
  const response = await fetch(
    `${BASE_URL}/signals/generate?symbol=${symbol}&use_cache=true`,
    { method: "POST" }
  );
  return response.json();
};

// Execute trade
const executeTrade = async (symbol) => {
  const response = await fetch(
    `${BASE_URL}/trades/execute?symbol=${symbol}`,
    { method: "POST" }
  );
  return response.json();
};

// Get portfolio
const getPortfolio = async () => {
  const response = await fetch(`${BASE_URL}/portfolio`);
  return response.json();
};

// Usage
(async () => {
  const signal = await generateSignal("AAPL");
  console.log(`Signal: ${signal.signal_type}`);

  if (signal.signal_type === "BUY") {
    const trade = await executeTrade("AAPL");
    console.log(`Executed: ${trade.quantity} shares`);
  }

  const portfolio = await getPortfolio();
  console.log(`Portfolio value: $${portfolio.portfolio_value}`);
})();
```

### cURL

```bash
# Generate signal
curl -X POST "http://localhost:8000/signals/generate?symbol=AAPL"

# Generate batch signals
curl -X POST "http://localhost:8000/signals/batch" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT", "GOOGL"], "use_cache": true}'

# Execute trade
curl -X POST "http://localhost:8000/trades/execute?symbol=AAPL&generate_signal=true"

# Get portfolio
curl "http://localhost:8000/portfolio/summary"

# Get positions
curl "http://localhost:8000/positions"

# Close position
curl -X DELETE "http://localhost:8000/positions/AAPL"
```

---

## Interactive Documentation

Visit **http://localhost:8000/docs** for interactive Swagger UI where you can:
- Try all endpoints
- See request/response schemas
- Test with different parameters
- Download OpenAPI spec

Alternative: **http://localhost:8000/redoc** for ReDoc documentation.

---

## WebSocket API (Future)

Coming soon: Real-time streaming of signals and trades via WebSocket.

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("New signal:", data);
};
```

---

## Best Practices

1. **Use caching**: Set `use_cache=true` to save costs and improve latency
2. **Batch requests**: Use `/signals/batch` for multiple symbols
3. **Check market hours**: Verify trades execute during market hours
4. **Monitor costs**: Track `metadata.total_cost_usd` in responses
5. **Handle errors**: Always check status codes and error messages
6. **Rate limit**: Don't hammer the API (be nice to LLM providers)
7. **Paper trade first**: Always test strategies before real money

---

## Support

- **API Issues**: Check logs with `docker-compose logs -f trading-api`
- **Documentation**: See `/docs` folder for guides
- **Examples**: See `/examples` folder for sample code