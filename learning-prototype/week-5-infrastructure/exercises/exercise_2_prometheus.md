# Exercise 2: Prometheus Metrics Collection

## Objective
Set up Prometheus to collect and monitor metrics from your trading system, including custom business metrics.

## Prerequisites
- Completed Exercise 1 (Docker setup)
- Trading services running
- Basic understanding of metrics and monitoring

## Part 1: Add Prometheus Client to Trading API

Install the Prometheus client library:

```bash
pip install prometheus-client
```

## Part 2: Implement Metrics in Your Trading API

Create `src/metrics.py`:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client import CollectorRegistry, multiprocess, CONTENT_TYPE_LATEST
import time

# Trading metrics
order_counter = Counter(
    'order_count_total',
    'Total number of orders placed',
    ['symbol', 'side', 'status']
)

order_latency = Histogram(
    'order_latency_seconds',
    'Order execution latency in seconds',
    ['symbol'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

position_value = Gauge(
    'position_value_dollars',
    'Current position value in dollars',
    ['symbol']
)

trade_pnl = Gauge(
    'trade_pnl_dollars',
    'Profit/loss on trades in dollars',
    ['symbol']
)

open_positions = Gauge(
    'open_positions_count',
    'Number of open positions'
)

# API metrics
http_requests = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# System metrics
db_connections = Gauge(
    'db_connection_pool_size',
    'Database connection pool size',
    ['status']
)

redis_cache_hits = Counter(
    'redis_cache_hits_total',
    'Total Redis cache hits'
)

redis_cache_misses = Counter(
    'redis_cache_misses_total',
    'Total Redis cache misses'
)

market_data_lag = Gauge(
    'market_data_lag_seconds',
    'Market data lag in seconds',
    ['symbol']
)

# Strategy metrics
strategy_signals = Counter(
    'strategy_signals_total',
    'Trading signals generated',
    ['strategy', 'signal_type']
)

risk_var = Gauge(
    'risk_var_dollars',
    'Value at Risk in dollars',
    ['confidence_level']
)

# Helper function to track request duration
class RequestTimer:
    def __init__(self, method, endpoint):
        self.method = method
        self.endpoint = endpoint
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        http_request_duration.labels(
            method=self.method,
            endpoint=self.endpoint
        ).observe(duration)
```

## Part 3: Add Metrics Endpoint to FastAPI

Update your `src/main.py`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.metrics import (
    http_requests, RequestTimer,
    order_counter, position_value, open_positions
)
import time

app = FastAPI(title="Trading API")

# Middleware to track all requests
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    http_requests.labels(
        method=method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()

    return response

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Example trading endpoint with metrics
@app.post("/orders")
async def create_order(symbol: str, side: str, quantity: float):
    with RequestTimer("POST", "/orders"):
        # Simulate order processing
        time.sleep(0.05)

        # Record metrics
        order_counter.labels(
            symbol=symbol,
            side=side,
            status="filled"
        ).inc()

        return {"status": "success", "symbol": symbol}
```

## Part 4: Verify Prometheus Configuration

Check that your `configs/prometheus-config.yml` includes the trading_api job:

```yaml
scrape_configs:
  - job_name: 'trading_api'
    scrape_interval: 10s
    static_configs:
      - targets: ['trading_api:8000']
    metrics_path: '/metrics'
```

## Part 5: Start Services and Verify Metrics

1. **Restart services to apply changes:**
   ```bash
   docker-compose -f docker-compose-basic.yml restart trading_api
   ```

2. **Check if metrics endpoint is working:**
   ```bash
   curl http://localhost:8000/metrics
   ```

3. **Access Prometheus UI:**
   Open browser to http://localhost:9090

4. **Verify targets are being scraped:**
   - Go to Status > Targets
   - Ensure trading_api shows as "UP"

## Part 6: Query Metrics in Prometheus

Try these PromQL queries in the Prometheus UI:

### HTTP Request Rate
```promql
rate(http_requests_total[5m])
```

### Order Volume by Symbol
```promql
sum by (symbol) (rate(order_count_total[5m]))
```

### 95th Percentile Request Latency
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Total Orders in Last Hour
```promql
increase(order_count_total[1h])
```

### Current Position Values
```promql
position_value_dollars
```

### Cache Hit Rate
```promql
rate(redis_cache_hits_total[5m]) /
(rate(redis_cache_hits_total[5m]) + rate(redis_cache_misses_total[5m]))
```

## Part 7: Generate Sample Metrics

Create a script to generate sample trading activity:

```python
import requests
import random
import time

symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
sides = ['BUY', 'SELL']

for i in range(100):
    symbol = random.choice(symbols)
    side = random.choice(sides)
    quantity = random.randint(1, 100)

    response = requests.post(
        'http://localhost:8000/orders',
        params={
            'symbol': symbol,
            'side': side,
            'quantity': quantity
        }
    )
    print(f"Order {i+1}: {symbol} {side} {quantity} - {response.status_code}")
    time.sleep(0.1)
```

Run the script and watch metrics update in Prometheus.

## Part 8: Create Alert Rules (Optional)

Create `configs/alert_rules.yml`:

```yaml
groups:
  - name: trading_alerts
    interval: 30s
    rules:
      # Alert if API response time is too high
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "95th percentile latency is {{ $value }}s"

      # Alert if order failure rate is high
      - alert: HighOrderFailureRate
        expr: rate(order_count_total{status="failed"}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High order failure rate"
          description: "Order failure rate is {{ $value }} per second"

      # Alert if position value drops significantly
      - alert: LargePositionLoss
        expr: trade_pnl_dollars < -10000
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Large position loss detected"
          description: "P&L is {{ $value }} dollars"
```

## Tasks to Complete

- [ ] Install prometheus-client library
- [ ] Create metrics.py with custom metrics
- [ ] Add metrics endpoint to FastAPI application
- [ ] Verify metrics are being scraped by Prometheus
- [ ] Execute PromQL queries successfully
- [ ] Generate sample trading activity
- [ ] Monitor metrics in Prometheus UI
- [ ] (Optional) Create and test alert rules

## Expected Outcomes

After completing this exercise, you should be able to:
1. Instrument Python code with Prometheus metrics
2. Expose metrics via HTTP endpoint
3. Write PromQL queries to analyze data
4. Understand different metric types (Counter, Gauge, Histogram)
5. Monitor trading-specific business metrics

## Next Steps

Proceed to Exercise 3 to create Grafana dashboards for visualization.
