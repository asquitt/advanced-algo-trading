# Exercise 1: Docker Setup for Trading System

## Objective
Set up a containerized trading environment using Docker Compose with PostgreSQL, Redis, and a trading API service.

## Prerequisites
- Docker installed (version 20.10+)
- Docker Compose installed (version 2.0+)
- Basic understanding of containers

## Part 1: Create Dockerfile for Trading API

Create a `Dockerfile` in the `learning-prototype/week-5-infrastructure/` directory:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Part 2: Create Database Initialization Script

Create `configs/init.sql`:

```sql
-- Create tables for trading system
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    avg_price DECIMAL(18, 8) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    volume BIGINT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_market_data_symbol ON market_data(symbol);
```

## Part 3: Start the Services

1. **Navigate to the configs directory:**
   ```bash
   cd /home/user/advanced-algo-trading/learning-prototype/week-5-infrastructure/configs
   ```

2. **Start all services:**
   ```bash
   docker-compose -f docker-compose-basic.yml up -d
   ```

3. **Verify all containers are running:**
   ```bash
   docker-compose -f docker-compose-basic.yml ps
   ```

4. **Check container logs:**
   ```bash
   docker-compose -f docker-compose-basic.yml logs -f trading_api
   ```

## Part 4: Test the Infrastructure

### Test PostgreSQL Connection
```bash
docker exec -it trading_postgres psql -U trader -d trading_db -c "\dt"
```

### Test Redis Connection
```bash
docker exec -it trading_redis redis-cli ping
```

Expected output: `PONG`

### Test Trading API
```bash
curl http://localhost:8000/health
```

## Part 5: Interact with Services

### Insert Sample Data into PostgreSQL
```bash
docker exec -it trading_postgres psql -U trader -d trading_db -c \
  "INSERT INTO trades (symbol, side, quantity, price) VALUES ('AAPL', 'BUY', 100, 150.25);"
```

### Query Data
```bash
docker exec -it trading_postgres psql -U trader -d trading_db -c \
  "SELECT * FROM trades;"
```

### Use Redis for Caching
```bash
# Set a value
docker exec -it trading_redis redis-cli SET last_price:AAPL 150.25

# Get a value
docker exec -it trading_redis redis-cli GET last_price:AAPL

# Set with expiration (60 seconds)
docker exec -it trading_redis redis-cli SETEX market_data:AAPL 60 '{"price": 150.25, "volume": 1000000}'
```

## Part 6: Monitor Resource Usage

```bash
# Check container resource usage
docker stats

# View specific container metrics
docker stats trading_postgres trading_redis trading_api
```

## Part 7: Clean Up

```bash
# Stop all services
docker-compose -f docker-compose-basic.yml down

# Remove volumes (WARNING: Deletes all data)
docker-compose -f docker-compose-basic.yml down -v
```

## Tasks to Complete

- [ ] Create Dockerfile for the trading API
- [ ] Create init.sql with table definitions
- [ ] Start all services using docker-compose
- [ ] Verify all containers are healthy
- [ ] Test PostgreSQL connection and query data
- [ ] Test Redis connection and caching
- [ ] Monitor container resource usage
- [ ] Successfully stop and restart services

## Troubleshooting

**Port already in use:**
```bash
# Find process using port 5432
lsof -i :5432
# Kill the process or change the port in docker-compose.yml
```

**Container fails to start:**
```bash
# Check logs
docker-compose -f docker-compose-basic.yml logs <service_name>

# Inspect container
docker inspect <container_name>
```

**Database connection refused:**
- Wait for PostgreSQL to fully initialize (check health status)
- Verify connection string in trading_api environment variables

## Expected Outcomes

After completing this exercise, you should have:
1. Running PostgreSQL, Redis, and trading API containers
2. Initialized database with trading tables
3. Understanding of Docker networking and health checks
4. Ability to debug containerized applications
5. Knowledge of Docker resource management

## Next Steps

Proceed to Exercise 2 to set up Prometheus for metrics collection.
