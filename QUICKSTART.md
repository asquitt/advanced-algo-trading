# Quick Start Guide - LLM Trading Platform

Complete guide to get the trading platform running locally in under 15 minutes.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Setup (TL;DR)](#quick-setup-tldr)
- [Detailed Setup](#detailed-setup)
- [Configuration](#configuration)
- [Running the Platform](#running-the-platform)
- [Accessing Services](#accessing-services)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **Python** (3.11+)
- **Git**
- **4GB+ RAM** available
- **10GB+ disk space**

### Optional (for API keys)
- Alpaca API account (paper trading)
- Alpha Vantage API key (news feeds)
- NewsAPI key (alternative news)
- Groq API key (LLM integration)

---

## Quick Setup (TL;DR)

```bash
# 1. Clone and enter directory
git clone <repo-url>
cd reimagined-winner

# 2. Copy environment template
cp .env.example .env

# 3. Add your API keys to .env (optional for testing)
# Edit .env and add:
# - ALPACA_API_KEY=your_key
# - ALPACA_SECRET_KEY=your_secret
# - ALPHA_VANTAGE_API_KEY=your_key

# 4. Start all services
docker-compose up -d

# 5. Verify services are running
docker-compose ps

# 6. Access the platform
# API: http://localhost:8000
# Grafana: http://localhost:3000 (admin/admin_password)
# Prometheus: http://localhost:9090
# Kibana: http://localhost:5601
```

---

## Detailed Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd reimagined-winner
```

### Step 2: Environment Configuration

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` with your favorite editor:

```bash
nano .env  # or vim, code, etc.
```

**Required configurations:**

```bash
# Environment
ENVIRONMENT=development

# Alpaca (Paper Trading) - REQUIRED for trading
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER_TRADING=true

# LLM Integration - REQUIRED for AI features
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_claude_key

# News Feeds - OPTIONAL (but recommended)
ALPHA_VANTAGE_API_KEY=your_av_key
NEWS_API_KEY=your_newsapi_key

# Database - Auto-configured in Docker
DATABASE_URL=postgresql://trading_user:trading_pass@postgres:5432/trading_db
TIMESCALE_URL=postgresql://trading_user:trading_pass@timescaledb:5432/timeseries_db

# Redis - Auto-configured
REDIS_URL=redis://redis:6379/0

# Kafka - Auto-configured
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
```

### Step 3: Install Python Dependencies (for local development)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Initialize Database Schemas

```bash
# Create database initialization scripts directory
mkdir -p scripts

# The database will be auto-initialized on first run
# No manual setup needed!
```

---

## Configuration

### Trading Configuration

Edit `config/trading_config.yaml`:

```yaml
trading:
  paper_trading: true  # ALWAYS use paper trading for safety
  initial_capital: 100000  # Starting capital
  max_position_size: 0.10  # 10% max per position
  max_leverage: 1.0  # No leverage

risk:
  max_drawdown: 0.15  # 15% max drawdown
  max_daily_loss: 0.05  # 5% max daily loss
  position_sizing: "kelly"  # kelly, equal_weight, volatility

strategies:
  enabled:
    - pairs_trading
    - regime_momentum
    - sentiment_intraday
```

### Monitoring Configuration

Prometheus is pre-configured in `monitoring/prometheus/prometheus.yml`.

Grafana dashboards are auto-loaded from `monitoring/grafana/dashboards/`.

---

## Running the Platform

### Start All Services

```bash
# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f trading-api

# Check service health
docker-compose ps
```

Expected output:

```
NAME                     STATUS              PORTS
trading-api              Up (healthy)        0.0.0.0:8000->8000/tcp
postgres                 Up (healthy)        0.0.0.0:5432->5432/tcp
timescaledb              Up (healthy)        0.0.0.0:5433->5432/tcp
redis                    Up                  0.0.0.0:6379->6379/tcp
kafka                    Up (healthy)        0.0.0.0:9092->9092/tcp
zookeeper                Up                  0.0.0.0:2181->2181/tcp
prometheus               Up                  0.0.0.0:9090->9090/tcp
grafana                  Up                  0.0.0.0:3000->3000/tcp
elasticsearch            Up (healthy)        0.0.0.0:9200->9200/tcp
logstash                 Up                  0.0.0.0:5000->5000/tcp
kibana                   Up (healthy)        0.0.0.0:5601->5601/tcp
mlflow                   Up                  0.0.0.0:5000->5000/tcp
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (CAUTION: deletes all data)
docker-compose down -v
```

---

## Accessing Services

### Main Services

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Trading API** | http://localhost:8000 | None | REST API endpoints |
| **API Docs** | http://localhost:8000/docs | None | Interactive API documentation |
| **Grafana** | http://localhost:3000 | admin / admin_password | Dashboards & visualization |
| **Prometheus** | http://localhost:9090 | None | Metrics & monitoring |
| **Kibana** | http://localhost:5601 | None | Log analysis |
| **MLflow** | http://localhost:5000 | None | Experiment tracking |

### Database Access

```bash
# PostgreSQL
docker exec -it postgres psql -U trading_user -d trading_db

# TimescaleDB
docker exec -it timescaledb psql -U trading_user -d timeseries_db

# Redis
docker exec -it redis redis-cli
```

### Kafka Topics

```bash
# List topics
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092

# Consume signals topic
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic trading_signals \
  --from-beginning
```

---

## Testing

### Run Test Suite

```bash
# Full test suite
pytest tests/ -v

# Run specific test category
pytest tests/test_backtesting.py -v
pytest tests/test_broker.py -v
pytest tests/test_strategies.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Backtesting Engine

```bash
# Run simple backtest
python examples/run_backtest.py
```

### Test Trading Strategies

```bash
# Test pairs trading
python examples/test_pairs_trading.py

# Test momentum strategy
python examples/test_regime_momentum.py

# Test sentiment strategy
python examples/test_sentiment_intraday.py
```

---

## Troubleshooting

### Common Issues

#### 1. Docker Services Won't Start

**Problem**: Services fail to start or exit immediately

**Solutions**:
```bash
# Check Docker is running
docker info

# Check logs for errors
docker-compose logs <service-name>

# Restart Docker daemon
sudo systemctl restart docker  # Linux
# or restart Docker Desktop on Mac/Windows

# Clean up and retry
docker-compose down
docker system prune -f
docker-compose up -d
```

#### 2. Port Already in Use

**Problem**: `Error: bind: address already in use`

**Solutions**:
```bash
# Find process using port
lsof -i :8000  # Replace 8000 with your port

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use different host port
```

#### 3. Kafka Connection Errors

**Problem**: `Connection refused to kafka:9092`

**Solutions**:
```bash
# Wait for Kafka to fully start (takes 30-60s)
docker-compose logs -f kafka

# Restart Kafka
docker-compose restart kafka

# Check Zookeeper is running
docker-compose ps zookeeper
```

#### 4. Database Connection Errors

**Problem**: `FATAL: password authentication failed`

**Solutions**:
```bash
# Reset database
docker-compose down -v
docker-compose up -d postgres

# Check credentials match .env file
docker exec -it postgres env | grep POSTGRES
```

#### 5. Out of Memory

**Problem**: Services crashing with OOM errors

**Solutions**:
```bash
# Increase Docker memory limit
# Docker Desktop -> Preferences -> Resources -> Memory: 8GB+

# Reduce Java heap for Kafka/Elasticsearch
# In docker-compose.yml:
environment:
  - "ES_JAVA_OPTS=-Xms256m -Xmx256m"  # Reduce from 512m
```

#### 6. Test Failures

**Problem**: Tests failing unexpectedly

**Solutions**:
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Clear pytest cache
pytest --cache-clear

# Run tests in isolation
pytest tests/test_backtesting.py -v --tb=short
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check Prometheus targets
open http://localhost:9090/targets

# Check Grafana datasources
open http://localhost:3000/datasources

# Check Elasticsearch cluster
curl http://localhost:9200/_cluster/health
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f trading-api

# Last 100 lines
docker-compose logs --tail=100 trading-api

# Filter by keyword
docker-compose logs trading-api | grep ERROR
```

---

## Next Steps

Once the platform is running:

1. **Explore API Documentation**
   - Visit http://localhost:8000/docs
   - Try the interactive endpoints

2. **View Grafana Dashboards**
   - Visit http://localhost:3000
   - Default dashboards:
     - Trading Overview
     - Risk Management
     - Execution Quality
     - System Health

3. **Run a Backtest**
   - See `examples/run_backtest.py`
   - Test different strategies
   - Analyze results

4. **Review Learning Prototype**
   - Check `learning-prototype/` directory
   - Follow 8-week curriculum
   - Build your own strategies

5. **Monitor System**
   - Prometheus metrics: http://localhost:9090
   - Logs in Kibana: http://localhost:5601
   - MLflow experiments: http://localhost:5000

---

## Support

- **Documentation**: Check `docs/` directory
- **Issues**: Create GitHub issue
- **Logs**: Check `logs/` directory
- **Tests**: Run `pytest tests/ -v`

---

## Safety Reminders

⚠️ **IMPORTANT**: This platform uses PAPER TRADING by default.

- Never use real money without thorough testing
- Always verify `ALPACA_PAPER_TRADING=true` in `.env`
- Test all strategies extensively before considering live trading
- Monitor risk metrics closely (drawdown, VaR, etc.)
- Set appropriate stop losses and position limits

---

**Happy Trading! 🚀**
