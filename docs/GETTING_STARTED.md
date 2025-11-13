# Getting Started Guide

This guide will walk you through setting up and using the LLM Trading Platform step-by-step.

## Prerequisites

### Required Software
- **Docker** (20.10+) and **Docker Compose** (2.0+)
  - [Install Docker](https://docs.docker.com/get-docker/)
- **Git** for version control

### API Keys (All Free Tiers Available)

#### 1. Groq API (Free, Fast LLM Inference)
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `gsk_...`)

**Why Groq?** Ultra-fast inference (~300 tokens/second) at very low cost. Perfect for simple tasks like sentiment analysis.

#### 2. Anthropic API (Claude for Complex Reasoning)
1. Go to [https://console.anthropic.com](https://console.anthropic.com)
2. Sign up for an account ($5 free credit on signup)
3. Navigate to API Keys
4. Create a new key
5. Copy the key (starts with `sk-ant-...`)

**Why Claude?** Best-in-class reasoning for complex financial analysis. We only use it sparingly to control costs.

#### 3. Alpaca API (Free Paper Trading)
1. Go to [https://alpaca.markets](https://alpaca.markets)
2. Sign up for a free account
3. Enable **Paper Trading** (important!)
4. Go to "Your API Keys" in the dashboard
5. Copy both the API Key and Secret Key

**Why Alpaca?** Free paper trading with real-time market data. Perfect for testing strategies without risking real money.

#### 4. Alpha Vantage (Optional, for backup data)
1. Go to [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Get a free API key (25 calls/day)

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd reimagined-winner

# Copy environment template
cp .env.example .env
```

## Step 2: Configure Environment

Edit the `.env` file with your API keys:

```bash
# Open in your favorite editor
nano .env
# or
vim .env
# or
code .env  # VS Code
```

Replace these values:
```bash
GROQ_API_KEY=gsk_your_actual_key_here
ANTHROPIC_API_KEY=sk-ant-your_actual_key_here
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
```

**Important**: Keep these keys secret! Never commit them to git.

## Step 3: Start the Platform

```bash
./scripts/start.sh
```

This script will:
1. Validate your environment file
2. Start all Docker services (PostgreSQL, Redis, Kafka, MLflow, etc.)
3. Start the Trading API
4. Display URLs for accessing services

**First startup takes 2-3 minutes** as Docker downloads images.

## Step 4: Verify Installation

### Check Service Health

```bash
# Check if API is running
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "broker": "healthy",
  "paper_trading": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Access the API Documentation

Open your browser to [http://localhost:8000/docs](http://localhost:8000/docs)

You should see the interactive Swagger UI with all available endpoints.

## Step 5: Generate Your First Trading Signal

### Via Web UI (Swagger)

1. Go to [http://localhost:8000/docs](http://localhost:8000/docs)
2. Find the `POST /signals/generate` endpoint
3. Click "Try it out"
4. Enter a symbol (e.g., `AAPL`)
5. Click "Execute"

### Via cURL

```bash
curl -X POST "http://localhost:8000/signals/generate?symbol=AAPL&use_cache=true"
```

### Via Python

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/signals/generate",
    params={
        "symbol": "AAPL",
        "use_cache": True  # Use cached analysis if available
    }
)

signal = response.json()

print(f"Symbol: {signal['symbol']}")
print(f"Signal Type: {signal['signal_type']}")  # BUY, SELL, or HOLD
print(f"AI Conviction: {signal['ai_conviction_score']:.2f}")
print(f"\nReasoning:\n{signal['reasoning']}")
```

## Step 6: Understanding the Output

A typical signal looks like:

```json
{
  "symbol": "AAPL",
  "signal_type": "BUY",
  "confidence_score": 0.82,
  "ai_conviction_score": 0.78,
  "fundamental_score": 0.85,
  "sentiment_score": 0.65,
  "technical_score": 0.5,
  "reasoning": "AI Conviction Score: 0.78/1.00\n\n📊 Fundamental Analysis:\n  - Score: 0.85\n  - Valuation: fairly_valued\n  - Thesis: Strong revenue growth in services segment...",
  "source_agent": "ensemble_strategy",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Key Fields:**
- `signal_type`: BUY, SELL, or HOLD recommendation
- `ai_conviction_score`: Overall AI confidence (0-1)
- `fundamental_score`: Score from financial analysis (0-1)
- `sentiment_score`: News sentiment score (0-1, converted from -1 to 1)
- `reasoning`: Human-readable explanation of the decision

## Step 7: Execute a Paper Trade (Optional)

⚠️ **Paper trading only! No real money involved.**

```bash
# Execute the signal (generates signal + executes trade)
curl -X POST "http://localhost:8000/signals/generate?symbol=AAPL&execute=true"
```

This will:
1. Generate a trading signal
2. Apply risk management rules
3. Execute a paper trade via Alpaca
4. Return the trade details

## Step 8: Monitor Your Portfolio

### Get Portfolio Summary
```bash
curl http://localhost:8000/portfolio/summary
```

### View Open Positions
```bash
curl http://localhost:8000/positions
```

### Check Account Balance
```bash
curl http://localhost:8000/account
```

## Step 9: View Experiment Tracking (MLflow)

1. Open [http://localhost:5000](http://localhost:5000)
2. Click on the "llm-trading-strategies" experiment
3. See all your trading signals with:
   - Parameters (weights, symbols)
   - Metrics (scores, costs, token usage)
   - Artifacts (full signal details)

This is powerful for:
- Comparing different strategy weights
- Analyzing what works and what doesn't
- Tracking API costs over time

## Step 10: View Monitoring (Grafana - Optional)

1. Open [http://localhost:3000](http://localhost:3000)
2. Login with `admin` / `admin`
3. Add Prometheus data source (http://prometheus:9090)
4. Create dashboards to visualize:
   - Trading signal rate
   - API latency
   - Trade execution success rate
   - Portfolio value over time

## Common Tasks

### Generate Signals for Multiple Stocks

```bash
curl -X POST "http://localhost:8000/signals/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "use_cache": true
  }'
```

### Close a Position

```bash
curl -X DELETE "http://localhost:8000/positions/AAPL"
```

### View Logs

```bash
# All logs
docker-compose logs -f

# Just the trading API
docker-compose logs -f trading-api

# Last 100 lines
docker-compose logs --tail=100 trading-api
```

## Troubleshooting

### API returns "Failed to get quote"
- **Cause**: Alpaca API rate limit or invalid symbol
- **Solution**: Wait a moment and try again, or check if the symbol exists

### "Kafka connection failed"
- **Cause**: Kafka not started or still initializing
- **Solution**: Wait 30 seconds after startup, Kafka takes time to initialize

### High API costs
- **Cause**: Cache is disabled or expired
- **Solution**:
  - Check Redis is running: `docker-compose ps redis`
  - Increase cache duration in `.env`: `CACHE_ANALYSIS_HOURS=48`
  - Enable `use_cache=true` in API calls

### MLflow UI not loading
- **Cause**: PostgreSQL not ready
- **Solution**: Check database: `docker-compose ps postgres`

## Next Steps

1. **Read the Architecture Docs**: [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Customize Agents**: [CUSTOM_AGENTS.md](CUSTOM_AGENTS.md)
3. **Run Backtests**: [BACKTESTING.md](BACKTESTING.md)
4. **Deploy to Production**: [DEPLOYMENT.md](DEPLOYMENT.md)

## Stopping the Platform

```bash
# Stop all services (preserves data)
./scripts/stop.sh

# Stop and remove all data
docker-compose down -v
```

## Getting Help

- **Check logs**: Most issues show up in logs
- **API Documentation**: http://localhost:8000/docs has examples
- **GitHub Issues**: Report bugs or ask questions

---

**Congratulations!** 🎉 You're now running an LLM-powered trading platform.

Remember: **This is for learning and paper trading only. Never risk real money without thorough testing!**
