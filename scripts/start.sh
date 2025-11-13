#!/bin/bash

# Startup script for LLM Trading Platform
# This script starts all services via Docker Compose

set -e  # Exit on error

echo "🚀 Starting LLM Trading Platform..."
echo "=" | head -c 60 | tr ' ' '='
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before proceeding!"
    echo "   Required: GROQ_API_KEY, ANTHROPIC_API_KEY, ALPACA_API_KEY"
    exit 1
fi

# Check for required API keys
if grep -q "your_groq_api_key_here" .env || \
   grep -q "your_anthropic_api_key_here" .env || \
   grep -q "your_alpaca_key_here" .env; then
    echo "⚠️  API keys not configured in .env file!"
    echo "   Please edit .env and add your API keys:"
    echo "   - GROQ_API_KEY (get free at: https://console.groq.com)"
    echo "   - ANTHROPIC_API_KEY (get at: https://console.anthropic.com)"
    echo "   - ALPACA_API_KEY (free paper trading: https://alpaca.markets)"
    exit 1
fi

echo "✅ Environment file validated"
echo ""

# Start Docker Compose services
echo "🐳 Starting Docker services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service health..."

# Check if trading-api is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Trading API is running"
else
    echo "⚠️  Trading API not responding yet (this is normal on first start)"
fi

echo ""
echo "=" | head -c 60 | tr ' ' '='
echo ""
echo "✨ LLM Trading Platform is starting up!"
echo ""
echo "📊 Access the services:"
echo "   - Trading API: http://localhost:8000"
echo "   - API Docs (Swagger): http://localhost:8000/docs"
echo "   - MLflow UI: http://localhost:5000"
echo "   - Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   - Prometheus: http://localhost:9090"
echo ""
echo "📝 View logs:"
echo "   docker-compose logs -f trading-api"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose down"
echo ""
echo "=" | head -c 60 | tr ' ' '='
