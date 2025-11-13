#!/bin/bash
###############################################################################
# Local Environment Setup Script
#
# This script sets up a minimal development environment for local testing
# without requiring Docker, Kafka, PostgreSQL, or Redis.
#
# Usage: ./scripts/setup_local_env.sh
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  LLM Trading Platform - Local Setup                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${YELLOW}[1/5] Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d'.' -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 11 ]); then
    echo -e "${RED}✗ Python 3.11+ required. Found: $python_version${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $python_version${NC}"

echo -e "${YELLOW}[2/5] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

echo -e "${YELLOW}[3/5] Activating virtual environment...${NC}"
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
echo -e "${GREEN}✓ Virtual environment activated${NC}"

echo -e "${YELLOW}[4/5] Installing dependencies...${NC}"
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo -e "${YELLOW}[5/5] Creating .env file...${NC}"
if [ ! -f ".env" ]; then
    cat > .env << 'ENV_EOF'
# LLM Trading Platform - Local Development Environment
# These are test keys for local development (no cost)

# API Keys (use test values for local testing)
GROQ_API_KEY=test_groq_key_replace_for_live
ANTHROPIC_API_KEY=test_anthropic_key_replace_for_live
ALPACA_API_KEY=test_alpaca_key_replace_for_paper_trading
ALPACA_SECRET_KEY=test_alpaca_secret_replace_for_paper_trading

# Paper Trading (ALWAYS true for safety!)
PAPER_TRADING=true

# Database (local SQLite for testing)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading_db
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_pass

# Redis (optional for local testing)
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka (optional for local testing)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Risk Management
MAX_POSITION_SIZE=10000.0
RISK_PER_TRADE=0.02
MAX_OPEN_POSITIONS=10

# LLM Cost Controls
MAX_TOKENS_PER_ANALYSIS=2000
CACHE_ANALYSIS_HOURS=24
USE_GROQ_FOR_SPEED=true
USE_ANTHROPIC_FOR_COMPLEX=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

# MLflow
MLFLOW_TRACKING_URI=file:///tmp/mlflow
ENV_EOF
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup Complete!                                             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run tests: ./scripts/test_local.sh"
echo "  3. Start development: uvicorn src.main:app --reload"
echo ""
