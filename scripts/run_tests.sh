#!/bin/bash

# Test runner script for LLM Trading Platform
# Runs different test suites with appropriate configurations

set -e  # Exit on error

echo "🧪 LLM Trading Platform - Test Runner"
echo "=" | head -c 60 | tr ' ' '='
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load test environment variables
if [ -f .env.test ]; then
    echo "Loading test environment variables from .env.test..."
    export $(cat .env.test | grep -v '^#' | xargs)
else
    # Set minimal required environment variables for testing
    export GROQ_API_KEY=${GROQ_API_KEY:-test_groq_key}
    export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-test_anthropic_key}
    export ALPACA_API_KEY=${ALPACA_API_KEY:-test_alpaca_key}
    export ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY:-test_alpaca_secret}
fi

# Function to run tests
run_tests() {
    local test_type=$1
    shift
    local test_args="$@"

    echo -e "${YELLOW}Running $test_type tests...${NC}"

    if eval "pytest $test_args"; then
        echo -e "${GREEN}✅ $test_type tests passed!${NC}"
        return 0
    else
        echo -e "${RED}❌ $test_type tests failed!${NC}"
        return 1
    fi
}

# Parse command line arguments
case "${1:-all}" in
    unit)
        echo "Running unit tests only..."
        run_tests "Unit" "tests/ -m unit -v"
        ;;

    integration)
        echo "Running integration tests..."
        echo "⚠️  Note: Integration tests may require services (Redis, Kafka, etc.)"
        run_tests "Integration" "tests/test_integration.py -v"
        ;;

    performance)
        echo "Running performance tests..."
        run_tests "Performance" "tests/test_performance.py -v"
        ;;

    hft)
        echo "Running HFT technique tests..."
        run_tests "HFT" "tests/test_hft_techniques.py -v"
        ;;

    regression)
        echo "Running regression tests..."
        run_tests "Regression" "tests/test_regression.py -v"
        ;;

    e2e)
        echo "Running end-to-end tests..."
        run_tests "E2E" "tests/test_e2e.py -v"
        ;;

    fast)
        echo "Running fast tests only (excluding slow tests)..."
        run_tests "Fast" "tests/ -v -m \"not slow\""
        ;;

    coverage)
        echo "Running tests with coverage report..."
        pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
        echo ""
        echo "📊 Coverage report generated in htmlcov/index.html"
        echo "   Open with: open htmlcov/index.html (macOS) or xdg-open htmlcov/index.html (Linux)"
        ;;

    all)
        echo "Running all tests..."

        # Run different test suites
        run_tests "Utils" "tests/test_utils.py -v" || true
        echo ""

        run_tests "Data Models" "tests/test_data_models.py -v" || true
        echo ""

        run_tests "HFT Techniques" "tests/test_hft_techniques.py -v" || true
        echo ""

        run_tests "LLM Agents" "tests/test_llm_agents.py -v" || true
        echo ""

        run_tests "Trading Engine" "tests/test_trading_engine.py -v" || true
        echo ""

        run_tests "Integration" "tests/test_integration.py -v" || true
        echo ""

        run_tests "Regression" "tests/test_regression.py -v" || true
        echo ""

        echo "Running end-to-end tests..."
        run_tests "E2E" "tests/test_e2e.py -v" || true
        echo ""

        echo "Running performance tests (may take a while)..."
        run_tests "Performance" "tests/test_performance.py -v" || true
        echo ""

        # Generate coverage report
        echo "Generating coverage report..."
        pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --quiet

        echo ""
        echo "=" | head -c 60 | tr ' ' '='
        echo -e "${GREEN}✅ All test suites complete!${NC}"
        echo "=" | head -c 60 | tr ' ' '='
        echo ""
        echo "📊 Coverage report: htmlcov/index.html"
        ;;

    watch)
        echo "Running tests in watch mode..."
        echo "⚠️  Install pytest-watch first: pip install pytest-watch"
        ptw tests/ -- -v
        ;;

    help|--help|-h)
        echo "Usage: $0 [test-suite]"
        echo ""
        echo "Available test suites:"
        echo "  all         - Run all tests (default)"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests"
        echo "  performance - Run performance tests"
        echo "  hft         - Run HFT technique tests"
        echo "  regression  - Run regression tests (prevent old bugs)"
        echo "  e2e         - Run end-to-end tests (full workflows)"
        echo "  fast        - Run fast tests only (exclude slow)"
        echo "  coverage    - Run tests with coverage report"
        echo "  watch       - Run tests in watch mode"
        echo "  help        - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Run all tests"
        echo "  $0 unit         # Run unit tests"
        echo "  $0 regression   # Run regression tests"
        echo "  $0 e2e          # Run end-to-end tests"
        echo "  $0 coverage     # Generate coverage report"
        ;;

    *)
        echo "❌ Unknown test suite: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

exit 0
