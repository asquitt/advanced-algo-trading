#!/bin/bash

################################################################################
# Week 1 Foundations - API Launcher Script
#
# This script helps you run your FastAPI trading application by:
# 1. Checking for virtual environment
# 2. Loading environment variables
# 3. Starting the uvicorn server
# 4. Providing helpful access information
#
# Usage: ./run_api.sh [options]
#
# Options:
#   --port PORT     Specify port (default: 8000)
#   --host HOST     Specify host (default: 0.0.0.0)
#   --reload        Enable auto-reload on code changes
#   --help          Show this help message
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Default configuration
PORT=8000
HOST="0.0.0.0"
RELOAD_FLAG="--reload"
APP_MODULE="starter-code.main:app"

# Directory paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WEEK1_DIR="$(dirname "$SCRIPT_DIR")"
STARTER_DIR="$WEEK1_DIR/starter-code"
ENV_FILE="$WEEK1_DIR/.env"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_failure() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_box() {
    local message="$1"
    local color="$2"
    echo -e "${color}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${color}║ $message${NC}"
    echo -e "${color}╚════════════════════════════════════════════════════════════╝${NC}"
}

show_help() {
    echo "Week 1 API Launcher - FastAPI Trading Application"
    echo ""
    echo "Usage: ./run_api.sh [options]"
    echo ""
    echo "Options:"
    echo "  --port PORT     Specify port (default: 8000)"
    echo "  --host HOST     Specify host (default: 0.0.0.0)"
    echo "  --reload        Enable auto-reload (default: on)"
    echo "  --no-reload     Disable auto-reload"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_api.sh                    # Run with defaults"
    echo "  ./run_api.sh --port 3000        # Run on port 3000"
    echo "  ./run_api.sh --no-reload        # Run without auto-reload"
    echo ""
    exit 0
}

################################################################################
# Parse Command Line Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --reload)
            RELOAD_FLAG="--reload"
            shift
            ;;
        --no-reload)
            RELOAD_FLAG=""
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Pre-flight Checks
################################################################################

preflight_checks() {
    print_header "Pre-flight Checks"

    # Check if Python is installed
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version)
        print_success "Python found: $python_version"
    else
        print_failure "Python 3 not found. Please install Python 3.11+"
        exit 1
    fi

    # Check if main.py exists
    if [ -f "$STARTER_DIR/main.py" ]; then
        print_success "Found main.py"
    else
        print_failure "main.py not found in $STARTER_DIR"
        exit 1
    fi

    # Check if FastAPI is installed
    if python3 -c "import fastapi" 2>/dev/null; then
        print_success "FastAPI installed"
    else
        print_failure "FastAPI not installed"
        print_info "Install with: pip install fastapi uvicorn"
        exit 1
    fi

    # Check if uvicorn is installed
    if python3 -c "import uvicorn" 2>/dev/null; then
        print_success "Uvicorn installed"
    else
        print_failure "Uvicorn not installed"
        print_info "Install with: pip install uvicorn"
        exit 1
    fi

    # Check for .env file
    if [ -f "$ENV_FILE" ]; then
        print_success "Found .env file"
        export $(grep -v '^#' "$ENV_FILE" | xargs) 2>/dev/null
        print_info "Environment variables loaded"
    else
        print_warning ".env file not found"
        print_info "API will run but Alpaca features may not work"
        print_info "Create .env with your Alpaca API keys (see config.py)"
    fi

    # Check if port is available
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $PORT is already in use"
        print_info "Use --port to specify a different port"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Port $PORT is available"
    fi
}

################################################################################
# Print API Information
################################################################################

print_api_info() {
    clear
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║           FastAPI Trading Application Launcher            ║"
    echo "║                  Week 1 - Foundations                      ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"

    print_header "Starting API Server"

    echo -e "${CYAN}Configuration:${NC}"
    echo "  Host:        $HOST"
    echo "  Port:        $PORT"
    echo "  Auto-reload: $([ -n "$RELOAD_FLAG" ] && echo "Enabled" || echo "Disabled")"
    echo "  Working Dir: $WEEK1_DIR"
    echo ""

    echo -e "${CYAN}Access Your API:${NC}"
    echo "  🌐 Swagger UI:     http://localhost:$PORT/docs"
    echo "  📚 ReDoc:          http://localhost:$PORT/redoc"
    echo "  🏠 Root:           http://localhost:$PORT/"
    echo "  ❤️  Health Check:  http://localhost:$PORT/health"
    echo ""

    echo -e "${CYAN}Available Endpoints:${NC}"
    echo "  GET  /                  - Welcome message"
    echo "  GET  /health            - Health check"
    echo "  GET  /signal/{symbol}   - Generate trading signal"
    echo "  POST /trade             - Execute paper trade"
    echo "  GET  /portfolio         - View portfolio status"
    echo ""

    echo -e "${CYAN}Example Usage:${NC}"
    echo "  # Get signal for AAPL"
    echo "  curl http://localhost:$PORT/signal/AAPL"
    echo ""
    echo "  # Execute trade"
    echo "  curl -X POST http://localhost:$PORT/trade \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"symbol\":\"AAPL\",\"side\":\"buy\",\"quantity\":10}'"
    echo ""
    echo "  # Check portfolio"
    echo "  curl http://localhost:$PORT/portfolio"
    echo ""

    echo -e "${YELLOW}Tip: Open http://localhost:$PORT/docs in your browser${NC}"
    echo -e "${YELLOW}     for interactive API documentation!${NC}"
    echo ""

    print_info "Press Ctrl+C to stop the server"
    echo ""
}

################################################################################
# Start Server
################################################################################

start_server() {
    print_header "Starting Server"

    # Change to week1 directory so imports work correctly
    cd "$WEEK1_DIR"

    # Add current directory to PYTHONPATH
    export PYTHONPATH="${PYTHONPATH}:${WEEK1_DIR}"

    print_info "Launching uvicorn..."
    echo ""

    # Start uvicorn
    if [ -n "$RELOAD_FLAG" ]; then
        python3 -m uvicorn "$APP_MODULE" \
            --host "$HOST" \
            --port "$PORT" \
            $RELOAD_FLAG
    else
        python3 -m uvicorn "$APP_MODULE" \
            --host "$HOST" \
            --port "$PORT"
    fi
}

################################################################################
# Cleanup on Exit
################################################################################

cleanup() {
    echo ""
    print_header "Shutting Down"
    print_info "Server stopped"
    print_info "Thank you for using Week 1 API!"
    echo ""
    exit 0
}

trap cleanup SIGINT SIGTERM

################################################################################
# Main Execution
################################################################################

main() {
    # Run pre-flight checks
    preflight_checks

    # Print API information
    print_api_info

    # Wait a moment before starting
    sleep 1

    # Start the server
    start_server
}

# Run main function
main "$@"
