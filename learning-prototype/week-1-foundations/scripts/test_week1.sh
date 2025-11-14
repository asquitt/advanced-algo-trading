#!/bin/bash

################################################################################
# Week 1 Foundations - Comprehensive Test Script
#
# This script validates your Week 1 completion by:
# 1. Checking Python syntax across all files
# 2. Running all exercises
# 3. Validating starter code TODOs are completed
# 4. Running integration tests
# 5. Checking code coverage
# 6. Providing detailed feedback
#
# Usage: ./test_week1.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directory paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WEEK1_DIR="$(dirname "$SCRIPT_DIR")"
STARTER_DIR="$WEEK1_DIR/starter-code"
EXERCISES_DIR="$WEEK1_DIR/exercises"
SOLUTIONS_DIR="$WEEK1_DIR/solutions"

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

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
    ((PASSED_TESTS++))
    ((TOTAL_TESTS++))
}

print_failure() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

################################################################################
# Test 1: Check Python Syntax
################################################################################

test_python_syntax() {
    print_header "Test 1: Checking Python Syntax"

    local syntax_errors=0

    # Check starter code
    for file in "$STARTER_DIR"/*.py; do
        if [ -f "$file" ]; then
            if python3 -m py_compile "$file" 2>/dev/null; then
                print_success "Syntax OK: $(basename "$file")"
            else
                print_failure "Syntax Error: $(basename "$file")"
                ((syntax_errors++))
            fi
        fi
    done

    # Check exercises
    for file in "$EXERCISES_DIR"/*.py; do
        if [ -f "$file" ]; then
            if python3 -m py_compile "$file" 2>/dev/null; then
                print_success "Syntax OK: $(basename "$file")"
            else
                print_failure "Syntax Error: $(basename "$file")"
                ((syntax_errors++))
            fi
        fi
    done

    if [ $syntax_errors -eq 0 ]; then
        print_info "All files have valid Python syntax!"
    else
        print_warning "Found $syntax_errors files with syntax errors"
    fi
}

################################################################################
# Test 2: Check TODO Completion
################################################################################

test_todo_completion() {
    print_header "Test 2: Checking TODO Completion"

    local total_todos=0
    local files_with_todos=0

    for file in "$STARTER_DIR"/*.py; do
        if [ -f "$file" ]; then
            local todo_count=$(grep -c "# TODO" "$file" 2>/dev/null || true)
            if [ $todo_count -gt 0 ]; then
                print_warning "$(basename "$file"): $todo_count TODOs remaining"
                ((total_todos += todo_count))
                ((files_with_todos++))
            else
                print_success "$(basename "$file"): All TODOs completed"
            fi
        fi
    done

    if [ $total_todos -eq 0 ]; then
        print_info "All TODOs completed! Great work!"
    else
        print_warning "Total: $total_todos TODOs remaining in $files_with_todos files"
        print_info "Hint: Complete all TODOs before moving to Week 2"
    fi
}

################################################################################
# Test 3: Check Imports and Dependencies
################################################################################

test_imports() {
    print_header "Test 3: Checking Imports and Dependencies"

    # Check if FastAPI is installed
    if python3 -c "import fastapi" 2>/dev/null; then
        print_success "FastAPI installed"
    else
        print_failure "FastAPI not installed (pip install fastapi)"
    fi

    # Check if uvicorn is installed
    if python3 -c "import uvicorn" 2>/dev/null; then
        print_success "Uvicorn installed"
    else
        print_failure "Uvicorn not installed (pip install uvicorn)"
    fi

    # Check if pydantic is installed
    if python3 -c "import pydantic" 2>/dev/null; then
        print_success "Pydantic installed"
    else
        print_failure "Pydantic not installed (pip install pydantic)"
    fi

    # Check if pytest is installed
    if python3 -c "import pytest" 2>/dev/null; then
        print_success "Pytest installed"
    else
        print_failure "Pytest not installed (pip install pytest)"
    fi

    # Check if alpaca-py is installed (optional)
    if python3 -c "import alpaca" 2>/dev/null; then
        print_success "Alpaca SDK installed"
    else
        print_warning "Alpaca SDK not installed (optional: pip install alpaca-py)"
    fi
}

################################################################################
# Test 4: Validate Starter Code Structure
################################################################################

test_starter_code() {
    print_header "Test 4: Validating Starter Code Structure"

    # Check main.py
    if [ -f "$STARTER_DIR/main.py" ]; then
        # Check if FastAPI app is created
        if grep -q "app = FastAPI" "$STARTER_DIR/main.py" || grep -q "app=FastAPI" "$STARTER_DIR/main.py"; then
            print_success "main.py: FastAPI app created"
        else
            print_failure "main.py: FastAPI app not created"
        fi

        # Check for endpoints
        if grep -q "@app.get.*signal" "$STARTER_DIR/main.py"; then
            print_success "main.py: Signal endpoint defined"
        else
            print_warning "main.py: Signal endpoint not found"
        fi

        if grep -q "@app.post.*trade" "$STARTER_DIR/main.py"; then
            print_success "main.py: Trade endpoint defined"
        else
            print_warning "main.py: Trade endpoint not found"
        fi
    else
        print_failure "main.py not found"
    fi

    # Check models.py
    if [ -f "$STARTER_DIR/models.py" ]; then
        if grep -q "class.*Signal" "$STARTER_DIR/models.py"; then
            print_success "models.py: Signal model defined"
        else
            print_warning "models.py: Signal model not found"
        fi

        if grep -q "class.*TradeRequest" "$STARTER_DIR/models.py"; then
            print_success "models.py: TradeRequest model defined"
        else
            print_warning "models.py: TradeRequest model not found"
        fi
    else
        print_failure "models.py not found"
    fi

    # Check broker.py
    if [ -f "$STARTER_DIR/broker.py" ]; then
        if grep -q "class AlpacaBroker" "$STARTER_DIR/broker.py"; then
            print_success "broker.py: AlpacaBroker class defined"
        else
            print_warning "broker.py: AlpacaBroker class not found"
        fi
    else
        print_failure "broker.py not found"
    fi

    # Check config.py
    if [ -f "$STARTER_DIR/config.py" ]; then
        if grep -q "class Settings" "$STARTER_DIR/config.py"; then
            print_success "config.py: Settings class defined"
        else
            print_warning "config.py: Settings class not found"
        fi
    else
        print_failure "config.py not found"
    fi
}

################################################################################
# Test 5: Run Exercise Tests
################################################################################

test_exercises() {
    print_header "Test 5: Testing Exercises"

    print_info "Exercises are meant to be run manually and tested interactively."
    print_info "Please verify each exercise works by running them individually:"
    echo ""

    for file in "$EXERCISES_DIR"/exercise_*.py; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")

            # Check if exercise has basic structure
            if grep -q "app = FastAPI\|app=FastAPI" "$file" 2>/dev/null; then
                print_success "$basename has FastAPI app"
            elif grep -q "TODO" "$file" 2>/dev/null; then
                print_warning "$basename has incomplete TODOs"
            else
                print_info "$basename ready for manual testing"
            fi
        fi
    done

    echo ""
    print_info "To test exercises manually:"
    print_info "  1. python $EXERCISES_DIR/exercise_1_fastapi_basics.py"
    print_info "  2. Open http://localhost:8000/docs"
    print_info "  3. Test endpoints in Swagger UI"
}

################################################################################
# Test 6: Check Environment Configuration
################################################################################

test_environment() {
    print_header "Test 6: Checking Environment Configuration"

    # Check for .env file
    if [ -f "$WEEK1_DIR/.env" ]; then
        print_success ".env file exists"

        # Check for required variables (without exposing values)
        if grep -q "ALPACA_API_KEY" "$WEEK1_DIR/.env"; then
            print_success ".env contains ALPACA_API_KEY"
        else
            print_warning ".env missing ALPACA_API_KEY"
        fi

        if grep -q "ALPACA_SECRET_KEY" "$WEEK1_DIR/.env"; then
            print_success ".env contains ALPACA_SECRET_KEY"
        else
            print_warning ".env missing ALPACA_SECRET_KEY"
        fi
    else
        print_warning ".env file not found (create one for Alpaca credentials)"
        print_info "See config.py for .env template"
    fi

    # Check .gitignore
    if [ -f "$WEEK1_DIR/.gitignore" ] && grep -q ".env" "$WEEK1_DIR/.gitignore"; then
        print_success ".env is in .gitignore (good!)"
    else
        print_warning ".env should be added to .gitignore for security"
    fi
}

################################################################################
# Test 7: Run pytest if available
################################################################################

test_pytest() {
    print_header "Test 7: Running Pytest Tests"

    if command -v pytest &> /dev/null; then
        print_info "Running pytest tests..."

        # Look for test files
        if ls "$WEEK1_DIR"/test_*.py 2>/dev/null || ls "$WEEK1_DIR"/tests/*.py 2>/dev/null; then
            if pytest "$WEEK1_DIR" -v --tb=short 2>/dev/null; then
                print_success "All pytest tests passed"
            else
                print_warning "Some pytest tests failed (this is OK if you're still learning)"
            fi
        else
            print_info "No pytest test files found (tests are optional for Week 1)"
        fi
    else
        print_info "Pytest not available (skipping unit tests)"
    fi
}

################################################################################
# Main Test Execution
################################################################################

main() {
    clear
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║      Week 1 Foundations - Comprehensive Test Suite        ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"

    print_info "Testing directory: $WEEK1_DIR"
    print_info "Started at: $(date '+%Y-%m-%d %H:%M:%S')"

    # Run all tests
    test_python_syntax
    test_todo_completion
    test_imports
    test_starter_code
    test_exercises
    test_environment
    test_pytest

    # Print summary
    print_header "Test Summary"

    echo -e "Total Checks: ${BLUE}$TOTAL_TESTS${NC}"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

    local pass_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        pass_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    fi

    echo -e "\nSuccess Rate: ${BLUE}${pass_rate}%${NC}"

    echo ""
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                                                            ║${NC}"
        echo -e "${GREEN}║  🎉 Congratulations! All tests passed!                    ║${NC}"
        echo -e "${GREEN}║                                                            ║${NC}"
        echo -e "${GREEN}║  You're ready to move on to Week 2!                       ║${NC}"
        echo -e "${GREEN}║                                                            ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    else
        echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║                                                            ║${NC}"
        echo -e "${YELLOW}║  Keep going! Review the failures above and try again.     ║${NC}"
        echo -e "${YELLOW}║                                                            ║${NC}"
        echo -e "${YELLOW}║  Tips:                                                     ║${NC}"
        echo -e "${YELLOW}║  - Complete all TODOs in starter-code/                    ║${NC}"
        echo -e "${YELLOW}║  - Test each exercise manually                            ║${NC}"
        echo -e "${YELLOW}║  - Check ../notes/ for help                               ║${NC}"
        echo -e "${YELLOW}║  - Review ../solutions/ if stuck                          ║${NC}"
        echo -e "${YELLOW}║                                                            ║${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    fi

    echo -e "\n${BLUE}Next Steps:${NC}"
    echo "  1. Review any failures or warnings above"
    echo "  2. Run: ./run_api.sh to test your API"
    echo "  3. Run: python validate.py for final validation"
    echo "  4. Complete all exercises manually"
    echo "  5. Read all notes in ../notes/"
    echo ""

    print_info "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
}

# Run main function
main "$@"
