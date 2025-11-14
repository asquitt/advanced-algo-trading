#!/bin/bash
################################################################################
# Week 2 LLM Integration - Comprehensive Test Suite
################################################################################
#
# This script runs comprehensive tests for Week 2:
# 1. Test LLM integration (Groq and Claude)
# 2. Test caching functionality
# 3. Test financial agent
# 4. Cost analysis
#
# Usage: ./test_week2.sh
#
# Requirements:
# - GROQ_API_KEY and ANTHROPIC_API_KEY environment variables
# - Python 3.8+
# - Dependencies: anthropic, groq, redis (optional)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_COST=0.0

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEEK2_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           Week 2: LLM Integration - Test Suite                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

################################################################################
# Test 1: Environment Setup
################################################################################

test_environment() {
    print_header "Test 1: Environment Setup"

    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python version: $PYTHON_VERSION"
    else
        print_error "Python 3 not found"
        return 1
    fi

    # Check API keys
    if [ -z "$GROQ_API_KEY" ]; then
        print_warning "GROQ_API_KEY not set (tests will be limited)"
    else
        print_success "GROQ_API_KEY is set"
    fi

    if [ -z "$ANTHROPIC_API_KEY" ]; then
        print_warning "ANTHROPIC_API_KEY not set (tests will be limited)"
    else
        print_success "ANTHROPIC_API_KEY is set"
    fi

    # Check required packages
    print_info "Checking required Python packages..."

    local packages=("anthropic" "groq")
    for package in "${packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            print_success "Package '$package' installed"
        else
            print_warning "Package '$package' not installed (run: pip install $package)"
        fi
    done

    # Check optional packages
    if python3 -c "import redis" 2>/dev/null; then
        print_success "Package 'redis' installed (caching enabled)"
    else
        print_info "Package 'redis' not installed (caching tests will be skipped)"
    fi
}

################################################################################
# Test 2: LLM Client Integration
################################################################################

test_llm_clients() {
    print_header "Test 2: LLM Client Integration"

    # Create a simple test script
    cat > /tmp/test_llm_clients.py << 'EOF'
import os
import sys
import asyncio
sys.path.insert(0, os.path.dirname(__file__))

async def test_clients():
    results = {"groq": None, "claude": None}

    # Test Groq
    if os.getenv("GROQ_API_KEY"):
        try:
            from starter_code.llm_client import GroqClient
            client = GroqClient(debug=False)
            response = await client.generate("What is 2+2? Answer in one word.")
            results["groq"] = {
                "success": True,
                "latency_ms": response.latency_ms,
                "tokens": response.tokens_used
            }
            print(f"✅ Groq: {response.latency_ms:.2f}ms, {response.tokens_used} tokens")
        except Exception as e:
            results["groq"] = {"success": False, "error": str(e)}
            print(f"❌ Groq failed: {e}")
    else:
        print("⚠️  Groq: API key not set, skipping")

    # Test Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from starter_code.llm_client import ClaudeClient
            client = ClaudeClient(debug=False)
            response = await client.generate("What is 2+2? Answer in one word.")
            results["claude"] = {
                "success": True,
                "latency_ms": response.latency_ms,
                "tokens": response.tokens_used
            }
            print(f"✅ Claude: {response.latency_ms:.2f}ms, {response.tokens_used} tokens")
        except Exception as e:
            results["claude"] = {"success": False, "error": str(e)}
            print(f"❌ Claude failed: {e}")
    else:
        print("⚠️  Claude: API key not set, skipping")

    return results

if __name__ == "__main__":
    results = asyncio.run(test_clients())

    # Calculate success
    success_count = sum(1 for r in results.values() if r and r.get("success"))
    total_count = sum(1 for r in results.values() if r is not None)

    if total_count == 0:
        print("\n⚠️  No API keys configured - cannot test LLM clients")
        sys.exit(2)
    elif success_count == total_count:
        print(f"\n✅ All {total_count} LLM client(s) working!")
        sys.exit(0)
    else:
        print(f"\n❌ {total_count - success_count}/{total_count} client(s) failed")
        sys.exit(1)
EOF

    # Run the test
    cd "$WEEK2_DIR"
    if python3 /tmp/test_llm_clients.py; then
        print_success "LLM clients are working"
    else
        TEST_EXIT=$?
        if [ $TEST_EXIT -eq 2 ]; then
            print_warning "LLM client tests skipped (no API keys)"
        else
            print_error "LLM client tests failed"
        fi
    fi

    rm -f /tmp/test_llm_clients.py
}

################################################################################
# Test 3: Caching Functionality
################################################################################

test_caching() {
    print_header "Test 3: Caching Functionality"

    # Check if Redis is available
    if ! python3 -c "import redis" 2>/dev/null; then
        print_info "Redis package not installed - skipping cache tests"
        print_info "Install with: pip install redis"
        return 0
    fi

    # Create cache test script
    cat > /tmp/test_cache.py << 'EOF'
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

try:
    from starter_code.cache import RedisCache

    # Try to connect to Redis
    cache = RedisCache()

    # Test basic operations
    test_key = "test:week2"
    test_value = {"data": "test"}

    # Set value
    cache.set(test_key, test_value, ttl=60)

    # Get value
    retrieved = cache.get(test_key)

    if retrieved == test_value:
        print("✅ Cache set/get working")

        # Test deletion
        cache.delete(test_key)
        if cache.get(test_key) is None:
            print("✅ Cache deletion working")
        else:
            print("❌ Cache deletion failed")
            sys.exit(1)
    else:
        print("❌ Cache retrieval failed")
        sys.exit(1)

    print("✅ All cache operations working")
    sys.exit(0)

except ImportError as e:
    print(f"⚠️  Cache module not fully implemented: {e}")
    sys.exit(2)
except Exception as e:
    print(f"❌ Cache test failed: {e}")
    print("   Make sure Redis is running: redis-server")
    sys.exit(1)
EOF

    cd "$WEEK2_DIR"
    if python3 /tmp/test_cache.py; then
        print_success "Caching functionality working"
    else
        TEST_EXIT=$?
        if [ $TEST_EXIT -eq 2 ]; then
            print_warning "Cache tests skipped (module not implemented)"
        else
            print_error "Cache tests failed (is Redis running?)"
        fi
    fi

    rm -f /tmp/test_cache.py
}

################################################################################
# Test 4: Financial Agent
################################################################################

test_financial_agent() {
    print_header "Test 4: Financial Agent"

    cat > /tmp/test_agent.py << 'EOF'
import os
import sys
import asyncio
sys.path.insert(0, os.path.dirname(__file__))

async def test_agent():
    if not os.getenv("GROQ_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  No API keys set - skipping agent tests")
        return 2

    try:
        from starter_code.financial_agent import FinancialAgent, StockMetrics

        # Create agent
        agent = FinancialAgent(debug=False, enable_cache=False)

        # Test data
        test_metrics = StockMetrics(
            symbol="TEST",
            price=100.0,
            pe_ratio=20.0,
            revenue_growth=10.0,
            profit_margin=15.0,
            debt_equity=1.0
        )

        # Analyze stock
        print("Testing stock analysis...")
        result = await agent.analyze_stock("TEST", test_metrics)

        # Validate result
        assert result.symbol == "TEST", "Symbol mismatch"
        assert 0 <= result.confidence <= 1.0, "Confidence out of range"
        assert 0 <= result.score <= 100, "Score out of range"
        assert result.reasoning, "Reasoning is empty"

        print(f"✅ Analysis completed:")
        print(f"   Recommendation: {result.recommendation.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Score: {result.score}/100")

        # Test metrics
        metrics = agent.get_metrics()
        print(f"✅ Agent metrics: {metrics['total_requests']} requests")

        return 0

    except ImportError as e:
        print(f"⚠️  Agent module not fully implemented: {e}")
        return 2
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(test_agent())
    sys.exit(exit_code)
EOF

    cd "$WEEK2_DIR"
    if python3 /tmp/test_agent.py; then
        print_success "Financial agent working"
    else
        TEST_EXIT=$?
        if [ $TEST_EXIT -eq 2 ]; then
            print_warning "Agent tests skipped (not fully implemented or no API keys)"
        else
            print_error "Agent tests failed"
        fi
    fi

    rm -f /tmp/test_agent.py
}

################################################################################
# Test 5: Cost Analysis
################################################################################

test_cost_analysis() {
    print_header "Test 5: Cost Analysis"

    print_info "Analyzing API costs..."

    # Estimated costs (based on README)
    GROQ_COST_PER_1K=0.0001  # $0.0001 per 1K tokens
    CLAUDE_INPUT_COST_PER_1M=3.0  # $3 per 1M input tokens
    CLAUDE_OUTPUT_COST_PER_1M=15.0  # $15 per 1M output tokens

    # Typical usage per week
    SIGNALS_PER_DAY=100
    DAYS_PER_WEEK=7
    AVG_TOKENS_PER_CALL=500
    CACHE_HIT_RATE=0.9  # 90% cache hit rate

    # Calculate weekly costs
    TOTAL_REQUESTS=$((SIGNALS_PER_DAY * DAYS_PER_WEEK))
    ACTUAL_API_CALLS=$(echo "$TOTAL_REQUESTS * (1 - $CACHE_HIT_RATE)" | bc)

    # Groq cost
    GROQ_TOKENS=$(echo "$ACTUAL_API_CALLS * $AVG_TOKENS_PER_CALL" | bc)
    GROQ_COST=$(echo "$GROQ_TOKENS / 1000 * $GROQ_COST_PER_1K" | bc -l)

    # Claude cost (approximate)
    CLAUDE_TOKENS=$(echo "$ACTUAL_API_CALLS * $AVG_TOKENS_PER_CALL" | bc)
    CLAUDE_COST=$(echo "$CLAUDE_TOKENS / 1000000 * ($CLAUDE_INPUT_COST_PER_1M + $CLAUDE_OUTPUT_COST_PER_1M) / 2" | bc -l)

    echo ""
    echo "Cost Estimates (per week):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "Total requests:          %d\n" $TOTAL_REQUESTS
    printf "Cache hit rate:          %.0f%%\n" $(echo "$CACHE_HIT_RATE * 100" | bc)
    printf "Actual API calls:        %.0f\n" $ACTUAL_API_CALLS
    echo ""
    printf "Groq cost/week:          \$%.4f\n" $GROQ_COST
    printf "Claude cost/week:        \$%.4f\n" $CLAUDE_COST
    echo ""

    # Check if under budget
    BUDGET=1.0

    if (( $(echo "$GROQ_COST < $BUDGET" | bc -l) )); then
        print_success "Groq cost (\$$GROQ_COST) is under \$$BUDGET/week budget"
    else
        print_error "Groq cost (\$$GROQ_COST) exceeds \$$BUDGET/week budget"
    fi

    if (( $(echo "$CLAUDE_COST < $BUDGET" | bc -l) )); then
        print_success "Claude cost (\$$CLAUDE_COST) is under \$$BUDGET/week budget"
    else
        print_warning "Claude cost (\$$CLAUDE_COST) exceeds \$$BUDGET/week budget (use Groq for simple tasks)"
    fi

    echo ""
    echo "💡 Tips to reduce costs:"
    echo "   • Use Groq for simple tasks (5-10x cheaper)"
    echo "   • Enable caching (90% cost reduction)"
    echo "   • Use Claude only for complex analysis"
    echo "   • Implement smart model selection based on task complexity"
}

################################################################################
# Test 6: Run Exercise Tests
################################################################################

test_exercises() {
    print_header "Test 6: Exercise Validation"

    cd "$WEEK2_DIR/exercises"

    # Check which exercises exist
    exercises=(
        "exercise_1_llm_basics.py"
        "exercise_2_prompts.py"
        "exercise_3_parsing.py"
        "exercise_4_agent.py"
        "exercise_5_optimization.py"
    )

    for exercise in "${exercises[@]}"; do
        if [ -f "$exercise" ]; then
            print_info "Found: $exercise"

            # Check if exercise has test function
            if grep -q "def test_" "$exercise"; then
                print_info "  → Contains test functions"
            else
                print_info "  → No test functions found (manual testing required)"
            fi
        else
            print_warning "Missing: $exercise"
        fi
    done

    cd "$WEEK2_DIR"
}

################################################################################
# Main Execution
################################################################################

main() {
    echo "Starting tests at: $(date)"
    echo ""

    # Run all tests
    test_environment
    test_llm_clients
    test_caching
    test_financial_agent
    test_cost_analysis
    test_exercises

    # Print summary
    print_header "Test Summary"

    TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))

    echo ""
    echo "Results:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "Tests passed:  ${GREEN}%d${NC}\n" $TESTS_PASSED
    printf "Tests failed:  ${RED}%d${NC}\n" $TESTS_FAILED
    printf "Total tests:   %d\n" $TOTAL_TESTS
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  🎉 All tests passed! Week 2 is looking good!                 ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Complete any remaining TODOs in starter code"
        echo "  2. Finish all 5 exercises"
        echo "  3. Run: ./scripts/validate.py"
        echo "  4. Move on to Week 3!"
        exit 0
    else
        echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ⚠️  Some tests failed - review the errors above              ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Common issues:"
        echo "  • Missing API keys (set GROQ_API_KEY and ANTHROPIC_API_KEY)"
        echo "  • TODOs not completed in starter code"
        echo "  • Missing dependencies (run: pip install -r requirements.txt)"
        echo "  • Redis not running (for cache tests)"
        exit 1
    fi
}

# Run main function
main
