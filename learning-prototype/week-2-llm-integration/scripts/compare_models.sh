#!/bin/bash
################################################################################
# Week 2 LLM Integration - Model Comparison Script
################################################################################
#
# This script compares Groq vs Claude across multiple dimensions:
# 1. Latency comparison
# 2. Cost comparison
# 3. Quality comparison
# 4. Recommendations on when to use each
#
# Usage: ./compare_models.sh [--iterations N] [--task TASK_TYPE]
#
# Options:
#   --iterations N     Number of test iterations (default: 5)
#   --task TASK_TYPE   Task type: simple|moderate|complex (default: all)
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Default parameters
ITERATIONS=5
TASK_TYPE="all"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEEK2_DIR="$(dirname "$SCRIPT_DIR")"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --task)
            TASK_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--iterations N] [--task TASK_TYPE]"
            echo ""
            echo "Options:"
            echo "  --iterations N     Number of test iterations (default: 5)"
            echo "  --task TASK_TYPE   Task type: simple|moderate|complex (default: all)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           Groq vs Claude - Comprehensive Comparison                ║"
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

print_subheader() {
    echo ""
    echo -e "${CYAN}─── $1${NC}"
}

################################################################################
# Check Prerequisites
################################################################################

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check API keys
    if [ -z "$GROQ_API_KEY" ]; then
        echo -e "${RED}❌ GROQ_API_KEY not set${NC}"
        echo "   Set it with: export GROQ_API_KEY='your-key-here'"
        exit 1
    else
        echo -e "${GREEN}✅ GROQ_API_KEY is set${NC}"
    fi

    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${RED}❌ ANTHROPIC_API_KEY not set${NC}"
        echo "   Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        exit 1
    else
        echo -e "${GREEN}✅ ANTHROPIC_API_KEY is set${NC}"
    fi

    # Check Python packages
    if ! python3 -c "import groq" 2>/dev/null; then
        echo -e "${RED}❌ groq package not installed${NC}"
        echo "   Install with: pip install groq"
        exit 1
    fi

    if ! python3 -c "import anthropic" 2>/dev/null; then
        echo -e "${RED}❌ anthropic package not installed${NC}"
        echo "   Install with: pip install anthropic"
        exit 1
    fi

    echo -e "${GREEN}✅ All prerequisites met${NC}"
}

################################################################################
# Comparison Tests
################################################################################

run_comparison() {
    print_header "Running Model Comparison"

    cat > /tmp/compare_models.py << 'PYTHON_SCRIPT'
import os
import sys
import time
import asyncio
import statistics
from typing import List, Dict, Any
import json

# Add week2 directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Try to import the LLM clients
try:
    from starter_code.llm_client import GroqClient, ClaudeClient, LLMResponse
except ImportError:
    # Fallback to direct API calls if starter code not implemented
    print("⚠️  Using direct API calls (starter code not fully implemented)")
    import groq
    import anthropic

    class SimpleGroqClient:
        def __init__(self):
            self.client = groq.AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

        async def generate(self, prompt, **kwargs):
            start = time.time()
            response = await self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            latency = (time.time() - start) * 1000
            return type('obj', (object,), {
                'content': response.choices[0].message.content,
                'latency_ms': latency,
                'tokens_used': response.usage.total_tokens,
                'model': 'mixtral-8x7b-32768'
            })

    class SimpleClaudeClient:
        def __init__(self):
            self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        async def generate(self, prompt, **kwargs):
            start = time.time()
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            latency = (time.time() - start) * 1000
            return type('obj', (object,), {
                'content': response.content[0].text,
                'latency_ms': latency,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
                'model': 'claude-3-haiku-20240307'
            })

    GroqClient = SimpleGroqClient
    ClaudeClient = SimpleClaudeClient


class ModelComparison:
    def __init__(self, iterations: int = 5):
        self.iterations = iterations
        self.groq_client = GroqClient()
        self.claude_client = ClaudeClient()

    async def compare_task(self, task_name: str, prompt: str) -> Dict[str, Any]:
        """Compare both models on a single task."""
        print(f"\n📊 Testing: {task_name}")
        print(f"   Prompt: {prompt[:60]}...")

        # Run Groq tests
        groq_results = []
        for i in range(self.iterations):
            try:
                response = await self.groq_client.generate(prompt)
                groq_results.append({
                    'latency_ms': response.latency_ms,
                    'tokens': response.tokens_used,
                    'content': response.content,
                    'model': response.model
                })
                print(f"   Groq {i+1}/{self.iterations}: {response.latency_ms:.0f}ms", end='\r')
            except Exception as e:
                print(f"\n   ❌ Groq iteration {i+1} failed: {e}")

        # Run Claude tests
        claude_results = []
        for i in range(self.iterations):
            try:
                response = await self.claude_client.generate(prompt)
                claude_results.append({
                    'latency_ms': response.latency_ms,
                    'tokens': response.tokens_used,
                    'content': response.content,
                    'model': response.model
                })
                print(f"   Claude {i+1}/{self.iterations}: {response.latency_ms:.0f}ms", end='\r')
            except Exception as e:
                print(f"\n   ❌ Claude iteration {i+1} failed: {e}")

        print()  # New line after progress

        return {
            'task_name': task_name,
            'prompt': prompt,
            'groq': self._analyze_results(groq_results),
            'claude': self._analyze_results(claude_results)
        }

    def _analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze test results and calculate statistics."""
        if not results:
            return {
                'success': False,
                'error': 'No successful requests'
            }

        latencies = [r['latency_ms'] for r in results]
        tokens = [r['tokens'] for r in results]

        return {
            'success': True,
            'iterations': len(results),
            'latency': {
                'avg': statistics.mean(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            'tokens': {
                'avg': statistics.mean(tokens),
                'min': min(tokens),
                'max': max(tokens),
                'total': sum(tokens)
            },
            'sample_response': results[0]['content'][:200] + '...' if len(results[0]['content']) > 200 else results[0]['content'],
            'model': results[0]['model']
        }

    def calculate_costs(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate estimated costs based on token usage."""
        # Pricing (approximate)
        GROQ_COST_PER_1K = 0.0001  # $0.0001 per 1K tokens
        CLAUDE_INPUT_COST_PER_1M = 0.25  # $0.25 per 1M input tokens (Haiku)
        CLAUDE_OUTPUT_COST_PER_1M = 1.25  # $1.25 per 1M output tokens (Haiku)

        groq_tokens = results['groq']['tokens']['total'] if results['groq']['success'] else 0
        claude_tokens = results['claude']['tokens']['total'] if results['claude']['success'] else 0

        groq_cost = (groq_tokens / 1000) * GROQ_COST_PER_1K
        # Approximate Claude cost (assuming 50/50 input/output split)
        claude_cost = (claude_tokens / 1000000) * ((CLAUDE_INPUT_COST_PER_1M + CLAUDE_OUTPUT_COST_PER_1M) / 2)

        return {
            'groq': groq_cost,
            'claude': claude_cost
        }

    def print_comparison(self, results: Dict[str, Any]):
        """Print formatted comparison results."""
        task_name = results['task_name']

        print(f"\n{'='*70}")
        print(f"Results: {task_name}")
        print(f"{'='*70}")

        # Latency comparison
        if results['groq']['success'] and results['claude']['success']:
            groq_lat = results['groq']['latency']['avg']
            claude_lat = results['claude']['latency']['avg']

            print(f"\n📊 Latency:")
            print(f"   Groq:   {groq_lat:6.1f}ms (±{results['groq']['latency']['stdev']:.1f}ms)")
            print(f"   Claude: {claude_lat:6.1f}ms (±{results['claude']['latency']['stdev']:.1f}ms)")

            speedup = claude_lat / groq_lat if groq_lat > 0 else 0
            if groq_lat < claude_lat:
                print(f"   → Groq is {speedup:.1f}x faster ⚡")
            else:
                print(f"   → Claude is {1/speedup:.1f}x faster ⚡")

            # Token comparison
            groq_tokens = results['groq']['tokens']['avg']
            claude_tokens = results['claude']['tokens']['avg']

            print(f"\n🎫 Tokens:")
            print(f"   Groq:   {groq_tokens:6.0f} tokens/request")
            print(f"   Claude: {claude_tokens:6.0f} tokens/request")

            # Cost comparison
            costs = self.calculate_costs(results)
            print(f"\n💰 Cost (for {self.iterations} requests):")
            print(f"   Groq:   ${costs['groq']:.6f}")
            print(f"   Claude: ${costs['claude']:.6f}")

            if costs['groq'] < costs['claude']:
                savings = ((costs['claude'] - costs['groq']) / costs['claude']) * 100
                print(f"   → Groq is {savings:.1f}% cheaper 💵")
            else:
                savings = ((costs['groq'] - costs['claude']) / costs['groq']) * 100
                print(f"   → Claude is {savings:.1f}% cheaper 💵")

            # Quality comparison (subjective - show sample responses)
            print(f"\n📝 Sample Responses:")
            print(f"\n   Groq ({results['groq']['model']}):")
            print(f"   {results['groq']['sample_response']}")
            print(f"\n   Claude ({results['claude']['model']}):")
            print(f"   {results['claude']['sample_response']}")

        else:
            if not results['groq']['success']:
                print(f"\n❌ Groq tests failed")
            if not results['claude']['success']:
                print(f"\n❌ Claude tests failed")


async def main():
    import sys
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    task_type = sys.argv[2] if len(sys.argv) > 2 else 'all'

    comparison = ModelComparison(iterations=iterations)

    # Define test tasks
    tasks = {
        'simple': [
            ("Simple Math", "What is 15 * 23? Answer with just the number."),
            ("Classification", "Is the sentiment of this text positive or negative? 'I love this product!' Answer with one word."),
        ],
        'moderate': [
            ("Stock Analysis", "Is AAPL stock a good buy right now? Provide a brief 2-sentence analysis."),
            ("Data Extraction", "Extract the key numbers from this text: 'Revenue was $50M, up 25% YoY, with margins of 15%'. Return as JSON."),
        ],
        'complex': [
            ("Complex Reasoning", "Analyze the trade-offs between momentum and value investing strategies. Provide 3 key differences."),
            ("Multi-step Problem", "If a portfolio has 60% stocks (12% annual return) and 40% bonds (4% return), what's the expected annual return? Show your calculation."),
        ]
    }

    # Select tasks based on task_type
    selected_tasks = []
    if task_type == 'all':
        for task_list in tasks.values():
            selected_tasks.extend(task_list)
    elif task_type in tasks:
        selected_tasks = tasks[task_type]
    else:
        print(f"❌ Invalid task type: {task_type}")
        print(f"   Valid options: simple, moderate, complex, all")
        sys.exit(1)

    # Run comparisons
    all_results = []
    for task_name, prompt in selected_tasks:
        result = await comparison.compare_task(task_name, prompt)
        comparison.print_comparison(result)
        all_results.append(result)

    # Print overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    # Calculate aggregate statistics
    groq_latencies = []
    claude_latencies = []
    groq_total_cost = 0
    claude_total_cost = 0

    for result in all_results:
        if result['groq']['success']:
            groq_latencies.append(result['groq']['latency']['avg'])
        if result['claude']['success']:
            claude_latencies.append(result['claude']['latency']['avg'])

        costs = comparison.calculate_costs(result)
        groq_total_cost += costs['groq']
        claude_total_cost += costs['claude']

    if groq_latencies and claude_latencies:
        print(f"\nAverage Latency:")
        print(f"   Groq:   {statistics.mean(groq_latencies):6.1f}ms")
        print(f"   Claude: {statistics.mean(claude_latencies):6.1f}ms")

        print(f"\nTotal Cost:")
        print(f"   Groq:   ${groq_total_cost:.6f}")
        print(f"   Claude: ${claude_total_cost:.6f}")

        # Recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print(f"{'='*70}")

        print("\n🚀 Use Groq when:")
        print("   • Speed is critical (5-10x faster)")
        print("   • Task is simple (math, classification, extraction)")
        print("   • Cost optimization is important (often 10-100x cheaper)")
        print("   • High request volume")

        print("\n🧠 Use Claude when:")
        print("   • Task requires complex reasoning")
        print("   • Quality and accuracy are paramount")
        print("   • Need detailed explanations")
        print("   • Working with nuanced financial analysis")

        print("\n💡 Best Practice:")
        print("   • Use Groq for 80% of requests (simple tasks)")
        print("   • Use Claude for 20% of requests (complex analysis)")
        print("   • Implement caching to reduce costs by 90%")
        print("   • Monitor quality metrics to validate model selection")

        # Weekly cost projections
        print(f"\n📊 Weekly Cost Projections (100 requests/day):")
        requests_per_week = 100 * 7

        groq_weekly = (groq_total_cost / len(all_results)) * requests_per_week
        claude_weekly = (claude_total_cost / len(all_results)) * requests_per_week

        # With caching
        cache_hit_rate = 0.9
        groq_weekly_cached = groq_weekly * (1 - cache_hit_rate)
        claude_weekly_cached = claude_weekly * (1 - cache_hit_rate)

        print(f"   Groq (no cache):   ${groq_weekly:.2f}/week")
        print(f"   Groq (90% cache):  ${groq_weekly_cached:.2f}/week")
        print(f"   Claude (no cache): ${claude_weekly:.2f}/week")
        print(f"   Claude (90% cache): ${claude_weekly_cached:.2f}/week")

        if groq_weekly_cached < 1.0:
            print(f"\n   ✅ Groq meets the <$1/week budget target!")
        if claude_weekly_cached < 1.0:
            print(f"\n   ✅ Claude meets the <$1/week budget target!")

    print()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON_SCRIPT

    # Run the comparison
    cd "$WEEK2_DIR"
    python3 /tmp/compare_models.py "$ITERATIONS" "$TASK_TYPE"

    # Cleanup
    rm -f /tmp/compare_models.py
}

################################################################################
# Main Execution
################################################################################

main() {
    check_prerequisites
    run_comparison

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ Comparison complete!                                       ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo "  • Review the recommendations above"
    echo "  • Implement smart model selection in your agent"
    echo "  • Test with different task types: $0 --task complex"
    echo "  • Run more iterations for better statistics: $0 --iterations 10"
}

main
