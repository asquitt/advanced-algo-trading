"""
Exercise 2: Prompt Engineering for Financial Analysis
======================================================

Objective: Write effective prompts for financial analysis

Time: 1.5 hours
Difficulty: Intermediate

What You'll Learn:
- How to structure prompts for financial analysis
- How to use few-shot examples effectively
- How to control output format (JSON)
- How to use temperature for different use cases
- How to iterate and improve prompts

Prerequisites:
- Completed Exercise 1
- Understanding of basic financial metrics (P/E, Revenue Growth)

Setup:
------
pip install requests anthropic pydantic
"""

import os
import json
from typing import Dict, Any, List, Optional
from enum import Enum


# ============================================================================
# TODO #1: Create stock analysis prompt template
# ============================================================================

def create_stock_analysis_prompt(
    symbol: str,
    metrics: Dict[str, float],
    include_system_prompt: bool = True
) -> tuple[str, str]:
    """
    Create a prompt for stock analysis.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        metrics: Dictionary of financial metrics
        include_system_prompt: Whether to include system prompt

    Returns:
        Tuple of (system_prompt, user_prompt)

    Expected metrics:
        - price: Current stock price
        - pe_ratio: Price-to-Earnings ratio
        - revenue_growth: Revenue growth rate (as decimal, e.g., 0.15 = 15%)
        - profit_margin: Profit margin (as decimal)
        - debt_to_equity: Debt-to-Equity ratio

    Hints:
    - System prompt: Define the AI's role and behavior
    - User prompt: Provide specific data and ask for analysis
    - Be specific about output format
    - Ask for reasoning, not just recommendations
    - Include risk analysis
    """

    # TODO: Create system prompt
    # Should define:
    # - Role: "You are a professional financial analyst"
    # - Behavior: Objective, data-driven, risk-aware
    # - Output requirements: JSON format, specific fields
    system_prompt = """
    # Fill in: Define the AI's role and output requirements
    """

    # TODO: Create user prompt with metrics
    # Should include:
    # - Stock symbol
    # - All metrics with clear labels
    # - Specific questions to answer
    # - Output format specification
    user_prompt = f"""
    # Fill in: Analyze {symbol} with provided metrics
    # Include all metrics from the dictionary
    # Ask for:
    # - Buy/Sell/Hold recommendation
    # - Confidence score (0-100)
    # - Reasoning
    # - Key risks
    # - Price target (optional)
    """

    return system_prompt, user_prompt


# ============================================================================
# TODO #2: Add few-shot examples to improve accuracy
# ============================================================================

def create_few_shot_prompt(
    symbol: str,
    metrics: Dict[str, float]
) -> tuple[str, str]:
    """
    Create prompt with few-shot examples to guide the model.

    Few-shot learning: Provide examples of input/output pairs to show
    the model exactly what format and quality you expect.

    Returns:
        Tuple of (system_prompt, user_prompt)

    Hints:
    - Include 2-3 examples of good analyses
    - Show both BUY and SELL examples
    - Demonstrate the exact JSON format you want
    - Examples should be realistic but can be simplified
    """

    system_prompt = """You are a professional financial analyst. Analyze stocks and provide recommendations in JSON format.

Output format:
{
    "symbol": "TICKER",
    "recommendation": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "reasoning": "Brief explanation",
    "risks": ["risk1", "risk2"],
    "price_target": number or null
}
"""

    # TODO: Add few-shot examples
    # Example structure:
    # INPUT: Symbol XYZ with metrics...
    # OUTPUT: {recommendation JSON}
    examples = """
Examples:

# TODO: Add Example 1 - STRONG BUY scenario
# Stock with low P/E, high growth, good margins
# Show complete input metrics and output JSON

# TODO: Add Example 2 - SELL scenario
# Stock with high P/E, negative growth, high debt
# Show complete input metrics and output JSON

# TODO: Add Example 3 - HOLD scenario
# Stock with mixed signals
# Show complete input metrics and output JSON
"""

    user_prompt = f"""
Now analyze the following stock:

Symbol: {symbol}
Metrics:
{json.dumps(metrics, indent=2)}

Provide your analysis in the same JSON format as the examples above.
"""

    return system_prompt, examples + user_prompt


# ============================================================================
# TODO #3: Test different temperature settings
# ============================================================================

def test_temperature_effects(
    symbol: str = "AAPL",
    metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Test how temperature affects response consistency and creativity.

    Temperature controls randomness:
    - 0.0: Deterministic, consistent, focused
    - 0.5: Balanced
    - 1.0: Creative, varied, exploratory
    - 2.0: Very random (rarely useful)

    Args:
        symbol: Stock to analyze
        metrics: Financial metrics (uses defaults if None)

    Returns:
        Dictionary with results for each temperature

    Hints:
    - Use same prompt for all temperatures
    - Call API multiple times per temperature
    - Look for consistency in recommendations
    - Low temp (0.0-0.3): Good for analysis
    - High temp (0.7-1.0): Good for creative scenarios
    """
    if metrics is None:
        metrics = {
            "price": 150.25,
            "pe_ratio": 28.5,
            "revenue_growth": 0.08,
            "profit_margin": 0.25,
            "debt_to_equity": 1.2
        }

    temperatures = [0.0, 0.5, 1.0]
    results = {}

    # TODO: For each temperature:
    # 1. Create the same prompt
    # 2. Call the API 3 times with that temperature
    # 3. Compare consistency of recommendations
    # 4. Store results

    # Pseudocode:
    # for temp in temperatures:
    #     responses = []
    #     for i in range(3):
    #         response = call_api_with_temperature(prompt, temp)
    #         responses.append(response)
    #
    #     # Check consistency
    #     recommendations = [r['recommendation'] for r in responses]
    #     consistency = len(set(recommendations)) == 1  # All same?
    #
    #     results[temp] = {
    #         'responses': responses,
    #         'consistent': consistency,
    #         'recommendations': recommendations
    #     }

    return results


# ============================================================================
# TODO #4: Format output as JSON reliably
# ============================================================================

def create_json_output_prompt(
    symbol: str,
    metrics: Dict[str, float]
) -> str:
    """
    Create a prompt that ensures JSON output.

    Strategies for reliable JSON:
    1. Explicitly request JSON format
    2. Show example JSON structure
    3. Use markdown code blocks: ```json
    4. List all required fields
    5. Specify data types

    Returns:
        Complete prompt string that will reliably produce JSON

    Hints:
    - Be very explicit: "You must respond with valid JSON"
    - Show the exact structure expected
    - Warn against extra text outside JSON
    - Consider asking for ```json code block
    """

    # TODO: Create prompt that maximizes JSON reliability
    prompt = f"""
# Fill in: Create a prompt that ensures JSON output

Requirements:
- Must be valid JSON (parseable by json.loads())
- Must include all required fields
- Should handle edge cases (null values, etc.)
- Should prevent extra commentary outside JSON

Template:
You must respond with ONLY valid JSON, no other text.

Required fields:
- symbol: string
- recommendation: "BUY" | "SELL" | "HOLD"
- confidence: number (0-100)
- reasoning: string
- risks: array of strings
- price_target: number or null

Stock to analyze:
Symbol: {symbol}
Metrics: {json.dumps(metrics, indent=2)}

Respond with JSON only, wrapped in ```json code block.
"""

    return prompt


# ============================================================================
# TODO #5: Implement prompt iteration and improvement
# ============================================================================

class PromptVersion:
    """Track different versions of a prompt for iteration."""

    def __init__(self, version: int, prompt: str, description: str):
        self.version = version
        self.prompt = prompt
        self.description = description
        self.results: List[Dict[str, Any]] = []

    def test(self, test_cases: List[Dict[str, Any]]) -> float:
        """
        Test this prompt version on multiple test cases.

        Returns:
            Success rate (0.0 to 1.0)

        TODO: Implement testing logic
        - For each test case, call API with this prompt
        - Check if response is valid JSON
        - Check if all required fields present
        - Check if recommendation makes sense given metrics
        - Calculate success rate
        """
        successes = 0
        total = len(test_cases)

        # TODO: Implement test logic
        # for test_case in test_cases:
        #     try:
        #         result = call_api(self.prompt.format(**test_case))
        #         if validate_response(result):
        #             successes += 1
        #     except Exception as e:
        #         print(f"Test failed: {e}")

        return successes / total if total > 0 else 0.0


def iterate_prompts() -> PromptVersion:
    """
    Demonstrate prompt iteration: start simple, improve based on results.

    Process:
    1. Version 1: Basic prompt
    2. Test and identify issues
    3. Version 2: Add examples
    4. Test and identify issues
    5. Version 3: Add format constraints
    6. Compare all versions

    Returns:
        Best performing prompt version
    """

    # TODO: Create Version 1 - Basic prompt
    v1 = PromptVersion(
        version=1,
        prompt="Analyze {symbol} and recommend BUY, SELL, or HOLD.",
        description="Basic prompt, no structure"
    )

    # TODO: Create Version 2 - Add structure
    v2 = PromptVersion(
        version=2,
        prompt="""Analyze {symbol} with these metrics: {metrics}

Provide:
- Recommendation (BUY/SELL/HOLD)
- Confidence (0-100)
- Reasoning
- Risks
""",
        description="Structured output, no format"
    )

    # TODO: Create Version 3 - Add JSON format
    v3 = PromptVersion(
        version=3,
        prompt="""Analyze {symbol} with these metrics: {metrics}

Respond with valid JSON:
{{
    "recommendation": "BUY|SELL|HOLD",
    "confidence": 0-100,
    "reasoning": "...",
    "risks": ["..."]
}}
""",
        description="JSON format required"
    )

    # TODO: Create Version 4 - Add few-shot examples
    v4 = PromptVersion(
        version=4,
        prompt="""You are a financial analyst. Analyze stocks and respond with JSON.

Example:
Input: MSFT, PE=30, Growth=0.12
Output: {{"recommendation": "BUY", "confidence": 75, "reasoning": "...", "risks": ["..."]}}

Now analyze {symbol} with metrics: {metrics}
""",
        description="Few-shot + JSON format"
    )

    # TODO: Test all versions and return best one
    test_cases = [
        {"symbol": "AAPL", "metrics": {"pe_ratio": 28, "growth": 0.08}},
        {"symbol": "TSLA", "metrics": {"pe_ratio": 65, "growth": 0.25}},
    ]

    versions = [v1, v2, v3, v4]
    best_version = v1
    best_score = 0.0

    # for version in versions:
    #     score = version.test(test_cases)
    #     print(f"Version {version.version}: {score:.1%} success rate")
    #     if score > best_score:
    #         best_score = score
    #         best_version = version

    return best_version


# ============================================================================
# Test Cases
# ============================================================================

def test_basic_prompt():
    """Test basic stock analysis prompt creation."""
    print("\n" + "=" * 80)
    print("TEST: Basic Stock Analysis Prompt")
    print("=" * 80)

    metrics = {
        "price": 150.25,
        "pe_ratio": 28.5,
        "revenue_growth": 0.08,
        "profit_margin": 0.25,
        "debt_to_equity": 1.2
    }

    system_prompt, user_prompt = create_stock_analysis_prompt("AAPL", metrics)

    print("System Prompt:")
    print(system_prompt[:200] + "...")
    print("\nUser Prompt:")
    print(user_prompt[:200] + "...")

    assert "AAPL" in user_prompt, "Symbol should be in prompt"
    assert "28.5" in user_prompt or "pe_ratio" in user_prompt.lower(), "Metrics should be in prompt"
    print("\n✅ Basic prompt test passed!")


def test_few_shot():
    """Test few-shot prompt creation."""
    print("\n" + "=" * 80)
    print("TEST: Few-Shot Prompt")
    print("=" * 80)

    metrics = {
        "price": 50.0,
        "pe_ratio": 15.0,
        "revenue_growth": 0.15,
        "profit_margin": 0.20,
        "debt_to_equity": 0.8
    }

    system_prompt, user_prompt = create_few_shot_prompt("MSFT", metrics)

    print("Prompt length:", len(user_prompt))
    print("Contains 'Example':", "Example" in user_prompt or "example" in user_prompt)

    assert len(user_prompt) > 500, "Few-shot prompt should include examples (be longer)"
    print("\n✅ Few-shot prompt test passed!")


def test_json_format():
    """Test JSON output prompt."""
    print("\n" + "=" * 80)
    print("TEST: JSON Format Prompt")
    print("=" * 80)

    metrics = {
        "price": 100.0,
        "pe_ratio": 20.0,
        "revenue_growth": 0.10,
        "profit_margin": 0.15,
        "debt_to_equity": 1.0
    }

    prompt = create_json_output_prompt("GOOGL", metrics)

    print("Prompt preview:")
    print(prompt[:300] + "...")

    assert "json" in prompt.lower(), "Should mention JSON"
    assert "GOOGL" in prompt, "Should include symbol"
    print("\n✅ JSON format prompt test passed!")


def test_temperature():
    """Test temperature effects (mock test)."""
    print("\n" + "=" * 80)
    print("TEST: Temperature Effects")
    print("=" * 80)

    print("""
    Temperature Guide:
    - 0.0: Deterministic, best for financial analysis
    - 0.5: Balanced creativity and consistency
    - 1.0: More creative, less consistent

    For trading signals, recommend temperature 0.0-0.3
    """)

    print("\n✅ Temperature concept test passed!")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║        Exercise 2: Prompt Engineering for Finance                 ║
    ║                                                                    ║
    ║  Learn to write effective prompts for financial analysis          ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Run all tests
    test_basic_prompt()
    test_few_shot()
    test_json_format()
    test_temperature()

    print("\n" + "=" * 80)
    print("EXAMPLE: Good vs Bad Prompts")
    print("=" * 80)

    print("""
    ❌ BAD PROMPT:
    "What do you think about AAPL?"

    Problems:
    - Too vague
    - No data provided
    - No output format
    - No specific question

    ✅ GOOD PROMPT:
    "You are a financial analyst. Analyze AAPL with the following metrics:
    - P/E Ratio: 28.5
    - Revenue Growth: 8% YoY
    - Profit Margin: 25%
    - Debt/Equity: 1.2

    Provide a BUY/SELL/HOLD recommendation with:
    1. Confidence score (0-100)
    2. Three key reasons
    3. Top 3 risks

    Format your response as JSON with fields: recommendation, confidence,
    reasoning, risks."

    Why it's better:
    ✅ Specific role defined
    ✅ All relevant data provided
    ✅ Clear output format
    ✅ Structured requirements
    ✅ Parseable response
    """)

    print("\n" + "=" * 80)
    print("EXERCISE COMPLETE!")
    print("=" * 80)
    print("""
    Key Learnings:
    ✅ System prompts define AI behavior
    ✅ Few-shot examples improve accuracy
    ✅ Temperature controls randomness (use low for finance)
    ✅ Explicit JSON formatting prevents parsing errors
    ✅ Prompt iteration improves results

    Best Practices:
    1. Be specific and provide context
    2. Use examples to show desired output
    3. Request structured formats (JSON)
    4. Use low temperature (0.0-0.3) for analysis
    5. Test and iterate on prompts

    Next: Exercise 3 - Response Parsing
    """)
