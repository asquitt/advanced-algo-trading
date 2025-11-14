"""
Exercise 3: Parse LLM Responses Reliably
=========================================

Objective: Handle LLM responses that may be malformed or inconsistent

Time: 1.5 hours
Difficulty: Intermediate

What You'll Learn:
- Extract JSON from markdown code blocks
- Handle malformed JSON responses
- Validate responses with Pydantic
- Implement error recovery strategies
- Build robust parsers for production

Prerequisites:
- Completed Exercise 1 and 2
- Understanding of JSON and Python type hints

Setup:
------
pip install pydantic requests anthropic
"""

import json
import re
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationError


# ============================================================================
# TODO #1: Define response schema with Pydantic
# ============================================================================

class Recommendation(str, Enum):
    """Valid recommendation types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StockAnalysis(BaseModel):
    """
    Validated stock analysis response.

    Use Pydantic to ensure responses have correct:
    - Field names
    - Data types
    - Value ranges
    - Required vs optional fields
    """

    # TODO: Define fields with proper types and validation

    symbol: str = Field(..., description="Stock ticker symbol")
    recommendation: Recommendation = Field(..., description="BUY, SELL, or HOLD")

    # TODO: Add confidence field (0-100)
    # Hint: Use Field with ge=0, le=100 for range validation
    confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")

    # TODO: Add reasoning field (non-empty string)
    reasoning: str = Field(..., min_length=10, description="Analysis reasoning")

    # TODO: Add risks field (list of strings)
    risks: List[str] = Field(default_factory=list, description="Identified risks")

    # TODO: Add optional price_target field
    price_target: Optional[float] = Field(None, gt=0, description="Price target if available")

    @field_validator('symbol')
    @classmethod
    def symbol_must_be_uppercase(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        # TODO: Convert symbol to uppercase and validate format
        # Should be 1-5 uppercase letters
        if not v:
            raise ValueError("Symbol cannot be empty")
        return v.upper()

    @field_validator('reasoning')
    @classmethod
    def reasoning_must_be_substantial(cls, v: str) -> str:
        """Ensure reasoning is meaningful."""
        # TODO: Check that reasoning has enough content
        # At least 10 characters, not just whitespace
        if len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters")
        return v.strip()

    @field_validator('risks')
    @classmethod
    def risks_must_be_valid(cls, v: List[str]) -> List[str]:
        """Ensure risks are non-empty strings."""
        # TODO: Filter out empty strings, ensure at least one risk
        valid_risks = [r.strip() for r in v if r.strip()]
        if not valid_risks:
            raise ValueError("At least one risk must be provided")
        return valid_risks


# ============================================================================
# TODO #2: Extract JSON from markdown code blocks
# ============================================================================

def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON from markdown code blocks.

    LLMs often wrap JSON in ```json ... ``` blocks.
    This function extracts the JSON content.

    Args:
        text: Raw LLM response that may contain markdown

    Returns:
        Extracted JSON string

    Examples:
        Input: "Here's the analysis:\n```json\n{\"key\": \"value\"}\n```"
        Output: "{\"key\": \"value\"}"

        Input: "```{\"key\": \"value\"}```"
        Output: "{\"key\": \"value\"}"

        Input: "{\"key\": \"value\"}"
        Output: "{\"key\": \"value\"}"

    Hints:
    - Use regex to find ```json or ``` blocks
    - Pattern: r'```(?:json)?\s*(\{.*?\})\s*```' with re.DOTALL
    - If no code block, try to find JSON directly
    - Handle multiple code blocks (take first)
    """

    # TODO: Try to extract from ```json code blocks
    # Pattern should match:
    # ```json
    # {...}
    # ```
    # OR
    # ```
    # {...}
    # ```

    # Hint: Use re.search with re.DOTALL flag
    # json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    # match = re.search(json_pattern, text, re.DOTALL)
    # if match:
    #     return match.group(1)

    # TODO: If no code block, try to find raw JSON
    # Look for content between { and }
    # json_pattern = r'\{.*\}'
    # match = re.search(json_pattern, text, re.DOTALL)
    # if match:
    #     return match.group(0)

    # If nothing found, return original text
    return text.strip()


# ============================================================================
# TODO #3: Parse and validate JSON responses
# ============================================================================

def parse_analysis_response(
    raw_response: str,
    strict: bool = True
) -> Union[StockAnalysis, Dict[str, Any]]:
    """
    Parse and validate an LLM response into a StockAnalysis object.

    Args:
        raw_response: Raw text from LLM
        strict: If True, raise exceptions. If False, return partial data.

    Returns:
        StockAnalysis object if valid, or dict with error info

    Process:
    1. Extract JSON from markdown
    2. Parse JSON string
    3. Validate with Pydantic
    4. Handle errors gracefully

    Hints:
    - Use extract_json_from_markdown first
    - Use json.loads to parse
    - Use StockAnalysis.model_validate or StockAnalysis(**data)
    - Catch JSONDecodeError and ValidationError
    """

    try:
        # TODO: Step 1 - Extract JSON from markdown
        # json_str = extract_json_from_markdown(raw_response)

        # TODO: Step 2 - Parse JSON string
        # data = json.loads(json_str)

        # TODO: Step 3 - Validate with Pydantic
        # analysis = StockAnalysis(**data)
        # return analysis

        pass

    except json.JSONDecodeError as e:
        # TODO: Handle JSON parsing errors
        if strict:
            raise ValueError(f"Invalid JSON: {e}")
        return {
            "error": "json_decode_error",
            "message": str(e),
            "raw_response": raw_response[:200]
        }

    except ValidationError as e:
        # TODO: Handle Pydantic validation errors
        if strict:
            raise ValueError(f"Validation error: {e}")
        return {
            "error": "validation_error",
            "message": str(e),
            "raw_response": raw_response[:200]
        }

    except Exception as e:
        # TODO: Handle unexpected errors
        if strict:
            raise
        return {
            "error": "unknown_error",
            "message": str(e),
            "raw_response": raw_response[:200]
        }


# ============================================================================
# TODO #4: Handle malformed responses with error recovery
# ============================================================================

def fix_common_json_errors(json_str: str) -> str:
    """
    Fix common JSON formatting errors from LLMs.

    Common issues:
    1. Trailing commas: {"key": "value",}
    2. Single quotes: {'key': 'value'}
    3. Unquoted keys: {key: "value"}
    4. Missing quotes: {key: value}
    5. Extra text before/after JSON

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Fixed JSON string

    Hints:
    - Remove trailing commas before } or ]
    - Replace single quotes with double quotes (carefully!)
    - These are band-aids; better prompts prevent issues
    """

    # TODO: Remove trailing commas
    # Pattern: ,\s*} or ,\s*]
    # json_str = re.sub(r',\s*}', '}', json_str)
    # json_str = re.sub(r',\s*]', ']', json_str)

    # TODO: Try to handle single quotes (risky with contractions!)
    # Only if no double quotes exist
    # if '"' not in json_str and "'" in json_str:
    #     json_str = json_str.replace("'", '"')

    # TODO: Remove common prefixes/suffixes
    # "Here's the JSON: {...}"
    # "```json{...}```"

    return json_str.strip()


def parse_with_recovery(raw_response: str) -> Optional[StockAnalysis]:
    """
    Parse response with multiple recovery strategies.

    Strategy:
    1. Try normal parsing
    2. If fails, extract JSON from markdown
    3. If fails, fix common JSON errors
    4. If fails, try regex extraction
    5. If all fail, return None

    This is production-grade error handling!

    Returns:
        StockAnalysis if successful, None otherwise
    """

    # TODO: Strategy 1 - Normal parse
    try:
        # result = parse_analysis_response(raw_response, strict=True)
        # if isinstance(result, StockAnalysis):
        #     return result
        pass
    except:
        pass

    # TODO: Strategy 2 - Extract from markdown
    try:
        # json_str = extract_json_from_markdown(raw_response)
        # data = json.loads(json_str)
        # return StockAnalysis(**data)
        pass
    except:
        pass

    # TODO: Strategy 3 - Fix common errors
    try:
        # json_str = extract_json_from_markdown(raw_response)
        # fixed_json = fix_common_json_errors(json_str)
        # data = json.loads(fixed_json)
        # return StockAnalysis(**data)
        pass
    except:
        pass

    # TODO: Strategy 4 - Aggressive regex extraction
    # Try to extract individual fields with regex
    try:
        # recommendation_match = re.search(r'"recommendation":\s*"(BUY|SELL|HOLD)"', raw_response)
        # confidence_match = re.search(r'"confidence":\s*(\d+)', raw_response)
        # ... extract other fields ...
        # If enough fields found, construct object
        pass
    except:
        pass

    # All strategies failed
    return None


# ============================================================================
# TODO #5: Build a robust response parser class
# ============================================================================

class ResponseParser:
    """
    Production-grade response parser with logging and metrics.
    """

    def __init__(self, enable_recovery: bool = True, log_errors: bool = True):
        """
        Initialize parser.

        Args:
            enable_recovery: Use error recovery strategies
            log_errors: Log parsing errors for debugging
        """
        self.enable_recovery = enable_recovery
        self.log_errors = log_errors
        self.stats = {
            "total_parses": 0,
            "successful": 0,
            "failed": 0,
            "recovered": 0
        }

    def parse(self, raw_response: str) -> Optional[StockAnalysis]:
        """
        Parse a response with full error handling.

        Returns:
            StockAnalysis if successful, None if all strategies fail
        """
        self.stats["total_parses"] += 1

        # TODO: Try normal parsing first
        try:
            result = parse_analysis_response(raw_response, strict=True)
            if isinstance(result, StockAnalysis):
                self.stats["successful"] += 1
                return result
        except Exception as e:
            if self.log_errors:
                print(f"Parse error: {e}")

        # TODO: If recovery enabled, try recovery strategies
        if self.enable_recovery:
            result = parse_with_recovery(raw_response)
            if result:
                self.stats["recovered"] += 1
                return result

        # TODO: All strategies failed
        self.stats["failed"] += 1
        if self.log_errors:
            print(f"Failed to parse response: {raw_response[:100]}...")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        success_rate = (
            (self.stats["successful"] + self.stats["recovered"]) /
            self.stats["total_parses"]
        ) if self.stats["total_parses"] > 0 else 0

        return {
            **self.stats,
            "success_rate": success_rate
        }


# ============================================================================
# Test Cases
# ============================================================================

def test_pydantic_validation():
    """Test Pydantic validation."""
    print("\n" + "=" * 80)
    print("TEST: Pydantic Validation")
    print("=" * 80)

    # Valid data
    valid_data = {
        "symbol": "aapl",  # Will be converted to uppercase
        "recommendation": "BUY",
        "confidence": 85,
        "reasoning": "Strong fundamentals and growth potential",
        "risks": ["Market volatility", "Competition"],
        "price_target": 175.50
    }

    try:
        analysis = StockAnalysis(**valid_data)
        print(f"✅ Valid data parsed successfully")
        print(f"   Symbol: {analysis.symbol}")
        print(f"   Recommendation: {analysis.recommendation}")
        print(f"   Confidence: {analysis.confidence}")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")

    # Invalid data - confidence out of range
    invalid_data = {
        "symbol": "TSLA",
        "recommendation": "BUY",
        "confidence": 150,  # Invalid! Must be 0-100
        "reasoning": "Good company",
        "risks": ["Risk 1"]
    }

    try:
        analysis = StockAnalysis(**invalid_data)
        print(f"❌ Should have failed validation")
    except ValidationError as e:
        print(f"✅ Correctly rejected invalid confidence: {e.error_count()} errors")


def test_json_extraction():
    """Test JSON extraction from markdown."""
    print("\n" + "=" * 80)
    print("TEST: JSON Extraction from Markdown")
    print("=" * 80)

    # Test case 1: JSON in code block
    text1 = """
    Here's my analysis:

    ```json
    {
        "symbol": "AAPL",
        "recommendation": "BUY",
        "confidence": 85
    }
    ```

    Hope this helps!
    """

    extracted = extract_json_from_markdown(text1)
    print("Test 1 - JSON in code block:")
    print(f"  Extracted: {extracted[:50]}...")
    assert "{" in extracted and "}" in extracted

    # Test case 2: Plain JSON
    text2 = '{"symbol": "MSFT", "recommendation": "HOLD"}'
    extracted = extract_json_from_markdown(text2)
    print("Test 2 - Plain JSON:")
    print(f"  Extracted: {extracted[:50]}...")
    assert extracted == text2

    print("\n✅ JSON extraction tests passed!")


def test_error_recovery():
    """Test error recovery strategies."""
    print("\n" + "=" * 80)
    print("TEST: Error Recovery")
    print("=" * 80)

    # Malformed JSON with trailing comma
    malformed = """{
        "symbol": "GOOGL",
        "recommendation": "BUY",
        "confidence": 90,
    }"""

    print("Original (malformed):")
    print(malformed)

    fixed = fix_common_json_errors(malformed)
    print("\nFixed:")
    print(fixed)

    try:
        data = json.loads(fixed)
        print("\n✅ Successfully parsed after fix!")
    except json.JSONDecodeError as e:
        print(f"\n❌ Still invalid: {e}")


def test_parser_class():
    """Test the ResponseParser class."""
    print("\n" + "=" * 80)
    print("TEST: ResponseParser Class")
    print("=" * 80)

    parser = ResponseParser(enable_recovery=True, log_errors=False)

    # Test 1: Valid response
    valid_response = """```json
    {
        "symbol": "AAPL",
        "recommendation": "BUY",
        "confidence": 85,
        "reasoning": "Strong fundamentals with solid growth trajectory",
        "risks": ["Market volatility", "Competition from Android"],
        "price_target": 180.0
    }
    ```"""

    result = parser.parse(valid_response)
    if result:
        print(f"✅ Test 1 passed: {result.symbol} - {result.recommendation}")
    else:
        print(f"❌ Test 1 failed")

    # Test 2: Response with extra text
    messy_response = """
    Based on my analysis, here's what I think:

    ```json
    {
        "symbol": "MSFT",
        "recommendation": "HOLD",
        "confidence": 70,
        "reasoning": "Fair valuation with moderate growth prospects",
        "risks": ["Cloud competition", "Regulatory scrutiny"]
    }
    ```

    Let me know if you need more details!
    """

    result = parser.parse(messy_response)
    if result:
        print(f"✅ Test 2 passed: {result.symbol} - {result.recommendation}")
    else:
        print(f"❌ Test 2 failed")

    # Print statistics
    stats = parser.get_stats()
    print(f"\nParser Statistics:")
    print(f"  Total parses: {stats['total_parses']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Recovered: {stats['recovered']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           Exercise 3: Parse LLM Responses Reliably                ║
    ║                                                                    ║
    ║  Learn to handle real-world LLM responses robustly                ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Run all tests
    test_pydantic_validation()
    test_json_extraction()
    test_error_recovery()
    test_parser_class()

    print("\n" + "=" * 80)
    print("COMMON LLM RESPONSE ISSUES")
    print("=" * 80)
    print("""
    1. ❌ Wrapped in markdown code blocks
       Response: "```json\\n{...}\\n```"
       Fix: Extract JSON from code blocks

    2. ❌ Extra explanatory text
       Response: "Here's my analysis: {...} Hope this helps!"
       Fix: Regex extraction of JSON content

    3. ❌ Trailing commas
       Response: {"key": "value",}
       Fix: Remove trailing commas with regex

    4. ❌ Single quotes instead of double quotes
       Response: {'key': 'value'}
       Fix: Replace single quotes (carefully!)

    5. ❌ Missing required fields
       Response: {"symbol": "AAPL"}  (missing other fields)
       Fix: Pydantic validation with clear error messages

    6. ❌ Invalid field values
       Response: {"confidence": 150}  (should be 0-100)
       Fix: Pydantic validators with range constraints

    Best Practices:
    ✅ Use Pydantic for schema validation
    ✅ Implement multiple parsing strategies
    ✅ Log errors for debugging
    ✅ Track success/failure metrics
    ✅ Improve prompts to prevent issues
    ✅ Always validate before using data
    """)

    print("\n" + "=" * 80)
    print("EXERCISE COMPLETE!")
    print("=" * 80)
    print("""
    Key Learnings:
    ✅ Pydantic provides robust validation
    ✅ LLMs often wrap JSON in markdown
    ✅ Multiple parsing strategies improve reliability
    ✅ Error recovery can salvage 90%+ of responses
    ✅ Better prompts prevent most issues

    Production Checklist:
    - [ ] Use Pydantic schemas for all responses
    - [ ] Extract JSON from markdown code blocks
    - [ ] Implement at least 2 recovery strategies
    - [ ] Log parsing errors for monitoring
    - [ ] Track success rates and improve prompts
    - [ ] Never trust unvalidated LLM output

    Next: Exercise 4 - Build Complete Financial Agent
    """)
