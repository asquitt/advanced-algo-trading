"""
Prompt Engineering for Financial Analysis

This module contains prompt templates and builders for financial analysis
using LLMs. It demonstrates best practices in prompt engineering including:
- System prompts for context
- Few-shot examples for better outputs
- Structured output formatting
- Dynamic prompt building

Learning Goals:
- Master prompt engineering techniques
- Build reusable prompt templates
- Structure LLM outputs as JSON
- Use few-shot learning effectively
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# =============================================================================
# TODO #1: Create prompt templates
# =============================================================================
# Instructions:
# 1. Create a PromptTemplate class
# 2. Add template string with placeholders
# 3. Implement a format() method to fill in values
# 4. Add validation for required parameters
# 5. Create multiple templates for different use cases
#
# Example templates to create:
# - Stock analysis
# - Risk assessment
# - Portfolio recommendation
# - Market sentiment
# =============================================================================


class AnalysisType(Enum):
    """Types of financial analysis"""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    RISK = "risk"
    PORTFOLIO = "portfolio"


@dataclass
class PromptTemplate:
    """
    A reusable prompt template with placeholders.

    Attributes:
        name: Template identifier
        template: Template string with {placeholder} syntax
        required_params: List of required parameter names
        description: What this template does
    """

    name: str
    template: str
    required_params: List[str]
    description: str

    def format(self, **kwargs) -> str:
        """
        Fill in template placeholders with values.

        TODO: Implement this method
        Steps:
        1. Validate all required_params are in kwargs
        2. Format template string with kwargs
        3. Return formatted string
        4. Raise ValueError if missing params

        Args:
            **kwargs: Parameter values to fill in

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If required parameters are missing
        """
        # TODO: Implement template formatting with validation
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # missing = [p for p in self.required_params if p not in kwargs]
        # if missing:
        #     raise ValueError(f"Missing required parameters: {missing}")
        #
        # return self.template.format(**kwargs)


# =============================================================================
# TODO #2: Add system prompts
# =============================================================================
# Instructions:
# 1. Create system prompts that define the LLM's role
# 2. Add domain-specific context (financial analysis)
# 3. Specify output format requirements
# 4. Add guidelines for reasoning
# 5. Create different system prompts for different tasks
#
# System prompts should:
# - Define the role (financial analyst)
# - Set the tone (professional, data-driven)
# - Specify constraints (be concise, cite data)
# - Define output structure (JSON format)
# =============================================================================


# TODO: Create system prompts for different analysis types

SYSTEM_PROMPTS = {
    "fundamental_analyst": """
TODO: Write a system prompt for fundamental analysis
Should include:
- Role: Expert financial analyst
- Task: Analyze company fundamentals
- Output: JSON with recommendation, confidence, reasoning
- Constraints: Be objective, cite specific metrics
""",

    "technical_analyst": """
TODO: Write a system prompt for technical analysis
Should include:
- Role: Technical analysis expert
- Task: Analyze price patterns and indicators
- Output: JSON with signals, entry/exit points
- Constraints: Focus on data, not speculation
""",

    "risk_analyst": """
TODO: Write a system prompt for risk assessment
Should include:
- Role: Risk management specialist
- Task: Identify and quantify risks
- Output: JSON with risk score, factors, mitigation
- Constraints: Be conservative, highlight uncertainties
""",
}

# EXAMPLE SYSTEM PROMPTS (uncomment to use):
# SYSTEM_PROMPTS = {
#     "fundamental_analyst": """You are an expert financial analyst specializing in fundamental analysis.
#
# Your task is to analyze stocks based on financial metrics and provide actionable recommendations.
#
# Guidelines:
# - Be objective and data-driven
# - Cite specific financial metrics (P/E, revenue growth, margins, etc.)
# - Consider both quantitative and qualitative factors
# - Provide clear buy/hold/sell recommendations
# - Explain your reasoning concisely
#
# Output format:
# Return your analysis as a JSON object with this structure:
# {
#   "recommendation": "BUY" | "HOLD" | "SELL",
#   "confidence": 0.0-1.0,
#   "reasoning": "Brief explanation",
#   "key_metrics": ["metric1", "metric2"],
#   "risks": ["risk1", "risk2"],
#   "score": 0-100
# }
#
# Be concise but thorough. Focus on actionable insights.""",
#
#     "technical_analyst": """You are a technical analysis expert with deep knowledge of chart patterns and indicators.
#
# Your task is to analyze price action and technical indicators to identify trading opportunities.
#
# Guidelines:
# - Focus on price patterns, support/resistance, and momentum
# - Use standard technical indicators (RSI, MACD, Moving Averages)
# - Identify clear entry and exit points
# - Specify stop-loss levels for risk management
# - Be specific about timeframes
#
# Output format:
# Return your analysis as JSON:
# {
#   "signal": "BUY" | "SELL" | "NEUTRAL",
#   "strength": 0.0-1.0,
#   "entry_price": float,
#   "stop_loss": float,
#   "take_profit": float,
#   "reasoning": "Brief explanation",
#   "indicators": {"RSI": value, "MACD": value}
# }""",
#
#     "risk_analyst": """You are a risk management specialist focused on portfolio protection.
#
# Your task is to identify, quantify, and communicate investment risks.
#
# Guidelines:
# - Identify all material risks (market, company-specific, macro)
# - Quantify risks where possible
# - Be conservative in your estimates
# - Suggest risk mitigation strategies
# - Highlight uncertainties and unknowns
#
# Output format:
# Return as JSON:
# {
#   "risk_score": 0-100,
#   "risk_level": "LOW" | "MEDIUM" | "HIGH",
#   "key_risks": ["risk1", "risk2"],
#   "mitigation": ["strategy1", "strategy2"],
#   "max_position_size": float (0.0-1.0),
#   "reasoning": "Brief explanation"
# }""",
# }


# =============================================================================
# TODO #3: Add few-shot examples
# =============================================================================
# Instructions:
# 1. Create example input/output pairs
# 2. Add 2-3 examples per analysis type
# 3. Show the desired output format
# 4. Include edge cases (high P/E, negative growth, etc.)
# 5. Create a function to inject examples into prompts
#
# Few-shot examples help the LLM understand:
# - Desired output structure
# - Level of detail expected
# - Reasoning style
# - Edge case handling
# =============================================================================


# TODO: Create few-shot examples for each analysis type

FEW_SHOT_EXAMPLES = {
    "fundamental_analysis": [
        {
            "input": """
TODO: Create example input
Should include stock symbol and financial metrics
""",
            "output": """
TODO: Create example output
Should be valid JSON matching the expected format
"""
        },
    ],
}

# EXAMPLE FEW-SHOT EXAMPLES (uncomment to use):
# FEW_SHOT_EXAMPLES = {
#     "fundamental_analysis": [
#         {
#             "input": """Analyze AAPL stock:
# - Price: $175.50
# - P/E Ratio: 28.5
# - Revenue Growth: 8.2%
# - Profit Margin: 25.3%
# - Debt/Equity: 1.5""",
#             "output": """{
#   "recommendation": "BUY",
#   "confidence": 0.75,
#   "reasoning": "Strong fundamentals with healthy margins and growth. P/E is reasonable for the tech sector. Moderate debt levels are manageable given strong cash flow.",
#   "key_metrics": ["profit_margin: 25.3%", "revenue_growth: 8.2%", "pe_ratio: 28.5"],
#   "risks": ["High P/E ratio", "Market volatility", "Competitive pressure"],
#   "score": 78
# }"""
#         },
#         {
#             "input": """Analyze XYZ stock:
# - Price: $25.00
# - P/E Ratio: 45.0
# - Revenue Growth: -5.0%
# - Profit Margin: 8.0%
# - Debt/Equity: 3.0""",
#             "output": """{
#   "recommendation": "SELL",
#   "confidence": 0.85,
#   "reasoning": "Poor fundamentals with declining revenue and high valuation. P/E of 45 is unjustified given negative growth. High debt increases financial risk.",
#   "key_metrics": ["revenue_growth: -5.0%", "pe_ratio: 45.0", "debt_equity: 3.0"],
#   "risks": ["Declining revenue", "High debt burden", "Overvaluation"],
#   "score": 25
# }"""
#         },
#     ],
#     "technical_analysis": [
#         {
#             "input": """Analyze TSLA:
# - Current Price: $250.00
# - RSI: 72
# - MACD: Bullish crossover
# - 50-day MA: $240.00
# - 200-day MA: $220.00""",
#             "output": """{
#   "signal": "BUY",
#   "strength": 0.70,
#   "entry_price": 250.00,
#   "stop_loss": 240.00,
#   "take_profit": 270.00,
#   "reasoning": "Bullish MACD crossover with price above both moving averages. RSI approaching overbought but momentum is strong.",
#   "indicators": {"RSI": 72, "MACD": "bullish", "trend": "upward"}
# }"""
#         },
#     ],
# }


def format_few_shot_examples(examples: List[Dict[str, str]]) -> str:
    """
    Format few-shot examples for inclusion in prompt.

    TODO: Implement this function
    Steps:
    1. Iterate through examples
    2. Format each as "Input: {input}\nOutput: {output}"
    3. Join with double newlines
    4. Return formatted string

    Args:
        examples: List of example dicts with 'input' and 'output'

    Returns:
        Formatted examples string
    """
    # TODO: Implement few-shot formatting
    pass

    # EXAMPLE IMPLEMENTATION (uncomment to use):
    # formatted = []
    # for i, example in enumerate(examples, 1):
    #     formatted.append(f"Example {i}:\nInput: {example['input']}\nOutput: {example['output']}")
    # return "\n\n".join(formatted)


# =============================================================================
# TODO #4: Create prompt builder
# =============================================================================
# Instructions:
# 1. Create a PromptBuilder class
# 2. Add methods: with_system(), with_examples(), with_user()
# 3. Implement a build() method that combines everything
# 4. Add validation for required components
# 5. Support method chaining for fluent API
#
# Example usage after implementation:
# prompt = (PromptBuilder()
#     .with_system("fundamental_analyst")
#     .with_examples("fundamental_analysis")
#     .with_user_input(symbol="AAPL", metrics=data)
#     .build())
# =============================================================================


class PromptBuilder:
    """
    Fluent API for building complete prompts.

    Combines system prompts, few-shot examples, and user input
    into a complete prompt ready for the LLM.

    Example:
        builder = PromptBuilder()
        prompt = (builder
            .with_system("fundamental_analyst")
            .with_examples("fundamental_analysis", count=2)
            .with_user_input(symbol="AAPL", data=metrics)
            .build())
    """

    def __init__(self):
        # TODO: Initialize builder state
        # - system_prompt: Optional[str]
        # - examples: List[Dict]
        # - user_input: Optional[str]
        # - template: Optional[PromptTemplate]
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # self.system_prompt: Optional[str] = None
        # self.examples: List[Dict[str, str]] = []
        # self.user_input: Optional[str] = None
        # self.template: Optional[PromptTemplate] = None

    def with_system(self, prompt_name: str) -> 'PromptBuilder':
        """
        Add a system prompt.

        TODO: Implement this method
        - Get system prompt from SYSTEM_PROMPTS dict
        - Store in self.system_prompt
        - Return self for chaining

        Args:
            prompt_name: Name of system prompt to use

        Returns:
            Self for method chaining
        """
        # TODO: Implement system prompt selection
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # if prompt_name not in SYSTEM_PROMPTS:
        #     raise ValueError(f"Unknown system prompt: {prompt_name}")
        # self.system_prompt = SYSTEM_PROMPTS[prompt_name]
        # return self

    def with_examples(
        self,
        example_type: str,
        count: Optional[int] = None
    ) -> 'PromptBuilder':
        """
        Add few-shot examples.

        TODO: Implement this method
        - Get examples from FEW_SHOT_EXAMPLES dict
        - Limit to 'count' if specified
        - Store in self.examples
        - Return self for chaining

        Args:
            example_type: Type of examples to include
            count: Max number of examples (None = all)

        Returns:
            Self for method chaining
        """
        # TODO: Implement example selection
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # if example_type not in FEW_SHOT_EXAMPLES:
        #     raise ValueError(f"Unknown example type: {example_type}")
        #
        # examples = FEW_SHOT_EXAMPLES[example_type]
        # if count:
        #     examples = examples[:count]
        #
        # self.examples = examples
        # return self

    def with_template(self, template: PromptTemplate) -> 'PromptBuilder':
        """
        Use a specific template.

        TODO: Implement this method
        - Store template in self.template
        - Return self for chaining

        Args:
            template: PromptTemplate to use

        Returns:
            Self for method chaining
        """
        # TODO: Implement template selection
        pass

    def with_user_input(self, **kwargs) -> 'PromptBuilder':
        """
        Add user input/data.

        TODO: Implement this method
        - Format kwargs as a readable string
        - If template exists, use template.format(**kwargs)
        - Otherwise, create simple key-value format
        - Store in self.user_input
        - Return self for chaining

        Args:
            **kwargs: User input data

        Returns:
            Self for method chaining
        """
        # TODO: Implement user input formatting
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # if self.template:
        #     self.user_input = self.template.format(**kwargs)
        # else:
        #     # Simple key-value format
        #     lines = [f"{k}: {v}" for k, v in kwargs.items()]
        #     self.user_input = "\n".join(lines)
        #
        # return self

    def build(self) -> Dict[str, str]:
        """
        Build the complete prompt.

        TODO: Implement this method
        Steps:
        1. Validate that user_input is set
        2. Build prompt components:
        3.   - Start with system_prompt
        4.   - Add formatted examples if present
        5.   - Add user_input
        6. Return dict with 'system' and 'user' keys

        Returns:
            Dict with 'system' and 'user' prompt components

        Raises:
            ValueError: If required components are missing
        """
        # TODO: Implement prompt building
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # if not self.user_input:
        #     raise ValueError("User input is required")
        #
        # user_prompt_parts = []
        #
        # # Add examples if present
        # if self.examples:
        #     user_prompt_parts.append("Here are some examples:\n")
        #     user_prompt_parts.append(format_few_shot_examples(self.examples))
        #     user_prompt_parts.append("\n\nNow analyze this:")
        #
        # # Add user input
        # user_prompt_parts.append(self.user_input)
        #
        # return {
        #     "system": self.system_prompt or "",
        #     "user": "\n".join(user_prompt_parts)
        # }

    def reset(self) -> 'PromptBuilder':
        """Reset the builder to initial state."""
        self.__init__()
        return self


# =============================================================================
# TODO #5: Add validation
# =============================================================================
# Instructions:
# 1. Create a validate_prompt() function
# 2. Check for common issues (too long, missing context, etc.)
# 3. Add a function to estimate token count
# 4. Create warnings for prompts that are too long
# 5. Add suggestions for improvement
# =============================================================================


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a prompt.

    TODO: Implement token estimation
    - Use rough estimate: ~4 characters per token
    - Or use tiktoken library for accurate counting
    - Return estimated token count

    Args:
        text: Prompt text

    Returns:
        Estimated token count
    """
    # TODO: Implement token estimation
    pass

    # EXAMPLE IMPLEMENTATION (uncomment to use):
    # # Rough estimate: 4 characters per token
    # return len(text) // 4

    # BETTER IMPLEMENTATION (requires tiktoken):
    # import tiktoken
    # enc = tiktoken.get_encoding("cl100k_base")
    # return len(enc.encode(text))


def validate_prompt(prompt: Dict[str, str]) -> List[str]:
    """
    Validate a prompt and return warnings/suggestions.

    TODO: Implement validation
    Checks:
    1. Total token count (warn if > 3000)
    2. System prompt exists and is not empty
    3. User prompt exists and is not empty
    4. User prompt has specific details (not too vague)
    5. Output format is specified

    Args:
        prompt: Dict with 'system' and 'user' keys

    Returns:
        List of warning/suggestion strings
    """
    # TODO: Implement validation logic
    pass

    # EXAMPLE IMPLEMENTATION (uncomment to use):
    # warnings = []
    #
    # # Check system prompt
    # if not prompt.get("system"):
    #     warnings.append("Missing system prompt - consider adding role context")
    #
    # # Check user prompt
    # if not prompt.get("user"):
    #     warnings.append("Missing user prompt")
    #
    # # Check total length
    # total_text = prompt.get("system", "") + prompt.get("user", "")
    # tokens = estimate_tokens(total_text)
    # if tokens > 3000:
    #     warnings.append(f"Prompt is very long ({tokens} tokens) - consider shortening")
    #
    # # Check for output format specification
    # if "json" not in total_text.lower():
    #     warnings.append("Consider specifying JSON output format")
    #
    # return warnings


# =============================================================================
# Pre-built Templates
# =============================================================================

# TODO: Create pre-built templates for common tasks

STOCK_ANALYSIS_TEMPLATE = PromptTemplate(
    name="stock_analysis",
    template="""Analyze {symbol} stock with these metrics:
- Current Price: ${price}
- P/E Ratio: {pe_ratio}
- Revenue Growth: {revenue_growth}%
- Profit Margin: {profit_margin}%
- Debt/Equity: {debt_equity}

Provide a buy/hold/sell recommendation with confidence score and reasoning.""",
    required_params=["symbol", "price", "pe_ratio", "revenue_growth", "profit_margin", "debt_equity"],
    description="Fundamental analysis of a stock"
)

# EXAMPLE TEMPLATES (uncomment to add more):
# PORTFOLIO_RECOMMENDATION_TEMPLATE = PromptTemplate(
#     name="portfolio_recommendation",
#     template="""Review this portfolio allocation:
# {portfolio_json}
#
# Risk tolerance: {risk_tolerance}
# Investment horizon: {time_horizon} years
# Goals: {goals}
#
# Suggest improvements to optimize risk/return.""",
#     required_params=["portfolio_json", "risk_tolerance", "time_horizon", "goals"],
#     description="Portfolio optimization recommendations"
# )


# =============================================================================
# Helper Functions
# =============================================================================

def create_stock_analysis_prompt(
    symbol: str,
    price: float,
    pe_ratio: float,
    revenue_growth: float,
    profit_margin: float,
    debt_equity: float,
    include_examples: bool = True
) -> Dict[str, str]:
    """
    Convenience function to create a stock analysis prompt.

    TODO: Implement this helper function
    Steps:
    1. Create PromptBuilder
    2. Add system prompt for fundamental analysis
    3. Optionally add examples
    4. Format template with provided data
    5. Build and return prompt

    Args:
        symbol: Stock ticker symbol
        price: Current stock price
        pe_ratio: Price-to-earnings ratio
        revenue_growth: Revenue growth percentage
        profit_margin: Profit margin percentage
        debt_equity: Debt-to-equity ratio
        include_examples: Whether to include few-shot examples

    Returns:
        Complete prompt dict ready for LLM
    """
    # TODO: Implement convenience function
    pass

    # EXAMPLE IMPLEMENTATION (uncomment to use):
    # builder = PromptBuilder()
    # builder.with_system("fundamental_analyst")
    #
    # if include_examples:
    #     builder.with_examples("fundamental_analysis", count=2)
    #
    # builder.with_template(STOCK_ANALYSIS_TEMPLATE)
    # builder.with_user_input(
    #     symbol=symbol,
    #     price=price,
    #     pe_ratio=pe_ratio,
    #     revenue_growth=revenue_growth,
    #     profit_margin=profit_margin,
    #     debt_equity=debt_equity
    # )
    #
    # return builder.build()


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """
    Example of how to use the prompt building system.

    Run this after implementing all TODOs to test your code.
    """
    print("=== Example 1: Manual Prompt Building ===")
    builder = PromptBuilder()
    prompt = (builder
        .with_system("fundamental_analyst")
        .with_examples("fundamental_analysis", count=2)
        .with_user_input(
            symbol="AAPL",
            price=175.50,
            pe_ratio=28.5,
            revenue_growth=8.2,
            profit_margin=25.3,
            debt_equity=1.5
        )
        .build())

    print(f"System: {prompt['system'][:100]}...")
    print(f"User: {prompt['user'][:200]}...")

    # Validate
    warnings = validate_prompt(prompt)
    if warnings:
        print(f"\nWarnings: {warnings}")

    print(f"\n=== Example 2: Using Template ===")
    prompt2 = create_stock_analysis_prompt(
        symbol="TSLA",
        price=250.00,
        pe_ratio=65.0,
        revenue_growth=15.0,
        profit_margin=12.0,
        debt_equity=0.8,
        include_examples=True
    )

    print(f"System: {prompt2['system'][:100]}...")
    print(f"User: {prompt2['user'][:200]}...")

    # Estimate tokens
    total_tokens = estimate_tokens(prompt2['system'] + prompt2['user'])
    print(f"\nEstimated tokens: {total_tokens}")


if __name__ == "__main__":
    # Uncomment to test your implementation
    # example_usage()
    pass
