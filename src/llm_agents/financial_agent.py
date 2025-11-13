"""
Financial Statement Analyzer Agent.

This agent uses LLMs to analyze:
- Balance sheets
- Income statements
- Cash flow statements
- Key financial ratios
- Year-over-year trends

It extracts insights that traditional algorithms might miss.
"""

import json
from typing import Dict, Any
from datetime import datetime
from src.llm_agents.base_agent import BaseLLMAgent
from src.data_layer.models import LLMAnalysis
from src.data_layer.market_data import market_data
from src.utils.logger import app_logger


class FinancialAnalyzerAgent(BaseLLMAgent):
    """
    Analyzes financial statements using LLMs to extract insights.

    This agent can identify:
    - Revenue growth trends
    - Profitability metrics
    - Debt levels and sustainability
    - Cash flow quality
    - Red flags in financials
    """

    def __init__(self):
        super().__init__(agent_name="financial_analyzer")

    def analyze(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> LLMAnalysis:
        """
        Perform deep financial analysis on a company.

        Args:
            symbol: Stock ticker
            use_cache: Whether to use cached results

        Returns:
            LLMAnalysis with financial insights and score
        """
        start_time = datetime.utcnow()

        # Check cache first
        if use_cache:
            cached = self._get_cached_analysis(symbol)
            if cached:
                return LLMAnalysis(
                    symbol=symbol,
                    agent_type=self.agent_name,
                    analysis_result=cached,
                    tokens_used=0,
                    api_cost=0.0,
                    processing_time_ms=0,
                    cache_hit=True,
                )

        app_logger.info(f"Starting financial analysis for {symbol}")

        # Gather financial data
        company_info = market_data.get_company_info(symbol)
        if not company_info:
            raise ValueError(f"Could not fetch company info for {symbol}")

        # Build analysis prompt
        prompt = self._build_financial_prompt(symbol, company_info)

        # Call LLM (complex analysis - use better model)
        response, tokens, cost = self._call_llm(
            prompt=prompt,
            complexity="complex",
            max_tokens=1500
        )

        # Parse response into structured format
        analysis_result = self._parse_financial_response(response, company_info)

        # Cache the result
        self._cache_analysis(symbol, analysis_result)

        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        app_logger.info(
            f"Financial analysis complete for {symbol}: "
            f"Score={analysis_result.get('score', 0):.2f}, "
            f"Cost=${cost:.4f}"
        )

        return LLMAnalysis(
            symbol=symbol,
            agent_type=self.agent_name,
            analysis_result=analysis_result,
            tokens_used=tokens,
            api_cost=cost,
            processing_time_ms=processing_time,
            cache_hit=False,
        )

    def _build_financial_prompt(
        self,
        symbol: str,
        company_info: Dict[str, Any]
    ) -> str:
        """
        Build a comprehensive prompt for financial analysis.

        Args:
            symbol: Stock ticker
            company_info: Company fundamental data

        Returns:
            Formatted prompt string
        """
        return f"""Analyze the financial health of {symbol} ({company_info.get('name', 'Unknown')}).

**Company Information:**
- Sector: {company_info.get('sector', 'Unknown')}
- Industry: {company_info.get('industry', 'Unknown')}
- Market Cap: ${company_info.get('market_cap', 0):,.0f}

**Key Metrics:**
- P/E Ratio: {company_info.get('pe_ratio', 'N/A')}
- Forward P/E: {company_info.get('forward_pe', 'N/A')}
- PEG Ratio: {company_info.get('peg_ratio', 'N/A')}
- Price to Book: {company_info.get('price_to_book', 'N/A')}
- Dividend Yield: {company_info.get('dividend_yield', 0):.2%}

**Task:**
1. Evaluate the company's financial health (0-100 score)
2. Identify key strengths and weaknesses
3. Assess valuation (overvalued, fairly valued, undervalued)
4. Note any red flags or concerns
5. Provide a one-sentence investment thesis

**Output Format (JSON):**
{{
    "score": <0-100>,
    "valuation": "<overvalued|fairly_valued|undervalued>",
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...],
    "red_flags": ["flag1", "flag2", ...],
    "investment_thesis": "one sentence summary",
    "confidence": <0.0-1.0>
}}

Be concise and analytical. Focus on facts over speculation.
"""

    def _parse_financial_response(
        self,
        response: str,
        company_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.

        Args:
            response: Raw LLM response
            company_info: Original company data

        Returns:
            Structured analysis result
        """
        try:
            # Try to extract JSON from response
            # LLMs sometimes wrap JSON in markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            parsed = json.loads(json_str)

            # Normalize score to 0-1 range
            score = parsed.get("score", 50) / 100.0

            return {
                "score": score,
                "valuation": parsed.get("valuation", "unknown"),
                "strengths": parsed.get("strengths", []),
                "weaknesses": parsed.get("weaknesses", []),
                "red_flags": parsed.get("red_flags", []),
                "investment_thesis": parsed.get("investment_thesis", ""),
                "confidence": parsed.get("confidence", 0.5),
                "metrics": {
                    "pe_ratio": company_info.get("pe_ratio"),
                    "peg_ratio": company_info.get("peg_ratio"),
                    "market_cap": company_info.get("market_cap"),
                },
                "analysis_date": datetime.utcnow().isoformat(),
            }

        except json.JSONDecodeError as e:
            app_logger.error(f"Failed to parse financial analysis JSON: {e}")
            # Fallback: return basic structure
            return {
                "score": 0.5,
                "valuation": "unknown",
                "strengths": [],
                "weaknesses": [],
                "red_flags": ["Failed to parse LLM response"],
                "investment_thesis": response[:200],  # First 200 chars
                "confidence": 0.3,
                "raw_response": response,
            }
