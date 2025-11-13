"""
News Sentiment Analyzer Agent.

This agent analyzes:
- News headlines and articles
- Social media sentiment (future)
- Market sentiment indicators
- News impact on stock price

Uses LLMs to understand nuanced language that traditional
sentiment analysis tools often miss.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from src.llm_agents.base_agent import BaseLLMAgent
from src.data_layer.models import LLMAnalysis, MarketNews
from src.data_layer.market_data import market_data
from src.utils.logger import app_logger


class SentimentAnalyzerAgent(BaseLLMAgent):
    """
    Analyzes market sentiment from news and social media.

    LLMs excel at:
    - Understanding context and nuance
    - Detecting sarcasm and sentiment shifts
    - Identifying hidden narratives
    - Weighing importance of different news items
    """

    def __init__(self):
        super().__init__(agent_name="sentiment_analyzer")

    def analyze(
        self,
        symbol: str,
        use_cache: bool = True,
        news_limit: int = 10
    ) -> LLMAnalysis:
        """
        Analyze sentiment for a stock based on recent news.

        Args:
            symbol: Stock ticker
            use_cache: Whether to use cached results
            news_limit: Number of news articles to analyze

        Returns:
            LLMAnalysis with sentiment score and insights
        """
        start_time = datetime.utcnow()

        # Check cache
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

        app_logger.info(f"Starting sentiment analysis for {symbol}")

        # Fetch recent news
        news_items = market_data.get_news(symbol, limit=news_limit)

        if not news_items:
            app_logger.warning(f"No news found for {symbol}")
            return self._create_neutral_analysis(symbol)

        # Build prompt with news
        prompt = self._build_sentiment_prompt(symbol, news_items)

        # Call LLM (simple task - use fast/cheap model)
        response, tokens, cost = self._call_llm(
            prompt=prompt,
            complexity="simple",
            max_tokens=1000
        )

        # Parse response
        analysis_result = self._parse_sentiment_response(response, news_items)

        # Cache result
        self._cache_analysis(symbol, analysis_result)

        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        app_logger.info(
            f"Sentiment analysis complete for {symbol}: "
            f"Sentiment={analysis_result.get('sentiment_score', 0):.2f}, "
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

    def _build_sentiment_prompt(
        self,
        symbol: str,
        news_items: List[MarketNews]
    ) -> str:
        """
        Build prompt for sentiment analysis.

        Args:
            symbol: Stock ticker
            news_items: List of news articles

        Returns:
            Formatted prompt
        """
        # Format news for the prompt
        news_text = "\n\n".join([
            f"**{i+1}. {item.headline}**\n"
            f"Source: {item.source} | Date: {item.published_at.strftime('%Y-%m-%d')}\n"
            f"Summary: {item.summary or 'N/A'}"
            for i, item in enumerate(news_items[:10])  # Limit to 10 to save tokens
        ])

        return f"""Analyze the sentiment of recent news about {symbol}.

**Recent News Articles:**
{news_text}

**Task:**
1. Determine overall sentiment (-1.0 to 1.0, where -1 is very negative, 0 is neutral, 1 is very positive)
2. Identify key themes in the news (positive and negative)
3. Assess the potential market impact (low, medium, high)
4. Note any significant events or catalysts
5. Provide a brief summary

**Output Format (JSON):**
{{
    "sentiment_score": <-1.0 to 1.0>,
    "sentiment_label": "<very_negative|negative|neutral|positive|very_positive>",
    "positive_themes": ["theme1", "theme2", ...],
    "negative_themes": ["theme1", "theme2", ...],
    "market_impact": "<low|medium|high>",
    "key_catalysts": ["catalyst1", "catalyst2", ...],
    "summary": "brief 1-2 sentence summary",
    "confidence": <0.0-1.0>
}}

Focus on factual analysis. Consider both explicit and implicit sentiment.
"""

    def _parse_sentiment_response(
        self,
        response: str,
        news_items: List[MarketNews]
    ) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.

        Args:
            response: Raw LLM response
            news_items: Original news items

        Returns:
            Structured analysis result
        """
        try:
            # Extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            parsed = json.loads(json_str)

            return {
                "sentiment_score": max(-1.0, min(1.0, parsed.get("sentiment_score", 0.0))),
                "sentiment_label": parsed.get("sentiment_label", "neutral"),
                "positive_themes": parsed.get("positive_themes", []),
                "negative_themes": parsed.get("negative_themes", []),
                "market_impact": parsed.get("market_impact", "low"),
                "key_catalysts": parsed.get("key_catalysts", []),
                "summary": parsed.get("summary", ""),
                "confidence": parsed.get("confidence", 0.5),
                "news_count": len(news_items),
                "analysis_date": datetime.utcnow().isoformat(),
            }

        except json.JSONDecodeError as e:
            app_logger.error(f"Failed to parse sentiment analysis JSON: {e}")
            # Fallback
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "positive_themes": [],
                "negative_themes": [],
                "market_impact": "low",
                "key_catalysts": [],
                "summary": response[:200],
                "confidence": 0.3,
                "raw_response": response,
            }

    def _create_neutral_analysis(self, symbol: str) -> LLMAnalysis:
        """
        Create a neutral analysis when no news is available.

        Args:
            symbol: Stock ticker

        Returns:
            Neutral LLMAnalysis
        """
        return LLMAnalysis(
            symbol=symbol,
            agent_type=self.agent_name,
            analysis_result={
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "positive_themes": [],
                "negative_themes": [],
                "market_impact": "low",
                "key_catalysts": [],
                "summary": "No recent news available for analysis",
                "confidence": 0.0,
                "news_count": 0,
            },
            tokens_used=0,
            api_cost=0.0,
            processing_time_ms=0,
            cache_hit=False,
        )
