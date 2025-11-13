"""
Unit tests for LLM agents.

Tests:
- Base agent functionality
- Financial analyzer
- Sentiment analyzer
- Ensemble strategy
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm_agents.base_agent import BaseLLMAgent
from src.llm_agents.financial_agent import FinancialAnalyzerAgent
from src.llm_agents.sentiment_agent import SentimentAnalyzerAgent
from src.llm_agents.ensemble_strategy import EnsembleStrategy
from src.data_layer.models import LLMAnalysis, SignalType


class TestBaseLLMAgent:
    """Test base LLM agent functionality."""

    def test_cost_estimation(self):
        """Test API cost estimation."""
        agent = FinancialAnalyzerAgent()

        # Groq cost (very cheap)
        groq_cost = agent._estimate_cost(1_000_000, "groq")
        assert groq_cost < 0.001  # Less than $0.001 per million tokens

        # Anthropic cost (more expensive)
        claude_cost = agent._estimate_cost(1_000_000, "anthropic_sonnet")
        assert claude_cost > 1.0  # More than $1 per million tokens

    @patch('src.llm_agents.base_agent.cache')
    def test_cache_usage(self, mock_cache):
        """Test that agents use cache."""
        agent = FinancialAnalyzerAgent()

        # Mock cache hit
        mock_cache.get.return_value = {"score": 0.8, "cached": True}

        cached_result = agent._get_cached_analysis("AAPL")

        assert cached_result is not None
        assert cached_result["cached"] is True

    @patch('src.llm_agents.base_agent.cache')
    def test_cache_miss(self, mock_cache):
        """Test cache miss returns None."""
        agent = FinancialAnalyzerAgent()

        # Mock cache miss
        mock_cache.get.return_value = None

        cached_result = agent._get_cached_analysis("AAPL")

        assert cached_result is None


class TestFinancialAnalyzerAgent:
    """Test financial analyzer agent."""

    @patch('src.llm_agents.financial_agent.market_data')
    @patch.object(FinancialAnalyzerAgent, '_call_llm')
    def test_financial_analysis(self, mock_llm, mock_market_data, sample_company_info):
        """Test financial analysis generation."""
        agent = FinancialAnalyzerAgent()

        # Mock company info
        mock_market_data.get_company_info.return_value = sample_company_info

        # Mock LLM response
        mock_llm.return_value = (
            '{"score": 75, "valuation": "fairly_valued", "strengths": ["Good margins"], "weaknesses": [], "red_flags": [], "investment_thesis": "Strong company", "confidence": 0.8}',
            1500,
            0.0045
        )

        # Analyze
        result = agent.analyze("AAPL", use_cache=False)

        assert isinstance(result, LLMAnalysis)
        assert result.symbol == "AAPL"
        assert result.agent_type == "financial_analyzer"
        assert result.tokens_used == 1500
        assert result.api_cost > 0

    def test_prompt_building(self, sample_company_info):
        """Test financial analysis prompt construction."""
        agent = FinancialAnalyzerAgent()

        prompt = agent._build_financial_prompt("AAPL", sample_company_info)

        # Should contain key information
        assert "AAPL" in prompt
        assert "Apple Inc." in prompt
        assert "Technology" in prompt
        assert "P/E Ratio" in prompt

    def test_response_parsing(self, sample_company_info, mock_llm_response):
        """Test parsing LLM response."""
        agent = FinancialAnalyzerAgent()

        import json
        response = json.dumps(mock_llm_response)

        parsed = agent._parse_financial_response(response, sample_company_info)

        assert "score" in parsed
        assert 0 <= parsed["score"] <= 1  # Normalized to 0-1
        assert "valuation" in parsed
        assert "strengths" in parsed

    def test_response_parsing_with_markdown(self, sample_company_info):
        """Test parsing response wrapped in markdown."""
        agent = FinancialAnalyzerAgent()

        # Response wrapped in markdown code blocks
        response = '''```json
{
    "score": 80,
    "valuation": "fairly_valued",
    "strengths": ["Test"],
    "weaknesses": [],
    "red_flags": [],
    "investment_thesis": "Good",
    "confidence": 0.85
}
```'''

        parsed = agent._parse_financial_response(response, sample_company_info)

        assert parsed["score"] == 0.80  # Normalized


class TestSentimentAnalyzerAgent:
    """Test sentiment analyzer agent."""

    @patch('src.llm_agents.sentiment_agent.market_data')
    @patch.object(SentimentAnalyzerAgent, '_call_llm')
    def test_sentiment_analysis(self, mock_llm, mock_market_data, sample_news):
        """Test sentiment analysis generation."""
        agent = SentimentAnalyzerAgent()

        # Mock news
        mock_market_data.get_news.return_value = sample_news

        # Mock LLM response
        mock_llm.return_value = (
            '{"sentiment_score": 0.6, "sentiment_label": "positive", "positive_themes": ["growth"], "negative_themes": [], "market_impact": "medium", "key_catalysts": ["earnings"], "summary": "Positive news", "confidence": 0.75}',
            800,
            0.0001
        )

        # Analyze
        result = agent.analyze("AAPL", use_cache=False)

        assert isinstance(result, LLMAnalysis)
        assert result.symbol == "AAPL"
        assert result.agent_type == "sentiment_analyzer"

    @patch('src.llm_agents.sentiment_agent.market_data')
    def test_sentiment_analysis_no_news(self, mock_market_data):
        """Test sentiment analysis with no news."""
        agent = SentimentAnalyzerAgent()

        # Mock no news
        mock_market_data.get_news.return_value = []

        # Should return neutral analysis
        result = agent.analyze("AAPL", use_cache=False)

        assert result.analysis_result["sentiment_score"] == 0.0
        assert result.analysis_result["sentiment_label"] == "neutral"

    def test_prompt_building(self, sample_news):
        """Test sentiment analysis prompt construction."""
        agent = SentimentAnalyzerAgent()

        prompt = agent._build_sentiment_prompt("AAPL", sample_news)

        # Should contain news headlines
        assert "Apple announces record earnings" in prompt
        assert "Apple stock upgraded" in prompt
        assert "AAPL" in prompt

    def test_sentiment_score_clamping(self):
        """Test sentiment score is clamped to [-1, 1]."""
        agent = SentimentAnalyzerAgent()

        # Response with out-of-range sentiment
        response = '{"sentiment_score": 2.0, "sentiment_label": "positive", "positive_themes": [], "negative_themes": [], "market_impact": "low", "key_catalysts": [], "summary": "", "confidence": 0.5}'

        import json
        parsed = agent._parse_sentiment_response(response, [])

        # Should be clamped to 1.0
        assert parsed["sentiment_score"] <= 1.0
        assert parsed["sentiment_score"] >= -1.0


class TestEnsembleStrategy:
    """Test ensemble strategy that combines agents."""

    def test_ensemble_initialization(self):
        """Test ensemble strategy initialization."""
        strategy = EnsembleStrategy(
            fundamental_weight=0.5,
            sentiment_weight=0.3,
            technical_weight=0.2,
        )

        # Weights should be normalized
        total = strategy.fundamental_weight + strategy.sentiment_weight + strategy.technical_weight
        assert abs(total - 1.0) < 0.001

    @patch('src.llm_agents.ensemble_strategy.mlflow')
    @patch.object(FinancialAnalyzerAgent, 'analyze')
    @patch.object(SentimentAnalyzerAgent, 'analyze')
    def test_signal_generation_buy(self, mock_sentiment, mock_financial, mock_mlflow):
        """Test generating BUY signal."""
        strategy = EnsembleStrategy()

        # Mock strong fundamental and sentiment scores
        mock_financial.return_value = LLMAnalysis(
            symbol="AAPL",
            agent_type="financial",
            analysis_result={"score": 0.85},  # High fundamental score
            tokens_used=1000,
            api_cost=0.003,
            processing_time_ms=500,
        )

        mock_sentiment.return_value = LLMAnalysis(
            symbol="AAPL",
            agent_type="sentiment",
            analysis_result={"sentiment_score": 0.6},  # Positive sentiment
            tokens_used=500,
            api_cost=0.0001,
            processing_time_ms=200,
        )

        # Mock MLflow
        mock_mlflow.start_run.return_value = MagicMock()

        # Generate signal
        signal = strategy.generate_signal("AAPL", use_cache=False, track_mlflow=False)

        # Should generate BUY signal with high conviction
        assert signal.signal_type == SignalType.BUY
        assert signal.ai_conviction_score > 0.7

    @patch('src.llm_agents.ensemble_strategy.mlflow')
    @patch.object(FinancialAnalyzerAgent, 'analyze')
    @patch.object(SentimentAnalyzerAgent, 'analyze')
    def test_signal_generation_sell(self, mock_sentiment, mock_financial, mock_mlflow):
        """Test generating SELL signal."""
        strategy = EnsembleStrategy()

        # Mock weak fundamental and sentiment scores
        mock_financial.return_value = LLMAnalysis(
            symbol="AAPL",
            agent_type="financial",
            analysis_result={"score": 0.2},  # Low fundamental score
            tokens_used=1000,
            api_cost=0.003,
            processing_time_ms=500,
        )

        mock_sentiment.return_value = LLMAnalysis(
            symbol="AAPL",
            agent_type="sentiment",
            analysis_result={"sentiment_score": -0.6},  # Negative sentiment
            tokens_used=500,
            api_cost=0.0001,
            processing_time_ms=200,
        )

        # Mock MLflow
        mock_mlflow.start_run.return_value = MagicMock()

        # Generate signal
        signal = strategy.generate_signal("AAPL", use_cache=False, track_mlflow=False)

        # Should generate SELL signal
        assert signal.signal_type == SignalType.SELL

    @patch('src.llm_agents.ensemble_strategy.mlflow')
    @patch.object(FinancialAnalyzerAgent, 'analyze')
    @patch.object(SentimentAnalyzerAgent, 'analyze')
    def test_signal_generation_hold(self, mock_sentiment, mock_financial, mock_mlflow):
        """Test generating HOLD signal."""
        strategy = EnsembleStrategy()

        # Mock neutral scores
        mock_financial.return_value = LLMAnalysis(
            symbol="AAPL",
            agent_type="financial",
            analysis_result={"score": 0.5},  # Neutral
            tokens_used=1000,
            api_cost=0.003,
            processing_time_ms=500,
        )

        mock_sentiment.return_value = LLMAnalysis(
            symbol="AAPL",
            agent_type="sentiment",
            analysis_result={"sentiment_score": 0.0},  # Neutral
            tokens_used=500,
            api_cost=0.0001,
            processing_time_ms=200,
        )

        # Mock MLflow
        mock_mlflow.start_run.return_value = MagicMock()

        # Generate signal
        signal = strategy.generate_signal("AAPL", use_cache=False, track_mlflow=False)

        # Should generate HOLD signal
        assert signal.signal_type == SignalType.HOLD

    @patch.object(FinancialAnalyzerAgent, 'analyze')
    @patch.object(SentimentAnalyzerAgent, 'analyze')
    def test_reasoning_building(self, mock_sentiment, mock_financial):
        """Test reasoning text construction."""
        strategy = EnsembleStrategy()

        financial_analysis = LLMAnalysis(
            symbol="AAPL",
            agent_type="financial",
            analysis_result={
                "score": 0.85,
                "valuation": "fairly_valued",
                "investment_thesis": "Strong growth",
                "strengths": ["Revenue", "Margins"],
                "weaknesses": ["Competition"],
            },
            tokens_used=1000,
            api_cost=0.003,
            processing_time_ms=500,
        )

        sentiment_analysis = LLMAnalysis(
            symbol="AAPL",
            agent_type="sentiment",
            analysis_result={
                "sentiment_score": 0.6,
                "sentiment_label": "positive",
                "market_impact": "high",
                "summary": "Positive news coverage",
            },
            tokens_used=500,
            api_cost=0.0001,
            processing_time_ms=200,
        )

        reasoning = strategy._build_reasoning(
            "AAPL",
            financial_analysis,
            sentiment_analysis,
            0.78
        )

        # Should contain key information
        assert "AAPL" in reasoning or "0.78" in reasoning
        assert "fairly_valued" in reasoning or "Fundamental" in reasoning
        assert "positive" in reasoning or "Sentiment" in reasoning
