"""
Ensemble Trading Strategy.

This combines insights from multiple LLM agents:
- Financial Analyzer: Fundamental analysis
- Sentiment Analyzer: Market sentiment
- (Future) Earnings Analyzer: Earnings call insights
- (Future) Technical Analyzer: Chart patterns

The ensemble approach is more robust than relying on a single signal.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from src.llm_agents.financial_agent import FinancialAnalyzerAgent
from src.llm_agents.sentiment_agent import SentimentAnalyzerAgent
from src.data_layer.models import TradingSignal, SignalType, LLMAnalysis
from src.utils.logger import app_logger
from src.utils.config import settings
import mlflow


class EnsembleStrategy:
    """
    Combines multiple LLM agents to generate trading signals.

    Weights can be adjusted based on market conditions:
    - Bull market: Higher weight on sentiment
    - Bear market: Higher weight on fundamentals
    - Earnings season: Higher weight on earnings analysis
    """

    def __init__(
        self,
        fundamental_weight: float = 0.5,
        sentiment_weight: float = 0.3,
        technical_weight: float = 0.2,
    ):
        """
        Initialize ensemble strategy.

        Args:
            fundamental_weight: Weight for fundamental analysis (0-1)
            sentiment_weight: Weight for sentiment analysis (0-1)
            technical_weight: Weight for technical analysis (0-1)
        """
        # Normalize weights
        total_weight = fundamental_weight + sentiment_weight + technical_weight
        self.fundamental_weight = fundamental_weight / total_weight
        self.sentiment_weight = sentiment_weight / total_weight
        self.technical_weight = technical_weight / total_weight

        # Initialize agents
        self.financial_agent = FinancialAnalyzerAgent()
        self.sentiment_agent = SentimentAnalyzerAgent()

        app_logger.info(
            f"Initialized ensemble strategy: "
            f"fundamental={self.fundamental_weight:.2f}, "
            f"sentiment={self.sentiment_weight:.2f}, "
            f"technical={self.technical_weight:.2f}"
        )

    def generate_signal(
        self,
        symbol: str,
        use_cache: bool = True,
        track_mlflow: bool = True
    ) -> TradingSignal:
        """
        Generate a trading signal by combining multiple agents.

        Args:
            symbol: Stock ticker
            use_cache: Whether to use cached analysis
            track_mlflow: Whether to log to MLflow

        Returns:
            TradingSignal with buy/sell/hold recommendation
        """
        app_logger.info(f"Generating ensemble signal for {symbol}")

        if track_mlflow:
            mlflow.set_experiment(settings.mlflow_experiment_name)
            run = mlflow.start_run(run_name=f"{symbol}_{datetime.utcnow().isoformat()}")

        try:
            # Run all agents in parallel (could be optimized with asyncio)
            financial_analysis = self.financial_agent.analyze(symbol, use_cache=use_cache)
            sentiment_analysis = self.sentiment_agent.analyze(symbol, use_cache=use_cache)

            # Extract scores
            fundamental_score = financial_analysis.analysis_result.get("score", 0.5)
            sentiment_score = (
                sentiment_analysis.analysis_result.get("sentiment_score", 0.0) + 1.0
            ) / 2.0  # Convert from [-1, 1] to [0, 1]

            # For now, technical score is neutral (0.5)
            # In production, you'd implement technical analysis
            technical_score = 0.5

            # Calculate weighted conviction score
            ai_conviction_score = (
                self.fundamental_weight * fundamental_score +
                self.sentiment_weight * sentiment_score +
                self.technical_weight * technical_score
            )

            # Determine signal type based on conviction
            if ai_conviction_score >= 0.7:
                signal_type = SignalType.BUY
                confidence = ai_conviction_score
            elif ai_conviction_score <= 0.3:
                signal_type = SignalType.SELL
                confidence = 1.0 - ai_conviction_score
            else:
                signal_type = SignalType.HOLD
                confidence = 1.0 - abs(ai_conviction_score - 0.5) * 2

            # Build reasoning
            reasoning = self._build_reasoning(
                symbol,
                financial_analysis,
                sentiment_analysis,
                ai_conviction_score
            )

            # Calculate total cost
            total_cost = (
                financial_analysis.api_cost +
                sentiment_analysis.api_cost
            )

            # Log to MLflow
            if track_mlflow:
                mlflow.log_params({
                    "symbol": symbol,
                    "fundamental_weight": self.fundamental_weight,
                    "sentiment_weight": self.sentiment_weight,
                })
                mlflow.log_metrics({
                    "fundamental_score": fundamental_score,
                    "sentiment_score": sentiment_score,
                    "ai_conviction_score": ai_conviction_score,
                    "confidence": confidence,
                    "total_cost": total_cost,
                })
                mlflow.set_tag("signal_type", signal_type.value)

            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence_score=confidence,
                ai_conviction_score=ai_conviction_score,
                fundamental_score=fundamental_score,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                reasoning=reasoning,
                source_agent="ensemble_strategy",
                metadata={
                    "financial_analysis": financial_analysis.analysis_result,
                    "sentiment_analysis": sentiment_analysis.analysis_result,
                    "total_cost_usd": total_cost,
                    "tokens_used": (
                        financial_analysis.tokens_used +
                        sentiment_analysis.tokens_used
                    ),
                }
            )

            app_logger.info(
                f"Signal generated for {symbol}: {signal_type.value} "
                f"(conviction={ai_conviction_score:.2f}, cost=${total_cost:.4f})"
            )

            return signal

        finally:
            if track_mlflow:
                mlflow.end_run()

    def _build_reasoning(
        self,
        symbol: str,
        financial: LLMAnalysis,
        sentiment: LLMAnalysis,
        conviction: float
    ) -> str:
        """
        Build human-readable reasoning for the signal.

        Args:
            symbol: Stock ticker
            financial: Financial analysis
            sentiment: Sentiment analysis
            conviction: Overall conviction score

        Returns:
            Reasoning string
        """
        fin_result = financial.analysis_result
        sent_result = sentiment.analysis_result

        # Build reasoning components
        parts = [
            f"AI Conviction Score: {conviction:.2f}/1.00",
            "",
            "📊 Fundamental Analysis:",
            f"  - Score: {fin_result.get('score', 0):.2f}",
            f"  - Valuation: {fin_result.get('valuation', 'unknown')}",
            f"  - Thesis: {fin_result.get('investment_thesis', 'N/A')}",
        ]

        if fin_result.get('strengths'):
            parts.append(f"  - Strengths: {', '.join(fin_result['strengths'][:2])}")

        if fin_result.get('weaknesses'):
            parts.append(f"  - Weaknesses: {', '.join(fin_result['weaknesses'][:2])}")

        parts.extend([
            "",
            "📰 Sentiment Analysis:",
            f"  - Sentiment: {sent_result.get('sentiment_label', 'neutral')} ({sent_result.get('sentiment_score', 0):.2f})",
            f"  - Market Impact: {sent_result.get('market_impact', 'low')}",
            f"  - Summary: {sent_result.get('summary', 'N/A')}",
        ])

        return "\n".join(parts)

    def backtest(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Backtest the ensemble strategy (placeholder).

        In production, this would:
        1. Load historical data
        2. Generate signals for each day
        3. Simulate trades
        4. Calculate performance metrics

        Args:
            symbols: List of stocks to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Backtest results
        """
        app_logger.warning("Backtesting not fully implemented yet")
        return {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
