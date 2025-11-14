"""
Sentiment Analysis Module
Extract sentiment from news articles using VADER and FinBERT

Learning objectives:
- Natural Language Processing basics
- Sentiment scoring techniques
- Handling financial text
- Aggregating sentiment signals
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import numpy as np

# TODO 1: Import VADER sentiment analyzer
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# TODO 2: Import FinBERT (optional - for advanced users)
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Data class for sentiment scores"""
    positive: float
    negative: float
    neutral: float
    compound: float
    method: str
    timestamp: datetime


class SentimentAnalyzer:
    """Main class for sentiment analysis"""

    def __init__(self, use_finbert: bool = False):
        # TODO 3: Initialize VADER sentiment analyzer
        self.vader = None

        # TODO 4: Initialize FinBERT if requested (optional)
        self.use_finbert = use_finbert
        self.finbert_tokenizer = None
        self.finbert_model = None

        # TODO 5: Define financial keywords and their weights
        self.bullish_keywords = []
        self.bearish_keywords = []
        self.keyword_weights = {}

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # TODO 6: Convert to lowercase
        text = None

        # TODO 7: Remove URLs
        # Pattern: http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+
        text = None

        # TODO 8: Remove special characters but keep sentence structure
        # Keep: letters, numbers, spaces, basic punctuation (.,!?)
        text = None

        # TODO 9: Remove extra whitespace
        text = None

        return text

    def analyze_vader(self, text: str) -> SentimentScore:
        """
        Analyze sentiment using VADER

        Args:
            text: Text to analyze

        Returns:
            SentimentScore object
        """
        # TODO 10: Preprocess text
        clean_text = None

        # TODO 11: Get VADER sentiment scores
        # Returns dict with 'pos', 'neg', 'neu', 'compound'
        scores = None

        # TODO 12: Create and return SentimentScore object
        return None

    def analyze_finbert(self, text: str) -> SentimentScore:
        """
        Analyze sentiment using FinBERT (financial BERT)

        Args:
            text: Text to analyze

        Returns:
            SentimentScore object
        """
        if not self.use_finbert:
            logger.warning("FinBERT not enabled")
            return None

        # TODO 13: Tokenize text for FinBERT
        # Max length: 512 tokens
        inputs = None

        # TODO 14: Get model predictions
        # No gradient calculation needed for inference
        outputs = None

        # TODO 15: Apply softmax to get probabilities
        probs = None

        # TODO 16: Extract positive, negative, neutral scores
        # FinBERT classes: [positive, negative, neutral]
        sentiment_score = None

        return sentiment_score

    def extract_financial_keywords(self, text: str) -> Dict[str, int]:
        """
        Extract and count financial keywords from text

        Args:
            text: Text to analyze

        Returns:
            Dictionary of keyword counts
        """
        # TODO 17: Convert text to lowercase
        text_lower = None

        # TODO 18: Count bullish keywords
        bullish_count = 0
        # Iterate through bullish_keywords and count occurrences

        # TODO 19: Count bearish keywords
        bearish_count = 0
        # Iterate through bearish_keywords and count occurrences

        # TODO 20: Return keyword counts
        return {
            "bullish": bullish_count,
            "bearish": bearish_count,
            "net_sentiment": bullish_count - bearish_count
        }

    def calculate_weighted_sentiment(
        self,
        text: str,
        vader_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> float:
        """
        Calculate weighted sentiment combining VADER and keywords

        Args:
            text: Text to analyze
            vader_weight: Weight for VADER score (0-1)
            keyword_weight: Weight for keyword score (0-1)

        Returns:
            Weighted sentiment score (-1 to 1)
        """
        # TODO 21: Get VADER sentiment
        vader_score = None

        # TODO 22: Get keyword sentiment
        keywords = None

        # TODO 23: Normalize keyword sentiment to -1 to 1 range
        # Assume max keywords per article is 10
        keyword_score = None

        # TODO 24: Calculate weighted average
        final_score = None

        return final_score

    def analyze_article(
        self,
        title: str,
        summary: str,
        title_weight: float = 0.6,
        summary_weight: float = 0.4
    ) -> SentimentScore:
        """
        Analyze sentiment of a news article

        Args:
            title: Article title
            summary: Article summary/description
            title_weight: Weight for title sentiment
            summary_weight: Weight for summary sentiment

        Returns:
            Combined SentimentScore
        """
        # TODO 25: Analyze title sentiment
        title_sentiment = None

        # TODO 26: Analyze summary sentiment
        summary_sentiment = None

        # TODO 27: Calculate weighted average of compound scores
        weighted_compound = None

        # TODO 28: Calculate weighted averages for pos, neg, neu
        weighted_pos = None
        weighted_neg = None
        weighted_neu = None

        # TODO 29: Create and return combined SentimentScore
        return None

    def batch_analyze(
        self,
        articles: List[Dict[str, str]]
    ) -> List[SentimentScore]:
        """
        Analyze sentiment for multiple articles

        Args:
            articles: List of dicts with 'title' and 'summary' keys

        Returns:
            List of SentimentScore objects
        """
        # TODO 30: Analyze each article
        scores = []
        for article in articles:
            # TODO 31: Get title and summary
            title = None
            summary = None

            # TODO 32: Analyze article
            score = None

            # TODO 33: Append to results
            pass

        return scores

    def aggregate_sentiment_by_symbol(
        self,
        articles_with_sentiment: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate sentiment scores by stock symbol

        Args:
            articles_with_sentiment: List of dicts with 'symbols' and 'sentiment' keys

        Returns:
            Dict mapping symbol to aggregated sentiment metrics
        """
        symbol_sentiments = {}

        # TODO 34: Group articles by symbol
        for article in articles_with_sentiment:
            symbols = None
            sentiment = None

            # TODO 35: Add sentiment to each symbol
            for symbol in symbols:
                # TODO 36: Initialize symbol entry if needed
                if symbol not in symbol_sentiments:
                    symbol_sentiments[symbol] = {
                        "scores": [],
                        "count": 0
                    }

                # TODO 37: Append sentiment score
                pass

        # TODO 38: Calculate aggregate statistics for each symbol
        aggregated = {}
        for symbol, data in symbol_sentiments.items():
            scores = data["scores"]

            # TODO 39: Calculate mean, median, std deviation
            aggregated[symbol] = {
                "mean_sentiment": None,
                "median_sentiment": None,
                "std_sentiment": None,
                "article_count": None,
                "bullish_ratio": None,  # Ratio of positive sentiment
            }

        return aggregated

    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment into categories

        Args:
            compound_score: Compound sentiment score (-1 to 1)

        Returns:
            Sentiment category: 'very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish'
        """
        # TODO 40: Classify based on thresholds
        # Very bullish: >= 0.5
        # Bullish: >= 0.1
        # Neutral: -0.1 to 0.1
        # Bearish: <= -0.1
        # Very bearish: <= -0.5
        pass


def main():
    """Example usage"""
    # Initialize analyzer
    analyzer = SentimentAnalyzer(use_finbert=False)

    # Sample news articles
    articles = [
        {
            "title": "Apple Reports Record Revenue, Beats Expectations",
            "summary": "Apple Inc. announced strong quarterly results with revenue exceeding analyst estimates. The company's performance was driven by robust iPhone sales and growing services revenue."
        },
        {
            "title": "Market Concerns Over Rising Interest Rates",
            "summary": "Investors worry as the Federal Reserve signals potential rate hikes. Markets experienced volatility with major indices posting losses."
        }
    ]

    # Analyze sentiments
    for article in articles:
        sentiment = analyzer.analyze_article(
            article["title"],
            article["summary"]
        )
        category = analyzer.classify_sentiment(sentiment.compound)

        print(f"\nTitle: {article['title']}")
        print(f"Sentiment: {category}")
        print(f"Compound Score: {sentiment.compound:.3f}")
        print(f"Positive: {sentiment.positive:.3f}")
        print(f"Negative: {sentiment.negative:.3f}")
        print(f"Neutral: {sentiment.neutral:.3f}")


if __name__ == "__main__":
    main()
