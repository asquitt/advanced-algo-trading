"""
News Integration Module
Fetch and parse news from multiple sources (Alpha Vantage, NewsAPI)

Learning objectives:
- API integration and rate limiting
- Data parsing and normalization
- Error handling and retries
- Caching strategies
"""

import requests
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSource(Enum):
    """Enumeration of supported news sources"""
    ALPHA_VANTAGE = "alpha_vantage"
    NEWS_API = "news_api"
    FINNHUB = "finnhub"


@dataclass
class NewsArticle:
    """Data class representing a news article"""
    title: str
    source: str
    published_at: datetime
    url: str
    summary: str
    symbols: List[str]
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None


class NewsIntegration:
    """Main class for integrating multiple news sources"""

    def __init__(self, alpha_vantage_key: str = None, newsapi_key: str = None):
        # TODO 1: Initialize API keys and base URLs
        self.alpha_vantage_key = None
        self.newsapi_key = None
        self.alpha_vantage_base_url = None
        self.newsapi_base_url = None

        # TODO 2: Initialize rate limiting parameters
        self.requests_per_minute = None
        self.last_request_time = None

        # TODO 3: Initialize cache for storing recent articles
        self.cache = {}
        self.cache_ttl = None  # Time-to-live in seconds

    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        # TODO 4: Implement rate limiting logic
        # Calculate time since last request
        # If requests are too frequent, sleep to enforce rate limit
        pass

    def _cache_key(self, source: str, symbols: List[str], hours: int) -> str:
        """Generate cache key for news queries"""
        # TODO 5: Create a unique cache key based on source, symbols, and time range
        # Example: "alpha_vantage_AAPL_GOOGL_24"
        pass

    def _get_from_cache(self, key: str) -> Optional[List[NewsArticle]]:
        """Retrieve articles from cache if available and not expired"""
        # TODO 6: Check if key exists in cache
        # TODO 7: Check if cached data is still valid (not expired)
        # TODO 8: Return cached articles or None
        pass

    def _save_to_cache(self, key: str, articles: List[NewsArticle]):
        """Save articles to cache with timestamp"""
        # TODO 9: Store articles in cache with current timestamp
        pass

    def fetch_alpha_vantage_news(
        self,
        symbols: List[str],
        hours: int = 24
    ) -> List[NewsArticle]:
        """
        Fetch news from Alpha Vantage News & Sentiment API

        Args:
            symbols: List of stock symbols
            hours: Number of hours to look back

        Returns:
            List of NewsArticle objects
        """
        # TODO 10: Check cache first
        cache_key = None

        # TODO 11: Build the API request URL
        # API endpoint: https://www.alphavantage.co/query
        # Function: NEWS_SENTIMENT
        # Parameters: tickers, apikey, time_from, time_to
        url = None

        try:
            # TODO 12: Enforce rate limiting before making request
            pass

            # TODO 13: Make HTTP request with timeout
            response = None

            # TODO 14: Check response status code and handle errors
            pass

            # TODO 15: Parse JSON response
            data = None

            # TODO 16: Extract articles from response
            # Alpha Vantage returns articles in 'feed' key
            raw_articles = None

            # TODO 17: Convert raw articles to NewsArticle objects
            articles = []
            for raw in raw_articles:
                # TODO 18: Parse each article field
                # Extract: title, source, time_published, url, summary
                # Extract ticker sentiment if available
                pass

            # TODO 19: Save articles to cache
            pass

            # TODO 20: Return articles
            return articles

        except requests.exceptions.Timeout:
            logger.error("Request timeout fetching Alpha Vantage news")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in fetch_alpha_vantage_news: {e}")
            return []

    def fetch_newsapi_news(
        self,
        symbols: List[str],
        hours: int = 24,
        language: str = "en"
    ) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI

        Args:
            symbols: List of stock symbols (used as keywords)
            hours: Number of hours to look back
            language: Language code (default: en)

        Returns:
            List of NewsArticle objects
        """
        # TODO 21: Check cache first
        pass

        # TODO 22: Build search query from symbols
        # Example: "AAPL OR GOOGL OR MSFT"
        query = None

        # TODO 23: Calculate time range (from_date and to_date)
        from_date = None
        to_date = None

        # TODO 24: Build API request URL
        # API endpoint: https://newsapi.org/v2/everything
        # Parameters: q, from, to, language, apiKey, sortBy
        url = None

        try:
            # TODO 25: Enforce rate limiting
            pass

            # TODO 26: Make HTTP request
            response = None

            # TODO 27: Check status and handle errors
            pass

            # TODO 28: Parse response JSON
            data = None
            raw_articles = None

            # TODO 29: Convert to NewsArticle objects
            articles = []
            for raw in raw_articles:
                # Parse: title, source.name, publishedAt, url, description
                pass

            # TODO 30: Save to cache and return
            return articles

        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            return []

    def fetch_all_sources(
        self,
        symbols: List[str],
        hours: int = 24
    ) -> List[NewsArticle]:
        """
        Fetch news from all available sources and combine results

        Args:
            symbols: List of stock symbols
            hours: Number of hours to look back

        Returns:
            Combined and deduplicated list of NewsArticle objects
        """
        all_articles = []

        # TODO 31: Fetch from Alpha Vantage if API key is available
        if self.alpha_vantage_key:
            pass

        # TODO 32: Fetch from NewsAPI if API key is available
        if self.newsapi_key:
            pass

        # TODO 33: Remove duplicate articles based on URL
        # Use a set to track seen URLs
        unique_articles = []

        # TODO 34: Sort articles by published date (newest first)
        pass

        return unique_articles

    def filter_by_relevance(
        self,
        articles: List[NewsArticle],
        min_relevance: float = 0.5
    ) -> List[NewsArticle]:
        """
        Filter articles by relevance score

        Args:
            articles: List of NewsArticle objects
            min_relevance: Minimum relevance score (0-1)

        Returns:
            Filtered list of articles
        """
        # TODO 35: Filter articles where relevance_score >= min_relevance
        pass

    def filter_by_keywords(
        self,
        articles: List[NewsArticle],
        keywords: List[str],
        case_sensitive: bool = False
    ) -> List[NewsArticle]:
        """
        Filter articles containing specific keywords

        Args:
            articles: List of NewsArticle objects
            keywords: List of keywords to search for
            case_sensitive: Whether search should be case-sensitive

        Returns:
            Filtered list of articles
        """
        # TODO 36: Filter articles that contain any of the keywords
        # Search in title and summary
        # Handle case sensitivity
        pass

    def get_article_count_by_symbol(
        self,
        articles: List[NewsArticle]
    ) -> Dict[str, int]:
        """
        Count articles mentioning each symbol

        Args:
            articles: List of NewsArticle objects

        Returns:
            Dictionary mapping symbol to article count
        """
        # TODO 37: Count articles for each symbol
        # Return dict like {"AAPL": 15, "GOOGL": 8}
        pass

    def get_recent_news_summary(
        self,
        symbols: List[str],
        hours: int = 24,
        max_articles: int = 10
    ) -> Dict[str, any]:
        """
        Get a summary of recent news for symbols

        Args:
            symbols: List of stock symbols
            hours: Number of hours to look back
            max_articles: Maximum number of articles to include

        Returns:
            Dictionary with summary statistics and top articles
        """
        # TODO 38: Fetch articles from all sources
        articles = []

        # TODO 39: Calculate summary statistics
        summary = {
            "total_articles": 0,
            "symbols": symbols,
            "time_range_hours": hours,
            "articles_by_symbol": {},
            "top_articles": []
        }

        # TODO 40: Return summary
        return summary


def main():
    """Example usage"""
    # Initialize with API keys (use environment variables in production)
    news = NewsIntegration(
        alpha_vantage_key="YOUR_ALPHA_VANTAGE_KEY",
        newsapi_key="YOUR_NEWSAPI_KEY"
    )

    # Fetch news for specific symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    articles = news.fetch_all_sources(symbols, hours=24)

    print(f"Found {len(articles)} articles")
    for article in articles[:5]:
        print(f"\n{article.title}")
        print(f"Source: {article.source}")
        print(f"Published: {article.published_at}")
        print(f"Symbols: {', '.join(article.symbols)}")

    # Get summary
    summary = news.get_recent_news_summary(symbols, hours=24)
    print(f"\nSummary: {json.dumps(summary, indent=2, default=str)}")


if __name__ == "__main__":
    main()
