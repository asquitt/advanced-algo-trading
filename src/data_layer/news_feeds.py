"""
News Feed Integration

Aggregates news from multiple sources for sentiment analysis and trading signals.

Supported sources:
- Alpha Vantage News API
- NewsAPI
- Reddit (via PRAW)
- Twitter/X (via API)

Author: LLM Trading Platform
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import asyncio
from loguru import logger
import aiohttp


@dataclass
class NewsArticle:
    """Data class for news articles."""
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str]
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    author: Optional[str] = None
    category: Optional[str] = None


class AlphaVantageNewsFeed:
    """
    Fetches news from Alpha Vantage News API.

    API: https://www.alphavantage.co/documentation/#news-sentiment
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage news feed.

        Args:
            api_key: Alpha Vantage API key (or from env var)
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"

        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")

        logger.info("AlphaVantageNewsFeed initialized")

    async def fetch_news(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch news articles from Alpha Vantage.

        Args:
            tickers: List of stock tickers to filter
            topics: List of topics (e.g., 'earnings', 'ipo', 'technology')
            time_from: Start time for articles
            time_to: End time for articles
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        if not self.api_key:
            logger.error("Cannot fetch news without API key")
            return []

        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.api_key,
            'limit': limit
        }

        if tickers:
            params['tickers'] = ','.join(tickers)

        if topics:
            params['topics'] = ','.join(topics)

        if time_from:
            params['time_from'] = time_from.strftime('%Y%m%dT%H%M')

        if time_to:
            params['time_to'] = time_to.strftime('%Y%m%dT%H%M')

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Alpha Vantage API error: {response.status}")
                        return []

                    data = await response.json()

                    if 'feed' not in data:
                        logger.warning("No news feed in Alpha Vantage response")
                        return []

                    articles = []
                    for item in data['feed']:
                        article = NewsArticle(
                            title=item.get('title', ''),
                            summary=item.get('summary', ''),
                            source=item.get('source', 'Alpha Vantage'),
                            url=item.get('url', ''),
                            published_at=datetime.strptime(
                                item.get('time_published', ''),
                                '%Y%m%dT%H%M%S'
                            ),
                            symbols=[t['ticker'] for t in item.get('ticker_sentiment', [])],
                            sentiment_score=self._parse_sentiment(item),
                            relevance_score=self._parse_relevance(item),
                            author=item.get('authors', [None])[0] if item.get('authors') else None,
                            category=item.get('category_within_source', None)
                        )
                        articles.append(article)

                    logger.info(f"Fetched {len(articles)} articles from Alpha Vantage")
                    return articles

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []

    def _parse_sentiment(self, item: Dict) -> Optional[float]:
        """Parse overall sentiment score from article."""
        sentiment_score = item.get('overall_sentiment_score')
        if sentiment_score is not None:
            return float(sentiment_score)
        return None

    def _parse_relevance(self, item: Dict) -> Optional[float]:
        """Parse average relevance score from tickers."""
        ticker_sentiments = item.get('ticker_sentiment', [])
        if ticker_sentiments:
            relevance_scores = [
                float(t.get('relevance_score', 0))
                for t in ticker_sentiments
            ]
            return sum(relevance_scores) / len(relevance_scores)
        return None


class NewsAPIFeed:
    """
    Fetches news from NewsAPI.org.

    API: https://newsapi.org/docs
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI feed.

        Args:
            api_key: NewsAPI key (or from env var)
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"

        if not self.api_key:
            logger.warning("NewsAPI key not provided")

        logger.info("NewsAPIFeed initialized")

    async def fetch_news(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page_size: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch news articles from NewsAPI.

        Args:
            query: Search query (e.g., 'Tesla OR TSLA')
            from_date: Start date for articles
            to_date: End date for articles
            language: Language code (default: 'en')
            sort_by: Sort order ('publishedAt', 'relevancy', 'popularity')
            page_size: Number of articles per page

        Returns:
            List of NewsArticle objects
        """
        if not self.api_key:
            logger.error("Cannot fetch news without API key")
            return []

        url = f"{self.base_url}/everything"

        params = {
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size,
            'apiKey': self.api_key
        }

        if from_date:
            params['from'] = from_date.strftime('%Y-%m-%d')

        if to_date:
            params['to'] = to_date.strftime('%Y-%m-%d')

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"NewsAPI error: {response.status}")
                        return []

                    data = await response.json()

                    if data.get('status') != 'ok':
                        logger.error(f"NewsAPI returned error: {data.get('message')}")
                        return []

                    articles = []
                    for item in data.get('articles', []):
                        article = NewsArticle(
                            title=item.get('title', ''),
                            summary=item.get('description', ''),
                            source=item.get('source', {}).get('name', 'NewsAPI'),
                            url=item.get('url', ''),
                            published_at=datetime.fromisoformat(
                                item.get('publishedAt', '').replace('Z', '+00:00')
                            ),
                            symbols=[],  # NewsAPI doesn't provide ticker symbols
                            author=item.get('author'),
                            category=None
                        )
                        articles.append(article)

                    logger.info(f"Fetched {len(articles)} articles from NewsAPI")
                    return articles

        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            return []


class NewsFeedAggregator:
    """
    Aggregates news from multiple sources.

    Provides unified interface for fetching and analyzing news.
    """

    def __init__(
        self,
        use_alpha_vantage: bool = True,
        use_newsapi: bool = True,
        alpha_vantage_key: Optional[str] = None,
        newsapi_key: Optional[str] = None
    ):
        """
        Initialize news aggregator.

        Args:
            use_alpha_vantage: Enable Alpha Vantage news
            use_newsapi: Enable NewsAPI
            alpha_vantage_key: Alpha Vantage API key
            newsapi_key: NewsAPI key
        """
        self.feeds = []

        if use_alpha_vantage:
            self.feeds.append(AlphaVantageNewsFeed(alpha_vantage_key))

        if use_newsapi:
            self.feeds.append(NewsAPIFeed(newsapi_key))

        logger.info(f"NewsFeedAggregator initialized with {len(self.feeds)} feeds")

    async def fetch_all_news(
        self,
        symbols: Optional[List[str]] = None,
        keywords: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit_per_source: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch news from all sources.

        Args:
            symbols: List of stock symbols
            keywords: Search keywords
            from_time: Start time
            to_time: End time
            limit_per_source: Maximum articles per source

        Returns:
            Aggregated list of NewsArticle objects
        """
        tasks = []

        for feed in self.feeds:
            if isinstance(feed, AlphaVantageNewsFeed):
                task = feed.fetch_news(
                    tickers=symbols,
                    time_from=from_time,
                    time_to=to_time,
                    limit=limit_per_source
                )
                tasks.append(task)

            elif isinstance(feed, NewsAPIFeed):
                query = ' OR '.join(symbols) if symbols else keywords
                if not query:
                    continue

                task = feed.fetch_news(
                    query=query,
                    from_date=from_time,
                    to_date=to_time,
                    page_size=limit_per_source
                )
                tasks.append(task)

        # Fetch all news in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Feed error: {result}")

        # Remove duplicates (by URL)
        unique_articles = {article.url: article for article in all_articles}
        all_articles = list(unique_articles.values())

        # Sort by published date (most recent first)
        all_articles.sort(key=lambda x: x.published_at, reverse=True)

        logger.info(f"Aggregated {len(all_articles)} unique articles")

        return all_articles

    def filter_by_symbols(
        self,
        articles: List[NewsArticle],
        symbols: List[str]
    ) -> List[NewsArticle]:
        """
        Filter articles by stock symbols.

        Args:
            articles: List of articles
            symbols: List of symbols to filter

        Returns:
            Filtered articles
        """
        filtered = [
            article for article in articles
            if any(symbol in article.symbols for symbol in symbols)
        ]

        logger.debug(f"Filtered to {len(filtered)} articles for symbols {symbols}")

        return filtered

    def filter_by_recency(
        self,
        articles: List[NewsArticle],
        hours: int = 24
    ) -> List[NewsArticle]:
        """
        Filter articles by recency.

        Args:
            articles: List of articles
            hours: Maximum age in hours

        Returns:
            Recent articles
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        recent = [
            article for article in articles
            if article.published_at >= cutoff
        ]

        logger.debug(f"Filtered to {len(recent)} articles from last {hours} hours")

        return recent

    def aggregate_sentiment(
        self,
        articles: List[NewsArticle],
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregate sentiment scores from articles.

        Args:
            articles: List of articles
            symbol: Optional symbol to filter

        Returns:
            Dictionary with aggregated sentiment metrics
        """
        if symbol:
            articles = [a for a in articles if symbol in a.symbols]

        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]

        if not sentiments:
            return {
                'avg_sentiment': 0.0,
                'sentiment_std': 0.0,
                'sentiment_trend': 0.0,
                'num_articles': 0
            }

        import numpy as np

        return {
            'avg_sentiment': float(np.mean(sentiments)),
            'sentiment_std': float(np.std(sentiments)),
            'sentiment_trend': float(np.polyfit(range(len(sentiments)), sentiments, 1)[0]),
            'num_articles': len(articles),
            'positive_count': sum(1 for s in sentiments if s > 0.2),
            'negative_count': sum(1 for s in sentiments if s < -0.2),
            'neutral_count': sum(1 for s in sentiments if -0.2 <= s <= 0.2)
        }
