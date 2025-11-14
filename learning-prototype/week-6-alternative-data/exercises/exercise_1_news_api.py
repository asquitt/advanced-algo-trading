"""
Exercise 1: News API Integration
Fetch and parse news from Alpha Vantage and NewsAPI

Tasks:
1. Complete the TODOs in news_integration.py
2. Test fetching news from Alpha Vantage
3. Test fetching news from NewsAPI
4. Implement caching mechanism
5. Handle rate limiting

Expected Output:
- Successfully fetch news articles for given symbols
- Display article titles, sources, and timestamps
- Verify caching is working
- Handle API errors gracefully
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starter_code.news_integration import NewsIntegration, NewsArticle
from datetime import datetime, timedelta
import json


def test_alpha_vantage():
    """Test Alpha Vantage news fetching"""
    print("=" * 80)
    print("Testing Alpha Vantage News API")
    print("=" * 80)

    # TODO: Initialize NewsIntegration with your Alpha Vantage API key
    # Get free API key from: https://www.alphavantage.co/support/#api-key
    news = NewsIntegration(
        alpha_vantage_key="YOUR_API_KEY_HERE"
    )

    # TODO: Fetch news for tech stocks
    symbols = ["AAPL", "GOOGL", "MSFT"]
    articles = news.fetch_alpha_vantage_news(symbols, hours=24)

    # TODO: Display results
    print(f"\nFetched {len(articles)} articles")
    for i, article in enumerate(articles[:5], 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Published: {article.published_at}")
        print(f"   Symbols: {', '.join(article.symbols)}")
        print(f"   URL: {article.url}")

    # TODO: Test caching - second call should be faster
    print("\n" + "-" * 80)
    print("Testing cache (second call should be instant)...")
    import time
    start = time.time()
    cached_articles = news.fetch_alpha_vantage_news(symbols, hours=24)
    elapsed = time.time() - start
    print(f"Fetched {len(cached_articles)} articles in {elapsed:.3f} seconds")


def test_newsapi():
    """Test NewsAPI fetching"""
    print("\n" + "=" * 80)
    print("Testing NewsAPI")
    print("=" * 80)

    # TODO: Initialize NewsIntegration with your NewsAPI key
    # Get free API key from: https://newsapi.org/register
    news = NewsIntegration(
        newsapi_key="YOUR_API_KEY_HERE"
    )

    # TODO: Fetch news for different symbols
    symbols = ["Tesla", "Bitcoin", "AI"]
    articles = news.fetch_newsapi_news(symbols, hours=48)

    # TODO: Display results
    print(f"\nFetched {len(articles)} articles")
    for i, article in enumerate(articles[:5], 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Published: {article.published_at}")
        print(f"   Summary: {article.summary[:100]}...")


def test_combined_sources():
    """Test fetching from all sources"""
    print("\n" + "=" * 80)
    print("Testing Combined News Sources")
    print("=" * 80)

    # TODO: Initialize with both API keys
    news = NewsIntegration(
        alpha_vantage_key="YOUR_ALPHA_VANTAGE_KEY",
        newsapi_key="YOUR_NEWSAPI_KEY"
    )

    # TODO: Fetch from all sources
    symbols = ["AAPL", "TSLA"]
    articles = news.fetch_all_sources(symbols, hours=24)

    # TODO: Display combined results
    print(f"\nFetched {len(articles)} total articles from all sources")

    # TODO: Show source distribution
    source_counts = {}
    for article in articles:
        source_counts[article.source] = source_counts.get(article.source, 0) + 1

    print("\nArticles by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")


def test_filtering():
    """Test article filtering"""
    print("\n" + "=" * 80)
    print("Testing Article Filtering")
    print("=" * 80)

    news = NewsIntegration(
        alpha_vantage_key="YOUR_ALPHA_VANTAGE_KEY"
    )

    # TODO: Fetch articles
    articles = news.fetch_all_sources(["AAPL", "GOOGL"], hours=48)

    # TODO: Filter by keywords
    keywords = ["revenue", "earnings", "profit"]
    filtered = news.filter_by_keywords(articles, keywords)
    print(f"\nFiltered to {len(filtered)} articles containing: {', '.join(keywords)}")

    # TODO: Get article count by symbol
    counts = news.get_article_count_by_symbol(articles)
    print("\nArticles per symbol:")
    for symbol, count in counts.items():
        print(f"  {symbol}: {count}")


def test_summary():
    """Test news summary generation"""
    print("\n" + "=" * 80)
    print("Testing News Summary")
    print("=" * 80)

    news = NewsIntegration(
        alpha_vantage_key="YOUR_ALPHA_VANTAGE_KEY",
        newsapi_key="YOUR_NEWSAPI_KEY"
    )

    # TODO: Generate summary
    symbols = ["AAPL", "MSFT", "GOOGL"]
    summary = news.get_recent_news_summary(symbols, hours=24, max_articles=10)

    # TODO: Display summary
    print(json.dumps(summary, indent=2, default=str))


def test_error_handling():
    """Test error handling"""
    print("\n" + "=" * 80)
    print("Testing Error Handling")
    print("=" * 80)

    # TODO: Test with invalid API key
    news = NewsIntegration(alpha_vantage_key="INVALID_KEY")
    articles = news.fetch_alpha_vantage_news(["AAPL"], hours=24)
    print(f"Result with invalid key: {len(articles)} articles (should be 0)")

    # TODO: Test with empty symbols list
    news = NewsIntegration(alpha_vantage_key="YOUR_KEY")
    articles = news.fetch_all_sources([], hours=24)
    print(f"Result with empty symbols: {len(articles)} articles (should be 0)")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("EXERCISE 1: NEWS API INTEGRATION")
    print("=" * 80)

    try:
        # TODO: Uncomment tests as you complete the corresponding TODOs
        # test_alpha_vantage()
        # test_newsapi()
        # test_combined_sources()
        # test_filtering()
        # test_summary()
        # test_error_handling()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# CHALLENGE TASKS:
# 1. Add support for Finnhub API (https://finnhub.io/)
# 2. Implement exponential backoff for rate limiting
# 3. Add persistent caching using Redis or SQLite
# 4. Create a background job to fetch news every hour
# 5. Add webhook support to notify when important news is detected
