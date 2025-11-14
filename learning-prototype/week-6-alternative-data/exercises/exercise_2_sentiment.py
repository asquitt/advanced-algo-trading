"""
Exercise 2: Sentiment Analysis
Extract sentiment scores from news articles using VADER and FinBERT

Prerequisites:
pip install vaderSentiment
pip install transformers torch  # For FinBERT (optional)

Tasks:
1. Complete the TODOs in sentiment_analysis.py
2. Analyze sample news articles
3. Compare VADER and FinBERT results
4. Test keyword extraction
5. Aggregate sentiment by symbol

Expected Output:
- Sentiment scores for each article
- Comparison of different methods
- Sentiment classification
- Aggregated sentiment by symbol
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starter_code.sentiment_analysis import SentimentAnalyzer, SentimentScore
import json


# Sample news articles for testing
SAMPLE_ARTICLES = [
    {
        "symbol": "AAPL",
        "title": "Apple Reports Record Breaking Quarter with Strong iPhone Sales",
        "summary": "Apple Inc. announced exceptional quarterly results, surpassing analyst expectations. The company's revenue grew 15% year-over-year, driven by robust iPhone 15 sales and expanding services business. CEO Tim Cook expressed optimism about future growth prospects."
    },
    {
        "symbol": "TSLA",
        "title": "Tesla Stock Plunges on Production Concerns and Recall Issues",
        "summary": "Tesla shares fell sharply after the company announced a major recall affecting hundreds of thousands of vehicles. Production delays at the new factory have raised concerns among investors about the company's ability to meet delivery targets."
    },
    {
        "symbol": "GOOGL",
        "title": "Google Announces New AI Features, Stock Remains Flat",
        "summary": "Alphabet Inc. unveiled several new artificial intelligence features for its search engine and cloud platform. While the announcements were well-received technically, investors showed muted response as the stock traded sideways."
    },
    {
        "symbol": "MSFT",
        "title": "Microsoft Beats Estimates, Cloud Revenue Soars 30%",
        "summary": "Microsoft Corporation reported strong quarterly earnings, beating both revenue and profit estimates. Azure cloud services revenue surged 30%, demonstrating the company's competitive position in enterprise cloud computing. Analysts raised price targets following the results."
    },
    {
        "symbol": "AMZN",
        "title": "Amazon Faces Regulatory Scrutiny Over Market Practices",
        "summary": "Amazon is under investigation by regulators over alleged anti-competitive practices. The company denies wrongdoing but faces potential fines and restrictions. Uncertainty around the outcome has weighed on the stock price."
    }
]


def test_vader_sentiment():
    """Test VADER sentiment analysis"""
    print("=" * 80)
    print("Testing VADER Sentiment Analysis")
    print("=" * 80)

    # TODO: Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(use_finbert=False)

    # TODO: Analyze each sample article
    print("\nAnalyzing sample articles with VADER:\n")
    for article in SAMPLE_ARTICLES:
        print(f"Symbol: {article['symbol']}")
        print(f"Title: {article['title']}\n")

        # TODO: Analyze article sentiment
        sentiment = analyzer.analyze_article(
            article['title'],
            article['summary']
        )

        # TODO: Classify sentiment
        category = analyzer.classify_sentiment(sentiment.compound)

        # TODO: Display results
        print(f"  Sentiment: {category}")
        print(f"  Compound: {sentiment.compound:.3f}")
        print(f"  Positive: {sentiment.positive:.3f}")
        print(f"  Negative: {sentiment.negative:.3f}")
        print(f"  Neutral:  {sentiment.neutral:.3f}")
        print("-" * 80)


def test_text_preprocessing():
    """Test text preprocessing"""
    print("\n" + "=" * 80)
    print("Testing Text Preprocessing")
    print("=" * 80)

    analyzer = SentimentAnalyzer()

    # TODO: Test preprocessing with messy text
    messy_texts = [
        "Check out this AMAZING stock!!! https://example.com #investing",
        "Stock fell 5%... disappointing results :(",
        "Normal   spacing     and    punctuation!!!",
        "BREAKING NEWS: Company announces merger!!!!"
    ]

    print("\nPreprocessing examples:\n")
    for text in messy_texts:
        cleaned = analyzer.preprocess_text(text)
        print(f"Original:  {text}")
        print(f"Cleaned:   {cleaned}")
        print()


def test_keyword_extraction():
    """Test financial keyword extraction"""
    print("=" * 80)
    print("Testing Financial Keyword Extraction")
    print("=" * 80)

    analyzer = SentimentAnalyzer()

    # TODO: Analyze keyword presence
    print("\nKeyword analysis:\n")
    for article in SAMPLE_ARTICLES:
        text = article['title'] + " " + article['summary']
        keywords = analyzer.extract_financial_keywords(text)

        print(f"{article['symbol']}: {article['title']}")
        print(f"  Bullish keywords: {keywords.get('bullish', 0)}")
        print(f"  Bearish keywords: {keywords.get('bearish', 0)}")
        print(f"  Net sentiment: {keywords.get('net_sentiment', 0)}")
        print()


def test_weighted_sentiment():
    """Test weighted sentiment combining methods"""
    print("=" * 80)
    print("Testing Weighted Sentiment")
    print("=" * 80)

    analyzer = SentimentAnalyzer()

    print("\nComparing VADER-only vs. Weighted sentiment:\n")
    for article in SAMPLE_ARTICLES[:3]:  # Just test first 3
        text = article['title'] + " " + article['summary']

        # TODO: Get VADER sentiment
        vader_score = analyzer.analyze_article(
            article['title'],
            article['summary']
        )

        # TODO: Get weighted sentiment
        weighted_score = analyzer.calculate_weighted_sentiment(
            text,
            vader_weight=0.7,
            keyword_weight=0.3
        )

        print(f"{article['symbol']}: {article['title']}")
        print(f"  VADER compound:    {vader_score.compound:.3f}")
        print(f"  Weighted score:    {weighted_score:.3f}")
        print()


def test_batch_analysis():
    """Test batch sentiment analysis"""
    print("=" * 80)
    print("Testing Batch Analysis")
    print("=" * 80)

    analyzer = SentimentAnalyzer()

    # TODO: Prepare articles for batch analysis
    articles_for_batch = [
        {"title": article["title"], "summary": article["summary"]}
        for article in SAMPLE_ARTICLES
    ]

    # TODO: Analyze batch
    sentiments = analyzer.batch_analyze(articles_for_batch)

    # TODO: Display results
    print(f"\nAnalyzed {len(sentiments)} articles in batch\n")
    for i, (article, sentiment) in enumerate(zip(SAMPLE_ARTICLES, sentiments), 1):
        category = analyzer.classify_sentiment(sentiment.compound)
        print(f"{i}. {article['symbol']}: {category} ({sentiment.compound:.3f})")


def test_symbol_aggregation():
    """Test sentiment aggregation by symbol"""
    print("\n" + "=" * 80)
    print("Testing Sentiment Aggregation by Symbol")
    print("=" * 80)

    analyzer = SentimentAnalyzer()

    # TODO: Create articles with sentiment scores
    articles_with_sentiment = []
    for article in SAMPLE_ARTICLES:
        sentiment = analyzer.analyze_article(
            article['title'],
            article['summary']
        )
        articles_with_sentiment.append({
            "symbols": [article['symbol']],
            "sentiment": sentiment.compound
        })

    # TODO: Add some duplicate symbols to test aggregation
    # Add another AAPL article
    extra_aapl = {
        "symbols": ["AAPL"],
        "sentiment": 0.5  # Moderate positive
    }
    articles_with_sentiment.append(extra_aapl)

    # TODO: Aggregate by symbol
    aggregated = analyzer.aggregate_sentiment_by_symbol(articles_with_sentiment)

    # TODO: Display aggregated results
    print("\nAggregated Sentiment by Symbol:\n")
    for symbol, metrics in sorted(aggregated.items()):
        print(f"{symbol}:")
        print(f"  Mean sentiment:   {metrics.get('mean_sentiment', 0):.3f}")
        print(f"  Median sentiment: {metrics.get('median_sentiment', 0):.3f}")
        print(f"  Std deviation:    {metrics.get('std_sentiment', 0):.3f}")
        print(f"  Article count:    {metrics.get('article_count', 0)}")
        print(f"  Bullish ratio:    {metrics.get('bullish_ratio', 0):.1%}")
        print()


def test_sentiment_classification():
    """Test sentiment classification categories"""
    print("=" * 80)
    print("Testing Sentiment Classification")
    print("=" * 80)

    analyzer = SentimentAnalyzer()

    # TODO: Test different sentiment levels
    test_scores = [
        (0.8, "Very strong positive"),
        (0.3, "Moderate positive"),
        (0.05, "Slightly positive/neutral"),
        (-0.05, "Slightly negative/neutral"),
        (-0.3, "Moderate negative"),
        (-0.8, "Very strong negative")
    ]

    print("\nSentiment classification thresholds:\n")
    for score, description in test_scores:
        category = analyzer.classify_sentiment(score)
        print(f"Score {score:+.2f} ({description:25s}) -> {category}")


def test_finbert_comparison():
    """Test FinBERT vs VADER comparison (optional)"""
    print("\n" + "=" * 80)
    print("Testing FinBERT vs VADER Comparison (Optional)")
    print("=" * 80)

    try:
        # TODO: Initialize with FinBERT
        analyzer_finbert = SentimentAnalyzer(use_finbert=True)
        analyzer_vader = SentimentAnalyzer(use_finbert=False)

        print("\nComparing VADER and FinBERT on financial text:\n")

        financial_texts = [
            "The company reported strong earnings growth and raised guidance.",
            "Analysts downgraded the stock citing margin compression concerns.",
            "The merger will create significant shareholder value."
        ]

        for text in financial_texts:
            vader_score = analyzer_vader.analyze_vader(text)
            finbert_score = analyzer_finbert.analyze_finbert(text)

            print(f"Text: {text}")
            print(f"  VADER:   {vader_score.compound:.3f}")
            print(f"  FinBERT: {finbert_score.compound:.3f}")
            print()

    except ImportError:
        print("\nFinBERT not available. Install transformers and torch to use.")
    except Exception as e:
        print(f"\nFinBERT test skipped: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("EXERCISE 2: SENTIMENT ANALYSIS")
    print("=" * 80)

    try:
        # TODO: Uncomment tests as you complete the corresponding TODOs
        # test_vader_sentiment()
        # test_text_preprocessing()
        # test_keyword_extraction()
        # test_weighted_sentiment()
        # test_batch_analysis()
        # test_symbol_aggregation()
        # test_sentiment_classification()
        # test_finbert_comparison()  # Optional

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
# 1. Implement custom financial sentiment lexicon
# 2. Add aspect-based sentiment analysis (sentiment per topic)
# 3. Train your own sentiment model on financial data
# 4. Implement sentiment momentum (tracking sentiment changes over time)
# 5. Add support for analyzing social media sentiment (Twitter, Reddit)
