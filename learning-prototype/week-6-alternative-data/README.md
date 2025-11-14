# Week 6: Alternative Data & News

Leverage news, sentiment, and alternative data for trading signals!

## Learning Objectives

By the end of this week, you will:

✅ Integrate news APIs (Alpha Vantage, NewsAPI)
✅ Extract sentiment from text using NLP
✅ Aggregate multi-source sentiment
✅ Build sentiment-driven trading signals
✅ Understand alternative data sources
✅ Implement data quality checks
✅ Handle real-time news feeds

## Why Alternative Data Matters

> "In the information age, ignorance is a choice." - Donny Miller

Alternative data provides:
- **Edge**: Information others don't have
- **Alpha**: Excess returns
- **Early signals**: Before price moves
- **Diversification**: Uncorrelated to price

**Examples**:
- Satellite imagery (parking lots)
- Credit card transactions
- Social media sentiment
- Web scraping (job postings)
- Weather data

## Prerequisites

- Python NLP basics
- API integration
- Async programming (asyncio)
- Completed Week 3-5

## Folder Structure

```
week-6-alternative-data/
├── README.md (you are here)
├── CONCEPTS.md (alternative data concepts)
├── starter-code/
│   ├── news_integration.py (20 TODOs)
│   ├── sentiment_analysis.py (20 TODOs)
│   └── signal_generation.py (15 TODOs)
├── exercises/
│   ├── exercise_1_news_api.py
│   ├── exercise_2_sentiment.py
│   ├── exercise_3_signals.py
│   └── exercise_4_backtest.py
├── solutions/
│   └── news_integration_complete.py
└── data/
    └── sample_news.json
```

## Learning Path

### Day 1: News API Integration (3-4 hours)

**What you'll learn**:
- Alpha Vantage News API
- NewsAPI.org integration
- Async data fetching
- Rate limiting
- Data deduplication

**Key concepts**:
```python
# Fetch news for multiple symbols
async def fetch_news(symbols):
    tasks = [fetch_alpha_vantage(sym) for sym in symbols]
    results = await asyncio.gather(*tasks)
    return aggregate_results(results)
```

### Day 2: Sentiment Analysis (4-5 hours)

**What you'll learn**:
- Lexicon-based sentiment (VADER)
- Transformer models (FinBERT)
- Sentiment scoring
- Context understanding
- Negation handling

**Methods**:

1. **Simple lexicon**:
```python
positive = ['bullish', 'growth', 'profit']
negative = ['bearish', 'loss', 'decline']
score = (pos_count - neg_count) / total_words
```

2. **FinBERT** (better):
```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis",
                    model="ProsusAI/finbert")
result = sentiment("Strong earnings beat expectations")
# {'label': 'positive', 'score': 0.95}
```

### Day 3: Signal Generation (3-4 hours)

**What you'll learn**:
- Sentiment aggregation
- Multi-source weighting
- Time decay
- Technical confirmation
- Signal filtering

**Signal logic**:
```python
def generate_signal(sentiment, price, volume):
    # 1. Check sentiment strength
    if sentiment < 0.5:
        return 0  # Not strong enough

    # 2. Technical confirmation
    if price < sma_50:
        return 0  # Downtrend, ignore

    # 3. Volume confirmation
    if volume < avg_volume * 1.5:
        return 0  # No volume surge

    # All checks passed
    return 1  # BUY signal
```

### Day 4: Alternative Data Sources (3-4 hours)

**What you'll learn**:
- Social media (Twitter, Reddit)
- SEC filings (EDGAR)
- Earnings call transcripts
- Analyst reports
- Economic calendars

**Example sources**:
- **Twitter**: Real-time sentiment
- **Reddit (r/wallstreetbets)**: Retail sentiment
- **StockTwits**: Trading community
- **Seeking Alpha**: Analysis articles

### Day 5: Backtesting & Integration (2-3 hours)

**Complete workflow**:
```bash
# 1. Fetch news
python fetch_news.py --symbols AAPL,MSFT,GOOGL

# 2. Extract sentiment
python analyze_sentiment.py --input news.json

# 3. Generate signals
python generate_signals.py --sentiment sentiment.json

# 4. Backtest
python backtest_sentiment_strategy.py
```

## Key Concepts

### 1. Sentiment Score

**Scale**: -1 (very negative) to +1 (very positive)

**Example**:
- "Record profits, exceeds expectations": +0.85
- "Slight revenue miss, guidance maintained": -0.15
- "Bankruptcy filing, massive layoffs": -0.95

### 2. Sentiment Aggregation

**Weighted by recency**:
```python
def aggregate_sentiment(articles):
    now = datetime.now()
    weights = []
    scores = []

    for article in articles:
        age_hours = (now - article.published_at).total_seconds() / 3600
        weight = 1.0 / (1 + age_hours)  # Decay over time
        weights.append(weight)
        scores.append(article.sentiment)

    return np.average(scores, weights=weights)
```

### 3. News Quality Filtering

**Filter out**:
- Duplicate articles
- Low relevance (< 0.3)
- Spam sources
- Republished content
- Non-English

### 4. Technical Confirmation

**Don't trade sentiment alone**!

```python
def confirm_sentiment_signal(sentiment, price_data):
    """Require technical confirmation."""

    # Bullish sentiment needs:
    # 1. Price above moving average
    # 2. Volume surge
    # 3. No recent gap down

    if sentiment > 0.5:
        if (price > sma_50 and
            volume > avg_volume * 1.5 and
            not has_gap_down(price_data)):
            return True

    return False
```

## Data Sources

### Free Tier

| Source | API | Free Tier | Best For |
|--------|-----|-----------|----------|
| Alpha Vantage | News | 25 req/day | News sentiment |
| NewsAPI | News | 100 req/day | General news |
| Reddit | PRAW | Unlimited | Retail sentiment |
| EDGAR | SEC | Unlimited | Filings |
| Yahoo Finance | yfinance | Unlimited | Basic data |

### Paid (Production)

| Source | Cost | Best For |
|--------|------|----------|
| Bloomberg Terminal | $2k/mo | Everything |
| Refinitiv | $1k+/mo | News, data |
| AlternativeData.org | Varies | Alt data |
| Quandl | $50+/mo | Financial data |

## Common Pitfalls

### Pitfall #1: Stale Sentiment
**Problem**: Using old news as fresh
**Solution**: Weight by recency, expire after 24h

### Pitfall #2: No Technical Confirmation
**Problem**: Trading on sentiment alone
**Solution**: Require price/volume confirmation

### Pitfall #3: Overfitting
**Problem**: Optimizing thresholds on past news
**Solution**: Walk-forward testing, hold-out set

### Pitfall #4: Survivorship Bias
**Problem**: Only analyzing successful companies
**Solution**: Include delisted/failed companies

### Pitfall #5: Look-Ahead Bias
**Problem**: Using future news in backtests
**Solution**: Careful timestamp handling

## Success Criteria

You've mastered Week 6 when you can:

✅ Fetch news from multiple sources
✅ Extract accurate sentiment scores
✅ Aggregate multi-source sentiment
✅ Generate trading signals from sentiment
✅ Confirm signals with technical analysis
✅ Backtest sentiment strategies
✅ Handle real-time news feeds
✅ Filter low-quality data

## Performance Expectations

**Sentiment strategy**:
- Sharpe ratio: 0.8-1.5
- Win rate: 50-60%
- Best in: Earnings season, major events
- Avoid: Low news flow periods

**Multi-source aggregation**:
- Improves Sharpe by 20-30%
- Reduces false signals
- Better risk-adjusted returns

## Resources

### APIs
- [Alpha Vantage](https://www.alphavantage.co/documentation/#news-sentiment)
- [NewsAPI](https://newsapi.org/docs)
- [Reddit API (PRAW)](https://praw.readthedocs.io/)

### NLP Libraries
- [FinBERT](https://huggingface.co/ProsusAI/finbert)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [spaCy](https://spacy.io/)

### Papers
- ["Textual Analysis in Finance"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3470272)
- ["News Sentiment and Stock Returns"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2193128)

### Books
- "Advances in Financial Machine Learning" (Ch. 18: Entropy Features)
- "Machine Learning for Algorithmic Trading" (Ch. 13: NLP)

**Next**: Week 7 - Cloud Deployment & Scaling ☁️
