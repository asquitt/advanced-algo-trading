"""
Market data fetching with multiple providers and fallbacks.

This module provides resilient market data fetching with:
- Primary source: Alpaca (free for paper trading accounts)
- Fallback: yfinance (free but rate-limited)
- Caching to reduce API calls and costs

Cost optimization:
- All data is cached in Redis
- Quotes cached for 15 seconds during market hours
- News cached for 1 hour
- Historical data cached for 24 hours
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import requests
from src.utils.config import settings
from src.utils.logger import app_logger
from src.utils.cache import cache, cached
from src.data_layer.models import MarketNews


class MarketDataProvider:
    """
    Unified market data provider with multiple sources and caching.

    This class abstracts away the complexity of dealing with multiple
    data providers and automatically handles fallbacks and caching.
    """

    def __init__(self):
        """Initialize market data provider."""
        self.alpaca_headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }
        self.alpha_vantage_key = settings.alpha_vantage_api_key

    @cached(ttl=15, key_prefix="quote")
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a symbol.

        Cached for 15 seconds to balance freshness and API costs.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with keys: price, bid, ask, volume, timestamp
        """
        try:
            # Try Alpaca first (fast and reliable for paper trading)
            url = f"{settings.alpaca_base_url}/v2/stocks/{symbol}/quotes/latest"
            response = requests.get(url, headers=self.alpaca_headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                quote = data.get("quote", {})
                app_logger.debug(f"Got quote for {symbol} from Alpaca")
                return {
                    "symbol": symbol,
                    "price": (quote.get("ap", 0) + quote.get("bp", 0)) / 2,  # Mid price
                    "bid": quote.get("bp"),
                    "ask": quote.get("ap"),
                    "bid_size": quote.get("bs"),
                    "ask_size": quote.get("as"),
                    "timestamp": quote.get("t"),
                }
        except Exception as e:
            app_logger.warning(f"Alpaca quote failed for {symbol}: {e}")

        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            app_logger.debug(f"Got quote for {symbol} from yfinance (fallback)")
            return {
                "symbol": symbol,
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "bid": info.get("bid"),
                "ask": info.get("ask"),
                "volume": info.get("volume"),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            app_logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    @cached(ttl=3600, key_prefix="historical")
    def get_historical_data(
        self,
        symbol: str,
        days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get historical OHLCV data.

        Cached for 1 hour since historical data doesn't change.

        Args:
            symbol: Stock ticker
            days: Number of days of history

        Returns:
            Dict with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")

            if df.empty:
                app_logger.warning(f"No historical data for {symbol}")
                return None

            app_logger.debug(f"Got {len(df)} days of historical data for {symbol}")
            return {
                "symbol": symbol,
                "data": df.to_dict(orient="records"),
                "start_date": df.index[0].isoformat(),
                "end_date": df.index[-1].isoformat(),
            }
        except Exception as e:
            app_logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    @cached(ttl=3600, key_prefix="news")
    def get_news(self, symbol: str, limit: int = 10) -> List[MarketNews]:
        """
        Get recent news articles for a symbol.

        Cached for 1 hour to avoid excessive API calls.

        Args:
            symbol: Stock ticker
            limit: Maximum number of articles

        Returns:
            List of MarketNews objects
        """
        news_items = []

        try:
            # Try Alpaca news API
            url = f"{settings.alpaca_base_url}/v1beta1/news"
            params = {
                "symbols": symbol,
                "limit": limit,
                "sort": "desc",
            }
            response = requests.get(
                url,
                headers=self.alpaca_headers,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                for item in data.get("news", []):
                    news_items.append(MarketNews(
                        symbol=symbol,
                        headline=item.get("headline", ""),
                        summary=item.get("summary"),
                        source=item.get("source", "unknown"),
                        url=item.get("url"),
                        published_at=datetime.fromisoformat(
                            item.get("created_at").replace("Z", "+00:00")
                        ),
                    ))
                app_logger.debug(f"Got {len(news_items)} news items for {symbol}")
        except Exception as e:
            app_logger.warning(f"Failed to get news for {symbol}: {e}")

        # Fallback: Try yfinance news
        if not news_items:
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                for item in news[:limit]:
                    news_items.append(MarketNews(
                        symbol=symbol,
                        headline=item.get("title", ""),
                        summary=item.get("summary"),
                        source=item.get("publisher", "unknown"),
                        url=item.get("link"),
                        published_at=datetime.fromtimestamp(
                            item.get("providerPublishTime", 0)
                        ),
                    ))
                app_logger.debug(f"Got {len(news_items)} news items from yfinance")
            except Exception as e:
                app_logger.error(f"All news sources failed for {symbol}: {e}")

        return news_items

    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company information and fundamentals.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with company info
        """
        cache_key = f"company_info:{symbol}"
        cached_info = cache.get(cache_key)
        if cached_info:
            return cached_info

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            company_info = {
                "symbol": symbol,
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "description": info.get("longBusinessSummary"),
            }

            # Cache for 24 hours (fundamental data changes slowly)
            cache.set(cache_key, company_info, ttl=86400)
            app_logger.debug(f"Got company info for {symbol}")
            return company_info
        except Exception as e:
            app_logger.error(f"Failed to get company info for {symbol}: {e}")
            return None

    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        try:
            url = f"{settings.alpaca_base_url}/v2/clock"
            response = requests.get(url, headers=self.alpaca_headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("is_open", False)
        except Exception as e:
            app_logger.error(f"Failed to check market status: {e}")

        # Fallback: Basic time-based check (US market hours)
        now = datetime.utcnow()
        # Simple check: weekdays 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
        if now.weekday() >= 5:  # Weekend
            return False
        hour = now.hour
        return 14 <= hour < 21  # Approximate market hours in UTC

    def get_sec_filings(self, symbol: str, filing_type: str = "10-K") -> List[Dict[str, Any]]:
        """
        Get SEC filings for a company.

        Note: This is a simplified implementation. For production,
        use the SEC EDGAR API or a paid service like Financial Modeling Prep.

        Args:
            symbol: Stock ticker
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)

        Returns:
            List of filing metadata
        """
        # This is a placeholder. In production, you would:
        # 1. Query SEC EDGAR API
        # 2. Parse filing URLs
        # 3. Extract relevant sections
        # 4. Cache the results

        app_logger.warning(
            f"SEC filing retrieval not fully implemented. "
            f"Symbol: {symbol}, Type: {filing_type}"
        )
        return []


# Global market data provider instance
market_data = MarketDataProvider()
