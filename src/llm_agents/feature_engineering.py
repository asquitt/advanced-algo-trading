"""
Advanced Feature Engineering for Signal Robustness

Enhances trading signals with:
1. Technical indicators (momentum, volatility, trend)
2. Market regime detection
3. Alternative data integration
4. Multi-timeframe analysis
5. Feature selection and importance ranking

Author: LLM Trading Platform
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"


@dataclass
class TechnicalFeatures:
    """Technical indicator features."""
    # Momentum indicators
    rsi_14: float  # Relative Strength Index (14-period)
    macd: float  # MACD value
    macd_signal: float  # MACD signal line
    macd_histogram: float  # MACD histogram
    stoch_k: float  # Stochastic %K
    stoch_d: float  # Stochastic %D
    momentum_10: float  # 10-day momentum
    roc_10: float  # 10-day rate of change

    # Trend indicators
    sma_20: float  # 20-day simple moving average
    sma_50: float  # 50-day simple moving average
    sma_200: float  # 200-day simple moving average
    ema_12: float  # 12-day exponential moving average
    ema_26: float  # 26-day exponential moving average
    adx: float  # Average Directional Index (trend strength)

    # Volatility indicators
    atr_14: float  # Average True Range (14-period)
    bb_upper: float  # Bollinger Band upper
    bb_middle: float  # Bollinger Band middle
    bb_lower: float  # Bollinger Band lower
    bb_width: float  # Bollinger Band width
    bb_position: float  # Price position in BB (0-1)

    # Volume indicators
    volume_sma_20: float  # 20-day average volume
    volume_ratio: float  # Current volume / average volume
    obv: float  # On-Balance Volume
    vwap: float  # Volume-Weighted Average Price

    # Price action
    current_price: float
    high_52w: float  # 52-week high
    low_52w: float  # 52-week low
    distance_from_52w_high: float  # % from 52w high
    distance_from_52w_low: float  # % from 52w low


@dataclass
class RegimeFeatures:
    """Market regime features."""
    primary_regime: MarketRegime
    regime_strength: float  # 0-1, confidence in regime
    regime_duration_days: int  # How long in current regime
    volatility_percentile: float  # Historical volatility percentile
    trend_strength: float  # -1 to 1 (bearish to bullish)
    is_consolidating: bool
    breakout_probability: float  # 0-1


@dataclass
class AlternativeFeatures:
    """Alternative data features."""
    # Sentiment
    social_sentiment_score: float  # -1 to 1
    news_sentiment_score: float  # -1 to 1
    sentiment_change_24h: float  # Change in sentiment

    # Market structure
    sector_relative_strength: float  # vs sector index
    market_relative_strength: float  # vs broad market
    correlation_to_spy: float  # Correlation to S&P 500

    # Options data (if available)
    put_call_ratio: Optional[float]
    implied_volatility: Optional[float]
    iv_rank: Optional[float]  # IV percentile rank

    # Insider activity
    insider_buy_signals: int  # Number of recent insider buys
    insider_sell_signals: int  # Number of recent insider sells


@dataclass
class MultiTimeframeFeatures:
    """Features across multiple timeframes."""
    daily_trend: str  # "bullish", "bearish", "neutral"
    weekly_trend: str
    monthly_trend: str
    alignment_score: float  # 0-1, how aligned timeframes are
    short_term_momentum: float
    medium_term_momentum: float
    long_term_momentum: float


@dataclass
class EnhancedFeatureSet:
    """Complete feature set for robust signals."""
    symbol: str
    timestamp: datetime
    technical: TechnicalFeatures
    regime: RegimeFeatures
    alternative: AlternativeFeatures
    multi_timeframe: MultiTimeframeFeatures
    feature_quality_score: float  # 0-1, data quality indicator


class TechnicalIndicators:
    """Calculate technical indicators."""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Neutral

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD, signal line, and histogram."""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0

        prices_arr = np.array(prices)

        # Calculate EMAs
        ema_fast = TechnicalIndicators._ema(prices_arr, fast)
        ema_slow = TechnicalIndicators._ema(prices_arr, slow)

        macd = ema_fast - ema_slow

        # Calculate signal line
        macd_values = [ema_fast - ema_slow]  # Simplified
        signal_line = TechnicalIndicators._ema(np.array(macd_values), signal)

        histogram = macd - signal_line

        return macd, signal_line, histogram

    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> float:
        """Calculate exponential moving average."""
        if len(prices) < period:
            return float(np.mean(prices))

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return float(ema)

    @staticmethod
    def calculate_atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> float:
        """Calculate Average True Range."""
        if len(closes) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)

        atr = np.mean(true_ranges[-period:])
        return float(atr)

    @staticmethod
    def calculate_bollinger_bands(
        prices: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(prices) < period:
            avg = np.mean(prices)
            return avg, avg, avg

        recent = prices[-period:]
        middle = np.mean(recent)
        std = np.std(recent)

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return float(upper), float(middle), float(lower)

    @staticmethod
    def calculate_adx(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> float:
        """Calculate Average Directional Index (trend strength)."""
        if len(closes) < period + 1:
            return 0.0

        # Simplified ADX calculation
        # Full implementation would use +DI and -DI

        # Calculate price movement
        movements = []
        for i in range(1, len(closes)):
            move = abs(closes[i] - closes[i-1]) / closes[i-1]
            movements.append(move)

        # ADX is essentially smoothed movement
        adx = np.mean(movements[-period:]) * 100

        return min(float(adx), 100.0)


class RegimeDetector:
    """Detect market regime for strategy adaptation."""

    def detect_regime(
        self,
        prices: List[float],
        volumes: List[int],
        period: int = 60
    ) -> RegimeFeatures:
        """
        Detect current market regime.

        Args:
            prices: Price history
            volumes: Volume history
            period: Lookback period for analysis

        Returns:
            RegimeFeatures with regime classification
        """
        if len(prices) < 20:
            return self._default_regime()

        recent_prices = prices[-period:] if len(prices) >= period else prices

        # Calculate volatility
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        # Historical volatility percentile
        if len(prices) >= 252:  # One year of data
            hist_vols = []
            for i in range(252, len(prices)):
                window_returns = np.diff(prices[i-60:i]) / prices[i-60:i-1]
                hist_vols.append(np.std(window_returns))
            percentile = np.percentile(hist_vols, 50)
            vol_percentile = min(volatility / percentile, 1.0) if percentile > 0 else 0.5
        else:
            vol_percentile = 0.5

        # Calculate trend
        sma_20 = np.mean(recent_prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        current_price = recent_prices[-1]

        trend_strength = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0

        # Detect regime
        primary_regime, regime_strength = self._classify_regime(
            prices=recent_prices,
            volatility=volatility,
            vol_percentile=vol_percentile,
            trend_strength=trend_strength
        )

        # Detect consolidation
        price_range = max(recent_prices[-20:]) - min(recent_prices[-20:])
        avg_price = np.mean(recent_prices[-20:])
        is_consolidating = (price_range / avg_price) < 0.05  # Less than 5% range

        # Estimate breakout probability
        if is_consolidating:
            # Higher probability after longer consolidation
            consolidation_days = self._count_consolidation_days(prices)
            breakout_probability = min(consolidation_days / 30.0, 0.8)
        else:
            breakout_probability = 0.2

        # Estimate regime duration
        regime_duration = self._estimate_regime_duration(prices, primary_regime)

        return RegimeFeatures(
            primary_regime=primary_regime,
            regime_strength=regime_strength,
            regime_duration_days=regime_duration,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            is_consolidating=is_consolidating,
            breakout_probability=breakout_probability
        )

    def _classify_regime(
        self,
        prices: List[float],
        volatility: float,
        vol_percentile: float,
        trend_strength: float
    ) -> Tuple[MarketRegime, float]:
        """Classify market regime."""
        # High/Low volatility
        if vol_percentile > 0.8:
            return MarketRegime.HIGH_VOLATILITY, vol_percentile
        elif vol_percentile < 0.2:
            return MarketRegime.LOW_VOLATILITY, 1 - vol_percentile

        # Trending
        if trend_strength > 0.1:
            return MarketRegime.TRENDING_UP, min(trend_strength * 5, 1.0)
        elif trend_strength < -0.1:
            return MarketRegime.TRENDING_DOWN, min(abs(trend_strength) * 5, 1.0)

        # Ranging
        return MarketRegime.RANGING, 0.6

    def _count_consolidation_days(self, prices: List[float]) -> int:
        """Count consecutive days in consolidation."""
        if len(prices) < 20:
            return 0

        days = 0
        for i in range(len(prices) - 20, 0, -1):
            window = prices[i:i+20]
            price_range = max(window) - min(window)
            avg_price = np.mean(window)

            if (price_range / avg_price) < 0.05:
                days += 1
            else:
                break

        return days

    def _estimate_regime_duration(
        self,
        prices: List[float],
        current_regime: MarketRegime
    ) -> int:
        """Estimate how long in current regime."""
        # Simplified - just return a reasonable default
        return 10

    def _default_regime(self) -> RegimeFeatures:
        """Default regime when insufficient data."""
        return RegimeFeatures(
            primary_regime=MarketRegime.RANGING,
            regime_strength=0.5,
            regime_duration_days=0,
            volatility_percentile=0.5,
            trend_strength=0.0,
            is_consolidating=False,
            breakout_probability=0.3
        )


class FeatureEngineer:
    """Main feature engineering class."""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.regime_detector = RegimeDetector()

    def create_features(
        self,
        symbol: str,
        prices: List[float],
        volumes: List[int],
        highs: List[float],
        lows: List[float],
        news_sentiment: float = 0.0,
        social_sentiment: float = 0.0
    ) -> EnhancedFeatureSet:
        """
        Create comprehensive feature set.

        Args:
            symbol: Trading symbol
            prices: Price history (close prices)
            volumes: Volume history
            highs: High prices
            lows: Low prices
            news_sentiment: News sentiment score (-1 to 1)
            social_sentiment: Social media sentiment (-1 to 1)

        Returns:
            EnhancedFeatureSet with all features
        """
        if len(prices) < 20:
            logger.warning(f"Insufficient data for {symbol}, using defaults")
            return self._default_features(symbol, prices, volumes)

        # Technical features
        technical = self._create_technical_features(
            prices, volumes, highs, lows
        )

        # Regime features
        regime = self.regime_detector.detect_regime(prices, volumes)

        # Alternative features
        alternative = self._create_alternative_features(
            symbol, prices, news_sentiment, social_sentiment
        )

        # Multi-timeframe features
        multi_timeframe = self._create_multi_timeframe_features(prices)

        # Calculate feature quality score
        quality_score = self._assess_feature_quality(
            len(prices), len(volumes), len(highs), len(lows)
        )

        return EnhancedFeatureSet(
            symbol=symbol,
            timestamp=datetime.now(),
            technical=technical,
            regime=regime,
            alternative=alternative,
            multi_timeframe=multi_timeframe,
            feature_quality_score=quality_score
        )

    def _create_technical_features(
        self,
        prices: List[float],
        volumes: List[int],
        highs: List[float],
        lows: List[float]
    ) -> TechnicalFeatures:
        """Create technical indicator features."""
        current_price = prices[-1]

        # Momentum
        rsi_14 = self.indicators.calculate_rsi(prices, 14)
        macd, macd_signal, macd_hist = self.indicators.calculate_macd(prices)

        # Stochastic
        if len(prices) >= 14:
            period_low = min(prices[-14:])
            period_high = max(prices[-14:])
            if period_high != period_low:
                stoch_k = ((current_price - period_low) / (period_high - period_low)) * 100
            else:
                stoch_k = 50.0
        else:
            stoch_k = 50.0
        stoch_d = stoch_k  # Simplified

        momentum_10 = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0.0
        roc_10 = momentum_10 * 100

        # Trend
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
        sma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else current_price
        ema_12 = self.indicators._ema(np.array(prices), 12)
        ema_26 = self.indicators._ema(np.array(prices), 26)
        adx = self.indicators.calculate_adx(highs, lows, prices)

        # Volatility
        atr_14 = self.indicators.calculate_atr(highs, lows, prices, 14)
        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(prices)
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
        bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)
                       if bb_upper != bb_lower else 0.5)

        # Volume
        volume_sma_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        volume_ratio = volumes[-1] / volume_sma_20 if volume_sma_20 > 0 else 1.0

        # OBV (simplified)
        obv = 0.0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]

        # VWAP (simplified - daily)
        if len(prices) >= 20 and len(volumes) >= 20:
            vwap = sum(p * v for p, v in zip(prices[-20:], volumes[-20:])) / sum(volumes[-20:])
        else:
            vwap = current_price

        # 52-week high/low
        high_52w = max(prices[-252:]) if len(prices) >= 252 else max(prices)
        low_52w = min(prices[-252:]) if len(prices) >= 252 else min(prices)
        distance_from_52w_high = (current_price / high_52w - 1) * 100
        distance_from_52w_low = (current_price / low_52w - 1) * 100

        return TechnicalFeatures(
            rsi_14=rsi_14,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_hist,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            momentum_10=momentum_10,
            roc_10=roc_10,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            adx=adx,
            atr_14=atr_14,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_width=bb_width,
            bb_position=bb_position,
            volume_sma_20=volume_sma_20,
            volume_ratio=volume_ratio,
            obv=obv,
            vwap=vwap,
            current_price=current_price,
            high_52w=high_52w,
            low_52w=low_52w,
            distance_from_52w_high=distance_from_52w_high,
            distance_from_52w_low=distance_from_52w_low
        )

    def _create_alternative_features(
        self,
        symbol: str,
        prices: List[float],
        news_sentiment: float,
        social_sentiment: float
    ) -> AlternativeFeatures:
        """Create alternative data features."""
        # In production, would fetch real data
        # For now, use provided sentiment and generate reasonable defaults

        # Sentiment change (would compare to 24h ago)
        sentiment_change_24h = 0.0  # Simplified

        # Relative strength (would compare to sector/market)
        sector_relative_strength = 0.0
        market_relative_strength = 0.0
        correlation_to_spy = 0.5  # Assume moderate correlation

        return AlternativeFeatures(
            social_sentiment_score=social_sentiment,
            news_sentiment_score=news_sentiment,
            sentiment_change_24h=sentiment_change_24h,
            sector_relative_strength=sector_relative_strength,
            market_relative_strength=market_relative_strength,
            correlation_to_spy=correlation_to_spy,
            put_call_ratio=None,  # Would need options data
            implied_volatility=None,
            iv_rank=None,
            insider_buy_signals=0,  # Would need insider data
            insider_sell_signals=0
        )

    def _create_multi_timeframe_features(
        self,
        prices: List[float]
    ) -> MultiTimeframeFeatures:
        """Create multi-timeframe features."""
        # Daily trend (last 20 days)
        if len(prices) >= 20:
            daily_change = (prices[-1] / prices[-20] - 1)
            if daily_change > 0.05:
                daily_trend = "bullish"
            elif daily_change < -0.05:
                daily_trend = "bearish"
            else:
                daily_trend = "neutral"
        else:
            daily_trend = "neutral"

        # Weekly trend (approximated as 5 trading days)
        if len(prices) >= 60:
            weekly_change = (prices[-1] / prices[-60] - 1)
            if weekly_change > 0.1:
                weekly_trend = "bullish"
            elif weekly_change < -0.1:
                weekly_trend = "bearish"
            else:
                weekly_trend = "neutral"
        else:
            weekly_trend = "neutral"

        # Monthly trend (approximated as 20 trading days)
        if len(prices) >= 120:
            monthly_change = (prices[-1] / prices[-120] - 1)
            if monthly_change > 0.15:
                monthly_trend = "bullish"
            elif monthly_change < -0.15:
                monthly_trend = "bearish"
            else:
                monthly_trend = "neutral"
        else:
            monthly_trend = "neutral"

        # Alignment score
        trends = [daily_trend, weekly_trend, monthly_trend]
        if all(t == "bullish" for t in trends):
            alignment_score = 1.0
        elif all(t == "bearish" for t in trends):
            alignment_score = 1.0
        elif all(t == "neutral" for t in trends):
            alignment_score = 0.5
        else:
            # Mixed trends
            bullish_count = sum(1 for t in trends if t == "bullish")
            bearish_count = sum(1 for t in trends if t == "bearish")
            alignment_score = abs(bullish_count - bearish_count) / 3.0

        # Momentum at different timeframes
        short_term_momentum = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0.0
        medium_term_momentum = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0.0
        long_term_momentum = (prices[-1] / prices[-60] - 1) if len(prices) >= 60 else 0.0

        return MultiTimeframeFeatures(
            daily_trend=daily_trend,
            weekly_trend=weekly_trend,
            monthly_trend=monthly_trend,
            alignment_score=alignment_score,
            short_term_momentum=short_term_momentum,
            medium_term_momentum=medium_term_momentum,
            long_term_momentum=long_term_momentum
        )

    def _assess_feature_quality(
        self,
        num_prices: int,
        num_volumes: int,
        num_highs: int,
        num_lows: int
    ) -> float:
        """Assess quality of input data for features."""
        # Ideal: 252 days of data (1 year)
        completeness = min(num_prices / 252.0, 1.0)

        # Check data consistency
        consistency = 1.0
        if num_prices != num_volumes:
            consistency -= 0.2
        if num_prices != num_highs:
            consistency -= 0.2
        if num_prices != num_lows:
            consistency -= 0.2

        quality_score = (completeness * 0.7 + consistency * 0.3)

        return max(0.0, min(1.0, quality_score))

    def _default_features(
        self,
        symbol: str,
        prices: List[float],
        volumes: List[int]
    ) -> EnhancedFeatureSet:
        """Return default features when insufficient data."""
        current_price = prices[-1] if prices else 0.0

        return EnhancedFeatureSet(
            symbol=symbol,
            timestamp=datetime.now(),
            technical=TechnicalFeatures(
                rsi_14=50.0, macd=0.0, macd_signal=0.0, macd_histogram=0.0,
                stoch_k=50.0, stoch_d=50.0, momentum_10=0.0, roc_10=0.0,
                sma_20=current_price, sma_50=current_price, sma_200=current_price,
                ema_12=current_price, ema_26=current_price, adx=0.0,
                atr_14=0.0, bb_upper=current_price, bb_middle=current_price,
                bb_lower=current_price, bb_width=0.0, bb_position=0.5,
                volume_sma_20=volumes[-1] if volumes else 0,
                volume_ratio=1.0, obv=0.0, vwap=current_price,
                current_price=current_price, high_52w=current_price,
                low_52w=current_price, distance_from_52w_high=0.0,
                distance_from_52w_low=0.0
            ),
            regime=self.regime_detector._default_regime(),
            alternative=AlternativeFeatures(
                social_sentiment_score=0.0, news_sentiment_score=0.0,
                sentiment_change_24h=0.0, sector_relative_strength=0.0,
                market_relative_strength=0.0, correlation_to_spy=0.5,
                put_call_ratio=None, implied_volatility=None, iv_rank=None,
                insider_buy_signals=0, insider_sell_signals=0
            ),
            multi_timeframe=MultiTimeframeFeatures(
                daily_trend="neutral", weekly_trend="neutral",
                monthly_trend="neutral", alignment_score=0.5,
                short_term_momentum=0.0, medium_term_momentum=0.0,
                long_term_momentum=0.0
            ),
            feature_quality_score=0.3
        )


# Singleton instance
feature_engineer = FeatureEngineer()
