"""
Regime-Adaptive Momentum Strategy - Starter Code with TODOs

Your mission: Implement a momentum strategy that adapts to market regimes!

This strategy:
1. Detects market regimes (volatility and trend)
2. Adjusts position sizing based on regime
3. Uses different momentum parameters per regime
4. Implements dynamic stop losses

Difficulty levels:
🟢 Easy: Basic implementation
🟡 Medium: Requires some thinking
🔴 Hard: Advanced concepts

Author: Learning Lab Week 4
"""

from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class VolatilityRegime(Enum):
    """Volatility regime states."""
    LOW = "low_vol"
    NORMAL = "normal_vol"
    HIGH = "high_vol"


class TrendRegime(Enum):
    """Trend regime states."""
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"


@dataclass
class RegimeMomentumConfig:
    """Configuration for regime-adaptive momentum strategy."""
    # Momentum parameters
    lookback_short: int = 20  # Short-term momentum lookback
    lookback_long: int = 50   # Long-term momentum lookback

    # Volatility regime thresholds (annualized)
    low_vol_threshold: float = 0.15
    high_vol_threshold: float = 0.25
    vol_lookback: int = 20

    # Trend regime parameters
    trend_ma_short: int = 20
    trend_ma_long: int = 50
    trend_threshold: float = 0.02  # 2% for bull/bear determination

    # Position sizing
    base_position_size: float = 1.0
    low_vol_multiplier: float = 1.5  # Trade bigger in low vol
    high_vol_multiplier: float = 0.5  # Trade smaller in high vol

    # Stop loss parameters
    base_stop_loss: float = 0.02  # 2% stop loss
    high_vol_stop_multiplier: float = 2.0  # Wider stops in high vol


class RegimeMomentumStrategy:
    """
    Regime-adaptive momentum trading strategy.

    Core concepts:
    1. Detect current market regime (volatility + trend)
    2. Calculate momentum signals
    3. Adjust position size based on regime
    4. Use regime-specific stop losses
    """

    def __init__(self, config: Optional[RegimeMomentumConfig] = None):
        """Initialize regime momentum strategy."""
        self.config = config or RegimeMomentumConfig()
        self.current_regime = None

    # ========================================
    # Part 1: Volatility Regime Detection
    # ========================================

    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling realized volatility (annualized).

        🟢 TODO #1: Calculate realized volatility

        Formula: vol = rolling_std(returns) * sqrt(252)

        HINT: Use pd.Series.rolling(window).std()
        HINT: Annualize by multiplying by sqrt(252)
        HINT: Use self.config.vol_lookback if lookback is None

        Args:
            returns: Daily returns series
            lookback: Lookback period (default from config)

        Returns:
            Annualized volatility series
        """
        # YOUR CODE HERE
        pass

    def detect_volatility_regime(
        self,
        volatility: pd.Series
    ) -> pd.Series:
        """
        Classify volatility regime (LOW, NORMAL, HIGH).

        🟢 TODO #2: Detect volatility regime

        Logic:
        - If vol < low_vol_threshold: LOW
        - If vol > high_vol_threshold: HIGH
        - Otherwise: NORMAL

        HINT: Use np.where() or pd.Series.apply()
        HINT: Return string values: "low_vol", "normal_vol", "high_vol"

        Args:
            volatility: Annualized volatility series

        Returns:
            Regime series ("low_vol", "normal_vol", "high_vol")
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 2: Trend Regime Detection
    # ========================================

    def calculate_moving_averages(
        self,
        prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate short and long-term moving averages.

        🟢 TODO #3: Calculate moving averages

        HINT: Use pd.Series.rolling(window).mean()
        HINT: Use self.config.trend_ma_short and trend_ma_long

        Args:
            prices: Price series

        Returns:
            (short_ma, long_ma) tuple
        """
        # YOUR CODE HERE
        pass

    def detect_trend_regime(
        self,
        prices: pd.Series,
        ma_short: Optional[pd.Series] = None,
        ma_long: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Classify trend regime (BULL, NEUTRAL, BEAR).

        🟡 TODO #4: Detect trend regime

        Logic:
        - If short_ma > long_ma * (1 + threshold): BULL
        - If short_ma < long_ma * (1 - threshold): BEAR
        - Otherwise: NEUTRAL

        HINT: Calculate MAs if not provided
        HINT: Use self.config.trend_threshold
        HINT: Return string values: "bull", "neutral", "bear"

        Args:
            prices: Price series
            ma_short: Short MA (optional, will calculate if None)
            ma_long: Long MA (optional, will calculate if None)

        Returns:
            Regime series ("bull", "neutral", "bear")
        """
        # YOUR CODE HERE
        pass

    def combine_regimes(
        self,
        vol_regime: pd.Series,
        trend_regime: pd.Series
    ) -> pd.DataFrame:
        """
        Combine volatility and trend regimes into single DataFrame.

        🟢 TODO #5: Combine regimes

        HINT: Create DataFrame with 'vol_regime' and 'trend_regime' columns
        HINT: Add 'combined_regime' column = vol_regime + '_' + trend_regime

        Args:
            vol_regime: Volatility regime series
            trend_regime: Trend regime series

        Returns:
            DataFrame with regime information
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 3: Momentum Calculation
    # ========================================

    def calculate_momentum(
        self,
        prices: pd.Series,
        lookback: int
    ) -> pd.Series:
        """
        Calculate price momentum.

        🟢 TODO #6: Calculate momentum

        Formula: momentum = (price / price_lagged) - 1

        HINT: Use pd.Series.shift(lookback)
        HINT: momentum = prices / prices.shift(lookback) - 1

        Args:
            prices: Price series
            lookback: Lookback period

        Returns:
            Momentum series
        """
        # YOUR CODE HERE
        pass

    def calculate_dual_momentum(
        self,
        prices: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate both short and long-term momentum.

        🟢 TODO #7: Calculate dual momentum

        HINT: Use calculate_momentum() twice
        HINT: Return DataFrame with 'mom_short' and 'mom_long' columns

        Args:
            prices: Price series

        Returns:
            DataFrame with short and long momentum
        """
        # YOUR CODE HERE
        pass

    def calculate_momentum_zscore(
        self,
        momentum: pd.Series,
        lookback: int = 60
    ) -> pd.Series:
        """
        Calculate z-score of momentum for signal generation.

        🟡 TODO #8: Calculate momentum z-score

        Formula: z = (momentum - rolling_mean) / rolling_std

        HINT: Similar to pairs trading z-score
        HINT: Use rolling mean and std

        Args:
            momentum: Momentum series
            lookback: Rolling window for z-score

        Returns:
            Z-score series
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 4: Signal Generation
    # ========================================

    def generate_base_signals(
        self,
        momentum_short: pd.Series,
        momentum_long: pd.Series
    ) -> pd.Series:
        """
        Generate base momentum signals (before regime adjustment).

        🟡 TODO #9: Generate base signals

        Logic:
        - If both momentums > 0: Signal = 1 (long)
        - If both momentums < 0: Signal = -1 (short)
        - Otherwise: Signal = 0 (flat)

        HINT: Use np.where() or boolean indexing
        HINT: Require agreement between short and long momentum

        Args:
            momentum_short: Short-term momentum
            momentum_long: Long-term momentum

        Returns:
            Base signal series (-1, 0, 1)
        """
        # YOUR CODE HERE
        pass

    def adjust_signals_for_trend(
        self,
        base_signals: pd.Series,
        trend_regime: pd.Series
    ) -> pd.Series:
        """
        Adjust signals based on trend regime.

        🟡 TODO #10: Adjust for trend regime

        Logic:
        - BULL regime: Only allow long signals (filter out shorts)
        - BEAR regime: Only allow short signals (filter out longs)
        - NEUTRAL regime: Allow both directions

        HINT: Use boolean masking
        HINT: base_signals[trend == 'bear' & base_signals > 0] = 0

        Args:
            base_signals: Base momentum signals
            trend_regime: Trend regime series

        Returns:
            Adjusted signals
        """
        # YOUR CODE HERE
        pass

    def adjust_signals_for_volatility(
        self,
        signals: pd.Series,
        vol_regime: pd.Series
    ) -> pd.Series:
        """
        Adjust signals based on volatility regime.

        🔴 TODO #11: Adjust for volatility regime

        This function doesn't change the direction, but we'll use
        vol regime for position sizing in the next function.

        For now, in HIGH volatility:
        - Be more conservative: require stronger momentum
        - You can filter out weak signals

        HINT: For simplicity, you can just return signals unchanged
        HINT: Advanced: filter signals when vol is too high

        Args:
            signals: Trading signals
            vol_regime: Volatility regime series

        Returns:
            Adjusted signals
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 5: Position Sizing
    # ========================================

    def calculate_regime_position_size(
        self,
        base_size: float,
        vol_regime: pd.Series
    ) -> pd.Series:
        """
        Calculate position size based on volatility regime.

        🟡 TODO #12: Calculate regime-based position size

        Logic:
        - LOW vol: base_size * low_vol_multiplier (trade bigger)
        - NORMAL vol: base_size * 1.0
        - HIGH vol: base_size * high_vol_multiplier (trade smaller)

        HINT: Use pd.Series.map() or apply()
        HINT: Map regime strings to multipliers

        Args:
            base_size: Base position size (typically 1.0)
            vol_regime: Volatility regime series

        Returns:
            Position size series
        """
        # YOUR CODE HERE
        pass

    def calculate_volatility_position_size(
        self,
        returns: pd.Series,
        target_volatility: float = 0.15,
        lookback: int = 20
    ) -> pd.Series:
        """
        Calculate position size to target specific volatility.

        🔴 TODO #13: Calculate volatility-targeted position size

        Formula: position_size = target_vol / realized_vol

        This ensures constant volatility exposure.

        HINT: Calculate realized vol first
        HINT: Clip position size to reasonable range (e.g., 0.5 to 2.0)

        Args:
            returns: Returns series
            target_volatility: Target annualized volatility
            lookback: Lookback for volatility calculation

        Returns:
            Position size series
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 6: Risk Management
    # ========================================

    def calculate_dynamic_stop_loss(
        self,
        vol_regime: pd.Series
    ) -> pd.Series:
        """
        Calculate dynamic stop loss based on volatility regime.

        🟡 TODO #14: Calculate dynamic stop loss

        Logic:
        - LOW/NORMAL vol: base_stop_loss
        - HIGH vol: base_stop_loss * high_vol_stop_multiplier

        Wider stops in high volatility to avoid getting stopped out
        by normal noise.

        HINT: Use self.config.base_stop_loss and high_vol_stop_multiplier

        Args:
            vol_regime: Volatility regime series

        Returns:
            Stop loss percentage series
        """
        # YOUR CODE HERE
        pass

    def apply_stop_loss(
        self,
        prices: pd.Series,
        signals: pd.Series,
        stop_loss_pct: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        """
        Apply stop loss logic to signals.

        🔴 TODO #15: Apply stop loss

        Logic:
        - Track entry price when signal changes
        - If price moves against us by > stop_loss_pct: exit (signal = 0)
        - Otherwise: maintain signal

        HINT: This is complex! You need to track entry prices
        HINT: For long (signal=1): exit if price < entry * (1 - stop_loss)
        HINT: For short (signal=-1): exit if price > entry * (1 + stop_loss)

        Args:
            prices: Price series
            signals: Trading signals
            stop_loss_pct: Stop loss percentage series
            entry_prices: Entry price series

        Returns:
            Signals with stop loss applied
        """
        # YOUR CODE HERE
        pass

    # ========================================
    # Part 7: Main Strategy Logic
    # ========================================

    def run_strategy(
        self,
        prices: pd.Series,
        use_regime_sizing: bool = True,
        use_stop_loss: bool = True
    ) -> Dict:
        """
        Run complete regime-adaptive momentum strategy.

        🔴 TODO #16: Implement main strategy

        Steps:
        1. Calculate returns
        2. Detect volatility regime
        3. Detect trend regime
        4. Calculate momentum (short and long)
        5. Generate base signals
        6. Adjust signals for trend regime
        7. Adjust signals for volatility regime
        8. Calculate position sizing
        9. Apply stop losses (if enabled)
        10. Return results dict

        HINT: Use all the functions you implemented above
        HINT: Chain the steps together

        Args:
            prices: Price series
            use_regime_sizing: Whether to adjust size by regime
            use_stop_loss: Whether to apply stop losses

        Returns:
            Dictionary with:
            - 'signals': Final trading signals
            - 'position_size': Position size series
            - 'vol_regime': Volatility regime
            - 'trend_regime': Trend regime
            - 'momentum_short': Short-term momentum
            - 'momentum_long': Long-term momentum
            - 'stop_loss': Stop loss levels
        """
        # YOUR CODE HERE
        pass

    def calculate_strategy_returns(
        self,
        prices: pd.Series,
        signals: pd.Series,
        position_size: pd.Series
    ) -> pd.Series:
        """
        Calculate strategy returns from signals and position sizing.

        🟡 TODO #17: Calculate returns

        Formula: strategy_return = signal (lagged) * market_return * position_size

        HINT: Calculate market returns: prices.pct_change()
        HINT: Lag signals by 1 to avoid look-ahead bias
        HINT: Multiply by position size for regime adjustment

        Args:
            prices: Price series
            signals: Trading signals
            position_size: Position size series

        Returns:
            Strategy returns series
        """
        # YOUR CODE HERE
        pass


# ========================================
# Self-Test Function
# ========================================

def test_implementation():
    """
    Test your regime momentum implementation!

    🎯 Run this function to check if your code works:
        python regime_momentum.py
    """
    print("🧪 Testing Regime Momentum Implementation...\n")

    # Generate synthetic price data
    np.random.seed(42)
    n = 500
    dates = pd.date_range(end='2024-01-01', periods=n, freq='D')

    # Simulate trending market with regime changes
    trend = np.linspace(0, 0.3, n)
    volatility = np.where(np.arange(n) < n//2, 0.01, 0.03)  # Vol regime change
    returns = trend/n + np.random.normal(0, volatility, n)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name='SPY')

    # Create strategy
    strategy = RegimeMomentumStrategy()

    # Test 1: Volatility
    print("Test 1: Volatility Calculation")
    try:
        returns_series = prices.pct_change()
        vol = strategy.calculate_realized_volatility(returns_series)
        print(f"✅ Volatility calculated: mean={vol.mean():.2%}, max={vol.max():.2%}")
    except:
        print("❌ calculate_realized_volatility() not implemented")

    # Test 2: Vol Regime
    print("\nTest 2: Volatility Regime Detection")
    try:
        vol_regime = strategy.detect_volatility_regime(vol)
        regime_counts = vol_regime.value_counts()
        print(f"✅ Vol regime detected:")
        for regime, count in regime_counts.items():
            print(f"   {regime}: {count} days")
    except:
        print("❌ detect_volatility_regime() not implemented")

    # Test 3: Moving Averages
    print("\nTest 3: Moving Averages")
    try:
        ma_short, ma_long = strategy.calculate_moving_averages(prices)
        print(f"✅ MAs calculated: short={ma_short.iloc[-1]:.2f}, long={ma_long.iloc[-1]:.2f}")
    except:
        print("❌ calculate_moving_averages() not implemented")

    # Test 4: Trend Regime
    print("\nTest 4: Trend Regime Detection")
    try:
        trend_regime = strategy.detect_trend_regime(prices)
        regime_counts = trend_regime.value_counts()
        print(f"✅ Trend regime detected:")
        for regime, count in regime_counts.items():
            print(f"   {regime}: {count} days")
    except:
        print("❌ detect_trend_regime() not implemented")

    # Test 5: Momentum
    print("\nTest 5: Momentum Calculation")
    try:
        momentum_df = strategy.calculate_dual_momentum(prices)
        print(f"✅ Momentum calculated:")
        print(f"   Short: {momentum_df['mom_short'].iloc[-1]:.2%}")
        print(f"   Long: {momentum_df['mom_long'].iloc[-1]:.2%}")
    except:
        print("❌ calculate_dual_momentum() not implemented")

    # Test 6: Signals
    print("\nTest 6: Signal Generation")
    try:
        base_signals = strategy.generate_base_signals(
            momentum_df['mom_short'],
            momentum_df['mom_long']
        )
        num_trades = base_signals.diff().abs().sum() / 2
        print(f"✅ Signals generated: {num_trades:.0f} trades")
    except:
        print("❌ generate_base_signals() not implemented")

    # Test 7: Position Sizing
    print("\nTest 7: Position Sizing")
    try:
        position_size = strategy.calculate_regime_position_size(1.0, vol_regime)
        print(f"✅ Position sizing: mean={position_size.mean():.2f}, max={position_size.max():.2f}")
    except:
        print("❌ calculate_regime_position_size() not implemented")

    # Test 8: Full Strategy
    print("\nTest 8: Full Strategy")
    try:
        result = strategy.run_strategy(prices)
        signals = result['signals']
        num_trades = signals.diff().abs().sum() / 2
        time_in_market = (signals != 0).sum() / len(signals) * 100
        print(f"✅ Strategy run successfully!")
        print(f"   Trades: {num_trades:.0f}")
        print(f"   Time in market: {time_in_market:.1f}%")
    except:
        print("❌ run_strategy() not implemented")

    # Test 9: Returns
    print("\nTest 9: Strategy Returns")
    try:
        returns = strategy.calculate_strategy_returns(
            prices,
            result['signals'],
            result['position_size']
        )
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        print(f"✅ Returns calculated!")
        print(f"   Total return: {total_return*100:.2f}%")
        print(f"   Sharpe ratio: {sharpe:.2f}")
    except:
        print("❌ calculate_strategy_returns() not implemented")

    print("\n" + "="*50)
    print("🎉 Testing complete!")
    print("="*50)


if __name__ == "__main__":
    test_implementation()
