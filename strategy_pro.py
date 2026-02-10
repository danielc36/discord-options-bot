"""
strategy_pro.py - Advanced ML Trading Strategy
Implements ensemble ML, regime detection, and multi-factor signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market condition types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class SignalStrength(Enum):
    """Signal confidence levels"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NONE = 0

@dataclass
class TradingSignal:
    """Structured trading signal with metadata"""
    direction: str  # BUY, SELL, HOLD
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float
    target_price: float
    stop_loss: float
    regime: MarketRegime
    risk_reward_ratio: float
    contributing_factors: Dict[str, int]  # Which indicators agree
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            "direction": self.direction,
            "strength": self.strength.name,
            "confidence": round(self.confidence, 3),
            "entry": round(self.entry_price, 2),
            "target": round(self.target_price, 2),
            "stop": round(self.stop_loss, 2),
            "regime": self.regime.value,
            "r_r_ratio": round(self.risk_reward_ratio, 2),
            "factors": self.contributing_factors
        }


class RegimeDetector:
    """Detects current market regime"""
    
    @staticmethod
    def detect_regime(df: pd.DataFrame) -> MarketRegime:
        """
        Analyze market conditions and classify regime
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Current market regime
        """
        last_row = df.iloc[-1]
        recent_df = df.tail(20)
        
        # Check volatility first
        if "atr_pct" in df.columns:
            avg_atr = recent_df["atr_pct"].mean()
            if avg_atr > 1.5:
                return MarketRegime.HIGH_VOLATILITY
            elif avg_atr < 0.5:
                return MarketRegime.LOW_VOLATILITY
        
        # Check trend
        if "adx" in df.columns and "ema_9" in df.columns and "ema_21" in df.columns:
            adx = last_row["adx"]
            ema9 = last_row["ema_9"]
            ema21 = last_row["ema_21"]
            
            if adx > 25:  # Strong trend
                if ema9 > ema21:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.RANGING
        
        # Default fallback
        return MarketRegime.RANGING


class MultiFactorStrategy:
    """Advanced multi-factor trading strategy"""
    
    def __init__(
        self,
        min_confidence: float = 0.65,
        min_signal_strength: int = 3,
        risk_reward_min: float = 1.5
    ):
        self.min_confidence = min_confidence
        self.min_signal_strength = min_signal_strength
        self.risk_reward_min = risk_reward_min
        self.regime_detector = RegimeDetector()
        
    def analyze(
        self, 
        df1m: pd.DataFrame, 
        df15m: pd.DataFrame,
        ml_model=None
    ) -> TradingSignal:
        """
        Generate comprehensive trading signal
        
        Args:
            df1m: 1-minute timeframe data with indicators
            df15m: 15-minute timeframe data with indicators
            ml_model: Optional ML model for confidence scoring
            
        Returns:
            TradingSignal with full analysis
        """
        
        # Get current market state
        price = df1m["Close"].iloc[-1]
        regime = self.regime_detector.detect_regime(df15m)
        
        # Multi-timeframe analysis
        factors_1m = self._analyze_timeframe(df1m, "1m")
        factors_15m = self._analyze_timeframe(df15m, "15m")
        
        # Combine factors with weighting
        combined_factors = self._combine_factors(factors_1m, factors_15m)
        
        # Determine direction and strength
        direction, strength = self._calculate_direction(combined_factors)
        
        # Calculate price targets
        atr = df1m["atr"].iloc[-1]
        target, stop = self._calculate_targets(price, direction, atr, regime)
        
        # ML confidence (if available)
        ml_confidence = 0.5  # Default neutral
        if ml_model is not None:
            try:
                features = self._build_ml_features(df1m, df15m)
                ml_confidence = ml_model.predict_proba(features)[0][1]
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # Calculate risk/reward
        if direction == "BUY":
            risk = price - stop
            reward = target - price
        elif direction == "SELL":
            risk = stop - price
            reward = price - target
        else:
            risk = reward = atr  # Default for HOLD
            
        risk_reward = reward / risk if risk > 0 else 0
        
        # Build final signal
        signal = TradingSignal(
            direction=direction,
            strength=strength,
            confidence=ml_confidence,
            entry_price=price,
            target_price=target,
            stop_loss=stop,
            regime=regime,
            risk_reward_ratio=risk_reward,
            contributing_factors=combined_factors
        )
        
        # Validate signal quality
        if not self._validate_signal(signal):
            signal.direction = "HOLD"
            signal.strength = SignalStrength.NONE
        
        return signal
    
    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, int]:
        """
        Analyze single timeframe for bullish/bearish factors
        
        Returns:
            Dictionary of factor scores (-1, 0, +1)
        """
        factors = {}
        last = df.iloc[-1]
        
        # Trend factors
        if "ema_9" in df.columns and "ema_21" in df.columns:
            factors[f"ema_cross_{timeframe}"] = 1 if last["ema_9"] > last["ema_21"] else -1
        
        if "macd_diff" in df.columns:
            factors[f"macd_{timeframe}"] = 1 if last["macd_diff"] > 0 else -1
        
        if "adx" in df.columns and "adx_pos" in df.columns and "adx_neg" in df.columns:
            if last["adx"] > 20:  # Only count if trend is strong
                factors[f"adx_{timeframe}"] = 1 if last["adx_pos"] > last["adx_neg"] else -1
        
        # Momentum factors
        if "rsi_14" in df.columns:
            rsi = last["rsi_14"]
            if 30 < rsi < 70:  # Valid range
                factors[f"rsi_{timeframe}"] = 1 if rsi > 50 else -1
        
        if "stoch_k" in df.columns:
            stoch = last["stoch_k"]
            if 20 < stoch < 80:  # Valid range
                factors[f"stoch_{timeframe}"] = 1 if stoch > 50 else -1
        
        # Volume factors
        if "vwap" in df.columns:
            factors[f"vwap_{timeframe}"] = 1 if last["Close"] > last["vwap"] else -1
        
        if "cmf" in df.columns:
            factors[f"cmf_{timeframe}"] = 1 if last["cmf"] > 0 else -1
        
        # Volatility positioning
        if "bb_pct" in df.columns:
            bb_pct = last["bb_pct"]
            if bb_pct < 0.2:
                factors[f"bb_oversold_{timeframe}"] = 1  # Bounce potential
            elif bb_pct > 0.8:
                factors[f"bb_overbought_{timeframe}"] = -1  # Pullback potential
        
        return factors
    
    def _combine_factors(
        self, 
        factors_1m: Dict[str, int], 
        factors_15m: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Combine multi-timeframe factors with weighting
        
        Higher timeframe (15m) gets more weight
        """
        combined = {}
        
        # Add 1m factors with 1x weight
        for key, value in factors_1m.items():
            combined[key] = value
        
        # Add 15m factors with 1.5x weight (higher timeframe more important)
        for key, value in factors_15m.items():
            # Weight higher timeframe more
            weighted_key = f"{key}_weighted"
            combined[weighted_key] = int(value * 1.5)
        
        return combined
    
    def _calculate_direction(
        self, 
        factors: Dict[str, int]
    ) -> Tuple[str, SignalStrength]:
        """
        Calculate overall direction from factor scores
        
        Returns:
            (direction, strength) tuple
        """
        # Sum all factor scores
        total_score = sum(factors.values())
        num_factors = len(factors)
        
        # Calculate average score
        avg_score = total_score / num_factors if num_factors > 0 else 0
        
        # Determine direction
        if avg_score > 0.3:
            direction = "BUY"
        elif avg_score < -0.3:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        # Determine strength based on factor agreement
        abs_score = abs(avg_score)
        
        if abs_score >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif abs_score >= 0.6:
            strength = SignalStrength.STRONG
        elif abs_score >= 0.4:
            strength = SignalStrength.MODERATE
        elif abs_score >= 0.2:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK
        
        return direction, strength
    
    def _calculate_targets(
        self,
        price: float,
        direction: str,
        atr: float,
        regime: MarketRegime
    ) -> Tuple[float, float]:
        """
        Calculate target and stop loss based on regime
        
        Returns:
            (target_price, stop_loss_price)
        """
        # Adjust multipliers based on regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            target_mult = 2.0
            stop_mult = 1.5
        elif regime == MarketRegime.LOW_VOLATILITY:
            target_mult = 1.0
            stop_mult = 0.8
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            target_mult = 2.5
            stop_mult = 1.0
        else:  # RANGING
            target_mult = 1.5
            stop_mult = 1.0
        
        if direction == "BUY":
            target = price + (atr * target_mult)
            stop = price - (atr * stop_mult)
        elif direction == "SELL":
            target = price - (atr * target_mult)
            stop = price + (atr * stop_mult)
        else:  # HOLD
            target = price
            stop = price
        
        return round(target, 2), round(stop, 2)
    
    def _build_ml_features(
        self, 
        df1m: pd.DataFrame, 
        df15m: pd.DataFrame
    ) -> pd.DataFrame:
        """Build feature vector for ML model"""
        last_1m = df1m.iloc[-1]
        last_15m = df15m.iloc[-1]
        
        features = {
            "stoch_1m": last_1m.get("stoch_k", 50),
            "bb_width_1m": last_1m.get("bb_width", 0),
            "atr_1m": last_1m.get("atr", 0),
            "std_1m": last_1m.get("std_20", 0),
            "vwap_1m": last_1m.get("vwap", last_1m["Close"]),
            "adx_15m": last_15m.get("adx", 0),
            "stoch_15m": last_15m.get("stoch_k", 50),
            "bb_width_15m": last_15m.get("bb_width", 0),
            "std_15m": last_15m.get("std_20", 0),
        }
        
        return pd.DataFrame([features])
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """
        Final validation of signal quality
        
        Returns:
            True if signal passes all filters
        """
        # Check minimum confidence
        if signal.confidence < self.min_confidence:
            logger.debug(f"Signal failed confidence check: {signal.confidence:.2%}")
            return False
        
        # Check minimum strength
        if signal.strength.value < self.min_signal_strength:
            logger.debug(f"Signal failed strength check: {signal.strength.name}")
            return False
        
        # Check risk/reward ratio
        if signal.risk_reward_ratio < self.risk_reward_min:
            logger.debug(f"Signal failed R:R check: {signal.risk_reward_ratio:.2f}")
            return False
        
        # Don't trade in extreme volatility unless very confident
        if signal.regime == MarketRegime.HIGH_VOLATILITY and signal.confidence < 0.75:
            logger.debug("High volatility requires higher confidence")
            return False
        
        return True


# Global instance
strategy = MultiFactorStrategy(
    min_confidence=0.65,
    min_signal_strength=3,
    risk_reward_min=1.5
)
