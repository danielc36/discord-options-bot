"""
indicators.py - Advanced Technical Indicators Suite
Implements 20+ professional-grade technical indicators with optimized parameters
"""

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.trend import ADXIndicator, MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class IndicatorSuite:
    """Comprehensive technical indicator calculator"""
    
    def __init__(self):
        self.min_periods = 25  # Minimum data points needed
        
    def add_all_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Add all technical indicators to dataframe
        
        Returns complete indicator suite or None if insufficient data
        """
        if len(df) < self.min_periods:
            logger.warning(f"Insufficient data for indicators: {len(df)} < {self.min_periods}")
            return None
            
        df = df.copy()
        
        try:
            # Trend Indicators
            df = self._add_trend_indicators(df)
            
            # Momentum Indicators
            df = self._add_momentum_indicators(df)
            
            # Volatility Indicators
            df = self._add_volatility_indicators(df)
            
            # Volume Indicators
            df = self._add_volume_indicators(df)
            
            # Custom Composite Indicators
            df = self._add_composite_indicators(df)
            
            # Clean up NaN values
            df = df.dropna()
            
            if len(df) < 10:
                logger.warning("Too few rows after indicator calculation")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return None
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        
        # Moving Averages (Multiple Timeframes)
        df["ema_9"] = EMAIndicator(close, window=9).ema_indicator()
        df["ema_21"] = EMAIndicator(close, window=21).ema_indicator()
        df["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
        df["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
        df["sma_200"] = SMAIndicator(close, window=200).sma_indicator()
        
        # MACD
        macd = MACD(close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        
        # ADX (Trend Strength)
        adx = ADXIndicator(high, low, close, window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()
        
        # Ichimoku Cloud (Advanced trend system)
        ichimoku = IchimokuIndicator(high, low)
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base"] = ichimoku.ichimoku_base_line()
        df["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators"""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        
        # RSI (Multiple periods for confluence)
        df["rsi_14"] = RSIIndicator(close, window=14).rsi()
        df["rsi_7"] = RSIIndicator(close, window=7).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        
        # Williams %R
        df["williams_r"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        
        # Rate of Change
        df["roc"] = ROCIndicator(close, window=12).roc()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures"""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        
        # Bollinger Bands
        bb = BollingerBands(close, window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()
        
        # Keltner Channels
        keltner = KeltnerChannel(high, low, close)
        df["keltner_high"] = keltner.keltner_channel_hband()
        df["keltner_low"] = keltner.keltner_channel_lband()
        
        # ATR (Average True Range)
        atr = AverageTrueRange(high, low, close, window=14)
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = (df["atr"] / close) * 100  # ATR as % of price
        
        # Historical Volatility
        df["std_20"] = close.rolling(20).std()
        df["std_50"] = close.rolling(50).std()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(high, low, close, volume)
        df["vwap"] = vwap.vwap
        
        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(close, volume)
        df["obv"] = obv.on_balance_volume()
        
        # Chaikin Money Flow
        cmf = ChaikinMoneyFlowIndicator(high, low, close, volume)
        df["cmf"] = cmf.chaikin_money_flow()
        
        # Volume Moving Average
        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma_20"]
        
        return df
    
    def _add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom composite indicators"""
        
        # Trend Strength Score (combines multiple trend indicators)
        df["trend_score"] = 0
        
        # EMA alignment
        if "ema_9" in df.columns and "ema_21" in df.columns:
            df["trend_score"] += np.where(df["ema_9"] > df["ema_21"], 1, -1)
        
        # Price vs VWAP
        if "vwap" in df.columns:
            df["trend_score"] += np.where(df["Close"] > df["vwap"], 1, -1)
        
        # ADX strength
        if "adx" in df.columns:
            df["trend_score"] += np.where(df["adx"] > 25, 1, 0)
        
        # MACD
        if "macd_diff" in df.columns:
            df["trend_score"] += np.where(df["macd_diff"] > 0, 1, -1)
        
        # Momentum Strength Score
        df["momentum_score"] = 0
        
        # RSI
        if "rsi_14" in df.columns:
            df["momentum_score"] += np.where(
                (df["rsi_14"] > 50) & (df["rsi_14"] < 80), 1,
                np.where((df["rsi_14"] < 50) & (df["rsi_14"] > 20), -1, 0)
            )
        
        # Stochastic
        if "stoch_k" in df.columns:
            df["momentum_score"] += np.where(
                (df["stoch_k"] > 50) & (df["stoch_k"] < 80), 1,
                np.where((df["stoch_k"] < 50) & (df["stoch_k"] > 20), -1, 0)
            )
        
        # Volatility Regime (categorize market volatility)
        if "atr_pct" in df.columns:
            df["volatility_regime"] = pd.cut(
                df["atr_pct"],
                bins=[0, 0.5, 1.0, 2.0, np.inf],
                labels=["very_low", "low", "medium", "high"]
            )
        
        # Support/Resistance Proximity
        if "bb_pct" in df.columns:
            df["bb_position"] = pd.cut(
                df["bb_pct"],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=["oversold", "low", "mid", "high", "overbought"]
            )
        
        return df


# Global instance
indicator_suite = IndicatorSuite()

def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Legacy function for backward compatibility"""
    return indicator_suite.add_all_indicators(df)
