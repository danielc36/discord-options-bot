"""
market.py - Production-Grade Market Data Handler
Handles data fetching, caching, validation, and preprocessing
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataError(Exception):
    """Custom exception for market data issues"""
    pass

class DataQualityChecker:
    """Validates data quality before use"""
    
    @staticmethod
    def check_data_integrity(df: pd.DataFrame, min_rows: int = 50) -> bool:
        """Validate dataframe has sufficient quality data"""
        if df is None or df.empty:
            logger.warning("DataFrame is None or empty")
            return False
            
        if len(df) < min_rows:
            logger.warning(f"Insufficient data: {len(df)} rows (need {min_rows})")
            return False
            
        # Check for excessive NaN values
        nan_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if nan_pct > 0.1:  # More than 10% NaN
            logger.warning(f"Too many NaN values: {nan_pct:.1%}")
            return False
            
        # Check for data gaps
        if 'Close' in df.columns:
            price_range = df['Close'].max() - df['Close'].min()
            if price_range == 0:
                logger.warning("No price movement detected")
                return False
                
        # Check for obvious data errors (negative prices, zero volume days)
        if 'Close' in df.columns and (df['Close'] <= 0).any():
            logger.warning("Negative or zero prices detected")
            return False
            
        return True
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """Remove statistical outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers_removed = df[
            (df[column] >= lower_bound) & 
            (df[column] <= upper_bound)
        ].copy()
        
        removed_count = len(df) - len(outliers_removed)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outliers from {column}")
            
        return outliers_removed


class MarketDataFetcher:
    """Advanced market data fetching with retry logic and caching"""
    
    def __init__(self, cache_duration_seconds: int = 60):
        self.cache_duration = cache_duration_seconds
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self.quality_checker = DataQualityChecker()
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache:
            return False
            
        _, timestamp = self._cache[cache_key]
        age = time.time() - timestamp
        
        return age < self.cache_duration
    
    def get_stock_data(
        self, 
        symbol: str, 
        interval: str = "1m", 
        period: str = "1d",
        retry_attempts: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with caching and retry logic
        
        Args:
            symbol: Stock ticker (e.g., 'SPY')
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            period: Historical period (1d, 5d, 1mo, 3mo, 1y)
            retry_attempts: Number of retry attempts on failure
            
        Returns:
            DataFrame with OHLCV data or None on failure
        """
        cache_key = f"{symbol}_{interval}_{period}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {cache_key}")
            df, _ = self._cache[cache_key]
            return df.copy()
        
        # Fetch with retry logic
        for attempt in range(retry_attempts):
            try:
                logger.info(f"Fetching {symbol} {interval} data (attempt {attempt + 1}/{retry_attempts})")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"Empty dataframe returned for {symbol}")
                    if attempt < retry_attempts - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    return None
                
                # Validate data quality
                if not self.quality_checker.check_data_integrity(df):
                    if attempt < retry_attempts - 1:
                        time.sleep(1)
                        continue
                    return None
                
                # Clean data
                df = df.dropna()
                df = self.quality_checker.detect_outliers(df, 'Close')
                
                # Cache the result
                self._cache[cache_key] = (df.copy(), time.time())
                
                logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
        
        return None
    
    def get_multi_timeframe_data(
        self, 
        symbol: str, 
        intervals: list = ["1m", "5m", "15m"]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple timeframes at once
        
        Returns:
            Dictionary mapping interval -> DataFrame
        """
        data = {}
        
        for interval in intervals:
            df = self.get_stock_data(symbol, interval=interval)
            if df is not None:
                data[interval] = df
                
        return data
    
    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        logger.info("Cache cleared")


# Global instance for easy import
market_data = MarketDataFetcher()

def get_stock_df(symbol: str, interval: str = "1m", period: str = "1d") -> Optional[pd.DataFrame]:
    """Legacy function for backward compatibility"""
    return market_data.get_stock_data(symbol, interval, period)
