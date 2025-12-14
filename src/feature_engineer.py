# src/feature_engineer.py
# Advanced Feature Engineering - 100+ features for 65-70% accuracy

import pandas as pd
import numpy as np
from scipy import stats
from config import *


class AdvancedFeatureEngineer:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def engineer_all_features(self, price_df, market_df=None, sentiment_df=None):
        """
        Create all features for maximum prediction power
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("ADVANCED FEATURE ENGINEERING")
            print(f"{'='*70}")
            print(f"Input rows: {len(price_df)}")
            print(f"Input columns: {len(price_df.columns)}")
        
        df = price_df.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Core features
        df = self._add_price_features(df)
        df = self._add_return_features(df)
        df = self._add_technical_indicators(df)
        df = self._add_volume_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_pattern_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_statistics(df)
        
        # Market context features
        if market_df is not None:
            df = self._add_market_features(df, market_df)
        
        # Sentiment features (if available)
        if sentiment_df is not None:
            df = self._add_sentiment_features(df, sentiment_df)
        
        # Target variables
        df = self._add_targets(df)
        
        if self.verbose:
            print(f"✅ Output rows: {len(df)}")
            print(f"✅ Output columns: {len(df.columns)}")
            print(f"✅ Total features created: {len(df.columns) - len(price_df.columns)}")
            print(f"{'='*70}\n")
        
        return df

    def _add_price_features(self, df):
        """Basic price-based features"""
        df['price_range'] = df['High'] - df['Low']
        df['price_range_pct'] = df['price_range'] / df['Close']
        
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['body_pct'] = df['body_size'] / df['price_range'].replace(0, np.nan)
        
        return df

    def _add_return_features(self, df):
        """Return calculations"""
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)
        df['return_60d'] = df['Close'].pct_change(60)
        
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Cumulative returns
        df['cum_return_20d'] = (1 + df['return_1d']).rolling(20).apply(lambda x: x.prod()) - 1
        
        return df

    def _add_technical_indicators(self, df):
        """Technical indicators"""
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss = -delta.clip(upper=0).rolling(RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(STOCH_PERIOD).min()
        high_14 = df['High'].rolling(STOCH_PERIOD).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # MACD
        ema_fast = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Moving Averages
        for period in SMA_PERIODS:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}']
        
        for period in EMA_PERIODS:
            df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Bollinger Bands
        bb_sma = df['Close'].rolling(BB_PERIOD).mean()
        bb_std = df['Close'].rolling(BB_PERIOD).std()
        df['bb_upper'] = bb_sma + BB_STD_DEV * bb_std
        df['bb_lower'] = bb_sma - BB_STD_DEV * bb_std
        df['bb_sma'] = bb_sma
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_sma']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(ATR_PERIOD).mean()
        df['atr_pct'] = df['atr'] / df['Close']
        
        # ADX (Average Directional Index)
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr_adx = tr.rolling(ADX_PERIOD).mean()
        plus_di = 100 * (plus_dm.rolling(ADX_PERIOD).mean() / atr_adx)
        minus_di = 100 * (minus_dm.rolling(ADX_PERIOD).mean() / atr_adx)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(ADX_PERIOD).mean()
        
        # CCI (Commodity Channel Index)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(CCI_PERIOD).mean()
        mad = lambda x: np.mean(np.abs(x - np.mean(x)))
        df['cci'] = (tp - sma_tp) / (0.015 * tp.rolling(CCI_PERIOD).apply(mad))
        
        return df

    def _add_volume_features(self, df):
        """Volume-based features"""
        df['volume_sma'] = df['Volume'].rolling(VOLUME_SMA_PERIOD).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['volume_change'] = df['Volume'].pct_change()
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        
        # Volume Price Trend
        df['vpt'] = (df['Volume'] * df['return_1d']).cumsum()
        
        # Money Flow Index
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        
        positive_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
        negative_mf = mf.where(tp < tp.shift(), 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return df

    def _add_momentum_features(self, df):
        """Momentum indicators"""
        for window in MOMENTUM_WINDOWS:
            df[f'momentum_{window}'] = df['Close'] - df['Close'].shift(window)
            df[f'roc_{window}'] = df['Close'].pct_change(window) * 100
        
        # Williams %R
        high_14 = df['High'].rolling(14).max()
        low_14 = df['Low'].rolling(14).min()
        df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
        
        return df

    def _add_volatility_features(self, df):
        """Volatility measures"""
        df['volatility_20'] = df['return_1d'].rolling(20).std()
        df['volatility_60'] = df['return_1d'].rolling(60).std()
        
        # Parkinson volatility (uses High/Low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(df['High'] / df['Low']) ** 2).rolling(20).mean()
        )
        
        # Historical volatility ratio
        df['hvr'] = df['volatility_20'] / df['volatility_60']
        
        return df

    def _add_pattern_features(self, df):
        """Price pattern recognition"""
        # Trend strength
        df['trend_strength'] = abs(df['Close'] - df['Close'].shift(20)) / df['atr']
        
        # Higher highs / Lower lows
        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        # Gap detection
        df['gap_up'] = ((df['Open'] > df['High'].shift(1))).astype(int)
        df['gap_down'] = ((df['Open'] < df['Low'].shift(1))).astype(int)
        
        return df

    def _add_lag_features(self, df):
        """Lagged features"""
        for lag in LAG_PERIODS:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            df[f'macd_lag_{lag}'] = df['macd'].shift(lag)
        
        return df

    def _add_rolling_statistics(self, df):
        """Rolling statistical features"""
        windows = [5, 10, 20]
        
        for window in windows:
            # Rolling mean
            df[f'close_mean_{window}'] = df['Close'].rolling(window).mean()
            
            # Rolling std
            df[f'close_std_{window}'] = df['Close'].rolling(window).std()
            
            # Rolling skewness
            df[f'return_skew_{window}'] = df['return_1d'].rolling(window).skew()
            
            # Rolling kurtosis
            df[f'return_kurt_{window}'] = df['return_1d'].rolling(window).kurt()
        
        return df

    def _add_market_features(self, df, market_df):
        """Market context features"""
        # Merge with market data on date
        market_data = market_df[['Date', 'Close']].copy()
        market_data.columns = ['Date', 'market_close']
        
        df = df.merge(market_data, on='Date', how='left')
        
        df['market_return'] = df['market_close'].pct_change()
        df['correlation_market'] = df['return_1d'].rolling(60).corr(df['market_return'])
        df['beta'] = df['return_1d'].rolling(60).cov(df['market_return']) / df['market_return'].rolling(60).var()
        
        return df

    def _add_sentiment_features(self, df, sentiment_df):
        """Sentiment features (if available)"""
        df = df.merge(sentiment_df, on='Date', how='left')
        df['sentiment'] = df['sentiment'].fillna(0)
        df['sentiment_sma'] = df['sentiment'].rolling(5).mean()
        return df

    def _add_targets(self, df):
        """Create target variables"""
        # Price targets
        df['target_price_1d'] = df['Close'].shift(-HORIZON_1D)
        df['target_price_5d'] = df['Close'].shift(-HORIZON_5D)
        
        # Return targets
        df['target_return_1d'] = (df['target_price_1d'] / df['Close']) - 1
        df['target_return_5d'] = (df['target_price_5d'] / df['Close']) - 1
        
        # Direction targets (classification)
        df['target_direction_1d'] = (df['target_return_1d'] > 0).astype(int)
        df['target_direction_5d'] = (df['target_return_5d'] > 0).astype(int)
        
        return df


def engineer_features_for_ticker(ticker, market_data=None):
    """
    Convenience function to engineer features for a single ticker
    """
    import os
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    # Load raw data
    raw_path = os.path.join(RAW_DATA_DIR, f"{ticker}_ohlcv.csv")
    if not os.path.exists(raw_path):
        print(f"❌ Raw data not found for {ticker}")
        return None
    
    df = pd.read_csv(raw_path)
    
    # 🔥 CRITICAL FIX: Clean the data
    # Remove header row if it exists as data
    df = df[df['Close'] != 'AAPL']  # Remove rows where Close is the ticker name
    df = df[df['Close'] != ticker]
    
    # Convert columns to numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Drop rows with NaN in critical columns
    df = df.dropna(subset=['Date', 'Close'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"✅ Cleaned data: {len(df)} rows")
    
    # Engineer features
    engineer = AdvancedFeatureEngineer(verbose=True)
    df_engineered = engineer.engineer_all_features(df, market_df=market_data)
    
    # Save
    output_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_engineered.csv")
    df_engineered.to_csv(output_path, index=False)
    
    print(f"✅ Saved to {output_path}")
    return df_engineered


if __name__ == "__main__":
    # Test feature engineering on AAPL
    print("Testing feature engineering on AAPL...")
    df = engineer_features_for_ticker("AAPL")
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")