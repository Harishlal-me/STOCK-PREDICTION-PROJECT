# src/feature_engineer.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import (
    PROCESSED_DATA_DIR, RSI_PERIOD, MACD_FAST, MACD_SLOW,
    MACD_SIGNAL, BB_PERIOD, BB_STD_DEV, SMA_PERIODS, EMA_PERIODS,
    VOLATILITY_PERIOD, LAG_DAYS, CORRELATION_WINDOW, VERBOSE,
    NORMALIZE_METHOD
)


# =========================
# DATA CLEANER
# =========================
class DataCleaner:
    def __init__(self, verbose=VERBOSE):
        self.verbose = verbose

    def clean_data(self, df):
        df = df.copy()

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Date' in df.columns:
            df = df.sort_values('Date')

        df = df.ffill().bfill()
        df = df.drop_duplicates(subset=['Date'])

        df['return'] = pd.to_numeric(df['Close'].pct_change(), errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['Close'])

        return df.reset_index(drop=True)


# =========================
# TECHNICAL INDICATORS
# =========================
class TechnicalIndicators:
    @staticmethod
    def compute_rsi(df, period=RSI_PERIOD):
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_macd(df):
        ema_fast = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    @staticmethod
    def compute_bollinger_bands(df):
        sma = df['Close'].rolling(BB_PERIOD).mean()
        std = df['Close'].rolling(BB_PERIOD).std()
        return sma + BB_STD_DEV * std, sma, sma - BB_STD_DEV * std

    @staticmethod
    def compute_moving_averages(df):
        for p in SMA_PERIODS:
            df[f'sma_{p}'] = df['Close'].rolling(p).mean()
        for p in EMA_PERIODS:
            df[f'ema_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
        return df

    @staticmethod
    def compute_volatility(df):
        df['volatility'] = df['Close'].pct_change().rolling(VOLATILITY_PERIOD).std()
        return df

    @staticmethod
    def compute_atr(df, period=14):
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                (df['High'] - df['Close'].shift()).abs(),
                (df['Low'] - df['Close'].shift()).abs()
            )
        )
        df['atr'] = tr.rolling(period).mean()
        return df

    @staticmethod
    def compute_obv(df):
        return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()


# =========================
# MARKET FEATURES
# =========================
class MarketFeatures:
    @staticmethod
    def add_returns(df):
        df['daily_return'] = pd.to_numeric(df['Close'].pct_change(), errors='coerce')
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df

    @staticmethod
    def add_price_ratios(df):
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['price_to_sma20'] = df['Close'] / df['sma_20']
        return df

    @staticmethod
    def add_volume_features(df):
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['volume_change'] = df['Volume'].pct_change()
        return df

    @staticmethod
    def add_market_correlation(df, macro_df):
        macro = macro_df.copy()

        if isinstance(macro.columns, pd.MultiIndex):
            macro.columns = [c[0] if isinstance(c, tuple) else c for c in macro.columns]

        if 'Date' not in macro.columns or 'SP500_Return' not in macro.columns:
            return df

        df = df.merge(macro[['Date', 'SP500_Return']], on='Date', how='left')

        df['correlation_sp500'] = (
            df['daily_return']
            .rolling(CORRELATION_WINDOW)
            .corr(df['SP500_Return'])
        )

        df['market_return_lag1'] = df['SP500_Return'].shift(1)
        return df


# =========================
# LAG FEATURES
# =========================
class LagFeatures:
    @staticmethod
    def add_lag_features(df, lag_days=LAG_DAYS):
        for lag in lag_days:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        return df


# =========================
# FEATURE ENGINEER
# =========================
class FeatureEngineer:
    def __init__(self, verbose=VERBOSE):
        self.verbose = verbose
        self.cleaner = DataCleaner(verbose)

    def engineer_features(self, price_df, macro_df=None, sentiment_df=None):
        df = price_df.copy()

        if self.verbose:
            print(f"  → Initial records: {len(df)}")

        df = self.cleaner.clean_data(df)
        df = MarketFeatures.add_returns(df)

        df['rsi'] = TechnicalIndicators.compute_rsi(df)
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalIndicators.compute_macd(df)
        df['bb_upper'], df['bb_sma'], df['bb_lower'] = TechnicalIndicators.compute_bollinger_bands(df)
        df = TechnicalIndicators.compute_moving_averages(df)
        df = TechnicalIndicators.compute_volatility(df)
        df = TechnicalIndicators.compute_atr(df)
        df['obv'] = TechnicalIndicators.compute_obv(df)

        df = MarketFeatures.add_price_ratios(df)
        df = MarketFeatures.add_volume_features(df)

        if macro_df is not None:
            df = MarketFeatures.add_market_correlation(df, macro_df)
            macro_clean = macro_df.copy()
            if isinstance(macro_clean.columns, pd.MultiIndex):
                macro_clean.columns = [c[0] if isinstance(c, tuple) else c for c in macro_clean.columns]
            df = df.merge(macro_clean, on='Date', how='left')
            df = df.ffill().bfill()

        if sentiment_df is not None and not sentiment_df.empty:
            df['date'] = pd.to_datetime(df['Date']).dt.date
            df = df.merge(
                sentiment_df[['date', 'sentiment_mean', 'sentiment_rolling', 'article_count']],
                on='date',
                how='left'
            )
            df = df.ffill().fillna(0)
            df = df.drop(columns=['date'])
        else:
            df['sentiment_mean'] = 0.0
            df['sentiment_rolling'] = 0.0
            df['article_count'] = 0

        df = LagFeatures.add_lag_features(df)
        df = df.dropna()

        if self.verbose:
            print(f"  → After engineering: {len(df)} records")

        return df

    def save_engineered_data(self, df, ticker):
        path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_engineered.csv")
        df.to_csv(path, index=False)
        print(f"✓ Saved engineered data to {path}")
        return path


# =========================
# TARGETS
# =========================
def create_regression_target(df, horizon=1):
    df['target_price'] = df['Close'].shift(-horizon)
    return df

def create_classification_target(df, horizon=1):
    df['target_direction'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    return df

def create_return_target(df, horizon=1):
    df['target_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df
