# notebook_03_fe.py
# Feature Engineering Notebook

"""
NOTEBOOK: 03_Feature_Engineering
PURPOSE: Engineer features from raw market + macro data
"""

# ============================================================
# Cell 1: Imports and Setup
# ============================================================

import sys
from pathlib import Path

sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_collector import MarketDataCollector, MacroDataCollector
from src.feature_engineer import FeatureEngineer
from config import STOCKS

# Results directory (SAFE)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("✓ Imports successful")

# ============================================================
# Cell 2: Load Data
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

market_collector = MarketDataCollector()
macro_collector = MacroDataCollector()

ticker = "AAPL"

price_df = market_collector.load_stock_data(ticker)
print(f"✓ Loaded {ticker}: {len(price_df)} records")

macro_df = macro_collector.fetch_all_macro_data(period="5y")
print(f"✓ Loaded macro data: {len(macro_df)} records")

# ============================================================
# Cell 3: Engineer Features
# ============================================================

print("\n" + "=" * 70)
print("ENGINEERING FEATURES")
print("=" * 70)

engineer = FeatureEngineer(verbose=True)

engineered_df = engineer.engineer_features(
    price_df=price_df,
    macro_df=macro_df,
    sentiment_df=None
)

print("\n✓ Feature engineering complete!")
print(f"  Shape: {engineered_df.shape}")
print(f"  Features: {engineered_df.shape[1]}")

# ============================================================
# Cell 4: Feature Groups
# ============================================================

print("\n" + "=" * 70)
print("ENGINEERED FEATURES")
print("=" * 70)

exclude = ['Date', 'Ticker', 'target_price', 'target_direction', 'target_return']
feature_cols = [c for c in engineered_df.columns if c not in exclude]
print(f"\nTotal features: {len(feature_cols)}")

technical_features = [
    col for col in feature_cols
    if any(x in col.lower() for x in ['rsi', 'macd', 'bb_', 'sma_', 'ema_', 'atr', 'obv', 'volatility'])
]

lag_features = [col for col in feature_cols if 'lag' in col.lower()]

market_features = [
    col for col in feature_cols
    if any(x in col.lower() for x in ['return', 'volume', 'ratio'])
]

macro_features = [
    col for col in feature_cols
    if any(x in col.lower() for x in ['vix', 'sp500', 'interest'])
]

sentiment_features = [col for col in feature_cols if 'sentiment' in col.lower()]

print("\nFeature Breakdown:")
print(f"  Technical indicators: {len(technical_features)}")
print(f"  Lag features: {len(lag_features)}")
print(f"  Market features: {len(market_features)}")
print(f"  Macro features: {len(macro_features)}")
print(f"  Sentiment features: {len(sentiment_features)}")

# ============================================================
# Cell 5: Feature Statistics
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

if technical_features:
    print("\nTechnical Indicators (sample):")
    print(engineered_df[technical_features[:5]].describe())

if market_features:
    print("\nMarket Features (sample):")
    print(engineered_df[market_features[:5]].describe())

# ============================================================
# Cell 6: Data Quality Check
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

missing = engineered_df[feature_cols].isnull().sum()
missing_pct = (missing / len(engineered_df) * 100).round(2)

print(f"Total records: {len(engineered_df)}")
print(f"Total missing values: {missing.sum()}")

if missing.sum() > 0:
    print("\nTop missing features:")
    print(missing[missing > 0].head(10))

# ============================================================
# Cell 7: Feature Correlation
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CORRELATION")
print("=" * 70)

corr_features = (
    technical_features[:5]
    + macro_features
    + ['daily_return', 'volatility']
)

corr_features = [c for c in corr_features if c in engineered_df.columns]

corr_matrix = engineered_df[corr_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title(f"{ticker} - Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_correlation_matrix.png", dpi=100, bbox_inches="tight")
plt.show()

print("✓ Saved correlation matrix")

# ============================================================
# Cell 8: Technical Indicator Visualizations
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(16, 10))

axes[0, 0].plot(engineered_df["Date"], engineered_df["Close"], label="Close")
axes[0, 0].plot(engineered_df["Date"], engineered_df["sma_20"], label="SMA 20")
axes[0, 0].plot(engineered_df["Date"], engineered_df["sma_50"], label="SMA 50")
axes[0, 0].set_title("Price & Moving Averages")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(engineered_df["Date"], engineered_df["Close"], label="Close")
axes[0, 1].fill_between(
    engineered_df["Date"],
    engineered_df["bb_lower"],
    engineered_df["bb_upper"],
    alpha=0.2,
    label="Bollinger Bands"
)
axes[0, 1].legend()
axes[0, 1].set_title("Bollinger Bands")
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(engineered_df["Date"], engineered_df["rsi"])
axes[1, 0].axhline(70, linestyle="--", color="r", alpha=0.5)
axes[1, 0].axhline(30, linestyle="--", color="g", alpha=0.5)
axes[1, 0].set_ylim(0, 100)
axes[1, 0].set_title("RSI")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(engineered_df["Date"], engineered_df["macd"], label="MACD")
axes[1, 1].plot(engineered_df["Date"], engineered_df["macd_signal"], label="Signal")
axes[1, 1].bar(engineered_df["Date"], engineered_df["macd_hist"], alpha=0.3)
axes[1, 1].legend()
axes[1, 1].set_title("MACD")
axes[1, 1].grid(alpha=0.3)

axes[2, 0].bar(engineered_df["Date"], engineered_df["Volume"], alpha=0.6)
axes[2, 0].plot(engineered_df["Date"], engineered_df["volume_sma"], color="r")
axes[2, 0].set_title("Volume")
axes[2, 0].grid(alpha=0.3)

axes[2, 1].plot(engineered_df["Date"], engineered_df["volatility"])
axes[2, 1].set_title("Volatility")
axes[2, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_technical_indicators.png", dpi=100, bbox_inches="tight")
plt.show()

print("✓ Saved technical indicators")

# ============================================================
# Cell 9: Create Targets (SAFE)
# ============================================================

print("\n" + "=" * 70)
print("CREATING TARGET VARIABLES")
print("=" * 70)

engineered_df["Close"] = pd.to_numeric(engineered_df["Close"], errors="coerce")
engineered_df = engineered_df.dropna(subset=["Close"])

from src.feature_engineer import (
    create_regression_target,
    create_classification_target,
    create_return_target
)

engineered_df = create_regression_target(engineered_df, horizon=1)
engineered_df = create_classification_target(engineered_df, horizon=1)
engineered_df = create_return_target(engineered_df, horizon=1)

print("✓ Targets created")

# ============================================================
# Cell 10: Feature Distributions
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].hist(engineered_df["daily_return"].dropna(), bins=50)
axes[0, 0].set_title("Daily Return")

axes[0, 1].hist(engineered_df["rsi"].dropna(), bins=50)
axes[0, 1].set_title("RSI")

axes[1, 0].hist(engineered_df["volatility"].dropna(), bins=50)
axes[1, 0].set_title("Volatility")

axes[1, 1].hist(engineered_df["target_return"].dropna(), bins=50)
axes[1, 1].set_title("Target Return")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_feature_distributions.png", dpi=100, bbox_inches="tight")
plt.show()

print("✓ Saved feature distributions")

# ============================================================
# Cell 11: Save Engineered Data
# ============================================================

print("\n" + "=" * 70)
print("SAVING ENGINEERED DATA")
print("=" * 70)

filepath = engineer.save_engineered_data(engineered_df, ticker)
print(f"✓ Saved engineered data to {filepath}")

print("\nProceed to notebook_04_training")
