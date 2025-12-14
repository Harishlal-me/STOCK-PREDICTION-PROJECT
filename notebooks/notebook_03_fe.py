"""
NOTEBOOK 03: Feature Engineering
Purpose: Engineer ML-ready features from OHLCV stock data
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.feature_engineer import FeatureEngineer
from config import RAW_DATA_DIR

sns.set_style("darkgrid")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("✓ Imports successful")

# =============================================================================
# LOAD ALL STOCK DATA
# =============================================================================

print("\n" + "=" * 70)
print("LOADING STOCK DATA")
print("=" * 70)

data_path = Path(RAW_DATA_DIR)
csv_files = list(data_path.glob("*_ohlcv.csv")) + list(data_path.glob("*_raw.csv"))

engineer = FeatureEngineer(verbose=True)
all_engineered = []

for file in csv_files:
    ticker = file.stem.split("_")[0]
    print(f"\nProcessing {ticker}")

    df = pd.read_csv(file, parse_dates=["Date"])

    # Force numeric conversion (CRITICAL)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Close"], inplace=True)

    if len(df) < 100:
        print(f"⚠ Skipping {ticker}: insufficient data ({len(df)} rows)")
        continue

    # Engineer features (NO macro, NO sentiment)
    engineered_df = engineer.engineer_features(
        price_df=df,
        macro_df=None,
        sentiment_df=None
    )

    engineered_df["Ticker"] = ticker
    all_engineered.append(engineered_df)

    print(f"✓ Engineered {ticker}: {engineered_df.shape}")

# =============================================================================
# COMBINE ALL STOCKS
# =============================================================================

engineered_df = pd.concat(all_engineered, ignore_index=True)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Total records : {len(engineered_df)}")
print(f"Total features: {engineered_df.shape[1]}")

# =============================================================================
# CREATE TARGET VARIABLES
# =============================================================================

print("\nCreating target variables...")

from src.feature_engineer import (
    create_regression_target,
    create_classification_target,
    create_return_target
)

engineered_df = create_regression_target(engineered_df, horizon=1)
engineered_df = create_classification_target(engineered_df, horizon=1)
engineered_df = create_return_target(engineered_df, horizon=1)

print("✓ Targets created")

# =============================================================================
# FEATURE GROUPS
# =============================================================================

exclude = ["Date", "Ticker", "target_price", "target_direction", "target_return"]
feature_cols = [c for c in engineered_df.columns if c not in exclude]

technical_features = [c for c in feature_cols if any(
    x in c.lower() for x in ["rsi", "macd", "sma", "ema", "bb_", "atr", "obv"]
)]

lag_features = [c for c in feature_cols if "lag" in c.lower()]
market_features = [c for c in feature_cols if any(
    x in c.lower() for x in ["return", "volume", "volatility"]
)]

print("\nFeature Breakdown")
print(f"  Technical : {len(technical_features)}")
print(f"  Lag       : {len(lag_features)}")
print(f"  Market    : {len(market_features)}")
print(f"  Total     : {len(feature_cols)}")

# =============================================================================
# CORRELATION CHECK (SAMPLE)
# =============================================================================

sample_features = technical_features[:6] + market_features[:4]
sample_features = [c for c in sample_features if c in engineered_df.columns]

plt.figure(figsize=(10, 8))
sns.heatmap(
    engineered_df[sample_features].corr(),
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Feature Correlation (Sample)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_feature_correlation.png", dpi=120)
plt.show()

print("✓ Saved correlation plot")

# =============================================================================
# SAVE ENGINEERED DATA
# =============================================================================

output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)

outfile = output_path / "engineered_features_all_stocks.csv"
engineered_df.to_csv(outfile, index=False)

print("\n" + "=" * 70)
print("ENGINEERED DATA SAVED")
print("=" * 70)
print(f"File: {outfile}")

print("\n➡ Next step: notebook_04_training.py")
print("✅ Notebook 03 finished successfully")   
