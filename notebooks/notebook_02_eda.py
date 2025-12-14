"""
NOTEBOOK 02: Exploratory Data Analysis (EDA)
Purpose: Explore and analyze collected stock data safely
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import RAW_DATA_DIR

sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (14, 6)

print("✓ Imports successful")

# =============================================================================
# STEP 1: LOAD ALL STOCK DATA
# =============================================================================
print("\n" + "=" * 70)
print("LOADING MARKET DATA")
print("=" * 70)

data_path = Path(RAW_DATA_DIR)
csv_files = list(data_path.glob("*_ohlcv.csv")) + list(data_path.glob("*_raw.csv"))

market_data = {}

for file in csv_files:
    ticker = file.stem.split("_")[0]
    df = pd.read_csv(file, parse_dates=["Date"])

    # Force numeric conversion (CRITICAL FIX)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no Close price
    df.dropna(subset=["Close"], inplace=True)

    if len(df) < 50:
        print(f"⚠ Skipping {ticker}: insufficient data ({len(df)} rows)")
        continue

    market_data[ticker] = df
    print(f"✓ Loaded {ticker}: {len(df)} rows")

# =============================================================================
# STEP 2: SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

for ticker, df in market_data.items():
    print(f"\n{ticker}")
    print(f"  Records     : {len(df)}")
    print(f"  Date range  : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Avg price   : ${df['Close'].mean():.2f}")
    print(f"  Price range : ${df['Low'].min():.2f} – ${df['High'].max():.2f}")
    print(f"  Avg volume  : {df['Volume'].mean():.0f}")

    df["Return"] = df["Close"].pct_change()
    print(f"  Avg return  : {df['Return'].mean():.4f}")
    print(f"  Volatility  : {df['Return'].std():.4f}")

# =============================================================================
# STEP 3: PRICE TRENDS (TOP 6 STOCKS)
# =============================================================================
print("\nCreating price trend plots...")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

top_stocks = list(market_data.keys())[:6]

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for ax, ticker in zip(axes, top_stocks):
    df = market_data[ticker]
    ax.plot(df["Date"], df["Close"], label="Close", linewidth=1.5)
    ax.fill_between(df["Date"], df["Low"], df["High"], alpha=0.2)
    ax.set_title(f"{ticker} Price Trend")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_price_trends.png", dpi=120)
plt.show()

print("✓ Saved: results/02_price_trends.png")

# =============================================================================
# STEP 4: VOLUME ANALYSIS
# =============================================================================
print("\nCreating volume analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

for ax, ticker in zip(axes, top_stocks[:4]):
    df = market_data[ticker]
    ax.bar(df["Date"], df["Volume"], alpha=0.6)
    ax.set_title(f"{ticker} Trading Volume")
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_volume_analysis.png", dpi=120)
plt.show()

print("✓ Saved: results/02_volume_analysis.png")

# =============================================================================
# STEP 5: RETURN DISTRIBUTION
# =============================================================================
print("\nCreating return distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

for ax, ticker in zip(axes, top_stocks[:4]):
    df = market_data[ticker]
    returns = df["Close"].pct_change().dropna()

    ax.hist(returns, bins=50, alpha=0.7)
    ax.axvline(returns.mean(), color="red", linestyle="--", label="Mean")
    ax.axvline(returns.median(), color="green", linestyle="--", label="Median")
    ax.set_title(f"{ticker} Return Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_return_distribution.png", dpi=120)
plt.show()

print("✓ Saved: results/02_return_distribution.png")

# =============================================================================
# STEP 6: VOLATILITY ANALYSIS
# =============================================================================
print("\nCreating volatility analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

for ax, ticker in zip(axes, top_stocks[:4]):
    df = market_data[ticker]
    df["Volatility_30"] = df["Close"].pct_change().rolling(30).std()
    ax.plot(df["Date"], df["Volatility_30"], linewidth=1.5)
    ax.set_title(f"{ticker} 30-Day Rolling Volatility")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_volatility_analysis.png", dpi=120)
plt.show()

print("✓ Saved: results/02_volatility_analysis.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"""
✓ Stocks analyzed     : {len(market_data)}
✓ Total data points   : {sum(len(df) for df in market_data.values())}
✓ Output directory    : results/

➡ Next step: notebook_03_feature_engineering.py
""")
