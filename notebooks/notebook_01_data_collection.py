# 01_data_collection.ipynb
# Jupyter notebook for data collection
# This is a text representation - use as template for actual .ipynb

"""
NOTEBOOK: 01_Data_Collection.ipynb
PURPOSE: Collect market, sentiment, and macro data

CELLS:
"""

# Cell 1: Setup and Imports
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from src.data_collector import DataOrchestrator, MarketDataCollector, MacroDataCollector
from src.sentiment_analyzer import SentimentOrchestrator
from config import STOCKS, DATA_PERIOD, VERBOSE
import warnings
warnings.filterwarnings('ignore')

print("✓ Imports successful")

# Cell 2: Configure Settings
config_info = {
    'Stocks': STOCKS,
    'Period': DATA_PERIOD,
    'Verbose': VERBOSE
}

print("Configuration:")
for key, value in config_info.items():
    print(f"  {key}: {value}")

# Cell 3: Initialize Data Collectors
market_collector = MarketDataCollector(verbose=True)
macro_collector = MacroDataCollector(verbose=True)
sentiment_orchestrator = SentimentOrchestrator(verbose=True)

print("✓ Collectors initialized")

# Cell 4: Collect Market Data (OHLCV)
print("=" * 70)
print("COLLECTING MARKET DATA")
print("=" * 70)

market_data = market_collector.fetch_multiple_stocks(STOCKS)

print(f"\n✓ Collected {len(market_data)} stocks")
for ticker, df in market_data.items():
    print(f"  {ticker}: {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")

# Cell 5: Examine Market Data
print("\nSample Market Data (AAPL):")
print(market_data['AAPL'].head())
print(f"\nShape: {market_data['AAPL'].shape}")
print(f"\nColumns: {market_data['AAPL'].columns.tolist()}")
print(f"\nData types:\n{market_data['AAPL'].dtypes}")

# Cell 6: Collect Macro Data
print("\n" + "=" * 70)
print("COLLECTING MACRO DATA")
print("=" * 70)

macro_data = macro_collector.fetch_all_macro_data(period=DATA_PERIOD)
print(f"\nMacro data shape: {macro_data.shape}")
print(f"Date range: {macro_data['Date'].min()} to {macro_data['Date'].max()}")
print(f"\nColumns: {macro_data.columns.tolist()}")
print(f"\nSample data:\n{macro_data.head()}")

# Cell 7: Plot VIX Trend
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(macro_data['Date'], macro_data['VIX'], linewidth=1)
axes[0, 0].set_title('VIX - Volatility Index')
axes[0, 0].set_ylabel('VIX')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(macro_data['Date'], macro_data['SP500_Close'], linewidth=1)
axes[0, 1].set_title('S&P 500 Close Price')
axes[0, 1].set_ylabel('Price')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(macro_data['Date'], macro_data['Interest_Rate'], linewidth=1)
axes[1, 0].set_title('10-Year Treasury Rate')
axes[1, 0].set_ylabel('Rate %')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(macro_data['Date'], macro_data['SP500_Return'], linewidth=0.5)
axes[1, 1].set_title('S&P 500 Daily Return')
axes[1, 1].set_ylabel('Return %')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
from pathlib import Path
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)
plt.savefig(RESULTS_DIR / "01_macro_data_overview.png", dpi=100, bbox_inches="tight")

print("✓ Saved macro overview plot")

# Cell 8: Collect Sentiment Data
print("\n" + "=" * 70)
print("COLLECTING SENTIMENT DATA")
print("=" * 70)

# Note: This requires valid NEWS_API_KEY in config.py
# Uncomment below if you have valid API key
"""
from src.data_collector import SentimentDataCollector
sentiment_collector = SentimentDataCollector(verbose=True)

sentiment_raw = {}
for ticker in STOCKS[:3]:  # Start with first 3 stocks
    articles = sentiment_collector.fetch_news_articles(ticker)
    if articles:
        df = sentiment_collector.articles_to_dataframe(articles)
        sentiment_raw[ticker] = df
        print(f"  {ticker}: {len(df)} articles")
"""

print("⚠️  Sentiment collection requires valid NewsAPI key")
print("   Update config.py with your API key from newsapi.org")

# Cell 9: Data Quality Check
print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

for ticker, df in list(market_data.items())[:3]:
    print(f"\n{ticker}:")
    print(f"  Total records: {len(df)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    low_col = df.filter(like='Low').min().min()
    high_col = df.filter(like='High').max().max()
    print(f"  Price range: ${low_col:.2f} - ${high_col:.2f}")
    
    # Check for extreme movements
    df['return'] = df['Close'].pct_change()
    extreme = df[abs(df['return']) > 0.1]
    print(f"  Extreme movements (>10%): {len(extreme)}")

# Cell 10: Summary Statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n✓ Market data collected: {len(market_data)} stocks")
print(f"✓ Macro data collected: {len(macro_data)} days")
print(f"✓ All data saved to data/raw/")
print(f"\nNext steps:")
print(f"  1. Run 02_eda.ipynb for exploratory analysis")
print(f"  2. Process sentiment data (requires valid API key)")
print(f"  3. Proceed to feature engineering")
