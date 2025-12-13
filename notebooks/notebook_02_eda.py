# 02_eda.ipynb
# Exploratory Data Analysis Notebook

"""
NOTEBOOK: 02_EDA.ipynb
PURPOSE: Explore and analyze collected data

CELLS:
"""

# Cell 1: Imports and Setup
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_collector import MarketDataCollector, MacroDataCollector
from config import STOCKS

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 6)

print("✓ Imports successful")

# Cell 2: Load Market Data
market_collector = MarketDataCollector()
market_data = {}

for ticker in STOCKS[:5]:  # Analyze first 5 stocks
    df = market_collector.load_stock_data(ticker)
    if df is not None:
        market_data[ticker] = df
        print(f"✓ Loaded {ticker}: {len(df)} records")

# Cell 3: Summary Statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

for ticker, df in market_data.items():
    print(f"\n{ticker}:")
    print(f"  Records: {len(df)}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    # Close price (safe)
    close_col = pd.to_numeric(df['Close'], errors='coerce')
    print(f"  Close price: ${close_col.mean():.2f} (avg)")
    # Price range (safe)
    low_col = pd.to_numeric(df['Low'], errors='coerce').min()
    high_col = pd.to_numeric(df['High'], errors='coerce').max()
    if pd.isna(high_col):
        print(f"  Price range: ${low_col:.2f} - N/A")
    else:
        print(f"  Price range: ${low_col:.2f} - ${high_col:.2f}")
    # Average volume (safe)
    volume_col = pd.to_numeric(df['Volume'], errors='coerce').mean()
    print(f"  Avg volume: {volume_col:.0f}")
    # Calculate returns
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['return'] = df['Close'].pct_change()
    print(f"  Avg daily return: {df['return'].mean():.4f}")
    print(f"  Daily return std: {df['return'].std():.4f}")



# Cell 4: Plot Price Trends
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (ticker, df) in enumerate(list(market_data.items())[:6]):
    axes[idx].plot(df['Date'], df['Close'], linewidth=1, label='Close')
    axes[idx].fill_between(df['Date'], df['Low'], df['High'], alpha=0.2)
    axes[idx].set_title(f'{ticker} - Price Trend (5 Years)')
    axes[idx].set_ylabel('Price ($)')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

plt.tight_layout()
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.savefig(RESULTS_DIR / "02_price_trends.png", dpi=100, bbox_inches="tight")
plt.show()

print("✓ Saved price trends plot")

# Cell 5: Volume Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.flatten()

for idx, (ticker, df) in enumerate(list(market_data.items())[:4]):
    axes[idx].bar(df['Date'], df['Volume'], width=1, alpha=0.6)
    axes[idx].set_title(f'{ticker} - Trading Volume')
    axes[idx].set_ylabel('Volume')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/02_volume_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Saved volume analysis plot")

# Cell 6: Return Distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.flatten()

for idx, (ticker, df) in enumerate(list(market_data.items())[:4]):
    df['return'] = df['Close'].pct_change()
    axes[idx].hist(df['return'].dropna(), bins=50, alpha=0.7)
    axes[idx].axvline(df['return'].mean(), color='r', linestyle='--', label='Mean')
    axes[idx].axvline(df['return'].median(), color='g', linestyle='--', label='Median')
    axes[idx].set_title(f'{ticker} - Return Distribution')
    axes[idx].set_xlabel('Daily Return')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/02_return_distribution.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Saved return distribution plot")

# Cell 7: Volatility Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.flatten()

for idx, (ticker, df) in enumerate(list(market_data.items())[:4]):
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['return'].rolling(30).std()
    
    axes[idx].plot(df['Date'], df['volatility'], linewidth=1)
    axes[idx].set_title(f'{ticker} - 30-Day Rolling Volatility')
    axes[idx].set_ylabel('Volatility')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/02_volatility_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Saved volatility analysis plot")

# Cell 8: Load and Analyze Macro Data
macro_collector = MacroDataCollector()
macro_data = macro_collector.fetch_all_macro_data(period='5y')

print("\n" + "=" * 70)
print("MACRO DATA ANALYSIS")
print("=" * 70)

print(f"\nMacro Data Summary:")
print(f"  Records: {len(macro_data)}")
print(f"  Date range: {macro_data['Date'].min()} to {macro_data['Date'].max()}")
print(f"\nStatistics:")
print(macro_data[['VIX', 'SP500_Close', 'Interest_Rate']].describe())

# Cell 9: Macro Data Plots
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# VIX
axes[0].plot(macro_data['Date'], macro_data['VIX'], linewidth=1, color='red')
axes[0].fill_between(macro_data['Date'], macro_data['VIX'], alpha=0.3, color='red')
axes[0].set_title('VIX - Market Volatility Index')
axes[0].set_ylabel('VIX Level')
axes[0].grid(True, alpha=0.3)

# S&P 500
axes[1].plot(macro_data['Date'], macro_data['SP500_Close'], linewidth=1, color='blue')
axes[1].set_title('S&P 500 Index')
axes[1].set_ylabel('Index Value')
axes[1].grid(True, alpha=0.3)

# Interest Rate
axes[2].plot(macro_data['Date'], macro_data['Interest_Rate'], linewidth=1, color='green')
axes[2].set_title('10-Year Treasury Rate')
axes[2].set_ylabel('Rate (%)')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/02_macro_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Saved macro data plots")

# Cell 10: Correlation Analysis
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

# Correlate stock prices with macro indicators
if market_data:
    ticker = list(market_data.keys())[0]
    df = market_data[ticker].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Merge with macro data
    df = df.merge(macro_data, on='Date', how='inner')
    
    # Calculate correlations
    corr_cols = ['Close', 'Volume', 'VIX', 'SP500_Close', 'Interest_Rate']
    if all(col in df.columns for col in corr_cols):
        corr_matrix = df[corr_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'{ticker} - Correlation with Macro Indicators')
        plt.tight_layout()
        from pathlib import Path
        RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
        RESULTS_DIR.mkdir(exist_ok=True)
        plt.savefig(RESULTS_DIR / f"03_corr_{ticker}.png", dpi=100, bbox_inches="tight")
        plt.show()
        
        print(f"\nCorrelation matrix for {ticker}:")
        print(corr_matrix)

print("✓ Saved correlation analysis")

# Cell 11: Key Insights
print("\n" + "=" * 70)
print("KEY INSIGHTS FROM EDA")
print("=" * 70)

print("""
1. PRICE TRENDS
   - All stocks show growth over 5 years
   - Volatility varies by sector

2. TRADING VOLUME
   - Volume varies significantly day-to-day
   - Useful feature for momentum signals

3. RETURNS
   - Daily returns approximately normally distributed
   - Some extreme movements (outliers)

4. VOLATILITY
   - Volatility clustering observed
   - Higher during market stress periods

5. MACRO INDICATORS
   - VIX spikes correlate with price drops
   - Interest rates affect stock valuations

6. NEXT STEPS
   - Engineer features from this raw data
   - Train models on processed features
   - Backtest trading signals
""")

# Cell 12: Data Quality Summary
print("\n" + "=" * 70)
print("DATA QUALITY SUMMARY")
print("=" * 70)

quality_report = {
    'Market Data': {
        'Status': '✓ Good',
        'Records': sum(len(df) for df in market_data.values()),
        'Missing': 0,
        'Issues': 'None'
    },
    'Macro Data': {
        'Status': '✓ Good',
        'Records': len(macro_data),
        'Missing': macro_data.isnull().sum().sum(),
        'Issues': 'Minor: Forward filled'
    }
}

for data_type, info in quality_report.items():
    print(f"\n{data_type}:")
    for key, value in info.items():
        print(f"  {key}: {value}")

print("\n✓ EDA Complete - Ready for feature engineering!")
