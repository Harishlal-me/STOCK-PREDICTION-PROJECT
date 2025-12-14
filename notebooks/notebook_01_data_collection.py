print("🚀 Notebook 01 started")

# src/data_collector.py
"""
Data Collector Module
Downloads stock market data using yfinance
"""

import os
import yfinance as yf
import pandas as pd
from config import START_DATE, END_DATE, RAW_DATA_DIR


class MarketDataCollector:
    def __init__(self):
        self.start_date = START_DATE
        self.end_date = END_DATE
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

    def fetch_stock_data(self, ticker, save=True):
        print("\n" + "=" * 70)
        print(f"FETCHING DATA FOR {ticker}")
        print("=" * 70)
        print(f"Date range: {self.start_date} to {self.end_date}")

        try:
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                print(f"❌ No data for {ticker}")
                return None

            df.reset_index(inplace=True)
            df["Ticker"] = ticker
            df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

            print(f"✓ Rows downloaded: {len(df)}")
            print(f"  Date range: {df['Date'].min()} → {df['Date'].max()}")
            print(f"  Latest Close: ${df['Close'].iloc[-1]:.2f}")  # ✅ FIXED

            if save:
                path = os.path.join(RAW_DATA_DIR, f"{ticker}_raw.csv")
                df.to_csv(path, index=False)
                print(f"✓ Saved to {path}")

            return df

        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return None

    def fetch_multiple_stocks(self, tickers):
        print("\n" + "=" * 70)
        print(f"FETCHING DATA FOR {len(tickers)} STOCKS")
        print("=" * 70)
        print("Tickers:", ", ".join(tickers))

        all_data = {}

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}")
            df = self.fetch_stock_data(ticker)

            if df is not None:
                all_data[ticker] = df
            else:
                print(f"⚠ Skipped {ticker}")

        print("\n" + "=" * 70)
        print(f"SUMMARY: {len(all_data)}/{len(tickers)} stocks downloaded")
        print("=" * 70)

        return all_data
print("✅ Notebook 01 finished successfully")
