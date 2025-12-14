# src/data_collector.py
# Improved version with better data cleaning and validation

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import START_DATE, END_DATE, RAW_DATA_DIR, TICKERS, MARKET_INDICES, SECTOR_ETFS


class ImprovedMarketDataCollector:
    def __init__(self):
        self.start_date = START_DATE
        self.end_date = END_DATE
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

    def fetch_stock_data(self, ticker, save=True):
        """
        Fetch stock data with improved cleaning and validation
        """
        print(f"\n{'='*70}")
        print(f"FETCHING: {ticker}")
        print(f"{'='*70}")
        print(f"Date range: {self.start_date} to {self.end_date}")

        try:
            # Download data
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                print(f"❌ No data returned for {ticker}")
                return None

            # Reset index to get Date as column
            df.reset_index(inplace=True)
            
            # 🔥 CRITICAL: Clean column names (handle multi-level columns)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure we have the required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return None

            # 🔥 DATA CLEANING
            # 1. Remove any rows where Close is NaN or 0
            df = df[df['Close'].notna()]
            df = df[df['Close'] > 0]
            
            # 2. Remove any rows with all zeros
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[~(df[numeric_cols] == 0).all(axis=1)]
            
            # 3. Forward fill any remaining NaN values
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
            
            # 4. Remove duplicates by date
            df = df.drop_duplicates(subset=['Date'], keep='last')
            
            # 5. Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            # 6. Validate OHLC relationships
            # High should be >= Low, Close, Open
            # Low should be <= High, Close, Open
            df = df[
                (df['High'] >= df['Low']) &
                (df['High'] >= df['Close']) &
                (df['High'] >= df['Open']) &
                (df['Low'] <= df['Close']) &
                (df['Low'] <= df['Open'])
            ]
            
            # 7. Remove extreme outliers (price changes > 50% in a day are suspicious)
            df['pct_change'] = df['Close'].pct_change()
            df = df[df['pct_change'].abs() < 0.5]
            df = df.drop(columns=['pct_change'])
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Reorder columns
            df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            print(f"✅ Rows downloaded: {len(df)}")
            print(f"   Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
            print(f"   Latest Close: ${df['Close'].iloc[-1]:.2f}")
            print(f"   Avg Volume: {df['Volume'].mean():,.0f}")

            if save:
                path = os.path.join(RAW_DATA_DIR, f"{ticker}_ohlcv.csv")
                df.to_csv(path, index=False)
                print(f"💾 Saved to {path}")

            return df

        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return None

    def fetch_multiple_stocks(self, tickers):
        """
        Fetch data for multiple stocks
        """
        print(f"\n{'='*70}")
        print(f"FETCHING DATA FOR {len(tickers)} STOCKS")
        print(f"{'='*70}")
        print("Tickers:", ", ".join(tickers))

        all_data = {}
        successful = 0
        failed = []

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            df = self.fetch_stock_data(ticker)

            if df is not None and len(df) > 0:
                all_data[ticker] = df
                successful += 1
            else:
                failed.append(ticker)
                print(f"⚠️  Skipped {ticker}")

        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Successful: {successful}/{len(tickers)}")
        if failed:
            print(f"❌ Failed: {', '.join(failed)}")
        print(f"{'='*70}")

        return all_data

    def fetch_market_data(self):
        """
        Fetch market indices and sector ETFs
        """
        print(f"\n{'='*70}")
        print("FETCHING MARKET INDICATORS")
        print(f"{'='*70}")
        
        market_data = {}
        
        # Fetch market indices
        print("\n📊 Market Indices:")
        for index in MARKET_INDICES:
            print(f"  Fetching {index}...")
            df = self.fetch_stock_data(index, save=True)
            if df is not None:
                market_data[index] = df
                print(f"  ✅ {index}: {len(df)} rows")
        
        # Fetch sector ETFs
        print("\n📈 Sector ETFs:")
        for etf in SECTOR_ETFS:
            print(f"  Fetching {etf}...")
            df = self.fetch_stock_data(etf, save=True)
            if df is not None:
                market_data[etf] = df
                print(f"  ✅ {etf}: {len(df)} rows")
        
        print(f"\n✅ Market data collection complete!")
        return market_data

    def update_existing_data(self, ticker):
        """
        Update existing data with new records only
        """
        path = os.path.join(RAW_DATA_DIR, f"{ticker}_ohlcv.csv")
        
        if not os.path.exists(path):
            print(f"No existing data for {ticker}, fetching all...")
            return self.fetch_stock_data(ticker)
        
        # Load existing data
        existing_df = pd.read_csv(path, parse_dates=['Date'])
        last_date = existing_df['Date'].max()
        
        print(f"Updating {ticker} from {last_date.date()}...")
        
        # Fetch new data
        new_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            new_df = yf.download(
                ticker,
                start=new_start,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            
            if new_df.empty:
                print(f"✅ {ticker} is up to date")
                return existing_df
            
            new_df.reset_index(inplace=True)
            if isinstance(new_df.columns, pd.MultiIndex):
                new_df.columns = new_df.columns.get_level_values(0)
            
            new_df['Ticker'] = ticker
            new_df = new_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Combine and deduplicate
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            # Save updated data
            combined_df.to_csv(path, index=False)
            
            print(f"✅ Added {len(new_df)} new rows to {ticker}")
            return combined_df
            
        except Exception as e:
            print(f"❌ Error updating {ticker}: {e}")
            return existing_df


# Convenience function
def fetch_all_data():
    """
    Fetch all stocks, indices, and sector ETFs
    """
    collector = ImprovedMarketDataCollector()
    
    # Fetch stocks
    stock_data = collector.fetch_multiple_stocks(TICKERS)
    
    # Fetch market data
    market_data = collector.fetch_market_data()
    
    return {**stock_data, **market_data}


if __name__ == "__main__":
    print("Starting data collection...")
    data = fetch_all_data()
    print(f"\n✅ Total datasets collected: {len(data)}")