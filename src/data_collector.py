# src/data_collector.py - Module to collect market, sentiment, and macro data

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
from config import (
    STOCKS, DATA_PERIOD, RAW_DATA_DIR, NEWS_API_KEY,
    SENTIMENT_DATA_DIR, SENTIMENT_LOOKBACK_DAYS, VERBOSE
)

class MarketDataCollector:
    """Collect OHLCV data from Yahoo Finance"""
    
    def __init__(self, output_dir=RAW_DATA_DIR, verbose=VERBOSE):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
    
    def fetch_single_stock(self, ticker, period=DATA_PERIOD):
        """Fetch OHLCV data for a single stock"""
        try:
            df = yf.download(ticker, period=period, progress=False)
            df = df.reset_index()
            df['Ticker'] = ticker
            
            if self.verbose:
                print(f"✓ Fetched {ticker}: {len(df)} records")
            
            return df
        
        except Exception as e:
            print(f"✗ Error fetching {ticker}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, stocks=STOCKS):
        """Fetch OHLCV data for multiple stocks"""
        all_data = {}
        
        for ticker in stocks:
            df = self.fetch_single_stock(ticker)
            if df is not None:
                all_data[ticker] = df
                
                # Save individual file
                filepath = os.path.join(self.output_dir, f'{ticker}_ohlcv.csv')
                df.to_csv(filepath, index=False)
                
                if self.verbose:
                    print(f"  → Saved to {filepath}")
        
        if self.verbose:
            print(f"\n✓ Collected {len(all_data)}/{len(stocks)} stocks\n")
        
        return all_data
    
    def load_stock_data(self, ticker):
        """Load previously downloaded stock data from CSV"""
        filepath = os.path.join(self.output_dir, f'{ticker}_ohlcv.csv')
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            print(f"✗ File not found: {filepath}")
            return None

class SentimentDataCollector:
    """Collect news sentiment data from NewsAPI"""
    
    def __init__(self, api_key=NEWS_API_KEY, output_dir=SENTIMENT_DATA_DIR, verbose=VERBOSE):
        self.api_key = api_key
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
        
        if api_key == "your_news_api_key_here":
            print("⚠️  WARNING: Please set your NewsAPI key in config.py")
    
    def fetch_news_articles(self, ticker, days=SENTIMENT_LOOKBACK_DAYS):
        """Fetch news articles for a ticker from NewsAPI"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.api_key,
            'pageSize': 100
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'articles' in data:
                articles = data['articles']
                if self.verbose:
                    print(f"✓ Fetched {len(articles)} articles for {ticker}")
                return articles
            else:
                print(f"✗ Error fetching news for {ticker}: {data.get('message', 'Unknown error')}")
                return []
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return []
    
    def articles_to_dataframe(self, articles):
        """Convert articles list to DataFrame"""
        if not articles:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'publishedAt': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', ''),
                'url': article.get('url', '')
            }
            for article in articles
        ])
        
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df['date'] = df['publishedAt'].dt.date
        
        return df

class MacroDataCollector:
    """Collect macroeconomic indicators"""
    
    def __init__(self, verbose=VERBOSE):
        self.verbose = verbose
    
    def fetch_vix(self, period='5y'):
        """Fetch VIX (Volatility Index)"""
        try:
            vix = yf.download('^VIX', period=period, progress=False)
            vix = vix.reset_index()
            vix.rename(columns={'Close': 'VIX'}, inplace=True)
            
            if self.verbose:
                print(f"✓ Fetched VIX: {len(vix)} records")
            
            return vix[['Date', 'VIX']]
        
        except Exception as e:
            print(f"✗ Error fetching VIX: {str(e)}")
            return None
    
    def fetch_sp500(self, period='5y'):
        """Fetch S&P 500 data for market correlation"""
        try:
            sp500 = yf.download('^GSPC', period=period, progress=False)
            sp500 = sp500.reset_index()
            sp500.rename(columns={'Close': 'SP500_Close'}, inplace=True)
            sp500['SP500_Return'] = sp500['SP500_Close'].pct_change()
            
            if self.verbose:
                print(f"✓ Fetched S&P 500: {len(sp500)} records")
            
            return sp500[['Date', 'SP500_Close', 'SP500_Return']]
        
        except Exception as e:
            print(f"✗ Error fetching S&P 500: {str(e)}")
            return None
    
    def fetch_interest_rate(self, period='5y'):
        """Fetch 10-Year Treasury Rate"""
        try:
            tnx = yf.download('^TNX', period=period, progress=False)
            tnx = tnx.reset_index()
            tnx.rename(columns={'Close': 'Interest_Rate'}, inplace=True)
            
            if self.verbose:
                print(f"✓ Fetched 10Y Treasury: {len(tnx)} records")
            
            return tnx[['Date', 'Interest_Rate']]
        
        except Exception as e:
            print(f"✗ Error fetching interest rate: {str(e)}")
            return None
    
    def fetch_all_macro_data(self, period='5y'):
        """Fetch all macroeconomic indicators"""
        vix = self.fetch_vix(period)
        sp500 = self.fetch_sp500(period)
        interest_rate = self.fetch_interest_rate(period)
        
        # Merge all data
        macro_data = vix.copy()
        
        if sp500 is not None:
            macro_data = macro_data.merge(sp500, on='Date', how='left')
        
        if interest_rate is not None:
            macro_data = macro_data.merge(interest_rate, on='Date', how='left')
        
        # Forward fill missing values
        macro_data = macro_data.fillna(method='ffill')
        
        if self.verbose:
            print(f"\n✓ Combined macro data: {len(macro_data)} records")
        
        return macro_data

class DataOrchestrator:
    """Orchestrate data collection from all sources"""
    
    def __init__(self, verbose=VERBOSE):
        self.verbose = verbose
        self.market_collector = MarketDataCollector(verbose=verbose)
        self.sentiment_collector = SentimentDataCollector(verbose=verbose)
        self.macro_collector = MacroDataCollector(verbose=verbose)
    
    def collect_all_data(self, stocks=STOCKS, period=DATA_PERIOD):
        """Collect market, sentiment, and macro data"""
        if self.verbose:
            print("=" * 70)
            print("STARTING DATA COLLECTION")
            print("=" * 70 + "\n")
        
        # Collect market data
        print("📊 Collecting Market Data (OHLCV)...")
        market_data = self.market_collector.fetch_multiple_stocks(stocks)
        
        # Collect macro data
        print("\n🌍 Collecting Macro Data...")
        macro_data = self.macro_collector.fetch_all_macro_data(period)
        
        # Collect sentiment data
        print("\n📰 Collecting Sentiment Data...")
        sentiment_data = {}
        for ticker in stocks:
            articles = self.sentiment_collector.fetch_news_articles(ticker)
            if articles:
                df = self.sentiment_collector.articles_to_dataframe(articles)
                sentiment_data[ticker] = df
                
                # Save to CSV
                filepath = os.path.join(SENTIMENT_DATA_DIR, f'{ticker}_sentiment_raw.csv')
                df.to_csv(filepath, index=False)
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("DATA COLLECTION COMPLETE")
            print("=" * 70 + "\n")
        
        return {
            'market': market_data,
            'macro': macro_data,
            'sentiment': sentiment_data
        }

if __name__ == "__main__":
    orchestrator = DataOrchestrator(verbose=True)
    data = orchestrator.collect_all_data(stocks=['AAPL', 'MSFT', 'GOOGL'], period='5y')
    print("\n✓ Data collection complete!")
