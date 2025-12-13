# src/sentiment_analyzer.py - Module for sentiment analysis

import pandas as pd
import numpy as np
import os
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from config import SENTIMENT_DATA_DIR, VERBOSE, SENTIMENT_MODEL

class TextBlobSentimentAnalyzer:
    """Simple sentiment analysis using TextBlob"""
    
    def __init__(self, verbose=VERBOSE):
        self.verbose = verbose
    
    def analyze_text(self, text):
        """Analyze sentiment of text using TextBlob"""
        if not text or not isinstance(text, str):
            return 0.0
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return float(polarity)
        except:
            return 0.0
    
    def analyze_documents(self, texts):
        """Analyze sentiment for multiple texts"""
        sentiments = [self.analyze_text(text) for text in texts]
        return sentiments

class FinBERTSentimentAnalyzer:
    """Financial sentiment analysis using FinBERT"""
    
    def __init__(self, verbose=VERBOSE):
        self.verbose = verbose
        self.model_name = "yiyanghkust/finbert-tone"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            if self.verbose:
                print(f"✓ Loaded FinBERT model on {self.device}")
        
        except Exception as e:
            print(f"✗ Error loading FinBERT: {str(e)}")
            self.model = None
    
    def analyze_text(self, text, max_length=512):
        """Analyze sentiment using FinBERT"""
        if not text or not isinstance(text, str) or self.model is None:
            return 0.0
        
        try:
            text = text[:max_length]
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            
            sentiment_score = probs[2] - probs[0]
            
            return float(sentiment_score)
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error analyzing text: {str(e)}")
            return 0.0
    
    def analyze_documents(self, texts, batch_size=8):
        """Analyze sentiment for multiple texts in batches"""
        sentiments = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_sentiments = [self.analyze_text(text) for text in batch]
            sentiments.extend(batch_sentiments)
        
        return sentiments

class SentimentProcessor:
    """Process raw sentiment data and create features"""
    
    def __init__(self, method=SENTIMENT_MODEL, verbose=VERBOSE):
        self.verbose = verbose
        
        if method.lower() == 'finbert':
            self.analyzer = FinBERTSentimentAnalyzer(verbose=verbose)
        else:
            self.analyzer = TextBlobSentimentAnalyzer(verbose=verbose)
        
        self.method = method
    
    def process_news_articles(self, articles_df, ticker):
        """Process news articles and compute sentiment"""
        if articles_df.empty:
            if self.verbose:
                print(f"⚠️  No articles for {ticker}")
            return pd.DataFrame()
        
        # Analyze sentiment
        articles_df['title_sentiment'] = self.analyzer.analyze_documents(
            articles_df['title'].fillna('').tolist()
        )
        
        articles_df['description_sentiment'] = self.analyzer.analyze_documents(
            articles_df['description'].fillna('').tolist()
        )
        
        # Average sentiment per article
        articles_df['sentiment'] = (articles_df['title_sentiment'] + articles_df['description_sentiment']) / 2
        
        # Aggregate by date
        daily_sentiment = articles_df.groupby('date').agg({
            'sentiment': ['mean', 'std', 'count'],
            'title_sentiment': 'mean',
            'description_sentiment': 'mean'
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'article_count',
                                    'title_sentiment', 'description_sentiment']
        
        daily_sentiment['ticker'] = ticker
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
        
        if self.verbose:
            print(f"✓ Processed {len(articles_df)} articles for {ticker}")
            print(f"  → {len(daily_sentiment)} unique dates")
            print(f"  → Avg sentiment: {daily_sentiment['sentiment_mean'].mean():.3f}")
        
        return daily_sentiment
    
    def compute_rolling_sentiment(self, daily_sentiment, window=7):
        """Compute rolling average sentiment"""
        daily_sentiment['sentiment_rolling'] = daily_sentiment['sentiment_mean'].rolling(
            window=window, min_periods=1
        ).mean()
        
        return daily_sentiment

class SentimentOrchestrator:
    """Orchestrate sentiment analysis for multiple stocks"""
    
    def __init__(self, method=SENTIMENT_MODEL, verbose=VERBOSE):
        self.verbose = verbose
        self.processor = SentimentProcessor(method=method, verbose=verbose)
    
    def process_all_sentiments(self, sentiment_data_dict):
        """Process sentiment for all stocks"""
        all_daily_sentiments = {}
        
        for ticker, articles_df in sentiment_data_dict.items():
            if not articles_df.empty:
                daily_sentiment = self.processor.process_news_articles(articles_df, ticker)
                
                if not daily_sentiment.empty:
                    daily_sentiment = self.processor.compute_rolling_sentiment(daily_sentiment, window=7)
                    all_daily_sentiments[ticker] = daily_sentiment
                    
                    # Save processed sentiment
                    filepath = os.path.join(SENTIMENT_DATA_DIR, f'{ticker}_sentiment_processed.csv')
                    daily_sentiment.to_csv(filepath, index=False)
                    
                    if self.verbose:
                        print(f"  → Saved to {filepath}\n")
        
        return all_daily_sentiments
    
    def get_latest_sentiment(self, daily_sentiment):
        """Get most recent sentiment for a stock"""
        if daily_sentiment.empty:
            return {
                'sentiment': 0.0,
                'sentiment_rolling': 0.0,
                'article_count': 0,
                'confidence': 0.0
            }
        
        latest = daily_sentiment.iloc[-1]
        
        return {
            'sentiment': float(latest['sentiment_mean']),
            'sentiment_rolling': float(latest['sentiment_rolling']),
            'article_count': int(latest['article_count']),
            'confidence': float(latest['article_count'] / max(1, daily_sentiment['article_count'].max()))
        }
    
    def load_processed_sentiment(self, ticker):
        """Load previously processed sentiment from CSV"""
        filepath = os.path.join(SENTIMENT_DATA_DIR, f'{ticker}_sentiment_processed.csv')
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df
        else:
            if self.verbose:
                print(f"⚠️  Sentiment file not found: {filepath}")
            return pd.DataFrame()

def combine_sentiment_with_price_data(price_df, sentiment_df):
    """Combine price data with sentiment data"""
    price_df['date'] = pd.to_datetime(price_df['Date']).dt.date
    
    combined = price_df.merge(
        sentiment_df,
        on='date',
        how='left'
    )
    
    # Forward fill missing sentiment values
    combined['sentiment_mean'] = combined['sentiment_mean'].fillna(method='ffill')
    combined['sentiment_rolling'] = combined['sentiment_rolling'].fillna(method='ffill')
    combined['article_count'] = combined['article_count'].fillna(0)
    
    # Fill remaining NaN with 0
    combined.fillna(0, inplace=True)
    
    return combined

if __name__ == "__main__":
    print("Sentiment Analysis Module")
    print("=" * 70)
