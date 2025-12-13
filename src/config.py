# src/config.py - Configuration file for Stock Price Prediction Project

import os
from datetime import datetime, timedelta

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SENTIMENT_DATA_DIR = os.path.join(DATA_DIR, 'sentiment')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SENTIMENT_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# STOCK SELECTION
# ============================================================================
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'META', 'AMZN', 'INFY', 'TCS', 'RELIANCE', 'WIPRO']
NUM_STOCKS = len(STOCKS)
DATA_PERIOD = '5y'
LOOKBACK_PERIOD = 30

# ============================================================================
# API KEYS
# ============================================================================
NEWS_API_KEY = "your_news_api_key_here"
FRED_API_KEY = "your_fred_api_key_here"

# ============================================================================
# DATA PROCESSING
# ============================================================================
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
NORMALIZE_METHOD = 'minmax'
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 0.3

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD_DEV = 2
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]
VOLATILITY_PERIOD = 14
LAG_DAYS = [1, 2, 3, 5, 7]
CORRELATION_WINDOW = 30
CORRELATION_MARKET_INDEX = '^GSPC'
SENTIMENT_LOOKBACK_DAYS = 365

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
LSTM_DROPOUT = 0.2
LSTM_ACTIVATION = 'relu'
DENSE_UNITS = 16
DENSE_ACTIVATION = 'relu'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.2
SEQUENCE_LENGTH = 30

# ============================================================================
# XGBOOST PARAMETERS
# ============================================================================
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
}

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
CONFIDENCE_THRESHOLD = 0.65
RISK_REWARD_RATIO = 1.5
VOLATILITY_MULTIPLIER = 2.0
INITIAL_CAPITAL = 10000
POSITION_SIZE_PERCENT = 0.1
MAX_POSITION_SIZE = 0.3

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================
SENTIMENT_MODEL = 'finbert'
SENTIMENT_THRESHOLD_POSITIVE = 0.5
SENTIMENT_THRESHOLD_NEGATIVE = -0.5

# ============================================================================
# RANDOM STATE & LOGGING
# ============================================================================
RANDOM_SEED = 42
VERBOSE = True
LOG_LEVEL = 'INFO'

# ============================================================================
# PLOTTING
# ============================================================================
PLOT_DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_FIGURE_SIZE = (15, 6)

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("PROJECT CONFIGURATION")
    print("=" * 70)
    print(f"\n📊 STOCKS: {', '.join(STOCKS)}")
    print(f"📅 DATA PERIOD: {DATA_PERIOD}")
    print(f"📁 DATA DIR: {DATA_DIR}")
    print(f"\n🔢 TRAIN/VAL/TEST SPLIT: {TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE}")
    print(f"🧠 LSTM: {LSTM_UNITS_1} → {LSTM_UNITS_2} units")
    print(f"⏳ SEQUENCE LENGTH: {SEQUENCE_LENGTH} days")
    print(f"💰 INITIAL CAPITAL: ${INITIAL_CAPITAL}")
    print(f"✅ CONFIDENCE THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    print_config()
