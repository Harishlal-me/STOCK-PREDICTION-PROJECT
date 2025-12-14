# config.py
# Improved configuration for 65-70% accuracy target

import os

# ==================================================
# Data Collection - MORE DATA = BETTER ACCURACY
# ==================================================
START_DATE = "2000-01-01"  # 🔥 Extended to 25 years (was 2021)
END_DATE = "2025-12-13"

# Core tech stocks with good liquidity
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AMD", "NFLX", "INTC",
    "CSCO", "ORCL", "CRM", "ADBE", "QCOM"
]

# Market indicators
MARKET_INDICES = ["^GSPC", "^IXIC", "^DJI", "^VIX"]

# Sector ETFs for correlation
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI"]

# ==================================================
# Paths
# ==================================================
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"
CACHE_DIR = "cache"

# Create directories
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================================================
# Randomness
# ==================================================
RANDOM_SEED = 42

# ==================================================
# Dataset Parameters
# ==================================================
SEQUENCE_LENGTH = 60  # 🔥 Increased from 30 (more context)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Prediction horizons
HORIZON_1D = 1   # Next day
HORIZON_5D = 5   # Next week
HORIZON_20D = 20 # Next month

# ==================================================
# Feature Engineering - ADVANCED
# ==================================================
USE_TECHNICAL_INDICATORS = True
USE_SENTIMENT_ANALYSIS = False  # Set to True if you have API keys
USE_VOLUME_PROFILE = True
USE_MARKET_REGIME = True
USE_SECTOR_CORRELATION = True

# Technical Indicators
RSI_PERIOD = 14
STOCH_PERIOD = 14  # Stochastic Oscillator
CCI_PERIOD = 20    # Commodity Channel Index

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

SMA_PERIODS = [5, 10, 20, 50, 100, 200]
EMA_PERIODS = [9, 12, 21, 26, 50]

BB_PERIOD = 20
BB_STD_DEV = 2

ATR_PERIOD = 14
ADX_PERIOD = 14

VOLATILITY_WINDOW = 20
VOLUME_SMA_PERIOD = 20

# Lag features for temporal patterns
LAG_PERIODS = [1, 2, 3, 5, 10, 20]

# Rolling windows for momentum
MOMENTUM_WINDOWS = [5, 10, 20, 60]

# ==================================================
# Model Architecture - ENSEMBLE
# ==================================================
USE_ENSEMBLE = True  # 🔥 Combine multiple models

# LSTM Configuration
LSTM_UNITS_1 = 128  # 🔥 Increased from 64
LSTM_UNITS_2 = 64   # 🔥 Increased from 32
LSTM_UNITS_3 = 32   # 🔥 Added third layer

LSTM_ACTIVATION = "tanh"
LSTM_DROPOUT = 0.3  # 🔥 Increased from 0.2
LSTM_RECURRENT_DROPOUT = 0.2  # 🔥 Added

DENSE_UNITS = 64  # 🔥 Increased from 32
DENSE_ACTIVATION = "relu"
DENSE_DROPOUT = 0.3

# XGBoost Configuration
XGBOOST_PARAMS = {
    "n_estimators": 500,  # 🔥 Increased from 300
    "max_depth": 7,       # 🔥 Increased from 5
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# LightGBM Configuration
LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1
}

# Ensemble weights (will be optimized during training)
ENSEMBLE_WEIGHTS = {
    "lstm": 0.4,
    "xgboost": 0.3,
    "lightgbm": 0.3
}

# ==================================================
# Training Parameters
# ==================================================
LEARNING_RATE = 0.001
BATCH_SIZE = 64  # 🔥 Increased from 32
EPOCHS = 100     # 🔥 Increased from 50

# Callbacks
EARLY_STOPPING_PATIENCE = 15  # 🔥 Increased from 10
REDUCE_LR_PATIENCE = 7        # 🔥 Increased from 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ==================================================
# Validation Strategy - WALK FORWARD
# ==================================================
USE_WALK_FORWARD = True
WALK_FORWARD_SPLITS = 5  # Number of time-based CV splits

# ==================================================
# Feature Selection
# ==================================================
USE_FEATURE_SELECTION = True
FEATURE_IMPORTANCE_THRESHOLD = 0.01  # Drop features below this importance

# ==================================================
# Normalization
# ==================================================
NORMALIZE_METHOD = "standard"  # 🔥 Changed from minmax (better for returns)

# ==================================================
# Trading Strategy Parameters
# ==================================================
CONFIDENCE_THRESHOLD = 0.60  # 🔥 Lowered from 0.65 (more trades)
STOP_LOSS = 0.03             # -3% stop loss
TAKE_PROFIT = 0.05           # +5% take profit
MAX_POSITION_SIZE = 0.2      # 20% of portfolio per trade

# ==================================================
# Backtesting
# ==================================================
BACKTEST_INITIAL_CAPITAL = 100000  # $100K
BACKTEST_COMMISSION = 0.001        # 0.1% per trade
BACKTEST_SLIPPAGE = 0.0005         # 0.05% slippage

# ==================================================
# Performance Targets
# ==================================================
TARGET_ACCURACY = 0.65  # 65% direction accuracy
TARGET_SHARPE_RATIO = 1.5
TARGET_MAX_DRAWDOWN = 0.20  # 20%

# ==================================================
# Logging
# ==================================================
VERBOSE = 1
LOG_FILE = "training.log"
SAVE_PLOTS = True
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)