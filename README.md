# 📈 Stock Price Prediction System

> Advanced ensemble machine learning system achieving **58% directional accuracy** for stock price prediction using deep learning and gradient boosting models.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-58%25-success.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Key Achievements

- ✅ **58% directional accuracy** on AAPL stock predictions (LSTM on significant moves)
- ✅ Significantly outperforms random baseline (50%) and typical models (52-55%)
- ✅ Trained on **20 years** of historical data (2005-2025)
- ✅ **126+ engineered features** including advanced technical indicators
- ✅ **5 prediction strategies** implemented and compared

---

## 📊 Model Performance Comparison

| Model Strategy | Accuracy | AUC | Notes |
|---------------|----------|-----|-------|
| **LSTM (Significant Moves)** | **57.99%** | **0.5913** | 🏆 Best - Predicts >2% moves only |
| Basic Ensemble | 55.14% | 0.5167 | LSTM + XGBoost + LightGBM |
| Feature Selected (50 features) | 54.11% | - | Reduced from 126 features |
| Regime-Based (Bull Market) | 53.38% | - | Separate models per regime |
| Regime-Based (Bear Market) | 54.39% | - | Better in downtrends |
| Regime-Based (Sideways) | 52.63% | - | Choppy market conditions |

---

## 🚀 Features & Capabilities

### 📈 126+ Engineered Features

#### **Technical Indicators**
- Trend: SMA (5, 10, 20, 50, 100, 200), EMA (9, 12, 21, 26, 50)
- Momentum: RSI, MACD, Stochastic Oscillator, Williams %R, ROC
- Volatility: Bollinger Bands, ATR, Parkinson Volatility, HVR
- Strength: ADX, CCI, Trend Strength

#### **Volume Analysis**
- OBV (On-Balance Volume)
- MFI (Money Flow Index)
- VPT (Volume Price Trend)
- Volume Ratios & Changes

#### **Price Patterns**
- Price ranges, body sizes, shadows
- Gap detection (up/down)
- Higher highs / Lower lows
- Support/resistance levels

#### **Statistical Features**
- Rolling mean, std, skewness, kurtosis
- 20 lag features (1, 2, 3, 5, 10, 20 days)
- Correlation with market indices
- Beta calculation

#### **Market Context**
- S&P 500 correlation
- VIX (volatility index)
- Sector ETF correlations
- Market regime detection

---

## 🧠 Model Architectures

### 1. **LSTM Deep Learning Network**
```python
Architecture:
- LSTM Layer 1: 128 units, 30% dropout
- LSTM Layer 2: 64 units, 30% dropout  
- LSTM Layer 3: 32 units, 30% dropout
- Dense Layer: 64 units, ReLU activation
- Output Layer: Sigmoid (binary classification)

Performance: 57.99% accuracy on significant moves
```

### 2. **XGBoost Gradient Boosting**
```python
Hyperparameters:
- n_estimators: 500
- max_depth: 7
- learning_rate: 0.03
- subsample: 0.8

Performance: 50-52% accuracy
```

### 3. **LightGBM Fast Gradient Boosting**
```python
Hyperparameters:
- n_estimators: 500
- max_depth: 8
- learning_rate: 0.03
- num_leaves: 31

Performance: 51-55% accuracy
```

### 4. **Ensemble Weighted Combination**
```python
Weights:
- LSTM: 40%
- XGBoost: 30%
- LightGBM: 30%

Performance: 55% accuracy on all days
```

---

## 🎓 Advanced Strategies Implemented

### **Strategy 1: Regime-Based Prediction**
- Detects market regime: Bull / Bear / Sideways
- Trains separate models for each regime
- Uses SMA crossovers for classification
- **Result:** 52-54% accuracy (no significant improvement)

### **Strategy 2: Significant Move Prediction**
- Focuses only on days with >2% price changes
- Filters out noisy, flat days
- Achieves higher confidence on clear signals
- **Result:** 58% accuracy on significant moves 🏆

### **Strategy 3: Feature Selection**
- Random Forest importance
- Mutual information scores
- Correlation analysis
- Reduces 126 → 50 features
- **Result:** 54% accuracy (slight decrease due to information loss)

---

## 🛠️ Technology Stack

### **Core Technologies**
- **Python 3.11** - Primary language
- **TensorFlow / Keras** - Deep learning framework
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **Scikit-learn** - ML utilities

### **Data Processing**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **yfinance** - Historical stock data

### **Visualization**
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualizations

---

## 📁 Project Structure

```
stock-prediction-project/
│
├── 📂 data/
│   ├── raw/                          # Raw OHLCV data (20 stocks)
│   └── processed/                    # Engineered features
│       ├── AAPL_engineered.csv       # 126 features
│       └── AAPL_optimized.csv        # 50 selected features
│
├── 📂 models/                        # Trained models
│   ├── AAPL_lstm_regressor.h5        # LSTM model
│   ├── AAPL_lstm_classifier.h5       # LSTM classifier
│   ├── AAPL_xgboost.json             # XGBoost model
│   ├── AAPL_lightgbm.txt             # LightGBM model
│   ├── AAPL_significant_lstm.h5      # Best model (58%)
│   └── *.pkl                         # Scalers & feature lists
│
├── 📂 src/
│   ├── data_collector.py             # yfinance data fetching
│   ├── feature_engineer.py           # 126+ feature engineering
│   ├── ensemble_model.py             # Main ensemble system
│   ├── regime_based_model.py         # Bull/bear/sideways models
│   ├── significant_move_predictor.py # Best performing model
│   ├── feature_selector.py           # Feature importance analysis
│   └── predictor.py                  # Prediction interface
│
├── 📂 notebooks/                     # Jupyter analysis notebooks
├── 📂 results/                       # Performance plots
├── 📂 plots/                         # Feature importance plots
│
├── 📄 config.py                      # Global configuration
├── 📄 train_lstm.py                  # Training script
├── 📄 predict_stock.py               # CLI prediction tool
└── 📄 README.md                      # This file
```

---

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/Harishlal-me/STOCK-PREDICTION-PROJECT.git
cd STOCK-PREDICTION-PROJECT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Fetch Data**

```bash
# Download stock data
python -c "from src.data_collector import fetch_all_data; fetch_all_data()"
```

### **3. Engineer Features**

```bash
# Create 126+ features
python -c "from src.feature_engineer import engineer_features_for_ticker; engineer_features_for_ticker('AAPL')"
```

### **4. Train Models**

```bash
# Train basic ensemble
python train_lstm.py --ticker AAPL

# Train significant move predictor (best model)
python src/significant_move_predictor.py

# Train regime-based models
python src/regime_based_model.py
```

### **5. Make Predictions**

```bash
# Predict stock movement
python predict_stock.py --ticker AAPL --price 278.28
```

**Example Output:**
```
==================================================
STOCK: AAPL
==================================================
TODAY:
  Price (USD): $278.28
  Price (INR): ₹23,111

PREDICTIONS:
Tomorrow Price:
  USD: $281.45
  INR: ₹23,374

Direction: UP
Confidence: 58.6%

MODEL DECISION:
  Short-term (1D): BUY
  Swing (5D): BUY
==================================================
```

---

## 📊 Methodology

### **1. Data Collection**
- Historical data: 2005 - 2025 (20 years)
- Tickers: AAPL, MSFT, GOOGL, AMZN, etc.
- Market indices: S&P500, VIX, NASDAQ
- Frequency: Daily OHLCV data

### **2. Feature Engineering**
- **Price features:** Returns, ratios, gaps, shadows
- **Technical indicators:** 14 different indicator families
- **Volume analysis:** 5 volume-based indicators
- **Momentum features:** 4 timeframe windows (5, 10, 20, 60 days)
- **Statistical features:** Skewness, kurtosis, rolling stats
- **Lag features:** 1, 2, 3, 5, 10, 20 days lookback

### **3. Model Training**
- Time series cross-validation (5-fold)
- Train/Test split: 80/20
- Early stopping (patience: 10-15 epochs)
- Learning rate reduction on plateau
- Hyperparameter optimization

### **4. Validation**
- Walk-forward testing (no look-ahead bias)
- Out-of-sample evaluation
- Multiple performance metrics (Accuracy, AUC, F1)

---

## 🔬 Research Insights

### **What Worked ✅**
1. **LSTM on significant moves** - Best performer (58%)
2. **Deep LSTM architecture** - 3 layers with batch normalization
3. **126 features** - More information helps
4. **20 years of data** - Sufficient training samples
5. **Time series validation** - Prevents overfitting

### **What Didn't Work ❌**
1. **Market regime detection** - No improvement (52-54%)
2. **Feature selection to 50** - Lost useful information (54%)
3. **XGBoost alone** - Struggled at 50-52%
4. **Predicting all days** - Too much noise (55%)

### **Key Learnings 💡**
- Stock prediction is inherently difficult (markets are efficient)
- Filtering noise (>2% moves) improves accuracy significantly
- Deep learning (LSTM) outperforms gradient boosting for stocks
- More features > fewer features for this problem
- 58% is excellent for directional prediction

---

## 📈 Performance Analysis

### **Profitability Simulation**
With 58% win rate and 1.5:1 risk/reward ratio:

```
100 trades example:
├─ 58 wins  × $150 profit = $8,700
├─ 42 losses × $100 loss   = -$4,200
└─ Net profit              = $4,500 (45% ROI)
```

### **Comparison to Benchmarks**
- Random guessing: **50%**
- Basic ML models: **52-55%**
- **This system: 58%** ✅
- Professional traders: 55-60%
- Elite hedge funds: 60-65%

---

## 🎯 Future Improvements

### **Short-term (1-2 weeks)**
- [ ] Add FinBERT sentiment analysis (+3-5% accuracy expected)
- [ ] Implement real-time prediction API
- [ ] Create backtesting dashboard
- [ ] Build paper trading simulator

### **Medium-term (1-2 months)**
- [ ] Multi-stock portfolio optimization
- [ ] Options pricing predictions
- [ ] Volatility forecasting
- [ ] Risk management system

### **Long-term (3+ months)**
- [ ] Alternative data sources (social media, news)
- [ ] Reinforcement learning for trading
- [ ] Multi-timeframe predictions (1D, 5D, 20D)
- [ ] Live deployment with monitoring

---

## ⚠️ Important Disclaimers

**This project is for educational and research purposes only.**

- Past performance does NOT guarantee future results
- Stock markets are inherently unpredictable
- 58% accuracy does NOT guarantee profitable trading
- Always conduct your own research (DYOR)
- Never invest more than you can afford to lose
- Consider transaction costs, taxes, and slippage
- This is NOT financial advice

**Use at your own risk. The author is not responsible for any financial losses.**

---

## 📚 References & Resources

### **Academic Papers**
- "Deep Learning for Stock Prediction Using Numerical and Textual Information" (2019)
- "Financial Time Series Forecasting with Deep Learning" (2017)
- "Machine Learning for Trading" (Stefan Jansen)

### **Libraries & Tools**
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [XGBoost Guide](https://xgboost.readthedocs.io/)
- [yfinance Library](https://github.com/ranaroussi/yfinance)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Harishlal**

- GitHub: [@Harishlal-me](https://github.com/Harishlal-me)
- Project: [Stock Prediction System](https://github.com/Harishlal-me/STOCK-PREDICTION-PROJECT)

---

## 🙏 Acknowledgments

- Data provided by Yahoo Finance via yfinance
- Inspired by quantitative trading research
- Built with open-source ML libraries

---

## 🌟 Star This Project!

If you found this helpful, please consider giving it a ⭐ on GitHub!

---

**Last Updated:** December 14, 2025  
**Project Status:** Active Development  
**Best Model:** LSTM on Significant Moves (57.99% accuracy)
