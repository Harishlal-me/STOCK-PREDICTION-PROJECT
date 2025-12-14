import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

print("=" * 80)
print("🎯 STOCK PREDICTION FOR ANY TICKER")
print("=" * 80)

# Get ticker and price from command line
ticker = "AAPL"
custom_price = None

if "--ticker" in sys.argv:
    ticker_idx = sys.argv.index("--ticker")
    ticker = sys.argv[ticker_idx + 1].upper()

if "--price" in sys.argv:
    price_idx = sys.argv.index("--price")
    custom_price = float(sys.argv[price_idx + 1])

print(f"\n📱 Ticker: {ticker}")

# Check if model exists
model_path = f"models/{ticker}_lstm.h5"
if not os.path.exists(model_path):
    print(f"\n❌ Error: No trained model for {ticker}")
    print(f"   Available models:")
    for file in os.listdir("models"):
        if file.endswith("_lstm.h5"):
            stock = file.replace("_lstm.h5", "")
            print(f"   - {stock}")
    sys.exit(1)

print(f"✅ Found model for {ticker}")

# Load model and scaler
try:
    model = tf.keras.models.load_model(model_path)
    scaler = pickle.load(open(f"models/{ticker}_scaler.pkl", 'rb'))
    selected_features = pickle.load(open(f"models/{ticker}_features.pkl", 'rb'))
    print(f"✅ Loaded model, scaler, and features")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Download stock data
print(f"\n📊 Downloading {ticker} data...")
try:
    df = yf.download(ticker, period="max", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.reset_index(drop=True)
    
    baseline_price = df['Close'].iloc[-1]
    print(f"✅ Downloaded {len(df)} days")
except Exception as e:
    print(f"❌ Error downloading data: {e}")
    sys.exit(1)

# Get prediction price
if custom_price:
    current_price = custom_price
else:
    current_price = baseline_price

usd_to_inr = 83.12
inr_price = current_price * usd_to_inr

print(f"\n💰 Current Price: ${current_price:.2f}")
print(f"   Baseline {ticker}: ${baseline_price:.2f}")

# Feature engineering
print(f"\n🔧 Engineering features...")

df['ret_1'] = df['Close'].pct_change(1)
df['ret_2'] = df['Close'].pct_change(2)
df['ret_5'] = df['Close'].pct_change(5)
df['ret_10'] = df['Close'].pct_change(10)

df['hl_ratio'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
df['close_pos'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)

df['vol_ma_20'] = df['Volume'].rolling(20).mean()
df.loc[:, 'vol_ratio'] = (df['Volume'] / (df['vol_ma_20'] + 1e-9)).values

for w in [5, 10, 20, 50]:
    df[f'sma_{w}'] = df['Close'].rolling(w).mean()

df['dist_sma_20'] = (df['Close'] - df['sma_20']) / (df['sma_20'] + 1e-9)
df['dist_sma_50'] = (df['Close'] - df['sma_50']) / (df['sma_50'] + 1e-9)

df['vol_10'] = df['ret_1'].rolling(10).std()
df['vol_20'] = df['ret_1'].rolling(20).std()

df['momentum_5'] = df['Close'] - df['Close'].shift(5)
df['momentum_10'] = df['Close'] - df['Close'].shift(10)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"✅ Features ready")

# Make prediction
try:
    X = df[selected_features].iloc[-60:].values.astype(np.float32)
    
    if len(X) < 60:
        print(f"⚠️  Only {len(X)} days available")
        X = np.pad(X, ((60-len(X), 0), (0, 0)), mode='mean')
    
    X_scaled = scaler.transform(X)
    X_reshaped = X_scaled.reshape(1, 60, len(selected_features))
    
    pred = model.predict(X_reshaped, verbose=0)
    base_prob = float(pred[0][0])
    
    # Adjust based on price level
    price_ratio = current_price / baseline_price
    
    if price_ratio > 1.15:
        adjusted_prob = 0.75
    elif price_ratio > 1.10:
        adjusted_prob = 0.68
    elif price_ratio > 1.05:
        adjusted_prob = 0.55
    elif price_ratio < 0.85:
        adjusted_prob = 0.15
    elif price_ratio < 0.90:
        adjusted_prob = 0.25
    elif price_ratio < 0.95:
        adjusted_prob = 0.40
    else:
        adjusted_prob = base_prob
    
    adjusted_prob = max(0.1, min(0.9, adjusted_prob))
    
    # Determine signal
    if adjusted_prob > 0.60:
        direction = "DOWN ⬇️"
        confidence = adjusted_prob
        price_change_pct = -(2.0 + (adjusted_prob - 0.5) * 4.0)
        decision = "🔴 SELL"
        reason = "Overbought - Pullback expected"
    elif adjusted_prob < 0.40:
        direction = "UP ⬆️"
        confidence = 1 - adjusted_prob
        price_change_pct = 2.0 + ((0.5 - adjusted_prob) * 4.0)
        decision = "🟢 BUY"
        reason = "Oversold - Recovery expected"
    else:
        direction = "NEUTRAL ➡️"
        confidence = 0.5
        price_change_pct = 0.0
        decision = "🟡 HOLD"
        reason = "Mixed signals - Awaiting confirmation"
    
    tomorrow_price = current_price * (1 + price_change_pct / 100)
    tomorrow_inr = tomorrow_price * usd_to_inr
    
    print("\n" + "=" * 80)
    print("📈 PREDICTION")
    print("=" * 80)
    print(f"\n🎯 Direction: {direction}")
    print(f"💯 Confidence: {confidence*100:.1f}%")
    print(f"📝 Reason: {reason}")
    print(f"\n💰 Estimated Tomorrow Price:")
    print(f"   USD: ${tomorrow_price:.2f}")
    print(f"   INR: ₹{tomorrow_inr:,.0f}")
    print(f"   Change: {price_change_pct:+.2f}%")
    
    print("\n" + "=" * 80)
    print(decision)
    print("=" * 80)
    
    print(f"\n💡 Analysis:")
    print(f"   • Ticker: {ticker}")
    print(f"   • Price Level: ${current_price:.2f}")
    print(f"   • vs Baseline: {(price_ratio - 1)*100:+.1f}%")
    print(f"   • Base Probability: {base_prob:.4f}")
    print(f"   • Adjusted Probability: {adjusted_prob:.4f}")
    print(f"   • Signal Strength: {'Strong' if confidence*100 >= 70 else 'Moderate' if confidence*100 >= 55 else 'Weak'}")
    
    print("\n" + "=" * 80)
    print("✅ Prediction complete!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()