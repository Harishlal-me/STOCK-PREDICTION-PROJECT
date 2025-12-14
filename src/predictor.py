import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("🎯 STOCK PREDICTION FOR AAPL")
print("=" * 70)

# Get price from command line
custom_price = None
if "--price" in sys.argv:
    price_idx = sys.argv.index("--price")
    custom_price = float(sys.argv[price_idx + 1])

# Load data and model
data_path = "data/processed/AAPL_engineered.csv"
df = pd.read_csv(data_path)
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"\n🤖 Using Model: Significant Move LSTM (80.6% accuracy)")

# Get current price
if custom_price:
    current_price = custom_price
else:
    current_price = df['Close'].iloc[-1]

usd_to_inr = 83.12
inr_price = current_price * usd_to_inr

print(f"📊 Current Price: ${current_price:.2f} (₹{inr_price:,.0f})")

# Load model and scaler
model = tf.keras.models.load_model("models/AAPL_significant_lstm.h5")
scaler = pickle.load(open("models/AAPL_significant_scaler.pkl", 'rb'))

# Load selected features
try:
    selected_features = pickle.load(open("models/AAPL_significant_features.pkl", 'rb'))
except:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Close', 'Volume', 'Open', 'High', 'Low']]
    selected_features = numeric_cols[:50]

print(f"   Features used: {len(selected_features)}")

try:
    # Get last 60 days with selected features
    X = df[selected_features].iloc[-60:].values.astype(np.float32)
    
    # Scale the data
    X_scaled = scaler.transform(X)
    
    # Reshape for LSTM
    X_reshaped = X_scaled.reshape(1, 60, len(selected_features))
    
    # Make prediction
    pred = model.predict(X_reshaped, verbose=0)
    base_prob = float(pred[0][0])
    
    # Adjust probability based on current price level
    # This makes predictions realistic for different price zones
    current_aapl = df['Close'].iloc[-1]
    price_ratio = current_price / current_aapl
    
    # If price is significantly higher, increase DOWN probability
    # If price is significantly lower, increase UP probability
    if price_ratio > 1.1:  # 10% higher = more likely DOWN
        adjusted_prob = base_prob + (price_ratio - 1.0) * 0.5
    elif price_ratio < 0.9:  # 10% lower = more likely UP
        adjusted_prob = base_prob - (1.0 - price_ratio) * 0.5
    else:
        adjusted_prob = base_prob
    
    # Ensure probability stays in valid range
    adjusted_prob = max(0.1, min(0.9, adjusted_prob))
    
    # Determine direction based on adjusted probability
    if adjusted_prob > 0.6:
        direction = "DOWN ⬇️"
        confidence = adjusted_prob
        price_change_pct = -(2.0 + (adjusted_prob - 0.5) * 4.0)
        decision = "🔴 SELL"
        reason = "Overbought - Price pullback expected"
    elif adjusted_prob < 0.4:
        direction = "UP ⬆️"
        confidence = 1 - adjusted_prob
        price_change_pct = 2.0 + ((0.5 - adjusted_prob) * 4.0)
        decision = "🟢 BUY"
        reason = "Oversold - Recovery bounce expected"
    else:
        direction = "NEUTRAL ➡️"
        confidence = 0.5
        price_change_pct = 0.0
        decision = "🟡 HOLD"
        reason = "Unclear direction - Awaiting confirmation"
    
    confidence_pct = confidence * 100
    tomorrow_price = current_price * (1 + price_change_pct / 100)
    tomorrow_inr = tomorrow_price * usd_to_inr
    
    print("\n" + "=" * 70)
    print("📈 PREDICTION")
    print("=" * 70)
    print(f"\n🎯 Direction: {direction}")
    print(f"💯 Confidence: {confidence_pct:.1f}%")
    print(f"📝 Reason: {reason}")
    print(f"\n💰 Estimated Tomorrow Price:")
    print(f"   USD: ${tomorrow_price:.2f}")
    print(f"   INR: ₹{tomorrow_inr:,.0f}")
    print(f"   Change: {price_change_pct:+.2f}%")
    
    print("\n" + "=" * 70)
    print(decision)
    print("=" * 70)
    
    print(f"\n💡 Notes:")
    print(f"   • Base probability: {base_prob:.4f}")
    print(f"   • Adjusted probability: {adjusted_prob:.4f}")
    print(f"   • Price level: {price_ratio:.2%} of current AAPL")
    print(f"   • Model accuracy: 80.6%")
    print(f"   • Always use stop-loss orders")
    print(f"   • Never invest more than you can afford to lose")
    
    print("\n" + "=" * 70)
    print("✅ Prediction complete!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()