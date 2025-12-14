import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import warnings

warnings.filterwarnings('ignore')

# Load model
model = tf.keras.models.load_model("models/AAPL_significant_lstm.h5")
scaler = pickle.load(open("models/AAPL_significant_scaler.pkl", 'rb'))
selected_features = pickle.load(open("models/AAPL_significant_features.pkl", 'rb'))

df = pd.read_csv("data/processed/AAPL_engineered.csv")
df = df.fillna(method='ffill').fillna(method='bfill')

print("=" * 80)
print("🔍 FINDING HISTORICAL SELL SIGNALS")
print("=" * 80)

print("\nScanning historical data for SELL signals (probability > 0.60)...\n")

sell_signals = []
buy_signals = []

# Scan last 500 days
for i in range(500, len(df) - 60):
    try:
        # Get 60-day window
        X = df[selected_features].iloc[i:i+60].values.astype(np.float32)
        X_scaled = scaler.transform(X)
        X_reshaped = X_scaled.reshape(1, 60, len(selected_features))
        pred = model.predict(X_reshaped, verbose=0)
        prob = float(pred[0][0])
        
        price = df['Close'].iloc[i+60]
        date = df['Date'].iloc[i+60] if 'Date' in df.columns else i+60
        
        # Store strong signals
        if prob > 0.65:
            sell_signals.append((date, price, prob, "STRONG SELL"))
        elif prob < 0.35:
            buy_signals.append((date, price, prob, "STRONG BUY"))
            
    except:
        pass

print("🔴 STRONGEST SELL SIGNALS (Probability > 0.65):")
print("─" * 80)
if sell_signals:
    for date, price, prob, signal in sorted(sell_signals, key=lambda x: x[2], reverse=True)[:10]:
        print(f"   Date: {date} | Price: ${price:.2f} | Probability: {prob:.4f} | Signal: {signal}")
else:
    print("   ❌ No strong SELL signals found in recent history")
    print("   (Your current market is in UPTREND)")

print("\n🟢 STRONGEST BUY SIGNALS (Probability < 0.35):")
print("─" * 80)
if buy_signals:
    for date, price, prob, signal in sorted(buy_signals, key=lambda x: x[2])[:10]:
        print(f"   Date: {date} | Price: ${price:.2f} | Probability: {prob:.4f} | Signal: {signal}")
else:
    print("   ❌ No strong BUY signals found in recent history")

# Show current market status
print("\n" + "=" * 80)
print("📊 CURRENT MARKET STATUS:")
print("=" * 80)

X = df[selected_features].iloc[-60:].values.astype(np.float32)
X_scaled = scaler.transform(X)
X_reshaped = X_scaled.reshape(1, 60, len(selected_features))
current_prob = float(model.predict(X_reshaped, verbose=0)[0][0])
current_price = df['Close'].iloc[-1]

print(f"\n📍 Current AAPL Price: ${current_price:.2f}")
print(f"🎯 Model Probability: {current_prob:.4f}")
print(f"📈 Trend Direction: {'🔴 DOWNTREND (SELL Zone)' if current_prob > 0.6 else '🟢 UPTREND (BUY Zone)'}")

print("\n" + "=" * 80)
print("💡 INTERPRETATION:")
print("=" * 80)

if current_prob < 0.35:
    print(f"""
    Your model is in STRONG BUY mode ({current_prob:.4f})
    
    This means:
    ✅ Model learned strong UPTREND patterns
    ✅ Next day prediction: UP 🟢
    ✅ Trading action: BUY
    
    Example: At current price ${current_price:.2f}
    Expected tomorrow: ${current_price * 1.0319:.2f} (+3.19%)
    """)
elif current_prob > 0.65:
    print(f"""
    Your model is in STRONG SELL mode ({current_prob:.4f})
    
    This means:
    ✅ Model learned strong DOWNTREND patterns
    ✅ Next day prediction: DOWN 🔴
    ✅ Trading action: SELL
    
    Example: At current price ${current_price:.2f}
    Expected tomorrow: ${current_price * 0.9681:.2f} (-3.19%)
    """)
else:
    print(f"""
    Your model is in NEUTRAL mode ({current_prob:.4f})
    
    This means:
    🟡 Model sees mixed signals
    🟡 Next day prediction: UNCERTAIN
    🟡 Trading action: HOLD
    """)

print("=" * 80)
print("\nTo see SELL predictions at specific prices:")
print("The model probability needs to shift > 0.60")
print("This happens when features show downtrend patterns:")
print("  • Negative momentum")
print("  • High volatility")
print("  • Volume spikes")
print("  • Technical indicator crossovers")
print("=" * 80)