import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

STOCKS = ['NVDA', 'META', 'MSFT', 'AAPL', 'GOOGL', 'AMZN']

print("=" * 90)
print("🚀 TRAINING PREDICTION MODELS FOR 6 MAJOR STOCKS")
print("=" * 90)

for ticker in STOCKS:
    print(f"\n\n{'=' * 90}")
    print(f"📊 TRAINING {ticker}")
    print('=' * 90)

    try:
        # Download data
        print(f"\n1️⃣ Downloading {ticker} data...")
        df = yf.download(ticker, period="max", progress=False)
        
        # Fix columns properly
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df = df.reset_index(drop=True)
        
        print(f"   ✅ Downloaded {len(df)} days")

        if len(df) < 300:
            print("   ⚠️ Not enough data")
            continue

        # Feature engineering
        print("\n2️⃣ Engineering features...")

        # Returns
        df['ret_1'] = df['Close'].pct_change(1)
        df['ret_2'] = df['Close'].pct_change(2)
        df['ret_5'] = df['Close'].pct_change(5)
        df['ret_10'] = df['Close'].pct_change(10)

        # Price structure
        df['hl_ratio'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
        df['close_pos'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)

        # Volume
        df['vol_ma_20'] = df['Volume'].rolling(20).mean()
        df.loc[:, 'vol_ratio'] = (df['Volume'] / (df['vol_ma_20'] + 1e-9)).values

        # Moving averages
        for w in [5, 10, 20, 50]:
            df[f'sma_{w}'] = df['Close'].rolling(w).mean()

        # Distance from MA
        df['dist_sma_20'] = (df['Close'] - df['sma_20']) / (df['sma_20'] + 1e-9)
        df['dist_sma_50'] = (df['Close'] - df['sma_50']) / (df['sma_50'] + 1e-9)

        # Volatility
        df['vol_10'] = df['ret_1'].rolling(10).std()
        df['vol_20'] = df['ret_1'].rolling(20).std()

        # Momentum
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)

        # Target
        df['next_ret'] = df['Close'].pct_change().shift(-1)
        df['target'] = (df['next_ret'].abs() > 0.02).astype(int)

        # Clean
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').dropna()

        # Select features
        feature_cols = [c for c in df.columns if c not in 
                       ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'next_ret', 'target']]

        print(f"   ✅ Created {len(feature_cols)} features")

        num_features = min(20, len(feature_cols))
        selected_features = (df[feature_cols].var().sort_values(ascending=False)
                            .head(num_features).index.tolist())

        print(f"   ✅ Using {num_features} features")

        # Create sequences
        print("\n3️⃣ Creating sequences...")
        X = df[selected_features].values.astype(np.float32)
        y = df['target'].values.astype(np.float32)

        SEQ_LEN = 60
        X_seq, y_seq = [], []

        for i in range(len(X) - SEQ_LEN):
            X_seq.append(X[i:i + SEQ_LEN])
            y_seq.append(y[i + SEQ_LEN])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        print(f"   ✅ Created {len(X_seq)} sequences")

        # Scale
        print("\n4️⃣ Scaling...")
        scaler = StandardScaler()
        X_flat = X_seq.reshape(-1, num_features)
        X_flat = scaler.fit_transform(X_flat)
        X_seq = X_flat.reshape(X_seq.shape)

        # Split
        print("\n5️⃣ Splitting (80/20)...")
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        print(f"   ✅ Train: {len(X_train)} | Test: {len(X_test)}")

        # Model
        print("\n6️⃣ Building model...")
        model = keras.Sequential([
            keras.layers.Input(shape=(SEQ_LEN, num_features)),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=keras.optimizers.Adam(0.001),
                     loss='binary_crossentropy', metrics=['accuracy'])

        # Train
        print("\n7️⃣ Training...")
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                 epochs=50, batch_size=32, verbose=0)

        # Evaluate
        train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

        print(f"   📊 Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

        # Save
        print("\n8️⃣ Saving...")
        os.makedirs("models", exist_ok=True)
        model.save(f"models/{ticker}_lstm.h5")
        pickle.dump(scaler, open(f"models/{ticker}_scaler.pkl", "wb"))
        pickle.dump(selected_features, open(f"models/{ticker}_features.pkl", "wb"))

        print(f"\n✅ {ticker} COMPLETE - Accuracy: {test_acc*100:.2f}%")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)[:80]}")

print("\n" + "=" * 90)
print("🎉 TRAINING FINISHED!")
print("=" * 90)

# Summary
for ticker in STOCKS:
    if os.path.exists(f"models/{ticker}_lstm.h5"):
        print(f"✅ {ticker}")
    else:
        print(f"❌ {ticker}")