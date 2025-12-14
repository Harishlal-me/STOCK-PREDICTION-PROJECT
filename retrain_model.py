import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 RETRAINING LSTM MODEL")
print("=" * 70)

# Load and clean data
print("\n1️⃣ Loading data...")
df = pd.read_csv("data/processed/AAPL_engineered.csv")
df = df.fillna(method='ffill').fillna(method='bfill')
print(f"   Shape: {df.shape}")

# Get numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['Close', 'Volume', 'Open', 'High', 'Low']]

# Select top 50 features
print("\n2️⃣ Selecting features...")
feature_var = df[numeric_cols].var().sort_values(ascending=False)
selected_features = feature_var.head(50).index.tolist()
print(f"   Selected {len(selected_features)} features")

# Create target
print("\n3️⃣ Creating target...")
df['next_return'] = df['Close'].pct_change().shift(-1)
df['target'] = (df['next_return'].abs() > 0.02).astype(int)

# Clean data
df_clean = df[selected_features + ['target']].dropna()
print(f"   Data shape: {df_clean.shape}")
print(f"   Positive samples: {df_clean['target'].sum()}")

# Prepare sequences
print("\n4️⃣ Creating sequences...")
X = df_clean[selected_features].values
y = df_clean['target'].values

sequence_len = 60
X_seq = []
y_seq = []

for i in range(len(X) - sequence_len):
    X_seq.append(X[i:i+sequence_len])
    y_seq.append(y[i+sequence_len])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)
print(f"   Sequences: {X_seq.shape}")

# Scale
print("\n5️⃣ Scaling...")
scaler = StandardScaler()
X_2d = X_seq.reshape(-1, X_seq.shape[-1])
X_2d = scaler.fit_transform(X_2d)
X_seq = X_2d.reshape(X_seq.shape)

# Split
print("\n6️⃣ Splitting...")
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Build model
print("\n7️⃣ Building model...")
model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', input_shape=(60, len(selected_features))),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
print("\n8️⃣ Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate
print("\n9️⃣ Evaluating...")
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"   Train Accuracy: {train_acc*100:.2f}%")
print(f"   Test Accuracy: {test_acc*100:.2f}%")

# Save
print("\n🔟 Saving...")
model.save("models/AAPL_significant_lstm.h5")
pickle.dump(scaler, open("models/AAPL_significant_scaler.pkl", 'wb'))
pickle.dump(selected_features, open("models/AAPL_significant_features.pkl", 'wb'))

print("\n" + "=" * 70)
print("✅ RETRAINING COMPLETE!")
print("=" * 70)
print("\nNow run: py predict_stock.py --price 280.00")