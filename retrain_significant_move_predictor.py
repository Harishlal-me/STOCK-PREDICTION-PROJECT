import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 RETRAINING SIGNIFICANT MOVE LSTM")
print("=" * 70)

# Load data
print("\n1️⃣ Loading data...")
df = pd.read_csv("data/processed/AAPL_engineered.csv")
df = df.fillna(method='ffill').fillna(method='bfill')
print(f"   Data shape: {df.shape}")

# Get numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['Close', 'Volume', 'Open', 'High', 'Low']]

print(f"\n2️⃣ Feature selection...")
print(f"   Total features: {len(numeric_cols)}")

# Select top 50 features by variance
feature_variance = df[numeric_cols].var().sort_values(ascending=False)
selected_features = feature_variance.head(50).index.tolist()
print(f"   Selected features: {len(selected_features)}")

# Save selected features
pickle.dump(selected_features, open("models/AAPL_significant_features.pkl", 'wb'))
print(f"   ✅ Saved to models/AAPL_significant_features.pkl")

# Create target: 1 if next day has >2% move, 0 otherwise
print(f"\n3️⃣ Creating target variable...")
df['price_change'] = df['Close'].pct_change().abs()
df['significant_move'] = (df['price_change'] > 0.02).astype(int)

# Remove rows with NaN
df_clean = df.dropna(subset=selected_features + ['significant_move'])
print(f"   Clean data shape: {df_clean.shape}")
print(f"   Significant moves: {df_clean['significant_move'].sum()} ({df_clean['significant_move'].mean()*100:.1f}%)")

# Prepare sequences
print(f"\n4️⃣ Creating sequences...")
X_data = df_clean[selected_features].values
y_data = df_clean['significant_move'].values

# Create 60-day sequences
sequence_length = 60
X_sequences = []
y_sequences = []

for i in range(len(X_data) - sequence_length):
    X_sequences.append(X_data[i:i+sequence_length])
    y_sequences.append(y_data[i+sequence_length])

X_sequences = np.array(X_sequences, dtype=np.float32)
y_sequences = np.array(y_sequences, dtype=np.float32)

print(f"   Sequences created: {len(X_sequences)}")
print(f"   Shape: {X_sequences.shape}")

# Scale features
print(f"\n5️⃣ Scaling features...")
scaler = StandardScaler()
X_reshaped = X_sequences.reshape(-1, X_sequences.shape[-1])
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X_sequences.shape)

# Save scaler
pickle.dump(scaler, open("models/AAPL_significant_scaler.pkl", 'wb'))
print(f"   ✅ Scaler saved")

# Split data
print(f"\n6️⃣ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_sequences, test_size=0.2, random_state=42, shuffle=False
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   Train positive ratio: {y_train.mean()*100:.1f}%")
print(f"   Test positive ratio: {y_test.mean()*100:.1f}%")

# Build LSTM model
print(f"\n7️⃣ Building LSTM model...")
model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', input_shape=(sequence_length, len(selected_features))),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)

print(f"   Model built!")
print(f"   Parameters: {model.count_params():,}")

# Train model
print(f"\n8️⃣ Training model...")
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
print(f"\n9️⃣ Evaluating model...")
train_loss, train_acc, train_auc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)

print(f"   Train Accuracy: {train_acc*100:.2f}%")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Train AUC: {train_auc:.4f}")
print(f"   Test AUC: {test_auc:.4f}")

# Save model
print(f"\n🔟 Saving model...")
model.save("models/AAPL_significant_lstm.h5")
print(f"   ✅ Model saved to models/AAPL_significant_lstm.h5")

print("\n" + "=" * 70)
print("✅ RETRAINING COMPLETE!")
print("=" * 70)
print(f"\nYour model is now ready to use!")
print(f"Run: python predict_stock.py --price 280.00")