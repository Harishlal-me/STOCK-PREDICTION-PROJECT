# ============================================================
# 04_model_training.py
# ============================================================

# Cell 1: Imports and Setup
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

from src.model_builder import (
    LSTMRegressor,
    LSTMClassifier,
    create_sequences,
    train_test_split_timeseries
)

from config import SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, RANDOM_SEED

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

print("✓ Imports successful")

# ============================================================
# Cell 2: Load Engineered Data
# ============================================================

print("=" * 70)
print("LOADING ENGINEERED DATA")
print("=" * 70)

ticker = "AAPL"
BASE_DIR = Path(__file__).resolve().parents[1]
processed_path = BASE_DIR / "data" / "processed" / f"{ticker}_engineered.csv"

print(f"Looking for file at: {processed_path}")

if not processed_path.exists():
    raise FileNotFoundError("Run notebook_03_feature_engineering first")

engineered_df = pd.read_csv(processed_path)
print(f"✓ Loaded {ticker} engineered data: {engineered_df.shape}")

# ============================================================
# Cell 3: Prepare Data
# ============================================================

print("\n" + "=" * 70)
print("PREPARING DATA")
print("=" * 70)

feature_cols = [
    col for col in engineered_df.columns
    if col not in [
        "Date", "Ticker",
        "target_price", "target_direction", "target_return"
    ]
]

df_clean = engineered_df.dropna()

print(f"Features: {len(feature_cols)}")
print(f"Clean records: {len(df_clean)}")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_clean[feature_cols])

print("✓ Data normalized")

# ============================================================
# Cell 4: Create Sequences
# ============================================================

print("\n" + "=" * 70)
print("CREATING SEQUENCES")
print("=" * 70)

data_for_sequences = np.hstack([
    X_scaled,
    df_clean[["target_price"]].values
])

X, y_price = create_sequences(
    data_for_sequences,
    lookback=SEQUENCE_LENGTH
)

print(f"✓ Sequences created")
print(f"  X shape: {X.shape}")
print(f"  y_price shape: {y_price.shape}")

# ✅ FIXED DIRECTION TARGET
y_direction = df_clean["target_direction"].iloc[SEQUENCE_LENGTH:].values

print(f"  y_direction shape: {y_direction.shape}")
print(f"  Direction distribution: {np.bincount(y_direction)}")

# ============================================================
# Cell 5: Train / Validation / Test Split
# ============================================================

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = (
    train_test_split_timeseries(X, y_price, train_size=0.70, val_size=0.15)
)

y_dir_train = y_direction[:len(X_train)]
y_dir_val = y_direction[len(X_train):len(X_train) + len(X_val)]
y_dir_test = y_direction[len(X_train) + len(X_val):]

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

# ============================================================
# Cell 6: Build & Train LSTM Regressor
# ============================================================

input_shape = (SEQUENCE_LENGTH, X_train.shape[2])

lstm_reg = LSTMRegressor(input_shape=input_shape)
lstm_reg.build()

history_reg = lstm_reg.train(
    X_train, y_train,
    X_val, y_val,
    epochs=EPOCHS
)

# ============================================================
# Cell 7: Build & Train LSTM Classifier
# ============================================================

lstm_clf = LSTMClassifier(input_shape=input_shape)
lstm_clf.build()

history_clf = lstm_clf.train(
    X_train, y_dir_train,
    X_val, y_dir_val,
    epochs=EPOCHS
)

# ============================================================
# Cell 8: Test Predictions
# ============================================================

y_pred_reg = lstm_reg.predict(X_test).flatten()
y_pred_clf = lstm_clf.predict(X_test).flatten()

y_pred_binary = (y_pred_clf > 0.5).astype(int)
accuracy = (y_pred_binary == y_dir_test).mean()

print(f"Classifier accuracy: {accuracy:.2%}")

# ============================================================
# Cell 9: Save Models
# ============================================================

models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

lstm_reg.get_model().save(models_dir / f"{ticker}_lstm_regressor.h5")
lstm_clf.get_model().save(models_dir / f"{ticker}_lstm_classifier.h5")

import pickle

# Save scaler
with open(models_dir / f"{ticker}_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ SAVE FEATURE COLUMNS (CRITICAL FIX)
with open(models_dir / f"{ticker}_feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("✓ Models, scaler, and feature columns saved")
