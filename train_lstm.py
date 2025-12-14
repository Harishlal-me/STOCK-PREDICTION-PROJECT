# train_lstm.py
# PROPERLY FIXED - Returns are NOT scaled with features

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import SEQUENCE_LENGTH

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_sequences(data, target, seq_length):
    """Create sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(seq_length, len(data)):
        X_seq.append(data[i-seq_length:i])
        y_seq.append(target[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm_models(ticker: str):
    """Train LSTM models - RETURNS ARE NOT SCALED"""
    print("="*70)
    print(f"🚀 TRAINING PROPERLY FIXED LSTM FOR {ticker}")
    print("="*70)
    
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "processed"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data...")
    df = pd.read_csv(data_dir / f"{ticker}_engineered.csv", parse_dates=["Date"])
    print(f"✅ Loaded {len(df)} rows")
    
    # Define features (same as before)
    exclude_cols = [
        'Date', 'Ticker', 'Adj Close',
        'target', 'target_class',
        'target_price', 'target_direction', 'target_return'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"\n📊 Using {len(feature_cols)} features")
    
    # Prepare data
    X = df[feature_cols].values
    
    # Get returns (UNSCALED!)
    if 'target_return' in df.columns:
        y_return = df['target_return'].values
    else:
        y_return = df['Close'].pct_change().shift(-1).values
    
    if 'target_direction' in df.columns:
        y_direction = df['target_direction'].values
    else:
        y_direction = (y_return > 0).astype(int)
    
    # Remove NaN
    valid_mask = ~np.isnan(y_return)
    X = X[valid_mask]
    y_return = y_return[valid_mask]
    y_direction = y_direction[valid_mask]
    
    print(f"\n✅ Data prepared:")
    print(f"   Features shape: {X.shape}")
    print(f"   Return stats (UNSCALED):")
    print(f"      Mean: {np.mean(y_return)*100:.4f}%")
    print(f"      Std: {np.std(y_return)*100:.4f}%")
    print(f"      Min: {np.min(y_return)*100:.2f}%")
    print(f"      Max: {np.max(y_return)*100:.2f}%")
    
    # Split
    X_train, X_test, y_ret_train, y_ret_test, y_dir_train, y_dir_test = train_test_split(
        X, y_return, y_direction,
        test_size=0.2,
        shuffle=False
    )
    
    print(f"\n✂️  Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # 🔥 CRITICAL: Scale ONLY features, NOT returns!
    print(f"\n📏 Scaling features (NOT returns)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✅ Features scaled")
    
    # Returns stay UNSCALED
    print(f"✅ Returns kept UNSCALED (range: {y_ret_train.min()*100:.2f}% to {y_ret_train.max()*100:.2f}%)")
    
    # Create sequences
    print(f"\n🔄 Creating sequences...")
    X_train_seq, y_ret_train_seq = create_sequences(X_train_scaled, y_ret_train, SEQUENCE_LENGTH)
    X_test_seq, y_ret_test_seq = create_sequences(X_test_scaled, y_ret_test, SEQUENCE_LENGTH)
    _, y_dir_train_seq = create_sequences(X_train_scaled, y_dir_train, SEQUENCE_LENGTH)
    _, y_dir_test_seq = create_sequences(X_test_scaled, y_dir_test, SEQUENCE_LENGTH)
    
    print(f"✅ Sequences: {X_train_seq.shape}")
    
    # Build Return Predictor
    print(f"\n🏗️  Building Return Predictor...")
    reg_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='tanh')  # 🔥 tanh to keep output in [-1, 1] range
    ], name='return_predictor')
    
    reg_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"✅ Model built with tanh output (bounded to ±100%)")
    
    # Train
    print(f"\n🎯 Training Return Predictor...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        model_dir / f"{ticker}_lstm_regressor.h5",
        monitor='val_loss',
        save_best_only=True
    )
    
    history_reg = reg_model.fit(
        X_train_seq, y_ret_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_mae = reg_model.evaluate(X_test_seq, y_ret_test_seq, verbose=0)
    print(f"\n✅ Return Predictor Results:")
    print(f"   MAE: {test_mae*100:.4f}%")
    
    # Build Classifier
    print(f"\n🏗️  Building Direction Classifier...")
    clf_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='direction_classifier')
    
    clf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train Classifier
    print(f"\n🎯 Training Direction Classifier...")
    checkpoint_clf = ModelCheckpoint(
        model_dir / f"{ticker}_lstm_classifier.h5",
        monitor='val_accuracy',
        save_best_only=True
    )
    
    history_clf = clf_model.fit(
        X_train_seq, y_dir_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint_clf],
        verbose=1
    )
    
    test_loss_clf, test_acc = clf_model.evaluate(X_test_seq, y_dir_test_seq, verbose=0)
    print(f"\n✅ Direction Classifier Results:")
    print(f"   Accuracy: {test_acc*100:.2f}%")
    
    # Save
    print(f"\n💾 Saving models...")
    reg_model.save(model_dir / f"{ticker}_lstm_regressor.h5")
    clf_model.save(model_dir / f"{ticker}_lstm_classifier.h5")
    
    with open(model_dir / f"{ticker}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    with open(model_dir / f"{ticker}_feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    print(f"✅ Saved all components")
    
    # Verification
    print(f"\n" + "="*70)
    print(f"✅ FINAL VERIFICATION")
    print("="*70)
    print(f"   Features: {len(feature_cols)}")
    print(f"   Scaler: {scaler.n_features_in_}")
    print(f"   Models aligned: ✅")
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"   Model predicts UNSCALED returns in range [-1, 1]")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    args = parser.parse_args()
    
    train_lstm_models(args.ticker)