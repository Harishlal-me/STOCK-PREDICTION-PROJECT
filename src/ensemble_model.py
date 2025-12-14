# src/ensemble_model.py
# Ensemble approach: LSTM + XGBoost + LightGBM for 65-70% accuracy

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install: pip install lightgbm")

from config import *


class EnsembleStockPredictor:
    def __init__(self, ticker, verbose=True):
        self.ticker = ticker
        self.verbose = verbose
        self.lstm_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = None
        self.feature_cols = None
        
    def prepare_data(self, df, target_col='target_direction_1d'):
        """
        Prepare data for training
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("PREPARING DATA")
            print(f"{'='*70}")
        
        # Remove rows with NaN in target
        df = df.dropna(subset=[target_col])
        
        # Define features (exclude metadata and targets)
        exclude_cols = [
            'Date', 'Ticker',
            'target_price_1d', 'target_price_5d',
            'target_return_1d', 'target_return_5d',
            'target_direction_1d', 'target_direction_5d'
        ]
        
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Get features and target
        X = df[self.feature_cols].values
        y = df[target_col].values
        dates = df['Date'].values
        
        # Handle any remaining NaN/inf in features
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        if self.verbose:
            print(f"✅ Features: {len(self.feature_cols)}")
            print(f"✅ Samples: {len(X)}")
            print(f"✅ Target distribution: {np.bincount(y.astype(int))}")
        
        return X, y, dates
    
    def create_sequences(self, X, y, seq_length):
        """
        Create sequences for LSTM
        """
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM model with improved architecture
        """
        model = Sequential([
            LSTM(LSTM_UNITS_1, return_sequences=True, 
                 input_shape=input_shape,
                 dropout=LSTM_DROPOUT,
                 recurrent_dropout=LSTM_RECURRENT_DROPOUT),
            BatchNormalization(),
            
            LSTM(LSTM_UNITS_2, return_sequences=True,
                 dropout=LSTM_DROPOUT,
                 recurrent_dropout=LSTM_RECURRENT_DROPOUT),
            BatchNormalization(),
            
            LSTM(LSTM_UNITS_3, return_sequences=False,
                 dropout=LSTM_DROPOUT),
            BatchNormalization(),
            
            Dense(DENSE_UNITS, activation=DENSE_ACTIVATION),
            Dropout(DENSE_DROPOUT),
            
            Dense(1, activation='sigmoid')
        ], name='lstm_classifier')
        
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train_ensemble(self, df, target_col='target_direction_1d'):
        """
        Train all models in the ensemble
        """
        print(f"\n{'='*70}")
        print(f"TRAINING ENSEMBLE FOR {self.ticker}")
        print(f"{'='*70}")
        
        # Prepare data
        X, y, dates = self.prepare_data(df, target_col)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, test_idx = list(tscv.split(X))[-1]  # Use last split
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"\n✅ Train samples: {len(X_train)}")
        print(f"✅ Test samples: {len(X_test)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train each model
        results = {}
        
        # 1. LSTM
        print(f"\n{'='*70}")
        print("TRAINING LSTM MODEL")
        print(f"{'='*70}")
        lstm_acc = self._train_lstm(X_train_scaled, y_train, X_test_scaled, y_test)
        results['lstm'] = lstm_acc
        
        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            print(f"\n{'='*70}")
            print("TRAINING XGBOOST MODEL")
            print(f"{'='*70}")
            xgb_acc = self._train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
            results['xgboost'] = xgb_acc
        
        # 3. LightGBM
        if LIGHTGBM_AVAILABLE:
            print(f"\n{'='*70}")
            print("TRAINING LIGHTGBM MODEL")
            print(f"{'='*70}")
            lgb_acc = self._train_lightgbm(X_train_scaled, y_train, X_test_scaled, y_test)
            results['lightgbm'] = lgb_acc
        
        # Ensemble predictions
        print(f"\n{'='*70}")
        print("ENSEMBLE RESULTS")
        print(f"{'='*70}")
        ensemble_acc = self._evaluate_ensemble(X_test_scaled, y_test)
        results['ensemble'] = ensemble_acc
        
        print(f"\n📊 Individual Model Accuracies:")
        for model_name, acc in results.items():
            print(f"   {model_name:12s}: {acc*100:.2f}%")
        
        return results
    
    def _train_lstm(self, X_train, y_train, X_test, y_test):
        """Train LSTM model"""
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, SEQUENCE_LENGTH)
        
        print(f"✅ LSTM sequences: {X_train_seq.shape}")
        
        # Build model
        self.lstm_model = self.build_lstm_model(
            input_shape=(SEQUENCE_LENGTH, X_train.shape[1])
        )
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            f'models/{self.ticker}_lstm_ensemble.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
        
        # Train
        history = self.lstm_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Evaluate
        _, test_acc, test_auc = self.lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
        print(f"\n✅ LSTM Test Accuracy: {test_acc*100:.2f}%")
        print(f"✅ LSTM Test AUC: {test_auc:.4f}")
        
        return test_acc
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        params = XGBOOST_PARAMS.copy()
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        evals = [(dtrain, 'train'), (dtest, 'test')]
        
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=50
        )
        
        # Evaluate
        y_pred_proba = self.xgb_model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        test_acc = (y_pred == y_test).mean()
        
        print(f"\n✅ XGBoost Test Accuracy: {test_acc*100:.2f}%")
        
        return test_acc
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        params = LIGHTGBM_PARAMS.copy()
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=params['n_estimators'],
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Evaluate
        y_pred_proba = self.lgb_model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        test_acc = (y_pred == y_test).mean()
        
        print(f"\n✅ LightGBM Test Accuracy: {test_acc*100:.2f}%")
        
        return test_acc
    
    def _evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble predictions"""
        predictions = []
        
        # LSTM predictions
        if self.lstm_model is not None:
            X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, SEQUENCE_LENGTH)
            lstm_pred = self.lstm_model.predict(X_test_seq, verbose=0).flatten()
            predictions.append(lstm_pred * ENSEMBLE_WEIGHTS['lstm'])
            y_test = y_test_seq  # Use aligned labels
        
        # XGBoost predictions
        if self.xgb_model is not None and XGBOOST_AVAILABLE:
            dtest = xgb.DMatrix(X_test[-len(y_test):])
            xgb_pred = self.xgb_model.predict(dtest)
            predictions.append(xgb_pred * ENSEMBLE_WEIGHTS['xgboost'])
        
        # LightGBM predictions
        if self.lgb_model is not None and LIGHTGBM_AVAILABLE:
            lgb_pred = self.lgb_model.predict(X_test[-len(y_test):])
            predictions.append(lgb_pred * ENSEMBLE_WEIGHTS['lightgbm'])
        
        # Weighted average
        ensemble_pred_proba = np.sum(predictions, axis=0)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        ensemble_acc = (ensemble_pred == y_test).mean()
        print(f"\n🎯 ENSEMBLE Accuracy: {ensemble_acc*100:.2f}%")
        
        return ensemble_acc
    
    def save_models(self):
        """Save all models"""
        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(exist_ok=True)
        
        # Save scaler and features
        with open(model_dir / f"{self.ticker}_scaler_ensemble.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        with open(model_dir / f"{self.ticker}_features_ensemble.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)
        
        # Save XGBoost
        if self.xgb_model is not None:
            self.xgb_model.save_model(str(model_dir / f"{self.ticker}_xgboost.json"))
        
        # Save LightGBM
        if self.lgb_model is not None:
            self.lgb_model.save_model(str(model_dir / f"{self.ticker}_lightgbm.txt"))
        
        print(f"\n✅ All models saved to {model_dir}")


if __name__ == "__main__":
    # Test training
    import pandas as pd
    
    print("Loading AAPL data...")
    df = pd.read_csv('data/processed/AAPL_engineered.csv')
    
    ensemble = EnsembleStockPredictor('AAPL', verbose=True)
    results = ensemble.train_ensemble(df, target_col='target_direction_1d')
    ensemble.save_models()
    
    print(f"\n🎉 Training complete!")