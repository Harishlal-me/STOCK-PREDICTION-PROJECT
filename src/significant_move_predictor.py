# src/significant_move_predictor.py
# Predict only significant moves (>2% change) - filters out noisy flat days

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

from config import *


class SignificantMovePredictor:
    """
    Predict significant price moves (>threshold) instead of any directional change
    
    Theory: Small moves (<2%) are mostly noise. By focusing on larger moves,
    we get clearer signals and higher accuracy.
    """
    
    def __init__(self, ticker, threshold=0.02, verbose=True):
        self.ticker = ticker
        self.threshold = threshold  # 2% default
        self.verbose = verbose
        
        self.lstm_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = None
        self.feature_cols = None
    
    def create_significant_targets(self, df):
        """
        Create targets for significant moves only
        
        Classes:
        - 0: No significant move (|return| < threshold)
        - 1: Significant UP move (return > threshold)
        - 2: Significant DOWN move (return < -threshold)
        """
        print(f"\n{'='*70}")
        print(f"CREATING SIGNIFICANT MOVE TARGETS (threshold={self.threshold*100}%)")
        print(f"{'='*70}")
        
        # Calculate returns if not present
        if 'target_return_1d' not in df.columns:
            df['target_return_1d'] = df['Close'].pct_change().shift(-1)
        
        # Create multi-class target
        df['target_significant'] = 0  # Default: no significant move
        df.loc[df['target_return_1d'] > self.threshold, 'target_significant'] = 1  # UP
        df.loc[df['target_return_1d'] < -self.threshold, 'target_significant'] = 2  # DOWN
        
        # Also create binary targets for each direction
        df['target_sig_up'] = (df['target_return_1d'] > self.threshold).astype(int)
        df['target_sig_down'] = (df['target_return_1d'] < -self.threshold).astype(int)
        
        # Statistics
        total = len(df)
        no_move = (df['target_significant'] == 0).sum()
        sig_up = (df['target_significant'] == 1).sum()
        sig_down = (df['target_significant'] == 2).sum()
        
        print(f"\n📊 Target Distribution:")
        print(f"   No significant move: {no_move:5d} ({no_move/total*100:5.1f}%)")
        print(f"   Significant UP:      {sig_up:5d} ({sig_up/total*100:5.1f}%)")
        print(f"   Significant DOWN:    {sig_down:5d} ({sig_down/total*100:5.1f}%)")
        
        # Filter to only significant move days for training
        df_filtered = df[df['target_significant'] != 0].copy()
        
        # Convert to binary: UP=1, DOWN=0
        df_filtered['target_binary'] = (df_filtered['target_significant'] == 1).astype(int)
        
        print(f"\n✅ Filtered to {len(df_filtered)} significant move days ({len(df_filtered)/total*100:.1f}%)")
        print(f"   This removes {total - len(df_filtered)} noisy days")
        
        return df_filtered
    
    def prepare_data(self, df, target_col='target_binary'):
        """Prepare filtered data"""
        df = df.dropna(subset=[target_col])
        
        exclude_cols = [
            'Date', 'Ticker',
            'target_price_1d', 'target_price_5d',
            'target_return_1d', 'target_return_5d',
            'target_direction_1d', 'target_direction_5d',
            'target_significant', 'target_sig_up', 'target_sig_down',
            'target_binary'
        ]
        
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_cols].values
        y = df[target_col].values
        
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        if self.verbose:
            print(f"\n✅ Features: {len(self.feature_cols)}")
            print(f"✅ Samples: {len(X)}")
            print(f"✅ Target distribution: UP={(y==1).sum()}, DOWN={(y==0).sum()}")
        
        return X, y
    
    def train_models(self, df):
        """
        Train ensemble on significant moves only
        """
        print(f"\n{'='*70}")
        print(f"TRAINING SIGNIFICANT MOVE PREDICTOR FOR {self.ticker}")
        print(f"{'='*70}")
        
        # Create targets and filter
        df_filtered = self.create_significant_targets(df)
        
        # Prepare data
        X, y = self.prepare_data(df_filtered)
        
        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n✅ Train samples: {len(X_train)}")
        print(f"✅ Test samples: {len(X_test)}")
        
        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train LSTM
        print(f"\n{'='*70}")
        print("TRAINING LSTM ON SIGNIFICANT MOVES")
        print(f"{'='*70}")
        lstm_acc = self._train_lstm(X_train_scaled, y_train, X_test_scaled, y_test)
        results['lstm'] = lstm_acc
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            print(f"\n{'='*70}")
            print("TRAINING XGBOOST ON SIGNIFICANT MOVES")
            print(f"{'='*70}")
            xgb_acc = self._train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
            results['xgboost'] = xgb_acc
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            print(f"\n{'='*70}")
            print("TRAINING LIGHTGBM ON SIGNIFICANT MOVES")
            print(f"{'='*70}")
            lgb_acc = self._train_lightgbm(X_train_scaled, y_train, X_test_scaled, y_test)
            results['lightgbm'] = lgb_acc
        
        # Ensemble
        print(f"\n{'='*70}")
        print("ENSEMBLE RESULTS ON SIGNIFICANT MOVES")
        print(f"{'='*70}")
        ensemble_acc = self._evaluate_ensemble(X_test_scaled, y_test)
        results['ensemble'] = ensemble_acc
        
        print(f"\n📊 Results on Significant Moves (>{self.threshold*100}%):")
        for model, acc in results.items():
            print(f"   {model:12s}: {acc*100:.2f}%")
        
        print(f"\n💡 Note: This model only predicts on days with >2% moves")
        print(f"   For other days, use 'HOLD' strategy")
        
        return results
    
    def _train_lstm(self, X_train, y_train, X_test, y_test):
        """Train LSTM"""
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X[i-seq_length:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQUENCE_LENGTH)
        
        self.lstm_model = Sequential([
            LSTM(128, return_sequences=True, 
                 input_shape=(SEQUENCE_LENGTH, X_train.shape[1]),
                 dropout=0.3),
            BatchNormalization(),
            LSTM(64, return_sequences=False, dropout=0.3),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        self.lstm_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        _, test_acc, test_auc = self.lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
        print(f"\n✅ LSTM Accuracy: {test_acc*100:.2f}%")
        print(f"✅ LSTM AUC: {test_auc:.4f}")
        
        return test_acc
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost"""
        params = XGBOOST_PARAMS.copy()
        params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        self.xgb_model = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=[(dtest, 'test')],
            early_stopping_rounds=30,
            verbose_eval=50
        )
        
        y_pred = (self.xgb_model.predict(dtest) > 0.5).astype(int)
        test_acc = (y_pred == y_test).mean()
        
        print(f"\n✅ XGBoost Accuracy: {test_acc*100:.2f}%")
        return test_acc
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM"""
        params = LIGHTGBM_PARAMS.copy()
        params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        self.lgb_model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
        )
        
        y_pred = (self.lgb_model.predict(X_test) > 0.5).astype(int)
        test_acc = (y_pred == y_test).mean()
        
        print(f"\n✅ LightGBM Accuracy: {test_acc*100:.2f}%")
        return test_acc
    
    def _evaluate_ensemble(self, X_test, y_test):
        """Ensemble predictions"""
        predictions = []
        
        if self.lstm_model is not None:
            def create_sequences(X, y, seq_length):
                X_seq, y_seq = [], []
                for i in range(seq_length, len(X)):
                    X_seq.append(X[i-seq_length:i])
                    y_seq.append(y[i])
                return np.array(X_seq), np.array(y_seq)
            
            X_test_seq, y_test = create_sequences(X_test, y_test, SEQUENCE_LENGTH)
            lstm_pred = self.lstm_model.predict(X_test_seq, verbose=0).flatten()
            predictions.append(lstm_pred * 0.4)
        
        if self.xgb_model is not None:
            dtest = xgb.DMatrix(X_test[-len(y_test):])
            xgb_pred = self.xgb_model.predict(dtest)
            predictions.append(xgb_pred * 0.3)
        
        if self.lgb_model is not None:
            lgb_pred = self.lgb_model.predict(X_test[-len(y_test):])
            predictions.append(lgb_pred * 0.3)
        
        ensemble_pred = np.sum(predictions, axis=0)
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        ensemble_acc = (ensemble_pred_binary == y_test).mean()
        print(f"\n🎯 ENSEMBLE Accuracy: {ensemble_acc*100:.2f}%")
        
        return ensemble_acc
    
    def save_models(self):
        """Save models"""
        model_dir = Path(MODEL_DIR)
        
        with open(model_dir / f"{self.ticker}_significant_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        with open(model_dir / f"{self.ticker}_significant_features.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)
        
        if self.lstm_model:
            self.lstm_model.save(model_dir / f"{self.ticker}_significant_lstm.h5")
        
        print(f"\n✅ Significant move models saved!")


if __name__ == "__main__":
    import pandas as pd
    
    print("Training significant move predictor for AAPL...")
    df = pd.read_csv('data/processed/AAPL_optimized.csv')
    
    predictor = SignificantMovePredictor('AAPL', threshold=0.02, verbose=True)
    results = predictor.train_models(df)
    predictor.save_models()
    
    print("\n🎉 Training complete!")