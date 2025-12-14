# src/regime_based_model.py
# Train separate models for different market regimes (bull vs bear vs sideways)

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

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from config import *


class RegimeBasedPredictor:
    """
    Train separate models for different market regimes
    Theory: Models perform better when trained on similar market conditions
    """
    
    def __init__(self, ticker, verbose=True):
        self.ticker = ticker
        self.verbose = verbose
        
        # Models for each regime
        self.bull_models = {'lstm': None, 'xgb': None, 'lgb': None}
        self.bear_models = {'lstm': None, 'xgb': None, 'lgb': None}
        self.sideways_models = {'lstm': None, 'xgb': None, 'lgb': None}
        
        # Scalers for each regime
        self.bull_scaler = None
        self.bear_scaler = None
        self.sideways_scaler = None
        
        self.feature_cols = None
    
    def detect_market_regime(self, df):
        """
        Classify market into Bull/Bear/Sideways
        
        Bull: Price > SMA200 and SMA50 > SMA200 (uptrend)
        Bear: Price < SMA200 and SMA50 < SMA200 (downtrend)
        Sideways: Everything else (choppy/ranging)
        """
        print(f"\n{'='*70}")
        print("DETECTING MARKET REGIMES")
        print(f"{'='*70}")
        
        # Ensure we have required indicators
        if 'sma_50' not in df.columns or 'sma_200' not in df.columns:
            print("❌ Missing SMA indicators. Adding them...")
            df['sma_50'] = df['Close'].rolling(50).mean()
            df['sma_200'] = df['Close'].rolling(200).mean()
        
        # Define regimes
        df['regime'] = 'sideways'  # Default
        
        # Bull market: price above SMA200 AND SMA50 above SMA200
        bull_mask = (df['Close'] > df['sma_200']) & (df['sma_50'] > df['sma_200'])
        df.loc[bull_mask, 'regime'] = 'bull'
        
        # Bear market: price below SMA200 AND SMA50 below SMA200
        bear_mask = (df['Close'] < df['sma_200']) & (df['sma_50'] < df['sma_200'])
        df.loc[bear_mask, 'regime'] = 'bear'
        
        # Add trend strength
        df['trend_strength'] = abs(df['sma_50'] - df['sma_200']) / df['sma_200']
        
        # Count regime distribution
        regime_counts = df['regime'].value_counts()
        total = len(df)
        
        print(f"\n📊 Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = (count / total) * 100
            print(f"   {regime.upper():10s}: {count:5d} days ({pct:5.1f}%)")
        
        return df
    
    def prepare_data(self, df, target_col='target_direction_1d'):
        """Prepare data with regime information"""
        df = df.dropna(subset=[target_col, 'regime'])
        
        exclude_cols = [
            'Date', 'Ticker', 'regime',
            'target_price_1d', 'target_price_5d',
            'target_return_1d', 'target_return_5d',
            'target_direction_1d', 'target_direction_5d'
        ]
        
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_cols].values
        y = df[target_col].values
        regimes = df['regime'].values
        
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        return X, y, regimes
    
    def train_regime_specific_models(self, df, target_col='target_direction_1d'):
        """
        Train separate ensemble models for each regime
        """
        print(f"\n{'='*70}")
        print(f"TRAINING REGIME-SPECIFIC MODELS FOR {self.ticker}")
        print(f"{'='*70}")
        
        # Detect regimes
        df = self.detect_market_regime(df)
        
        # Prepare data
        X, y, regimes = self.prepare_data(df, target_col)
        
        results = {}
        
        # Train models for each regime
        for regime_name in ['bull', 'bear', 'sideways']:
            print(f"\n{'='*70}")
            print(f"TRAINING {regime_name.upper()} MARKET MODELS")
            print(f"{'='*70}")
            
            # Filter data for this regime
            regime_mask = regimes == regime_name
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) < 500:
                print(f"⚠️  Not enough data for {regime_name} regime ({len(X_regime)} samples)")
                print(f"   Skipping...")
                continue
            
            print(f"✅ Training samples: {len(X_regime)}")
            print(f"✅ Target distribution: {np.bincount(y_regime.astype(int))}")
            
            # Time series split
            split_idx = int(len(X_regime) * 0.8)
            X_train, X_test = X_regime[:split_idx], X_regime[split_idx:]
            y_train, y_test = y_regime[:split_idx], y_regime[split_idx:]
            
            # Scale
            if regime_name == 'bull':
                self.bull_scaler = StandardScaler()
                X_train_scaled = self.bull_scaler.fit_transform(X_train)
                X_test_scaled = self.bull_scaler.transform(X_test)
            elif regime_name == 'bear':
                self.bear_scaler = StandardScaler()
                X_train_scaled = self.bear_scaler.fit_transform(X_train)
                X_test_scaled = self.bear_scaler.transform(X_test)
            else:
                self.sideways_scaler = StandardScaler()
                X_train_scaled = self.sideways_scaler.fit_transform(X_train)
                X_test_scaled = self.sideways_scaler.transform(X_test)
            
            # Train LSTM
            print(f"\n🔷 Training LSTM for {regime_name} market...")
            lstm_acc = self._train_regime_lstm(
                X_train_scaled, y_train, X_test_scaled, y_test, regime_name
            )
            
            # Train XGBoost
            if XGBOOST_AVAILABLE:
                print(f"\n🔷 Training XGBoost for {regime_name} market...")
                xgb_acc = self._train_regime_xgboost(
                    X_train_scaled, y_train, X_test_scaled, y_test, regime_name
                )
            else:
                xgb_acc = None
            
            # Train LightGBM
            if LIGHTGBM_AVAILABLE:
                print(f"\n🔷 Training LightGBM for {regime_name} market...")
                lgb_acc = self._train_regime_lightgbm(
                    X_train_scaled, y_train, X_test_scaled, y_test, regime_name
                )
            else:
                lgb_acc = None
            
            # Store results
            results[regime_name] = {
                'lstm': lstm_acc,
                'xgb': xgb_acc,
                'lgb': lgb_acc
            }
        
        # Print summary
        print(f"\n{'='*70}")
        print("REGIME-SPECIFIC RESULTS")
        print(f"{'='*70}")
        
        for regime, accuracies in results.items():
            print(f"\n{regime.upper()} MARKET:")
            for model, acc in accuracies.items():
                if acc is not None:
                    print(f"   {model:10s}: {acc*100:.2f}%")
        
        return results
    
    def _train_regime_lstm(self, X_train, y_train, X_test, y_test, regime):
        """Train LSTM for specific regime"""
        from src.ensemble_model import EnsembleStockPredictor
        
        # Create sequences
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X[i-seq_length:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQUENCE_LENGTH)
        
        # Build model
        model = Sequential([
            LSTM(64, return_sequences=True, 
                 input_shape=(SEQUENCE_LENGTH, X_train.shape[1]),
                 dropout=0.3),
            BatchNormalization(),
            LSTM(32, return_sequences=False, dropout=0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Store model
        if regime == 'bull':
            self.bull_models['lstm'] = model
        elif regime == 'bear':
            self.bear_models['lstm'] = model
        else:
            self.sideways_models['lstm'] = model
        
        # Evaluate
        _, test_acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
        print(f"   ✅ Accuracy: {test_acc*100:.2f}%")
        
        return test_acc
    
    def _train_regime_xgboost(self, X_train, y_train, X_test, y_test, regime):
        """Train XGBoost for specific regime"""
        params = XGBOOST_PARAMS.copy()
        params['n_estimators'] = 200  # Fewer trees for regime-specific
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=200,
            evals=[(dtest, 'test')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Store model
        if regime == 'bull':
            self.bull_models['xgb'] = model
        elif regime == 'bear':
            self.bear_models['xgb'] = model
        else:
            self.sideways_models['xgb'] = model
        
        # Evaluate
        y_pred = (model.predict(dtest) > 0.5).astype(int)
        test_acc = (y_pred == y_test).mean()
        print(f"   ✅ Accuracy: {test_acc*100:.2f}%")
        
        return test_acc
    
    def _train_regime_lightgbm(self, X_train, y_train, X_test, y_test, regime):
        """Train LightGBM for specific regime"""
        params = LIGHTGBM_PARAMS.copy()
        params['n_estimators'] = 200
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params, train_data,
            num_boost_round=200,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # Store model
        if regime == 'bull':
            self.bull_models['lgb'] = model
        elif regime == 'bear':
            self.bear_models['lgb'] = model
        else:
            self.sideways_models['lgb'] = model
        
        # Evaluate
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        test_acc = (y_pred == y_test).mean()
        print(f"   ✅ Accuracy: {test_acc*100:.2f}%")
        
        return test_acc
    
    def save_models(self):
        """Save all regime-specific models"""
        model_dir = Path(MODEL_DIR)
        
        # Save scalers
        for regime, scaler in [('bull', self.bull_scaler), 
                                ('bear', self.bear_scaler),
                                ('sideways', self.sideways_scaler)]:
            if scaler is not None:
                with open(model_dir / f"{self.ticker}_{regime}_scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)
        
        # Save features
        with open(model_dir / f"{self.ticker}_regime_features.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)
        
        print(f"\n✅ All regime models saved!")


if __name__ == "__main__":
    import pandas as pd
    
    print("Training regime-based models for AAPL...")
    df = pd.read_csv('data/processed/AAPL_optimized.csv')
    
    predictor = RegimeBasedPredictor('AAPL', verbose=True)
    results = predictor.train_regime_specific_models(df)
    predictor.save_models()
    
    print("\n🎉 Regime-based training complete!")