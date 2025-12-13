# src/model_builder.py - Build and train ML/DL models

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Flatten, Conv1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from config import (
    LSTM_UNITS_1, LSTM_UNITS_2, LSTM_DROPOUT, LSTM_ACTIVATION,
    DENSE_UNITS, DENSE_ACTIVATION, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, SEQUENCE_LENGTH, XGBOOST_PARAMS, RANDOM_SEED,
    VERBOSE
)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class LSTMRegressor:
    """LSTM model for price prediction (regression)"""
    
    def __init__(self, input_shape, units=[LSTM_UNITS_1, LSTM_UNITS_2], 
                 dropout=LSTM_DROPOUT, verbose=VERBOSE):
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.verbose = verbose
        self.model = None
        self.history = None
    
    def build(self):
        """Build LSTM regression model"""
        model = Sequential([
            LSTM(self.units[0], activation=LSTM_ACTIVATION, 
                 input_shape=self.input_shape, return_sequences=True),
            Dropout(self.dropout),
            
            LSTM(self.units[1], activation=LSTM_ACTIVATION, 
                 return_sequences=False),
            Dropout(self.dropout),
            
            Dense(DENSE_UNITS, activation=DENSE_ACTIVATION),
            Dropout(self.dropout),
            
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        if self.verbose:
            print("✓ LSTM Regressor built")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS):
        """Train LSTM model"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                             min_lr=1e-6, verbose=1)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        if self.verbose:
            print(f"✓ Training complete")
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def get_model(self):
        """Get Keras model"""
        return self.model

class LSTMClassifier:
    """LSTM model for direction prediction (classification)"""
    
    def __init__(self, input_shape, units=[LSTM_UNITS_1, LSTM_UNITS_2],
                 dropout=LSTM_DROPOUT, verbose=VERBOSE):
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.verbose = verbose
        self.model = None
        self.history = None
    
    def build(self):
        """Build LSTM classification model"""
        model = Sequential([
            LSTM(self.units[0], activation=LSTM_ACTIVATION,
                 input_shape=self.input_shape, return_sequences=True),
            Dropout(self.dropout),
            
            LSTM(self.units[1], activation=LSTM_ACTIVATION,
                 return_sequences=False),
            Dropout(self.dropout),
            
            Dense(DENSE_UNITS, activation=DENSE_ACTIVATION),
            Dropout(self.dropout),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        if self.verbose:
            print("✓ LSTM Classifier built")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS):
        """Train LSTM classifier"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                             min_lr=1e-6, verbose=1)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        if self.verbose:
            print(f"✓ Training complete")
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def get_model(self):
        """Get Keras model"""
        return self.model

class TransformerRegressor:
    """Transformer model for stock price prediction"""
    
    def __init__(self, input_shape, num_heads=4, ff_dim=128, verbose=VERBOSE):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.verbose = verbose
        self.model = None
        self.history = None
    
    def build(self):
        """Build Transformer model"""
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.input_shape[-1]
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # Feed-forward network
        x = Dense(self.ff_dim, activation='relu')(x)
        x = Dense(self.input_shape[-1])(x)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        if self.verbose:
            print("✓ Transformer Regressor built")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS):
        """Train Transformer model"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                             min_lr=1e-6, verbose=1)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        if self.verbose:
            print(f"✓ Training complete")
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def get_model(self):
        """Get Keras model"""
        return self.model

class CNNLSTMRegressor:
    """Hybrid CNN-LSTM model for time-series prediction"""
    
    def __init__(self, input_shape, verbose=VERBOSE):
        self.input_shape = input_shape
        self.verbose = verbose
        self.model = None
        self.history = None
    
    def build(self):
        """Build CNN-LSTM model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu',
                  input_shape=self.input_shape, padding='same'),
            Dropout(0.2),
            
            LSTM(LSTM_UNITS_1, activation=LSTM_ACTIVATION, return_sequences=True),
            Dropout(LSTM_DROPOUT),
            
            LSTM(LSTM_UNITS_2, activation=LSTM_ACTIVATION, return_sequences=False),
            Dropout(LSTM_DROPOUT),
            
            Dense(DENSE_UNITS, activation=DENSE_ACTIVATION),
            Dropout(0.2),
            
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        if self.verbose:
            print("✓ CNN-LSTM Regressor built")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS):
        """Train CNN-LSTM model"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                         restore_best_weights=True, verbose=1)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def get_model(self):
        """Get Keras model"""
        return self.model

class XGBoostModel:
    """XGBoost for baseline comparison"""
    
    def __init__(self, params=XGBOOST_PARAMS, verbose=VERBOSE):
        self.params = params
        self.verbose = verbose
        self.model = None
    
    def build_regressor(self):
        """Build XGBoost regressor"""
        self.model = xgb.XGBRegressor(**self.params)
        
        if self.verbose:
            print("✓ XGBoost Regressor built")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        if self.verbose:
            print("✓ XGBoost training complete")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_model(self):
        """Get XGBoost model"""
        return self.model

class RandomForestModel:
    """Random Forest for baseline"""
    
    def __init__(self, n_estimators=100, max_depth=10, verbose=VERBOSE):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.verbose = verbose
        self.model = None
    
    def build(self):
        """Build Random Forest model"""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
        if self.verbose:
            print("✓ Random Forest built")
        
        return self.model
    
    def train(self, X_train, y_train):
        """Train Random Forest"""
        self.model.fit(X_train, y_train)
        
        if self.verbose:
            print("✓ Random Forest training complete")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_model(self):
        """Get model"""
        return self.model

def create_sequences(data, lookback=SEQUENCE_LENGTH):
    """Create sequences for LSTM/Transformer"""
    X, y = [], []
    
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback, -1])
    
    return np.array(X), np.array(y)

def train_test_split_timeseries(X, y, train_size=0.7, val_size=0.15):
    """Chronological train-test split"""
    total = len(X)
    train_idx = int(total * train_size)
    val_idx = int(total * (train_size + val_size))
    
    return (
        (X[:train_idx], y[:train_idx]),
        (X[train_idx:val_idx], y[train_idx:val_idx]),
        (X[val_idx:], y[val_idx:])
    )

if __name__ == "__main__":
    print("Model Builder Module")
