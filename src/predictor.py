# src/predictor.py - Prediction and Trading Signal Module

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
from config import (
    CONFIDENCE_THRESHOLD, RISK_REWARD_RATIO, VOLATILITY_MULTIPLIER,
    SEQUENCE_LENGTH, MODELS_DIR, VERBOSE
)

class StockPredictor:
    """Main predictor class for making stock price predictions"""
    
    def __init__(self, lstm_model=None, classifier_model=None, 
                 scaler=None, lookback=SEQUENCE_LENGTH, verbose=VERBOSE):
        self.lstm_model = lstm_model
        self.classifier_model = classifier_model
        self.scaler = scaler
        self.lookback = lookback
        self.verbose = verbose
    
    def predict_price(self, X):
        """Predict next-day price"""
        if self.lstm_model is None:
            raise ValueError("LSTM model not loaded")
        
        prediction = self.lstm_model.predict(X, verbose=0)[0][0]
        return float(prediction)
    
    def predict_direction(self, X):
        """Predict direction (UP/DOWN)"""
        if self.classifier_model is None:
            raise ValueError("Classifier model not loaded")
        
        prob = self.classifier_model.predict(X, verbose=0)[0][0]
        direction = "UP" if prob > 0.5 else "DOWN"
        confidence = prob if direction == "UP" else 1 - prob
        
        return direction, float(confidence)
    
    def get_confidence_score(self, predicted_price, current_price):
        """Calculate prediction confidence score"""
        price_change_pct = abs((predicted_price - current_price) / current_price)
        confidence = min(price_change_pct, 1.0)
        return float(confidence)

class TradingSignalGenerator:
    """Generate trading signals (BUY/SELL/HOLD)"""
    
    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD, verbose=VERBOSE):
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
    
    def generate_signal(self, predicted_price, current_price, 
                       confidence, direction=None):
        """Generate BUY/SELL/HOLD signal"""
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Strong buy signal
        if (direction == "UP" and confidence > self.confidence_threshold and 
            price_change_pct > 0.01):
            signal = "BUY"
        
        # Strong sell signal
        elif (direction == "DOWN" and confidence > self.confidence_threshold and 
              price_change_pct < -0.01):
            signal = "SELL"
        
        # Hold for uncertain cases
        else:
            signal = "HOLD"
        
        return signal
    
    def generate_trade_levels(self, entry_price, predicted_price, 
                             current_volatility):
        """Generate entry, target, and stop-loss levels"""
        target_price = predicted_price
        
        # Stop loss: 2x volatility below entry
        stop_loss = entry_price * (1 - VOLATILITY_MULTIPLIER * current_volatility)
        
        # Calculate risk-reward
        if entry_price != stop_loss:
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0
        
        return {
            'entry_price': float(entry_price),
            'target_price': float(target_price),
            'stop_loss': float(max(stop_loss, entry_price * 0.95)),
            'risk_reward_ratio': float(risk_reward)
        }

class RiskManager:
    """Manage risk for trading positions"""
    
    def __init__(self, initial_capital=10000, max_risk_per_trade=0.02, 
                 max_position_size=0.3, verbose=VERBOSE):
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.verbose = verbose
        self.current_capital = initial_capital
    
    def calculate_position_size(self, entry_price, stop_loss, 
                               current_capital=None):
        """Calculate position size based on risk"""
        if current_capital is None:
            current_capital = self.current_capital
        
        risk_per_share = entry_price - stop_loss
        max_loss = current_capital * self.max_risk_per_trade
        
        position_shares = max_loss / risk_per_share if risk_per_share > 0 else 0
        position_value = position_shares * entry_price
        position_pct = position_value / current_capital
        
        # Cap at max position size
        if position_pct > self.max_position_size:
            position_shares = (current_capital * self.max_position_size) / entry_price
            position_value = position_shares * entry_price
            position_pct = self.max_position_size
        
        return {
            'position_shares': float(position_shares),
            'position_value': float(position_value),
            'position_pct': float(position_pct),
            'risk_amount': float(position_shares * risk_per_share)
        }
    
    def update_capital(self, trade_profit):
        """Update capital after a trade"""
        self.current_capital += trade_profit
        return self.current_capital

class PredictionPipeline:
    """Complete prediction pipeline"""
    
    def __init__(self, lstm_model, classifier_model, scaler, 
                 ticker, verbose=VERBOSE):
        self.predictor = StockPredictor(
            lstm_model=lstm_model,
            classifier_model=classifier_model,
            scaler=scaler,
            verbose=verbose
        )
        self.signal_gen = TradingSignalGenerator(verbose=verbose)
        self.risk_mgr = RiskManager(verbose=verbose)
        self.ticker = ticker
        self.verbose = verbose
    
    def predict(self, features_sequence, current_price, current_volatility,
               xgb_model=None):
        """Complete prediction and signal generation"""
        X_seq = features_sequence
        
        # Price prediction
        predicted_price = self.predictor.predict_price(X_seq)
        
        # Direction prediction
        direction, direction_confidence = self.predictor.predict_direction(X_seq)
        
        # Confidence score
        confidence = self.predictor.get_confidence_score(predicted_price, current_price)
        
        # Generate trading signal
        signal = self.signal_gen.generate_signal(
            predicted_price, current_price, confidence, direction
        )
        
        # Generate trade levels
        trade_levels = self.signal_gen.generate_trade_levels(
            current_price, predicted_price, current_volatility
        )
        
        # Position sizing
        position_info = self.risk_mgr.calculate_position_size(
            trade_levels['entry_price'],
            trade_levels['stop_loss']
        )
        
        output = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change_pct': float((predicted_price - current_price) / current_price * 100),
            'direction': direction,
            'direction_confidence': float(direction_confidence),
            'confidence': float(confidence),
            'signal': signal,
            'trade_levels': trade_levels,
            'position_sizing': position_info
        }
        
        return output
    
    def save_prediction(self, prediction, output_dir='results'):
        """Save prediction to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f'{self.ticker}_predictions.csv')
        
        # Flatten nested dicts for CSV
        flat_pred = {
            'ticker': prediction['ticker'],
            'timestamp': prediction['timestamp'],
            'current_price': prediction['current_price'],
            'predicted_price': prediction['predicted_price'],
            'price_change_pct': prediction['price_change_pct'],
            'direction': prediction['direction'],
            'direction_confidence': prediction['direction_confidence'],
            'confidence': prediction['confidence'],
            'signal': prediction['signal'],
            'entry_price': prediction['trade_levels']['entry_price'],
            'target_price': prediction['trade_levels']['target_price'],
            'stop_loss': prediction['trade_levels']['stop_loss'],
            'risk_reward_ratio': prediction['trade_levels']['risk_reward_ratio'],
            'position_shares': prediction['position_sizing']['position_shares'],
            'position_value': prediction['position_sizing']['position_value']
        }
        
        # Append to CSV
        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            df_new = pd.DataFrame([flat_pred])
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = pd.DataFrame([flat_pred])
        
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"✓ Prediction saved to {filepath}")
        
        return filepath

def format_prediction_output(prediction):
    """Format prediction for display"""
    output = f"""
╔════════════════════════════════════════════════════════════════╗
║         STOCK PRICE PREDICTION - {prediction['ticker']}
╚════════════════════════════════════════════════════════════════╝

📊 PRICE FORECAST
  Current Price:        ${prediction['current_price']:.2f}
  Predicted Price:      ${prediction['predicted_price']:.2f}
  Expected Change:      {prediction['price_change_pct']:+.2f}%

🎯 DIRECTION & CONFIDENCE
  Direction:            {prediction['direction']}
  Direction Confidence: {prediction['direction_confidence']:.1%}
  Model Confidence:     {prediction['confidence']:.1%}

💹 TRADING SIGNAL
  Signal:               {prediction['signal']}
  
📋 TRADE LEVELS
  Entry Price:          ${prediction['trade_levels']['entry_price']:.2f}
  Target Price:         ${prediction['trade_levels']['target_price']:.2f}
  Stop Loss:            ${prediction['trade_levels']['stop_loss']:.2f}
  Risk/Reward Ratio:    {prediction['trade_levels']['risk_reward_ratio']:.2f}

📈 POSITION SIZING
  Position Size:        {prediction['position_sizing']['position_pct']:.1%} of capital
  Shares:               {prediction['position_sizing']['position_shares']:.0f}
  Position Value:       ${prediction['position_sizing']['position_value']:.2f}
  Risk Amount:          ${prediction['position_sizing']['risk_amount']:.2f}

⏰ Timestamp: {prediction['timestamp']}
════════════════════════════════════════════════════════════════
"""
    return output

if __name__ == "__main__":
    print("Stock Price Predictor Module")
