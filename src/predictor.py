# src/predictor.py
# For models trained with UNSCALED returns

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from config import SEQUENCE_LENGTH

USD_TO_INR = 83.05


class StockPredictor:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()

        base_dir = Path(__file__).resolve().parents[1]
        model_dir = base_dir / "models"
        data_dir = base_dir / "data" / "processed"

        # Load models
        self.reg_model = tf.keras.models.load_model(
            model_dir / f"{self.ticker}_lstm_regressor.h5",
            compile=False
        )

        self.clf_model = tf.keras.models.load_model(
            model_dir / f"{self.ticker}_lstm_classifier.h5",
            compile=False
        )

        # Load scaler and features
        with open(model_dir / f"{self.ticker}_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(model_dir / f"{self.ticker}_feature_cols.pkl", "rb") as f:
            self.feature_cols = pickle.load(f)

        # Load data
        self.df = pd.read_csv(
            data_dir / f"{self.ticker}_engineered.csv",
            parse_dates=["Date"]
        )

    def _build_sequence(self):
        """Build input sequence"""
        df = self.df.copy()

        # Align features
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        df = df[self.feature_cols].tail(SEQUENCE_LENGTH)

        if len(df) < SEQUENCE_LENGTH:
            raise ValueError(f"Need {SEQUENCE_LENGTH} rows")

        X_scaled = self.scaler.transform(df)
        return X_scaled.reshape(1, SEQUENCE_LENGTH, X_scaled.shape[1])

    def predict_prices(self, current_price: float):
        """Predict prices using UNSCALED returns"""
        X = self._build_sequence()

        # Model outputs UNSCALED return with tanh activation (range: -1 to +1)
        predicted_return = float(self.reg_model.predict(X, verbose=0)[0][0])
        
        print(f"\n" + "="*60)
        print(f"📊 Price Prediction")
        print("="*60)
        print(f"Current price: ${current_price:.2f}")
        print(f"Predicted return: {predicted_return*100:.2f}%")
        
        # Calculate tomorrow's price
        tomorrow = current_price * (1 + predicted_return)
        change_pct = ((tomorrow / current_price) - 1) * 100
        
        print(f"Tomorrow's price: ${tomorrow:.2f}")
        print(f"Change: ${tomorrow - current_price:+.2f} ({change_pct:+.2f}%)")
        
        # 5-day projection
        recent_returns = self.df["daily_return"].tail(30).mean()
        five_day = current_price * ((1 + recent_returns) ** 5)
        
        print(f"\n30-day avg return: {recent_returns*100:.2f}%")
        print(f"5-day projection: ${five_day:.2f}")
        print("="*60 + "\n")

        return tomorrow, float(five_day)

    def predict_direction(self):
        """Predict direction with confidence"""
        X = self._build_sequence()
        prob_up = float(self.clf_model.predict(X, verbose=0)[0][0])
        
        direction = "UP" if prob_up >= 0.5 else "DOWN"
        confidence = prob_up if direction == "UP" else (1 - prob_up)
        
        return direction, confidence * 100

    def predict(self, today_price: float):
        """Main prediction API"""
        tomorrow_usd, five_day_usd = self.predict_prices(today_price)
        direction, confidence = self.predict_direction()

        return {
            "ticker": self.ticker,
            "today_usd": today_price,
            "today_inr": today_price * USD_TO_INR,
            "tomorrow_usd": tomorrow_usd,
            "tomorrow_inr": tomorrow_usd * USD_TO_INR,
            "five_day_usd": five_day_usd,
            "five_day_inr": five_day_usd * USD_TO_INR,
            "direction": direction,
            "confidence": confidence,
            "decision_1d": "BUY" if tomorrow_usd > today_price else "SELL",
            "decision_5d": "BUY" if direction == "UP" else "SELL",
        }