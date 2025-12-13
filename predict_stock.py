import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from keras.models import load_model

from config import SEQUENCE_LENGTH


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def confidence_label(return_pct: float) -> str:
    if abs(return_pct) >= 2.0:
        return "HIGH"
    elif abs(return_pct) >= 0.8:
        return "MEDIUM"
    return "LOW"


def investment_decision(return_pct: float) -> str:
    if return_pct >= 0.5:
        return "✅ BUY"
    elif return_pct <= -0.5:
        return "❌ SELL"
    return "⚠️ HOLD"


# --------------------------------------------------
# Main prediction logic
# --------------------------------------------------

def predict_stock(ticker: str, days_ahead: int, today_price: float):
    BASE_DIR = Path(__file__).resolve().parent

    model_dir = BASE_DIR / "models"
    data_dir = BASE_DIR / "data" / "processed"

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    lstm_reg = load_model(
        model_dir / f"{ticker}_lstm_regressor.h5",
        compile=False
    )

    lstm_clf = load_model(
        model_dir / f"{ticker}_lstm_classifier.h5",
        compile=False
    )

    # Load scaler
    with open(model_dir / f"{ticker}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load EXACT feature columns (59)
    with open(model_dir / f"{ticker}_feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # --------------------------------------------------
    # Load engineered data
    # --------------------------------------------------
    df = pd.read_csv(data_dir / f"{ticker}_engineered.csv")
    df = df.dropna().reset_index(drop=True)

    # Scale features
    X_scaled = scaler.transform(df[feature_cols])  # (N, 59)

    # --------------------------------------------------
    # 🔑 ADD target_price feature (60th feature)
    # --------------------------------------------------
    last_target = np.full((X_scaled.shape[0], 1), today_price)
    X_full = np.hstack([X_scaled, last_target])  # (N, 60)

    # Build last sequence (1, 30, 60)
    seq = X_full[-SEQUENCE_LENGTH:].reshape(
        1, SEQUENCE_LENGTH, X_full.shape[1]
    )

    # --------------------------------------------------
    # Predict future price (rolling)
    # --------------------------------------------------
    predicted_price = today_price

    for _ in range(days_ahead):
        next_price = lstm_reg.predict(seq, verbose=0)[0][0]

        # Shift sequence
        seq = np.roll(seq, shift=-1, axis=1)

        # Keep features stable
        seq[0, -1, :-1] = seq[0, -2, :-1]

        # Update target_price feature
        seq[0, -1, -1] = next_price

        predicted_price = next_price

    # --------------------------------------------------
    # Direction probability
    # --------------------------------------------------
    prob_up = lstm_clf.predict(seq, verbose=0)[0][0]

    expected_return = ((predicted_price - today_price) / today_price) * 100

    decision = investment_decision(expected_return)
    confidence = confidence_label(expected_return)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STOCK: {ticker}")
    print("=" * 60)
    print(f"Today's Price:                 ${today_price:.2f}")
    print(f"Predicted Price ({days_ahead} day(s)): ${predicted_price:.2f}")
    print(f"Expected Return:               {expected_return:+.2f}%")
    print()
    print(f"MODEL DECISION:                {decision}")
    print(f"CONFIDENCE LEVEL:              {confidence}")
    print(f"UP Probability:                {prob_up:.2%}")
    print("=" * 60 + "\n")


# --------------------------------------------------
# CLI entry point
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock price prediction & investment decision tool"
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, META)"
    )

    parser.add_argument(
        "--price",
        type=float,
        required=True,
        help="Today's stock price"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Days ahead to predict (default: 1)"
    )

    args = parser.parse_args()

    predict_stock(
        ticker=args.ticker.upper(),
        days_ahead=args.days,
        today_price=args.price
    )
