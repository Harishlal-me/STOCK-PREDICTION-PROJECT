# predict_stock.py
"""
CLI entry point for stock prediction
"""

import argparse
from src.predictor import StockPredictor


def main():
    parser = argparse.ArgumentParser(description="Stock Price Predictor")

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., AAPL)"
    )

    parser.add_argument(
        "--price",
        type=float,
        required=True,
        help="Today's stock price in USD"
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()
    today_price = args.price

    # -----------------------------
    # Run prediction
    # -----------------------------
    predictor = StockPredictor(ticker)
    result = predictor.predict(today_price)

    # -----------------------------
    # Display output
    # -----------------------------
    print("\n" + "=" * 50)
    print(f"STOCK: {result['ticker']}")
    print("=" * 50)

    print("\nTODAY:")
    print(f"  Price (USD): ${result['today_usd']:.2f}")
    print(f"  Price (INR): ₹{result['today_inr']:.0f}")

    print("\nPREDICTIONS:")
    print("Tomorrow Price:")
    print(f"  USD: ${result['tomorrow_usd']:.2f}")
    print(f"  INR: ₹{result['tomorrow_inr']:.0f}")

    print("\n5-Day Price:")
    print(f"  USD: ${result['five_day_usd']:.2f}")
    print(f"  INR: ₹{result['five_day_inr']:.0f}")

    print("\nDIRECTION (5-Day):")
    print(f"  Direction: {result['direction']}")
    print(f"  Confidence: {result['confidence']:.1f}%")

    print("\nMODEL DECISION:")
    print(f"  Short-term (1D): {result['decision_1d']}")
    print(f"  Swing (5D): {result['decision_5d']}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
