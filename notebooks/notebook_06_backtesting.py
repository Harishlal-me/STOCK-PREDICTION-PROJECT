# 06_backtesting.py
# Backtesting and Trading Strategy Evaluation

# ============================================================
# Cell 1: Imports and Setup
# ============================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import pickle
import warnings

from src.model_builder import create_sequences, train_test_split_timeseries
from src.backtest import EvaluationReporter
from src.predictor import TradingSignalGenerator

warnings.filterwarnings("ignore")

print("✓ Imports successful")

# ============================================================
# Cell 2: Load Models and Data
# ============================================================

print("=" * 70)
print("LOADING MODELS AND TEST DATA")
print("=" * 70)

ticker = "AAPL"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

lstm_reg = load_model(MODEL_DIR / f"{ticker}_lstm_regressor.h5", compile=False)
lstm_clf = load_model(MODEL_DIR / f"{ticker}_lstm_classifier.h5", compile=False)

with open(MODEL_DIR / f"{ticker}_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

engineered_df = pd.read_csv(DATA_DIR / f"{ticker}_engineered.csv")
engineered_df["Date"] = pd.to_datetime(engineered_df["Date"])

print("✓ Models and scaler loaded")

# ============================================================
# Cell 3: Prepare Test Data
# ============================================================

feature_cols = [
    c for c in engineered_df.columns
    if c not in ["Date", "Ticker", "target_price", "target_direction", "target_return"]
]

df_clean = engineered_df.dropna()
X_scaled = scaler.transform(df_clean[feature_cols])

SEQUENCE_LENGTH = 30

data_for_seq = np.hstack([
    X_scaled,
    df_clean[["target_price"]].values
])

X, y_price = create_sequences(data_for_seq, lookback=SEQUENCE_LENGTH)

(X_train, y_train), (_, _), (X_test, y_test) = train_test_split_timeseries(
    X, y_price, train_size=0.7, val_size=0.15
)

test_dates = df_clean["Date"].iloc[-len(y_test):].values
test_prices = df_clean["Close"].iloc[-len(y_test):].values

print(f"✓ Test samples: {len(X_test)}")
print(f"  Date range: {test_dates[0]} → {test_dates[-1]}")

# ============================================================
# Cell 4: Make Predictions
# ============================================================

print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

y_pred_price = lstm_reg.predict(X_test, verbose=0).flatten()
y_pred_prob = lstm_clf.predict(X_test, verbose=0).flatten()

print("✓ Predictions completed")

# ============================================================
# Cell 5: Generate Trading Signals
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TRADING SIGNALS")
print("=" * 70)

signal_gen = TradingSignalGenerator(confidence_threshold=0.65)

signals = []
confidences = []

for i in range(len(test_prices)):
    current_price = test_prices[i]
    predicted_price = y_pred_price[i]
    confidence = abs(predicted_price - current_price) / current_price
    direction = "UP" if y_pred_prob[i] > 0.5 else "DOWN"

    signal = signal_gen.generate_signal(
        predicted_price, current_price, confidence, direction
    )

    signals.append(signal)
    confidences.append(min(confidence, 1.0))

signals = np.array(signals)
confidences = np.array(confidences)

print(f"BUY: {(signals=='BUY').sum()} | SELL: {(signals=='SELL').sum()} | HOLD: {(signals=='HOLD').sum()}")

# ============================================================
# Cell 6: Run Backtest
# ============================================================

print("\n" + "=" * 70)
print("RUNNING BACKTEST")
print("=" * 70)

initial_capital = 10_000
capital = initial_capital
portfolio_values = [capital]

position = None   # will store dict when in position
trades = []

for i in range(len(test_prices) - 1):
    price_today = test_prices[i]
    price_next = test_prices[i + 1]
    signal = signals[i]

    # ---- BUY ----
    if signal == "BUY" and position is None:
        position = {
            "entry_price": price_today,
            "date": test_dates[i]
        }

        trades.append({
            "action": "BUY",
            "price": price_today,
            "date": test_dates[i]
        })

    # ---- SELL ----
    elif signal == "SELL" and position is not None:
        exit_price = price_today
        profit = exit_price - position["entry_price"]
        profit_pct = (profit / position["entry_price"]) * 100
        capital += profit

        trades.append({
            "action": "SELL",
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "profit": profit,
            "profit_pct": profit_pct,
            "date": test_dates[i]
        })

        position = None

    # ---- Portfolio value update ----
    if position is not None:
        unrealized = price_next - position["entry_price"]
        portfolio_values.append(capital + unrealized)
    else:
        portfolio_values.append(capital)

# ============================================================
# Cell 7: Backtest Metrics
# ============================================================

portfolio_values = np.array(portfolio_values)
returns = np.diff(portfolio_values) / portfolio_values[:-1]

total_return = (capital - initial_capital) / initial_capital
annual_return = np.mean(returns) * 252
volatility = np.std(returns) * np.sqrt(252)
sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0

cummax = np.maximum.accumulate(portfolio_values)
drawdown = (portfolio_values - cummax) / cummax
max_drawdown = drawdown.min()

print("\nBACKTEST METRICS")
print(f"Total Return : {total_return:+.2%}")
print(f"Sharpe Ratio : {sharpe_ratio:.2f}")
print(f"Max Drawdown : {max_drawdown:.2%}")

# ============================================================
# Cell 8: Portfolio Plot
# ============================================================

plt.figure(figsize=(14, 5))
plt.plot(portfolio_values, label="Portfolio Value")
plt.axhline(initial_capital, linestyle="--", color="red", alpha=0.5)
plt.title("Portfolio Value Over Time")
plt.ylabel("Capital ($)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "06_portfolio_performance.png", dpi=100)
plt.show()

print("✓ Saved portfolio performance plot")
# ===== Calculate trade statistics =====

completed_trades = [t for t in trades if t['action'] == 'SELL']

num_trades = len(completed_trades)

if num_trades > 0:
    wins = [t for t in completed_trades if t['profit_pct'] > 0]
    losses = [t for t in completed_trades if t['profit_pct'] < 0]

    win_rate = len(wins) / num_trades
    avg_win = np.mean([t['profit_pct'] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t['profit_pct'] for t in losses]) if losses else 0.0

    profit_factor = (
        abs(sum(t['profit'] for t in wins) /
            sum(t['profit'] for t in losses))
        if losses else np.inf
    )
else:
    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0

# ===== Portfolio metrics =====

portfolio_array = np.array(portfolio_values)
returns = np.diff(portfolio_array) / portfolio_array[:-1]

volatility = np.std(returns) * np.sqrt(252)
sharpe = (
    (np.mean(returns) * 252 - 0.02) /
    (np.std(returns) * np.sqrt(252))
    if np.std(returns) > 0 else 0.0
)

cummax = np.maximum.accumulate(portfolio_array)
drawdown = (portfolio_array - cummax) / cummax
max_drawdown = np.min(drawdown)

# ============================================================
# Cell 9: Report
# ============================================================

reporter = EvaluationReporter()
# ---- Calmar Ratio (SAFE) ----
if max_drawdown != 0:
    calmar_ratio = annual_return / abs(max_drawdown)
else:
    calmar_ratio = 0.0

trading_metrics = {
    'total_return': total_return,
    'annual_return': annual_return,
    'num_trades': len(completed_trades),
    'win_rate': win_rate,
    'avg_win': avg_win,
    'avg_loss': avg_loss,
    'profit_factor': profit_factor,
    'volatility': volatility,
    'sharpe_ratio': sharpe,
    'max_drawdown': max_drawdown,
    'calmar_ratio': calmar_ratio   # ✅ FIX
}


reporter.print_backtest_report(trading_metrics, trades[:10])
reporter.save_evaluation_report({"backtest": trading_metrics}, ticker)

print("\n✓ BACKTESTING COMPLETE")
print("✓ ALL NOTEBOOKS FINISHED SUCCESSFULLY 🎉")
