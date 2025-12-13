# ============================================================
# 05_evaluation.ipynb
# Model Evaluation Notebook
# ============================================================

"""
NOTEBOOK: 05_Evaluation.ipynb
PURPOSE: Comprehensive model evaluation and metrics
"""

# ------------------------------------------------------------
# Cell 1: Imports and Setup
# ------------------------------------------------------------
import sys
from pathlib import Path
sys.path.append("../")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from keras.models import load_model
from sklearn.metrics import confusion_matrix

from src.backtest import MetricsCalculator, EvaluationReporter
from src.model_builder import create_sequences, train_test_split_timeseries

warnings.filterwarnings("ignore")

print("✓ Imports successful")

# ------------------------------------------------------------
# Cell 2: Load Models and Data
# ------------------------------------------------------------
print("=" * 70)
print("LOADING MODELS AND DATA")
print("=" * 70)

ticker = "AAPL"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# ✅ IMPORTANT FIX: compile=False
lstm_reg = load_model(
    MODEL_DIR / f"{ticker}_lstm_regressor.h5",
    compile=False
)
lstm_clf = load_model(
    MODEL_DIR / f"{ticker}_lstm_classifier.h5",
    compile=False
)

with open(MODEL_DIR / f"{ticker}_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"✓ Models and scaler loaded for {ticker}")

# ------------------------------------------------------------
# Cell 3: Load Engineered Data & Create Sequences
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("PREPARING TEST DATA")
print("=" * 70)

engineered_df = pd.read_csv(DATA_DIR / f"{ticker}_engineered.csv")

feature_cols = [
    col for col in engineered_df.columns
    if col not in [
        "Date", "Ticker",
        "target_price", "target_direction", "target_return"
    ]
]

df_clean = engineered_df.dropna().reset_index(drop=True)

X_scaled = scaler.transform(df_clean[feature_cols])

SEQUENCE_LENGTH = 30

data_for_seq = np.hstack([
    X_scaled,
    df_clean[["target_price"]].values
])

X, y_price = create_sequences(
    data_for_seq,
    lookback=SEQUENCE_LENGTH
)

y_direction = (
    df_clean["target_price"]
    .iloc[SEQUENCE_LENGTH:].values
    > df_clean["Close"]
    .iloc[SEQUENCE_LENGTH:].values
).astype(int)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = (
    train_test_split_timeseries(X, y_price)
)

y_dir_test = y_direction[len(X_train) + len(X_val):]

print(f"✓ Test samples: {len(X_test)}")

# ------------------------------------------------------------
# Cell 4: Make Predictions
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

y_pred_price = lstm_reg.predict(X_test, verbose=0).flatten()
y_pred_prob = lstm_clf.predict(X_test, verbose=0).flatten()
y_pred_direction = (y_pred_prob > 0.5).astype(int)

print("✓ Predictions completed")

# ------------------------------------------------------------
# Cell 5: Regression Metrics
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("REGRESSION METRICS")
print("=" * 70)

reg_metrics = MetricsCalculator.calculate_regression_metrics(
    y_test,
    y_pred_price
)

for k, v in reg_metrics.items():
    print(f"{k:15s}: {v:.4f}")

# ------------------------------------------------------------
# Cell 6: Classification Metrics
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("CLASSIFICATION METRICS")
print("=" * 70)

clf_metrics = MetricsCalculator.calculate_classification_metrics(
    y_dir_test,
    y_pred_direction,
    y_pred_prob
)

for k, v in clf_metrics.items():
    if v is not None:
        print(f"{k:15s}: {v:.4f}")

# ------------------------------------------------------------
# Cell 7: Confusion Matrix
# ------------------------------------------------------------
cm = confusion_matrix(y_dir_test, y_pred_direction)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["DOWN", "UP"],
    yticklabels=["DOWN", "UP"]
)
plt.title("Confusion Matrix - Direction Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(BASE_DIR / "results" / "05_confusion_matrix.png", dpi=100)
plt.show()

# ------------------------------------------------------------
# Cell 8: Directional Accuracy
# ------------------------------------------------------------
dir_accuracy = MetricsCalculator.calculate_directional_accuracy(
    y_test,
    y_pred_price
)

print("\n" + "=" * 70)
print("DIRECTIONAL ACCURACY")
print("=" * 70)
print(f"Directional Accuracy: {dir_accuracy:.2f}%")

# ------------------------------------------------------------
# Cell 9: Evaluation Report
# ------------------------------------------------------------
reporter = EvaluationReporter()

full_report = {
    "regression_metrics": reg_metrics,
    "classification_metrics": clf_metrics,
    "directional_accuracy": dir_accuracy,
    "timestamp": pd.Timestamp.now().isoformat()
}

reporter.print_evaluation_report(full_report)
reporter.save_evaluation_report(full_report, ticker)

print("\n✓ Evaluation completed successfully")
print("Proceed to 06_backtesting.ipynb")
