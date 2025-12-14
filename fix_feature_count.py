# fix_feature_count.py
# This adds the missing 60th feature to your feature_cols.pkl

import pickle
import pandas as pd
from pathlib import Path

ticker = "AAPL"
base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "models"
data_dir = base_dir / "data" / "processed"

# Load current features
with open(model_dir / f"{ticker}_feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

print(f"📊 Current features: {len(feature_cols)}")

# Load the engineered data to see what we have
df = pd.read_csv(data_dir / f"{ticker}_engineered.csv")

# The issue: model expects 60, we have 59
# Most likely causes:
# 1. Ticker column was used during training
# 2. A duplicate SP500_Return column existed during training
# 3. Adj Close was used but not saved

# Check what could be the 60th feature
candidates = []

# Check if Ticker exists and is numeric (was it one-hot encoded?)
if 'Ticker' in df.columns:
    candidates.append('Ticker')

# Check for Adj Close (sometimes used instead of just Close)
if 'Adj Close' in df.columns:
    candidates.append('Adj Close')

print(f"\n🎯 Candidate features to add:")
for i, col in enumerate(candidates, 1):
    print(f"  {i}. {col}")

# Most likely scenario: Model was trained with both OHLCV columns
# and your code had Adj Close in addition to Close
if 'Adj Close' in df.columns and 'Adj Close' not in feature_cols:
    print(f"\n✅ FOUND IT! 'Adj Close' exists in data but not in saved features")
    print(f"   This is the 60th feature.")
    
    # Add Adj Close right after Close to maintain logical order
    close_idx = feature_cols.index('Close')
    feature_cols.insert(close_idx + 1, 'Adj Close')
    
elif 'Ticker' in df.columns and 'Ticker' not in feature_cols:
    print(f"\n⚠️  Adding 'Ticker' - but this is unusual for a feature")
    feature_cols.append('Ticker')
else:
    # Last resort: check training data structure
    print(f"\n⚠️  Could not identify missing feature automatically")
    print(f"   Most likely cause: pandas merge created duplicate columns during training")
    print(f"   The model has an internal representation we can't see.")
    print(f"\n💡 RECOMMENDATION: Retrain the model to fix this properly")
    exit(1)

print(f"\n📊 Updated features: {len(feature_cols)}")
print(f"\n📝 Full feature list:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# Verify we have exactly 60 now
if len(feature_cols) == 60:
    print(f"\n✅ SUCCESS! Feature count is now 60")
    
    # Save updated features
    with open(model_dir / f"{ticker}_feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    print(f"💾 Saved updated feature_cols.pkl")
    print(f"\n🎉 Now try running: py predict_stock.py --ticker AAPL --price 190")
else:
    print(f"\n❌ ERROR: Feature count is {len(feature_cols)}, expected 60")
    print(f"   This requires manual inspection of your training code")