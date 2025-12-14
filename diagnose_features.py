# diagnose_features.py
# Run this to find the missing feature

import pickle
import pandas as pd
from pathlib import Path

ticker = "AAPL"
base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "models"
data_dir = base_dir / "data" / "processed"

# Load saved features
with open(model_dir / f"{ticker}_feature_cols.pkl", "rb") as f:
    saved_features = pickle.load(f)

# Load engineered data
df = pd.read_csv(data_dir / f"{ticker}_engineered.csv")

# Compare
print(f"📊 Saved features: {len(saved_features)}")
print(f"📊 Available columns: {len(df.columns)}")
print(f"\n🔍 Saved features list:")
print(saved_features)

print(f"\n🔍 All available columns:")
print(df.columns.tolist())

# Find what's in CSV but not in saved features
extra_in_csv = set(df.columns) - set(saved_features)
print(f"\n✅ Columns in CSV but NOT in saved features:")
print(extra_in_csv)

# Find what's in saved features but not in CSV
missing_in_csv = set(saved_features) - set(df.columns)
print(f"\n⚠️  Features saved but NOT in CSV:")
print(missing_in_csv)

# Most likely candidate for missing feature (excluding non-feature columns)
non_feature_cols = ['Date', 'Adj Close', 'target', 'target_class']
potential_features = [col for col in df.columns if col not in non_feature_cols and col not in saved_features]

print(f"\n🎯 Most likely missing feature(s):")
print(potential_features)