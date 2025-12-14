# intelligent_fix.py
# This checks the SCALER to find what features were actually used during training

import pickle
import pandas as pd
import tensorflow as tf
from pathlib import Path

ticker = "AAPL"
base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "models"
data_dir = base_dir / "data" / "processed"

print("="*70)
print("🔬 SCALER-BASED FEATURE DETECTIVE")
print("="*70)

# Load scaler (it knows what features it was trained on!)
with open(model_dir / f"{ticker}_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load current saved features
with open(model_dir / f"{ticker}_feature_cols.pkl", "rb") as f:
    saved_features = pickle.load(f)

# Load model
model = tf.keras.models.load_model(model_dir / f"{ticker}_lstm_regressor.h5", compile=False)

# Load data
df = pd.read_csv(data_dir / f"{ticker}_engineered.csv")

print(f"\n📊 Current State:")
print(f"   • Saved features: {len(saved_features)}")
print(f"   • Scaler expects: {scaler.n_features_in_}")
print(f"   • Model expects: {model.input_shape[2]}")

# The scaler has feature_names_in_ if fitted with pandas DataFrame
if hasattr(scaler, 'feature_names_in_'):
    scaler_features = list(scaler.feature_names_in_)
    print(f"   • Scaler trained on: {len(scaler_features)} features")
    
    print(f"\n✅ FOUND IT! The scaler knows the exact features used during training!")
    
    # Compare with saved features
    missing_in_saved = set(scaler_features) - set(saved_features)
    extra_in_saved = set(saved_features) - set(scaler_features)
    
    if missing_in_saved:
        print(f"\n🎯 Features in scaler but NOT in saved list:")
        for feat in missing_in_saved:
            print(f"   ❌ {feat}")
    
    if extra_in_saved:
        print(f"\n⚠️  Features in saved list but NOT in scaler:")
        for feat in extra_in_saved:
            print(f"   ⚠️  {feat}")
    
    if len(scaler_features) == scaler.n_features_in_:
        print(f"\n✅ Using scaler's feature list as ground truth")
        
        # Backup original
        backup_path = model_dir / f"{ticker}_feature_cols_BACKUP.pkl"
        with open(model_dir / f"{ticker}_feature_cols.pkl", "rb") as f:
            pickle.dump(saved_features, open(backup_path, "wb"))
        print(f"💾 Backed up original to: {backup_path.name}")
        
        # Save correct features from scaler
        with open(model_dir / f"{ticker}_feature_cols.pkl", "wb") as f:
            pickle.dump(scaler_features, f)
        
        print(f"💾 Saved corrected feature_cols.pkl with {len(scaler_features)} features")
        
        print(f"\n📝 Correct feature list ({len(scaler_features)} features):")
        for i, col in enumerate(scaler_features, 1):
            marker = "⭐" if col in missing_in_saved else "✅" if col in saved_features else "  "
            print(f"{marker} {i:2d}. {col}")
        
        print(f"\n" + "="*70)
        print(f"🎉 FIX COMPLETE!")
        print("="*70)
        print(f"\n✅ Now try: py predict_stock.py --ticker AAPL --price 190")
    else:
        print(f"\n❌ Scaler feature count mismatch - this shouldn't happen!")
        
else:
    print(f"\n⚠️  Scaler doesn't have feature_names_in_ attribute")
    print(f"   This means it was fitted with numpy arrays, not a DataFrame")
    print(f"\n💡 Solution: We need to check your train_lstm.py")
    print(f"   The scaler expects {scaler.n_features_in_} features")
    
    # Try to infer from data
    exclude_cols = ['Date', 'Ticker', 'target', 'target_class', 
                    'target_price', 'target_direction', 'target_return']
    potential = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n🔍 Potential features from CSV ({len(potential)}):")
    
    if len(potential) == scaler.n_features_in_:
        print(f"✅ Found {len(potential)} potential features - matches scaler!")
        
        # Save these as features
        backup_path = model_dir / f"{ticker}_feature_cols_BACKUP.pkl"
        with open(model_dir / f"{ticker}_feature_cols.pkl", "wb") as f_backup:
            pickle.dump(saved_features, f_backup)
        
        with open(model_dir / f"{ticker}_feature_cols.pkl", "wb") as f:
            pickle.dump(potential, f)
        
        print(f"💾 Saved potential features as feature_cols.pkl")
        print(f"\n✅ Try: py predict_stock.py --ticker AAPL --price 190")
    else:
        print(f"❌ Found {len(potential)} but need {scaler.n_features_in_}")
        print(f"\n💡 RECOMMENDATION: Retrain the model with proper feature tracking")