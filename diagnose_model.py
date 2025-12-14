import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

print("=" * 70)
print("🔍 DIAGNOSING MODEL TRAINING ISSUES")
print("=" * 70)

# Check 1: Data file exists and has data
print("\n1️⃣ Checking Training Data...")
data_path = "data/processed/AAPL_engineered.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"✅ Data file found: {data_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"\n   First few columns: {df.columns[:10].tolist()}")
    print(f"\n   Data types:\n{df.dtypes.value_counts()}")
else:
    print(f"❌ Data file NOT found: {data_path}")

# Check 2: Scaler file
print("\n2️⃣ Checking Scaler File...")
scaler_path = "models/AAPL_significant_scaler.pkl"
if os.path.exists(scaler_path):
    try:
        scaler = pickle.load(open(scaler_path, 'rb'))
        print(f"✅ Scaler file found: {scaler_path}")
        print(f"   Scaler type: {type(scaler)}")
        if hasattr(scaler, 'n_features_in_'):
            print(f"   Features expected: {scaler.n_features_in_}")
        else:
            print(f"   ⚠️ Scaler doesn't have n_features_in_ (old version?)")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
else:
    print(f"❌ Scaler file NOT found: {scaler_path}")

# Check 3: LSTM model file
print("\n3️⃣ Checking LSTM Model File...")
model_path = "models/AAPL_significant_lstm.h5"
if os.path.exists(model_path):
    size = os.path.getsize(model_path)
    print(f"✅ Model file found: {model_path}")
    print(f"   File size: {size:,} bytes")
    if size < 500000:
        print(f"   ⚠️ Model file seems small (might be incomplete)")
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"   ✅ Model loads successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Trainable params: {model.count_params():,}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
else:
    print(f"❌ Model file NOT found: {model_path}")

# Check 4: Sample prediction to see what's happening
print("\n4️⃣ Testing Sample Prediction...")
try:
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get feature columns (exclude target/date columns)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close', 'daily_return', 'target']]
    
    print(f"   Total features: {len(feature_cols)}")
    
    # Load scaler and model
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = tf.keras.models.load_model(model_path)
    
    # Get last sequence
    last_data = df[feature_cols].iloc[-60:].values  # Last 60 days
    
    if np.isnan(last_data).any():
        print(f"   ⚠️ WARNING: Data contains NaN values!")
        print(f"   NaN count: {np.isnan(last_data).sum()}")
    else:
        print(f"   ✅ No NaN values in data")
    
    # Reshape for LSTM
    X = last_data.reshape(1, last_data.shape[0], last_data.shape[1])
    
    # Predict
    pred = model.predict(X, verbose=0)
    print(f"   Raw prediction: {pred[0]}")
    print(f"   Predicted probability: {pred[0][0]:.4f}")
    
    if pred[0][0] < 0.1 or pred[0][0] > 0.9:
        print(f"   ⚠️ WARNING: Probability is at extreme (likely overfitted)")
    elif 0.4 < pred[0][0] < 0.6:
        print(f"   ✅ Prediction is reasonable (near 50%)")
    
except Exception as e:
    print(f"   ❌ Error testing prediction: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("💡 RECOMMENDATION:")
print("=" * 70)
print("""
If you see issues above, you likely need to retrain the model.
Run: python src/significant_move_predictor.py

This will retrain with proper validation and create working models.
""")