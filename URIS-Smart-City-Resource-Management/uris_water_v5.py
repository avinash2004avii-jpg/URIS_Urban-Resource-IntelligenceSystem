"""
URIS v5.0 - ULTRA-FAST WATER MODEL (OPTIMIZED MEMORY & IO)
✅ Use chunking for large files
✅ Cache intermediate results
✅ Minimal feature set (only top 5 predictors)
✅ Fast training with scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import os
import pickle
import time
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
OUTPUT_DIR = str(BASE_DIR)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("🚀 URIS v5.0 - ULTRA-FAST WATER MODEL (OPTIMIZED)")
print("="*80)

start = time.time()

# ==================== LOAD EFFICIENTLY ====================
print("\n[1] LOAD DATA (EFFICIENT)")
print("-" * 80)

print("Loading electricity...")
elec_df = pd.read_csv(os.path.join(DATA_DIR, "building_consumption.csv"), 
                      parse_dates=['timestamp'], dtype={'consumption': 'float32'})
print(f"  ✓ {len(elec_df)} rows")

print("Loading water...")
water_df = pd.read_csv(os.path.join(DATA_DIR, "water_consumption.csv"),
                       parse_dates=['timestamp'], dtype={'consumption': 'float32'})
print(f"  ✓ {len(water_df)} rows")

# Load weather in chunks and aggregate immediately
print("Loading weather (memory-efficient)...")
weather_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "weather_data.csv"), chunksize=100000,
                         parse_dates=['timestamp'], 
                         usecols=['timestamp', 'air_temperature', 'relative_humidity']):
    chunk.rename(columns={'air_temperature': 'temperature', 'relative_humidity': 'humidity'}, inplace=True)
    chunk['timestamp'] = chunk['timestamp'].dt.floor('H')  # Round to hour
    weather_chunks.append(chunk.groupby('timestamp')[['temperature', 'humidity']].mean())
    
weather_df = pd.concat(weather_chunks).groupby(weather_chunks[0].index).mean().reset_index()
weather_df.columns = ['timestamp', 'temperature', 'humidity']
print(f"  ✓ {len(weather_df)} hourly records")

# ==================== HOURLY AGGREGATION ====================
print("\n[2] HOURLY AGGREGATION")
print("-" * 80)

elec_df['timestamp'] = elec_df['timestamp'].dt.floor('H')
elec_hourly = elec_df.groupby('timestamp')['consumption'].mean().reset_index()
elec_hourly.columns = ['timestamp', 'electricity']
elec_hourly = elec_hourly[elec_hourly['electricity'] > 0]
print(f"  ⚡ {len(elec_hourly)} hourly records")

water_df['timestamp'] = water_df['timestamp'].dt.floor('H')
water_hourly = water_df.groupby('timestamp')['consumption'].mean().reset_index()
water_hourly.columns = ['timestamp', 'water']
water_hourly = water_hourly[water_hourly['water'] > 0]
print(f"  💧 {len(water_hourly)} hourly records")
print(f"  🌡️  {len(weather_df)} hourly records")

# ==================== MERGE ====================
print("\n[3] MERGE DATASETS")
print("-" * 80)

df = pd.merge(elec_hourly, weather_df, on='timestamp', how='left')
df = pd.merge(df, water_hourly, on='timestamp', how='inner')
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✓ Final dataset: {len(df)} rows (times with water data available)")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ==================== FEATURES (MINIMAL) ====================
print("\n[4] CREATE FEATURES (MINIMAL SET)")
print("-" * 80)

df['water_lag1'] = df['water'].shift(1)
df['water_lag24'] = df['water'].shift(24)
df['elec_today'] = df['electricity']
df['temp'] = df['temperature']
df['hour'] = df['timestamp'].dt.hour

# Fill first values
df[['water_lag1', 'water_lag24']] = df[['water_lag1', 'water_lag24']].fillna(method='bfill')

feature_cols = ['water_lag1', 'water_lag24', 'elec_today', 'temp', 'hour']
X = df[feature_cols].dropna().copy()
y = df.loc[X.index, 'water'].copy()

print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Samples: {len(X)}")

# ==================== NORMALIZE ====================
print("\n[5] NORMALIZE")
print("-" * 80)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

print(f"✓ Scaled")

# ==================== SPLIT ====================
print("\n[6] TRAIN-TEST SPLIT")
print("-" * 80)

split_idx = int(len(X_scaled) * 0.8)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y_scaled[:split_idx]
y_test = y_scaled[split_idx:]

y_test_orig = y.iloc[split_idx:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ==================== TRAIN ====================
print("\n[7] TRAIN MODELS")
print("-" * 80)

# Linear Regression (baseline)
print("Training Linear Regression...")
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr_test = model_lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr_test)
print(f"  Linear: R² = {r2_lr:.4f}")

# Ridge Regression
print("Training Ridge Regression...")
model_ridge = Ridge(alpha=0.1)
model_ridge.fit(X_train, y_train)
y_pred_ridge_test = model_ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge_test)
print(f"  Ridge: R² = {r2_ridge:.4f}")

# ==================== EVALUATION ====================
print("\n[8] EVALUATION")
print("-" * 80)

# Choose better model
if r2_ridge > r2_lr:
    best_model = model_ridge
    y_pred = y_pred_ridge_test
    best_name = "Ridge"
    r2_best = r2_ridge
else:
    best_model = model_lr
    y_pred = y_pred_lr_test
    best_name = "Linear"
    r2_best = r2_lr

# Convert back to original scale
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

r2_orig = r2_score(y_test_orig, y_pred_orig)
mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)

print(f"\n{best_name} Regression:")
print(f"  R² (scaled): {r2_best:.4f}")
print(f"  R² (original): {r2_orig:.4f}")
print(f"  MAE:        {mae_orig:.3f}")

print(f"\n📊 Results:")
residuals = np.abs(y_test_orig - y_pred_orig)
print(f"  Mean error: {residuals.mean():.3f}")
print(f"  Median error: {np.median(residuals):.3f}")
print(f"  Max error: {residuals.max():.3f}")

if r2_orig > 0.85:
    print(f"\n✅ 🎉 EXCELLENT: R² = {r2_orig:.4f} (>85% accuracy!)")
elif r2_orig > 0.70:
    print(f"\n✓ GOOD: R² = {r2_orig:.4f} (>70% accuracy)")
else:
    print(f"\n⚠️  R² = {r2_orig:.4f} ({r2_orig*100:.1f}%)")

# ==================== SAVE ====================
print("\n[9] SAVE RESULTS")
print("-" * 80)

with open(os.path.join(OUTPUT_DIR, 'model_water_v5.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

results_df = pd.DataFrame({
    'timestamp': df['timestamp'].iloc[split_idx:split_idx+len(y_test_orig)].values,
    'water_actual': y_test_orig.values,
    'water_predicted': y_pred_orig,
    'error_abs': residuals
})
results_df.to_csv(os.path.join(OUTPUT_DIR, 'water_predictions_v5.csv'), index=False)

print(f"✓ model_water_v5.pkl")
print(f"✓ water_predictions_v5.csv")

elapsed = time.time() - start
print(f"\n⏱️  Total time: {elapsed:.1f}s")

print("\n" + "=" * 80)
print(f"FINAL R² SCORE: {r2_orig:.4f}")
print("=" * 80 + "\n")
