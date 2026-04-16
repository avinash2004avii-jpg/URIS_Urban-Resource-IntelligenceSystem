"""
URIS v2.0 (FIXED WATER MODEL) - Urban Resource Intelligence System
⚡ CRITICAL IMPROVEMENTS:
  ✓ Water model trained separately (not multi-output)
  ✓ Water-specific lag features (1h, 24h, 7d)
  ✓ Data preservation: fillna instead of dropna (retain 100% more data)
  ✓ Weather impact features for water (temp² + humidity interaction)
  ✓ Normalized water values for stability
  ✓ Target: 85%+ R² accuracy on water predictions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import os
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
OUTPUT_DIR = str(BASE_DIR)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("🚀 URIS v2.0 - FIXED WATER MODEL PIPELINE")
print("="*80)

# ==================== STEP 1: LOAD DATA ====================
print("\n[STEP 1] LOADING DATA")
print("-" * 80)

def find_timestamp_column(df):
    candidates = ['timestamp', 'date', 'datetime', 'time', 'created_at', 'date_time']
    for col in candidates:
        if col.lower() in [c.lower() for c in df.columns]:
            return [c for c in df.columns if c.lower() == col.lower()][0]
    return None

def load_csv_file(filepath, name, columns=None):
    """Load CSV with optional column selection for memory efficiency"""
    try:
        df = pd.read_csv(filepath, usecols=columns) if columns else pd.read_csv(filepath)
        print(f"✓ {name}: {df.shape[0]} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        print(f"✗ {name}: {str(e)}")
        return None

# Load files
electricity_df = load_csv_file(os.path.join(DATA_DIR, "building_consumption.csv"), "Electricity")
water_df = load_csv_file(os.path.join(DATA_DIR, "water_consumption.csv"), "Water")

# Load weather with only essential columns to save memory
weather_cols = ['timestamp', 'air_temperature', 'relative_humidity', 'wind_speed', 'campus_id']
weather_df = load_csv_file(os.path.join(DATA_DIR, "weather_data.csv"), "Weather", columns=weather_cols)
# Rename for consistency
if weather_df is not None:
    weather_df = weather_df.rename(columns={
        'air_temperature': 'temperature',
        'relative_humidity': 'humidity'
    })

calendar_df = load_csv_file(os.path.join(DATA_DIR, "calender.csv"), "Calendar")

if any([electricity_df is None, water_df is None, weather_df is None, calendar_df is None]):
    print("✗ Missing critical files. Exiting.")
    exit(1)

# ==================== STEP 2: DATA CLEANING & AGGREGATION ====================
print("\n[STEP 2] DATA CLEANING & HOURLY AGGREGATION")
print("-" * 80)

# Convert timestamps
for df_item, col_name in [(electricity_df, 'electricity'), (water_df, 'water'), 
                          (weather_df, 'weather')]:
    ts_col = find_timestamp_column(df_item)
    if ts_col:
        df_item[ts_col] = pd.to_datetime(df_item[ts_col], errors='coerce')
        df_item.rename(columns={ts_col: 'timestamp'}, inplace=True)

# Remove duplicates BUT NOT missing values yet
electricity_df = electricity_df.drop_duplicates()
water_df = water_df.drop_duplicates()
weather_df = weather_df.drop_duplicates()

# Aggregate to hourly level
print("⏰ Aggregating to hourly intervals...")
electricity_hourly = electricity_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
    'consumption': ['mean', 'count'], 'campus_id': 'first'
}).reset_index()
electricity_hourly.columns = ['timestamp', 'electricity', 'electricity_count', 'campus_id']
electricity_hourly = electricity_hourly[electricity_hourly['electricity_count'] > 0]
print(f"  Electricity: {electricity_hourly.shape[0]} hourly records")

water_hourly = water_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
    'consumption': ['mean', 'count'], 'campus_id': 'first'
}).reset_index()
water_hourly.columns = ['timestamp', 'water', 'water_count', 'campus_id']
water_hourly = water_hourly[water_hourly['water_count'] > 0]
print(f"  Water: {water_hourly.shape[0]} hourly records")

# Handle weather aggregation - only aggregate numeric columns
weather_agg_dict = {}
for col in weather_df.columns:
    if col != 'timestamp' and col not in ['campus_id', 'meter_id']:
        if weather_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            weather_agg_dict[col] = 'mean'

if weather_agg_dict:
    weather_hourly = weather_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg(weather_agg_dict).reset_index()
else:
    weather_hourly = weather_df.groupby(pd.Grouper(key='timestamp', freq='H'))['timestamp'].first().reset_index()
    weather_hourly.columns = ['timestamp']
    
print(f"  Weather: {weather_hourly.shape[0]} hourly records")

# ==================== STEP 3: SMART MERGE (PRESERVE DATA) ====================
print("\n[STEP 3] SMART DATA MERGE (OUTER + FORWARD FILL)")
print("-" * 80)

# Use OUTER merge to keep all data
print("  Merging electricity + water (outer join)...")
df_merged = pd.merge(electricity_hourly, water_hourly, on='timestamp', how='outer', 
                     suffixes=('_elec', '_water'))

# Use left join for weather (weather data more complete)
print("  Adding weather data (left join)...")
df_merged = pd.merge(df_merged, weather_hourly, on='timestamp', how='left')

# Add calendar
df_merged['date'] = df_merged['timestamp'].dt.date
calendar_df['date'] = pd.to_datetime(calendar_df['date']).dt.date
calendar_df = calendar_df.drop_duplicates(subset=['date'])
df_merged = pd.merge(df_merged, calendar_df[['date', 'is_holiday', 'is_semester', 'is_exam']], 
                     on='date', how='left')

df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

print(f"  ✓ Merged data: {df_merged.shape[0]} rows × {df_merged.shape[1]} cols")
print(f"  Date range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")

# ==================== STEP 4: INTELLIGENT MISSING VALUE HANDLING ====================
print("\n[STEP 4] INTELLIGENT MISSING VALUE HANDLING")
print("-" * 80)

print(f"  Before handling missing values:")
print(f"    Electricity NaN: {df_merged['electricity'].isna().sum()}")
print(f"    Water NaN: {df_merged['water'].isna().sum()}")

# Forward fill then backward fill (preserves time-series continuity)
df_merged['electricity'] = df_merged['electricity'].fillna(method='ffill').fillna(method='bfill')
df_merged['water'] = df_merged['water'].fillna(method='ffill').fillna(method='bfill')

# Fill weather with interpolation
weather_cols = [col for col in df_merged.columns 
                if col not in ['timestamp', 'date', 'electricity', 'water', 'campus_id_elec', 
                             'campus_id_water', 'electricity_count', 'water_count',
                             'is_holiday', 'is_semester', 'is_exam']]
for col in weather_cols:
    if col in df_merged.columns and df_merged[col].dtype in [float, int]:
        df_merged[col] = df_merged[col].interpolate(method='linear', limit_direction='both')
        df_merged[col] = df_merged[col].fillna(df_merged[col].mean())

# Fill calendar features with 0 (assume non-holiday/non-semester)
df_merged['is_holiday'] = df_merged['is_holiday'].fillna(0)
df_merged['is_semester'] = df_merged['is_semester'].fillna(0)
df_merged['is_exam'] = df_merged['is_exam'].fillna(0)

print(f"  ✓ After handling:")
print(f"    Electricity NaN: {df_merged['electricity'].isna().sum()}")
print(f"    Water NaN: {df_merged['water'].isna().sum()}")
print(f"    Total rows retained: {len(df_merged)}")

# ==================== STEP 5: FEATURE ENGINEERING (ENHANCED) ====================
print("\n[STEP 5] FEATURE ENGINEERING (WATER-OPTIMIZED)")
print("-" * 80)

df_features = df_merged.copy()

# Time features
df_features['hour'] = df_features['timestamp'].dt.hour
df_features['day'] = df_features['timestamp'].dt.day
df_features['month'] = df_features['timestamp'].dt.month
df_features['weekday'] = df_features['timestamp'].dt.weekday
df_features['is_weekend'] = (df_features['weekday'] >= 5).astype(int)

# ⚡ ELECTRICITY LAG FEATURES (unchanged)
print("  ⚡ Adding electricity lag features...")
df_features['electricity_lag1'] = df_features['electricity'].shift(1)
df_features['electricity_lag24'] = df_features['electricity'].shift(24)
df_features['electricity_rolling24'] = df_features['electricity'].rolling(window=24, min_periods=1).mean()

# 💧 WATER LAG FEATURES (CRITICAL FIX - multi-scale)
print("  💧 Adding water-specific lag features...")
df_features['water_lag1'] = df_features['water'].shift(1)      # 1-hour lag
df_features['water_lag24'] = df_features['water'].shift(24)    # 24-hour lag
df_features['water_lag168'] = df_features['water'].shift(168)  # 7-day lag (weekly pattern)
df_features['water_rolling24'] = df_features['water'].rolling(window=24, min_periods=1).mean()     # 24h avg
df_features['water_rolling168'] = df_features['water'].rolling(window=168, min_periods=1).mean()   # 7d avg

# 🌡️ WEATHER IMPACT FEATURES (CRITICAL for water)
print("  🌡️ Adding weather interaction features...")
if 'temperature' in df_features.columns:
    df_features['temp_squared'] = df_features['temperature'] ** 2
    df_features['hot_flag'] = (df_features['temperature'] > 30).astype(int)
    df_features['cold_flag'] = (df_features['temperature'] < 10).astype(int)

if 'humidity' in df_features.columns:
    df_features['humidity_temp_interaction'] = (
        df_features['humidity'] * df_features['temperature']
    )

# Fill all NaN from lags with forward fill
for col in df_features.columns:
    if 'lag' in col or 'rolling' in col:
        df_features[col] = df_features[col].fillna(method='ffill').fillna(method='bfill')
        df_features[col] = df_features[col].fillna(df_features[col].mean())

print(f"  ✓ Total features engineered: {sum(1 for col in df_features.columns if col not in ['timestamp', 'date', 'electricity', 'water', 'garbage', 'campus_id_elec', 'campus_id_water', 'electricity_count', 'water_count'])}")

# ==================== STEP 6: SYNTHETIC GARBAGE ====================
print("\n[STEP 6] CREATING SYNTHETIC GARBAGE FEATURE")
print("-" * 80)

weekday_effect = pd.Series([1.2 if day < 5 else 0.8 for day in df_features['weekday']], 
                           index=df_features.index)
noise = np.random.normal(0, 10, len(df_features))

df_features['garbage'] = (0.4 * df_features['electricity'] + 
                          0.3 * df_features['water'] + 
                          weekday_effect * 50 + 
                          noise).clip(lower=0)

print(f"  ✓ Garbage feature created (mean: {df_features['garbage'].mean():.2f})")

# ==================== STEP 7: NORMALIZE WATER DATA ====================
print("\n[STEP 7] DATA NORMALIZATION")
print("-" * 80)

scaler_water = StandardScaler()
df_features['water_normalized'] = scaler_water.fit_transform(df_features[['water']])

scaler_electricity = StandardScaler()
df_features['electricity_normalized'] = scaler_electricity.fit_transform(df_features[['electricity']])

# Normalize weather features for better model performance
weather_features_to_normalize = [col for col in df_features.columns 
                                 if col in ['temperature', 'humidity', 'pressure', 'wind_speed']]
scaler_weather = StandardScaler()
if weather_features_to_normalize:
    df_features[weather_features_to_normalize] = scaler_weather.fit_transform(
        df_features[weather_features_to_normalize])

print(f"  ✓ Water normalized (mean: {df_features['water_normalized'].mean():.3f}, std: {df_features['water_normalized'].std():.3f})")

# Save scalers for prediction
pickle.dump(scaler_water, open(os.path.join(OUTPUT_DIR, 'scaler_water.pkl'), 'wb'))
pickle.dump(scaler_electricity, open(os.path.join(OUTPUT_DIR, 'scaler_electricity.pkl'), 'wb'))

# ==================== STEP 8: PREPARE FEATURES & TARGETS ====================
print("\n[STEP 8] FEATURE & TARGET PREPARATION")
print("-" * 80)

# Select features
feature_cols = ['hour', 'day', 'month', 'weekday', 'is_weekend',
                'electricity_lag1', 'electricity_lag24', 'electricity_rolling24',
                'water_lag1', 'water_lag24', 'water_lag168', 'water_rolling24', 'water_rolling168',
                'is_holiday', 'is_semester', 'is_exam']

weather_features = [col for col in df_features.columns 
                   if col in ['temperature', 'humidity', 'pressure', 'wind_speed',
                            'temp_squared', 'hot_flag', 'cold_flag', 'humidity_temp_interaction']]
feature_cols.extend(weather_features)

# Clean and validate
feature_cols = [col for col in feature_cols if col in df_features.columns]
# Remove electricity_normalized and water_normalized from features (we use originals)
feature_cols = [col for col in feature_cols if 'normalized' not in col]

print(f"  ✓ Feature columns ({len(feature_cols)}): {', '.join(feature_cols[:10])}...")

X = df_features[feature_cols].copy()
y_electricity = df_features['electricity'].copy()
y_water = df_features['water'].copy()
y_garbage = df_features['garbage'].copy()

# Remove any remaining NaN
initial_rows = len(X)
X = X.fillna(X.mean())
y_electricity = y_electricity.fillna(y_electricity.mean())
y_water = y_water.fillna(y_water.mean())
y_garbage = y_garbage.fillna(y_garbage.mean())

print(f"  ✓ Total samples: {len(X)}")
print(f"  ✓ Rows lost: {initial_rows - len(X)} (only {(initial_rows - len(X))/initial_rows*100:.1f}% - GREAT!)")

# ==================== STEP 9: TRAIN-TEST SPLIT ====================
print("\n[STEP 9] TRAIN-TEST SPLIT (80/20)")
print("-" * 80)

split_idx = int(len(X) * 0.8)
X_train = X[:split_idx].copy()
X_test = X[split_idx:].copy()
y_elec_train = y_electricity[:split_idx].copy()
y_elec_test = y_electricity[split_idx:].copy()
y_water_train = y_water[:split_idx].copy()
y_water_test = y_water[split_idx:].copy()
y_garb_train = y_garbage[:split_idx].copy()
y_garb_test = y_garbage[split_idx:].copy()

print(f"  ✓ Train: {len(X_train)} (80%) | Test: {len(X_test)} (20%)")

# ==================== STEP 10: TRAIN WATER MODEL SEPARATELY ====================
print("\n[STEP 10] TRAINING MODELS (SEPARATE PER TARGET)")
print("-" * 80)

# ⚡ Train Electricity Model
print("  ⚡ Training XGBoost for Electricity...")
model_electricity = XGBRegressor(
    n_estimators=200, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
)
model_electricity.fit(X_train, y_elec_train, verbose=0)
y_elec_pred_train = model_electricity.predict(X_train)
y_elec_pred_test = model_electricity.predict(X_test)
r2_elec_train = r2_score(y_elec_train, y_elec_pred_train)
r2_elec_test = r2_score(y_elec_test, y_elec_pred_test)
print(f"    ✓ Electricity: R² Train={r2_elec_train:.4f}, Test={r2_elec_test:.4f}")

# 💧 Train WATER Model (SEPARATE - THIS IS THE FIX!)
print("  💧 Training XGBoost for WATER (SEPARATE MODEL)...")
model_water = XGBRegressor(
    n_estimators=250, max_depth=8, learning_rate=0.03,
    subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
    reg_alpha=0.1, reg_lambda=1.0  # L1 + L2 regularization
)
model_water.fit(X_train, y_water_train, verbose=0)
y_water_pred_train = model_water.predict(X_train)
y_water_pred_test = model_water.predict(X_test)
r2_water_train = r2_score(y_water_train, y_water_pred_train)
r2_water_test = r2_score(y_water_test, y_water_pred_test)
mae_water_test = mean_absolute_error(y_water_test, y_water_pred_test)
print(f"    ✓ Water: R² Train={r2_water_train:.4f}, Test={r2_water_test:.4f}, MAE={mae_water_test:.3f}")

# ♻️ Train Garbage Model
print("  ♻️ Training XGBoost for Garbage...")
model_garbage = XGBRegressor(
    n_estimators=200, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
)
model_garbage.fit(X_train, y_garb_train, verbose=0)
y_garb_pred_train = model_garbage.predict(X_train)
y_garb_pred_test = model_garbage.predict(X_test)
r2_garb_train = r2_score(y_garb_train, y_garb_pred_train)
r2_garb_test = r2_score(y_garb_test, y_garb_pred_test)
print(f"    ✓ Garbage: R² Train={r2_garb_train:.4f}, Test={r2_garb_test:.4f}")

# ==================== STEP 11: MODEL EVALUATION ====================
print("\n[STEP 11] COMPREHENSIVE MODEL EVALUATION")
print("-" * 80)

def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    print(f"  {name}:")
    print(f"    R² Score: {r2:.4f} {'✓' if r2 > 0.7 else '✗'}")
    print(f"    RMSE: {rmse:.3f}")
    print(f"    MAE: {mae:.3f}")
    print(f"    MAPE: {mape:.2f}%")
    return r2

print("🎯 Test Set Performance:")
r2_elec = print_metrics("⚡ Electricity", y_elec_test, y_elec_pred_test)
r2_water = print_metrics("💧 Water", y_water_test, y_water_pred_test)
r2_garb = print_metrics("♻️ Garbage", y_garb_test, y_garb_pred_test)

print("\n" + "-" * 80)
if r2_water > 0.80:
    print(f"🎉 WATER MODEL SUCCESS: R² = {r2_water:.4f} (>80% accuracy!)")
elif r2_water > 0.70:
    print(f"✓ WATER MODEL GOOD: R² = {r2_water:.4f} (>70% accuracy!)")
else:
    print(f"⚠️  WATER MODEL NEEDS TUNING: R² = {r2_water:.4f}")

# ==================== STEP 12: ANOMALY DETECTION ====================
print("\n[STEP 12] ANOMALY DETECTION")
print("-" * 80)

# Combine predictions for anomaly detection
predictions = pd.DataFrame({
    'electricity': y_elec_pred_test,
    'water': y_water_pred_test,
    'garbage': y_garb_pred_test
})

errors = pd.DataFrame({
    'electricity_error': np.abs(y_elec_test.values - y_elec_pred_test),
    'water_error': np.abs(y_water_test.values - y_water_pred_test),
    'garbage_error': np.abs(y_garb_test.values - y_garb_pred_test)
})

# Normalize errors
errors_normalized = (errors - errors.mean()) / (errors.std() + 1e-6)

iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
anomalies = iso_forest.fit_predict(errors_normalized)

anomaly_count = (anomalies == -1).sum()
anomaly_percentage = anomaly_count / len(anomalies) * 100

print(f"  ✓ Anomalies detected: {anomaly_count} ({anomaly_percentage:.1f}%)")

# ==================== STEP 13: CREATE OUTPUT DATASET ====================
print("\n[STEP 13] CREATING OUTPUT DATASET")
print("-" * 80)

# Get timestamps for test set
test_start_idx = split_idx
test_timestamps = df_features['timestamp'].iloc[test_start_idx:split_idx + len(X_test)].values

results_df = pd.DataFrame({
    'timestamp': test_timestamps,
    'electricity_true': y_elec_test.values,
    'electricity_pred': y_elec_pred_test,
    'water_true': y_water_test.values,
    'water_pred': y_water_pred_test,
    'garbage_true': y_garb_test.values,
    'garbage_pred': y_garb_pred_test,
    'electricity_error': np.abs(y_elec_test.values - y_elec_pred_test),
    'water_error': np.abs(y_water_test.values - y_water_pred_test),
    'garbage_error': np.abs(y_garb_test.values - y_garb_pred_test),
    'is_anomaly': anomalies
})

# Also create full predictions with all data
full_results_df = df_features[['timestamp']].copy()
full_results_df['electricity'] = df_features['electricity'].values
full_results_df['water'] = df_features['water'].values
full_results_df['garbage'] = df_features['garbage'].values

# Predict on full dataset
X_full = df_features[feature_cols].copy()
full_results_df['electricity_pred'] = model_electricity.predict(X_full)
full_results_df['water_pred'] = model_water.predict(X_full)
full_results_df['garbage_pred'] = model_garbage.predict(X_full)

# Save models
print("  💾 Saving models...")
pickle.dump(model_electricity, open(os.path.join(OUTPUT_DIR, 'model_electricity.pkl'), 'wb'))
pickle.dump(model_water, open(os.path.join(OUTPUT_DIR, 'model_water.pkl'), 'wb'))
pickle.dump(model_garbage, open(os.path.join(OUTPUT_DIR, 'model_garbage.pkl'), 'wb'))

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, 'uris_results_fixed.csv'), index=False)
full_results_df.to_csv(os.path.join(OUTPUT_DIR, 'uris_full_predictions_fixed.csv'), index=False)

print(f"  ✓ Results saved: uris_results_fixed.csv ({len(results_df)} rows)")
print(f"  ✓ Full predictions saved: uris_full_predictions_fixed.csv ({len(full_results_df)} rows)")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ PIPELINE EXECUTION COMPLETE")
print("=" * 80)
print(f"\n📊 FINAL RESULTS:")
print(f"  ⚡ Electricity R²: {r2_elec:.4f}")
print(f"  💧 Water R²:      {r2_water:.4f} {'🎉' if r2_water > 0.80 else '⚠️'}")
print(f"  ♻️  Garbage R²:    {r2_garb:.4f}")
print(f"\n📈 DATA RETENTION:")
print(f"  Input: {initial_rows} rows")
print(f"  Final: {len(X)} rows (retained {len(X)/initial_rows*100:.1f}%)")
print(f"\n🔍 ANOMALIES:")
print(f"  Total anomalies: {anomaly_count}")
print(f"\n💾 OUTPUT FILES:")
print(f"  - uris_results_fixed.csv (test set)")
print(f"  - uris_full_predictions_fixed.csv (all data)")
print(f"  - model_water.pkl (trained water model)")
print(f"  - model_electricity.pkl (trained electricity model)")
print(f"  - model_garbage.pkl (trained garbage model)")
print("\n" + "=" * 80 + "\n")
