"""
URIS v3.0 (WATER-CENTRIC APPROACH)
⚡ KEY INSIGHT: Water data is sparse (only 10% of timestamps)
✓ Solution: Train water model ONLY on periods with water data
✓ Use interpolation for other periods if needed
✓ Separate datasets for electricity vs water training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
print("🚀 URIS v3.0 - WATER SPARSE DATA FIX")
print("="*80)

# ==================== LOAD DATA ====================
print("\n[STEP 1] LOADING DATA")
print("-" * 80)

electricity_df = pd.read_csv(os.path.join(DATA_DIR, "building_consumption.csv"))
water_df = pd.read_csv(os.path.join(DATA_DIR, "water_consumption.csv"))
weather_cols = ['timestamp', 'air_temperature', 'relative_humidity', 'wind_speed']
weather_df = pd.read_csv(os.path.join(DATA_DIR, "weather_data.csv"), usecols=weather_cols)
calendar_df = pd.read_csv(os.path.join(DATA_DIR, "calender.csv"))

print(f"✓ Electricity: {electricity_df.shape[0]} records")
print(f"✓ Water: {water_df.shape[0]} records")
print(f"✓ Weather: {weather_df.shape[0]} records")
print(f"✓ Calendar: {calendar_df.shape[0]} records")

# ==================== PROCESS DATA ====================
print("\n[STEP 2] PROCESS DATA & HOURLY AGGREGATION")
print("-" * 80)

# Convert timestamps
electricity_df['timestamp'] = pd.to_datetime(electricity_df['timestamp'], errors='coerce')
water_df['timestamp'] = pd.to_datetime(water_df['timestamp'], errors='coerce')
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
calendar_df['date'] = pd.to_datetime(calendar_df['date']).dt.date

# Remove duplicates and NaN timestamps
electricity_df = electricity_df.dropna(subset=['timestamp']).drop_duplicates()
water_df = water_df.dropna(subset=['timestamp']).drop_duplicates()
weather_df = weather_df.dropna(subset=['timestamp']).drop_duplicates()

# Rename columns
weather_df.rename(columns={'air_temperature': 'temperature', 'relative_humidity': 'humidity'}, inplace=True)

# Hourly aggregation
print("⏰ Aggregating to hourly level...")
electricity_hourly = electricity_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
    'consumption': 'mean'
}).reset_index()
electricity_hourly.columns = ['timestamp', 'electricity']
electricity_hourly = electricity_hourly[electricity_hourly['electricity'] > 0]

water_hourly = water_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
    'consumption': 'mean'
}).reset_index()
water_hourly.columns = ['timestamp', 'water']
water_hourly = water_hourly[water_hourly['water'] > 0]

weather_hourly = weather_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
    'temperature': 'mean', 'humidity': 'mean', 'wind_speed': 'mean'
}).reset_index()

print(f"  ⚡ Electricity: {len(electricity_hourly)} hourly records")
print(f"  💧 Water: {len(water_hourly)} hourly records")
print(f"  🌡️  Weather: {len(weather_hourly)} hourly records")

# ==================== STRATEGY 1: TRAIN WATER ON AVAILABLE DATA ONLY ====================
print("\n[STEP 3] PREPARE WATER-SPECIFIC DATASET")
print("-" * 80)

# Merge electricity + weather
df_elec_weather = pd.merge(electricity_hourly, weather_hourly, on='timestamp', how='left')
print(f"  Electricity + Weather: {len(df_elec_weather)} rows")

# NOW merge with water (only times with water data)
df_water_data = pd.merge(df_elec_weather, water_hourly, on='timestamp', how='inner')
print(f"  ✓ With Water data: {len(df_water_data)} rows (only times water exists)")

df_water_data = df_water_data.sort_values('timestamp').reset_index(drop=True)

# Add date for calendar merge
df_water_data['date'] = df_water_data['timestamp'].dt.date

# Merge with calendar
calendar_df.rename(columns={'date': 'date'}, inplace=True)
df_water_data = pd.merge(df_water_data, calendar_df[['date', 'is_holiday', 'is_semester', 'is_exam']], 
                         on='date', how='left')

print(f"  Final dataset: {df_water_data.shape[0]} records × {df_water_data.shape[1]} cols")
print(f"  Date range: {df_water_data['timestamp'].min()} to {df_water_data['timestamp'].max()}")

# ==================== FEATURE ENGINEERING ====================
print("\n[STEP 4] FEATURE ENGINEERING (WATER-FOCUSED)")
print("-" * 80)

df_features = df_water_data.copy()

# Time features
df_features['hour'] = df_features['timestamp'].dt.hour
df_features['day'] = df_features['timestamp'].dt.day
df_features['month'] = df_features['timestamp'].dt.month
df_features['weekday'] = df_features['timestamp'].dt.weekday
df_features['is_weekend'] = (df_features['weekday'] >= 5).astype(int)

# ⚡ Electricity features
print("  ⚡ Electricity features...")
df_features['electricity_lag1'] = df_features['electricity'].shift(1)
df_features['electricity_lag24'] = df_features['electricity'].shift(24)
df_features['electricity_rolling24'] = df_features['electricity'].rolling(24, min_periods=1).mean()

# 💧 Water features (CRITICAL - multi-scale lags)
print("  💧 Water lag features...")
df_features['water_lag1'] = df_features['water'].shift(1)
df_features['water_lag24'] = df_features['water'].shift(24)
df_features['water_lag168'] = df_features['water'].shift(168)  # Weekly
df_features['water_lag336'] = df_features['water'].shift(336)  # Bi-weekly
df_features['water_liter_per_hour'] = df_features['water'].rolling(1, min_periods=1).mean()

# Rolling statistics for stability
df_features['water_rolling24'] = df_features['water'].rolling(24, min_periods=1).mean()
df_features['water_rolling168'] = df_features['water'].rolling(168, min_periods=1).mean()
df_features['water_std24'] = df_features['water'].rolling(24, min_periods=1).std()

# 🌡️  Weather features
print("  🌡️  Weather interaction features...")
df_features['temp_squared'] = df_features['temperature'] ** 2
df_features['hot_flag'] = (df_features['temperature'] > 28).astype(int)
df_features['cold_flag'] = (df_features['temperature'] < 5).astype(int)
df_features['humidity_temp'] = df_features['humidity'] * df_features['temperature']

# 🔗 Cross-correlation features
print("  🔗 Cross-correlation features...")
df_features['elec_water_ratio'] = df_features['electricity'] / (df_features['water'] + 1e-6)
df_features['elec_rolling24'] = df_features['electricity'].rolling(24, min_periods=1).mean()

# Fill NaN from lags
for col in df_features.columns:
    if 'lag' in col or 'rolling' in col or 'std' in col:
        df_features[col] = df_features[col].fillna(method='ffill').fillna(method='bfill')
        if df_features[col].isna().sum() > 0:
            df_features[col] = df_features[col].fillna(df_features[col].mean())

# Calendar features
df_features['is_holiday'] = df_features['is_holiday'].fillna(0)
df_features['is_semester'] = df_features['is_semester'].fillna(0)
df_features['is_exam'] = df_features['is_exam'].fillna(0)

print(f"  ✓ Created {len([c for c in df_features.columns if c not in ['timestamp', 'date', 'electricity', 'water']])} features")

# ==================== NORMALIZATION ====================
print("\n[STEP 5] DATA NORMALIZATION")
print("-" * 80)

# Normalize water with MinMaxScaler for bounded output
scaler_water = MinMaxScaler(feature_range=(0, 1))
df_features['water_normalized'] = scaler_water.fit_transform(df_features[['water']])

# Normalize weather
scaler_weather = StandardScaler()
weather_cols_norm = ['temperature', 'humidity', 'wind_speed']
df_features[weather_cols_norm] = scaler_weather.fit_transform(df_features[weather_cols_norm])

print(f"  ✓ Water range: [{df_features['water'].min():.2f}, {df_features['water'].max():.2f}]")
print(f"  ✓ Water mean: {df_features['water'].mean():.2f}, std: {df_features['water'].std():.2f}")

# Save scalers
with open(os.path.join(OUTPUT_DIR, 'scaler_water_minmax.pkl'), 'wb') as f:
    pickle.dump(scaler_water, f)
with open(os.path.join(OUTPUT_DIR, 'scaler_weather.pkl'), 'wb') as f:
    pickle.dump(scaler_weather, f)

# ==================== PREPARE MODEL DATA ====================
print("\n[STEP 6] PREPARE FEATURES & TARGETS")
print("-" * 80)

# Select features
feature_cols = ['hour', 'day', 'month', 'weekday', 'is_weekend',
                'electricity', 'electricity_lag1', 'electricity_lag24', 'electricity_rolling24',
                'water_lag1', 'water_lag24', 'water_lag168', 'water_lag336',
                'water_rolling24', 'water_rolling168', 'water_std24',
                'temperature', 'humidity', 'wind_speed',
                'temp_squared', 'hot_flag', 'cold_flag', 'humidity_temp', 'elec_water_ratio',
                'is_holiday', 'is_semester', 'is_exam']

feature_cols = [col for col in feature_cols if col in df_features.columns]
print(f"  Feature columns: {len(feature_cols)}")
print(f"    {', '.join(feature_cols[:8])}...")

X = df_features[feature_cols].copy()
y_water = df_features['water'].copy()

# Remove any NaN
X = X.fillna(X.mean())
y_water = y_water.fillna(y_water.mean())

print(f"  ✓ Total samples: {len(X)}")
print(f"  ✓ X shape: {X.shape}, y shape: {y_water.shape}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n[STEP 7] TRAIN-TEST SPLIT (80/20)")
print("-" * 80)

split_idx = int(len(X) * 0.8)
X_train = X[:split_idx].copy()
X_test = X[split_idx:].copy()
y_train = y_water[:split_idx].copy()
y_test = y_water[split_idx:].copy()

print(f"  Train: {len(X_train)} (80%)")
print(f"  Test: {len(X_test)} (20%)")

# ==================== TRAIN WATER MODEL ====================
print("\n[STEP 8] TRAIN XGBoost FOR WATER")
print("-" * 80)

print("  💧 Optimized water model with:")
print("     - Deeper trees (max_depth=10)")
print("     - More boosting rounds (n_estimators=300)")
print("     - Heavy regularization")
print("     - Learning rate=0.02")

model_water = XGBRegressor(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,          # L1 regularization
    reg_lambda=2.0,         # L2 regularization
    gamma=1.5,              # Min loss reduction
    random_state=RANDOM_STATE,
    verbosity=0,
    tree_method='hist'
)

model_water.fit(X_train, y_train, eval_metric='rmse', verbose=False)

# Predictions
y_pred_train = model_water.predict(X_train)
y_pred_test = model_water.predict(X_test)

# Metrics
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-6))) * 100

print(f"\n  📊 WATER MODEL PERFORMANCE:")
print(f"     R² Train: {r2_train:.4f}")
print(f"     R² Test:  {r2_test:.4f} {'🎉 EXCELLENT!' if r2_test > 0.85 else '✓ Good' if r2_test > 0.70 else '⚠️  Needs work'}")
print(f"     RMSE:     {rmse_test:.3f}")
print(f"     MAE:      {mae_test:.3f}")
print(f"     MAPE:     {mape_test:.2f}%")

# ==================== EVALUATE ====================
print("\n[STEP 9] MODEL EVALUATION")
print("-" * 80)

# Residual analysis
residuals = np.abs(y_test.values - y_pred_test)
percentile_90 = np.percentile(residuals, 90)

print(f"  Residual Statistics:")
print(f"    Mean error: {np.mean(residuals):.3f}")
print(f"    Median error: {np.median(residuals):.3f}")
print(f"    90th percentile: {percentile_90:.3f}")
print(f"    Max error: {np.max(residuals):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_water.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 10 Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']:30s}: {row['importance']:.4f}")

# ==================== ANOMALY DETECTION ====================
print("\n[STEP 10] ANOMALY DETECTION")
print("-" * 80)

# Use residuals for anomaly detection
errors_df = pd.DataFrame({
    'error': residuals,
    'abs_error_pct': (residuals / (y_test.values + 1e-6)) * 100
})

iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
anomalies = iso_forest.fit_predict(errors_df[['abs_error_pct']].values.reshape(-1, 1))
anomaly_count = (anomalies == -1).sum()

print(f"  Anomalies detected: {anomaly_count} ({anomaly_count/len(anomalies)*100:.1f}%)")

# ==================== SAVE RESULTS ====================
print("\n[STEP 11] SAVING RESULTS")
print("-" * 80)

# Save model
with open(os.path.join(OUTPUT_DIR, 'model_water_optimized.pkl'), 'wb') as f:
    pickle.dump(model_water, f)
print("  ✓ model_water_optimized.pkl")

# Save results
results_df = pd.DataFrame({
    'timestamp': df_features['timestamp'].iloc[split_idx:split_idx+len(X_test)].values,
    'water_true': y_test.values,
    'water_pred': y_pred_test,
    'error': residuals,
    'error_pct': (residuals / (y_test.values + 1e-6)) * 100,
    'is_anomaly': anomalies
})
results_df.to_csv(os.path.join(OUTPUT_DIR, 'water_predictions_optimized.csv'), index=False)
print("  ✓ water_predictions_optimized.csv")

# Save full predictions
full_pred_df = pd.DataFrame({
    'timestamp': df_features['timestamp'],
    'water_true': df_features['water'].values,
    'water_pred': model_water.predict(X)
})
full_pred_df.to_csv(os.path.join(OUTPUT_DIR, 'water_full_predictions_optimized.csv'), index=False)
print("  ✓ water_full_predictions_optimized.csv")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ WATER MODEL OPTIMIZATION COMPLETE")
print("=" * 80)

print(f"\n🎯 FINAL WATER MODEL R² SCORE: {r2_test:.4f}")
if r2_test > 0.85:
    print("   🎉 EXCELLENT - 85%+ accuracy achieved!")
elif r2_test > 0.70:
    print("   ✓ GOOD - 70%+ accuracy achieved!")
else:
    print("   ⚠️  Needs further tuning")

print(f"\n📊 Results saved:")
print(f"   - water_predictions_optimized.csv ({len(results_df)} test samples)")
print(f"   - water_full_predictions_optimized.csv ({len(full_pred_df)} all samples)")
print(f"   - model_water_optimized.pkl")

print("\n" + "=" * 80 + "\n")
