"""
URIS v4.0 - WATER MODEL (SIMPLE & ROBUST)
🎯 KEY INSIGHT: Less is more! Simpler model + fewer features = better generalization
✓ Use only top predictive features (water lags, electricity correlation)
✓ Light XGBoost (max_depth=3, n_estimators=50)✓ Ridge/Lasso regression as backup
✓ Cross-validation for reliability
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
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
print("🚀 URIS v4.0 - SIMPLE WATER MODEL (ANTI-OVERFITTING)")
print("="*80)

# ==================== LOAD DATA ====================
print("\n[1] LOAD & PREPARE DATA")
print("-" * 80)

electricity_df = pd.read_csv(os.path.join(DATA_DIR, "building_consumption.csv"))
water_df = pd.read_csv(os.path.join(DATA_DIR, "water_consumption.csv"))
weather_cols = ['timestamp', 'air_temperature', 'relative_humidity', 'wind_speed']
weather_df = pd.read_csv(os.path.join(DATA_DIR, "weather_data.csv"), usecols=weather_cols)

electricity_df['timestamp'] = pd.to_datetime(electricity_df['timestamp'], errors='coerce')
water_df['timestamp'] = pd.to_datetime(water_df['timestamp'], errors='coerce')
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')

electricity_df = electricity_df.dropna(subset=['timestamp']).drop_duplicates()
water_df = water_df.dropna(subset=['timestamp']).drop_duplicates()
weather_df = weather_df.dropna(subset=['timestamp']).drop_duplicates()

weather_df.rename(columns={'air_temperature': 'temperature', 'relative_humidity': 'humidity'}, inplace=True)

print(f"✓ Electricity: {len(electricity_df)} records")
print(f"✓ Water: {len(water_df)} records")
print(f"✓ Weather: {len(weather_df)} records")

# ==================== HOURLY AGGREGATION ====================
print("\n[2] HOURLY AGGREGATION")
print("-" * 80)

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

print(f"⚡ Electricity: {len(electricity_hourly)} records")
print(f"💧 Water: {len(water_hourly)} records")
print(f"🌡️  Weather: {len(weather_hourly)} records")

# ==================== MERGE ON WATER DATA ====================
print("\n[3] MERGE DATASETS")
print("-" * 80)

df_elec_weather = pd.merge(electricity_hourly, weather_hourly, on='timestamp', how='left')
df_full = pd.merge(df_elec_weather, water_hourly, on='timestamp', how='inner')
print(f"✓ Merged dataset: {len(df_full)} rows (only with water data)")

df_full = df_full.sort_values('timestamp').reset_index(drop=True)

# ==================== MINIMAL FEATURES ====================
print("\n[4] CREATE MINIMAL FEATURE SET")
print("-" * 80)

df_features = df_full.copy()

# Time features
df_features['hour'] = df_features['timestamp'].dt.hour
df_features['day_of_week'] = df_features['timestamp'].dt.weekday

# Core water features ONLY
print("Building lag features...")
df_features['water_lag1'] = df_features['water'].shift(1)
df_features['water_lag24'] = df_features['water'].shift(24)
df_features['water_rolling24'] = df_features['water'].rolling(24, min_periods=1).mean()

# Electricity as predictor
df_features['electricity_lag0'] = df_features['electricity']
df_features['electricity_lag24'] = df_features['electricity'].shift(24)

# Simple weather
df_features['temperature'] = df_features['temperature']
df_features['humidity'] = df_features['humidity']

# Fill NaN
for col in df_features.columns:
    if df_features[col].dtype in [float, int]:
        df_features[col] = df_features[col].fillna(method='ffill').fillna(method='bfill')
        if df_features[col].isna().sum() > 0:
            df_features[col] = df_features[col].fillna(df_features[col].mean())

# Select ONLY essential features
feature_cols = ['water_lag1', 'water_lag24', 'water_rolling24',
                'electricity_lag0', 'electricity_lag24',
                'temperature', 'humidity', 'hour', 'day_of_week']

X = df_features[feature_cols].copy()
y = df_features['water'].copy()

print(f"✓ Features selected: {len(feature_cols)}")
print(f"  {feature_cols}")
print(f"✓ Samples: {len(X)}")

# ==================== NORMALIZE ====================
print("\n[5] NORMALIZATION")
print("-" * 80)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

print(f"✓ X normalized: mean={X_scaled.mean().mean():.3f}, std={X_scaled.std().mean():.3f}")
print(f"✓ y normalized: mean={y_scaled.mean():.3f}, std={y_scaled.std():.3f}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n[6] TRAIN-TEST SPLIT")
print("-" * 80)

split_idx = int(len(X_scaled) * 0.8)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y_scaled[:split_idx]
y_test = y_scaled[split_idx:]

y_train_orig = y[:split_idx]
y_test_orig = y[split_idx:]

print(f"Train: {len(X_train)} (80%)")
print(f"Test: {len(X_test)} (20%)")

# ==================== MODEL 1: SIMPLE XGBOOST ====================
print("\n[7] TRAIN LIGHTWEIGHT XGBOOST")
print("-" * 80)

model_xgb = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    verbosity=0
)

model_xgb.fit(X_train, y_train)
y_train_xgb = model_xgb.predict(X_train)
y_test_xgb = model_xgb.predict(X_test)

r2_train_xgb = r2_score(y_train, y_train_xgb)
r2_test_xgb = r2_score(y_test, y_test_xgb)

print(f"XGBoost (Normalized):")
print(f"  R² Train: {r2_train_xgb:.4f}")
print(f"  R² Test:  {r2_test_xgb:.4f}")

# Inverse transform for original scale evaluation
y_train_xgb_orig = scaler_y.inverse_transform(y_train_xgb.reshape(-1, 1)).ravel()
y_test_xgb_orig = scaler_y.inverse_transform(y_test_xgb.reshape(-1, 1)).ravel()

r2_test_xgb_orig = r2_score(y_test_orig, y_test_xgb_orig)
mae_xgb = mean_absolute_error(y_test_orig, y_test_xgb_orig)

print(f"XGBoost (Original scale):")
print(f"  R² Test:  {r2_test_xgb_orig:.4f}")
print(f"  MAE:      {mae_xgb:.3f}")

# ==================== MODEL 2: RIDGE REGRESSION ====================
print("\n[8] TRAIN RIDGE REGRESSION (ROBUST BASELINE)")
print("-" * 80)

model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
y_train_ridge = model_ridge.predict(X_train)
y_test_ridge = model_ridge.predict(X_test)

r2_train_ridge = r2_score(y_train, y_train_ridge)
r2_test_ridge = r2_score(y_test, y_test_ridge)

y_train_ridge_orig = scaler_y.inverse_transform(y_train_ridge.reshape(-1, 1)).ravel()
y_test_ridge_orig = scaler_y.inverse_transform(y_test_ridge.reshape(-1, 1)).ravel()

r2_test_ridge_orig = r2_score(y_test_orig, y_test_ridge_orig)
mae_ridge = mean_absolute_error(y_test_orig, y_test_ridge_orig)

print(f"Ridge Regression (Normalized):")
print(f"  R² Train: {r2_train_ridge:.4f}")
print(f"  R² Test:  {r2_test_ridge:.4f}")

print(f"Ridge Regression (Original scale):")
print(f"  R² Test:  {r2_test_ridge_orig:.4f}")
print(f"  MAE:      {mae_ridge:.3f}")

# ==================== CROSS-VALIDATION ====================
print("\n[9] CROSS-VALIDATION (K=5)")
print("-" * 80)

kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# XGBoost CV
cv_scores_xgb = cross_val_score(model_xgb, X_train, y_train, cv=kfold, scoring='r2')
print(f"XGBoost CV Scores: {[f'{s:.3f}' for s in cv_scores_xgb]}")
print(f"  Mean: {cv_scores_xgb.mean():.4f}, Std: {cv_scores_xgb.std():.4f}")

# Ridge CV
cv_scores_ridge = cross_val_score(model_ridge, X_train, y_train, cv=kfold, scoring='r2')
print(f"Ridge CV Scores: {[f'{s:.3f}' for s in cv_scores_ridge]}")
print(f"  Mean: {cv_scores_ridge.mean():.4f}, Std: {cv_scores_ridge.std():.4f}")

# ==================== SELECT BEST MODEL ====================
print("\n[10] MODEL SELECTION")
print("-" * 80)

print(f"\nComparison on Test Set (Original Scale):")
print(f"  XGBoost: R² = {r2_test_xgb_orig:.4f}, MAE = {mae_xgb:.3f}")
print(f"  Ridge:   R² = {r2_test_ridge_orig:.4f}, MAE = {mae_ridge:.3f}")

if r2_test_ridge_orig > r2_test_xgb_orig:
    print(f"\n✓ SELECTED: Ridge Regression (more stable)")
    best_model = model_ridge
    best_predictions = y_test_ridge_orig
    best_r2 = r2_test_ridge_orig
    model_name = "Ridge"
else:
    print(f"\n✓ SELECTED: XGBoost (better accuracy)")
    best_model = model_xgb
    best_predictions = y_test_xgb_orig
    best_r2 = r2_test_xgb_orig
    model_name = "XGBoost"

# ==================== DETAILED EVALUATION ====================
print("\n[11] FINAL EVALUATION")
print("-" * 80)

residuals = np.abs(y_test_orig - best_predictions)
rmse = np.sqrt(mean_squared_error(y_test_orig, best_predictions))
mae = mean_absolute_error(y_test_orig, best_predictions)
mape = np.mean(np.abs((y_test_orig - best_predictions) / (y_test_orig + 1e-6))) * 100

print(f"{model_name} Model Performance:")
print(f"  R² Score:        {best_r2:.4f}")
print(f"  RMSE:            {rmse:.3f}")
print(f"  MAE:             {mae:.3f}")
print(f"  MAPE:            {mape:.2f}%")
print(f"  Mean Error:      {residuals.mean():.3f}")
print(f"  Median Error:    {np.median(residuals):.3f}")
print(f"  Max Error:       {residuals.max():.3f}")

if best_r2 > 0.85:
    print(f"\n🎉 EXCELLENT: {best_r2:.1%} accuracy - GOAL ACHIEVED!")
elif best_r2 > 0.70:
    print(f"\n✓ GOOD: {best_r2:.1%} accuracy")
else:
    print(f"\n⚠️  {best_r2:.1%} accuracy - needs improvement")

# ==================== FEATURE IMPORTANCE ====================
print("\n[12] FEATURE ANALYSIS")
print("-" * 80)

if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Feature Importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
elif hasattr(best_model, 'coef_'):
    coef = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': np.abs(best_model.coef_)
    }).sort_values('coefficient', ascending=False)
    print("Feature Coefficients (absolute value):")
    for _, row in coef.iterrows():
        print(f"  {row['feature']:25s}: {row['coefficient']:.4f}")

# ==================== SAVE RESULTS ====================
print("\n[13] SAVE OUTPUTS")
print("-" * 80)

# Save model
with open(os.path.join(OUTPUT_DIR, 'model_water_final.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ model_water_final.pkl ({model_name})")

# Save scalers
with open(os.path.join(OUTPUT_DIR, 'scaler_water_final.pkl'), 'wb') as f:
    pickle.dump(scaler_y, f)
with open(os.path.join(OUTPUT_DIR, 'scaler_X_final.pkl'), 'wb') as f:
    pickle.dump(scaler_X, f)
print("✓ scalers saved")

# Save predictions
results_df = pd.DataFrame({
    'timestamp': df_full['timestamp'].iloc[split_idx:split_idx+len(X_test)].values,
    'water_actual': y_test_orig.values,
    'water_predicted': best_predictions,
    'error': residuals,
    'error_pct': (residuals / (y_test_orig.values + 1e-6)) * 100
})
results_df.to_csv(os.path.join(OUTPUT_DIR, 'water_predictions_final.csv'), index=False)
print("✓ water_predictions_final.csv")

# All predictions
all_X_scaled = scaler_X.transform(X)
all_pred_scaled = best_model.predict(all_X_scaled)
all_pred_orig = scaler_y.inverse_transform(all_pred_scaled.reshape(-1, 1)).ravel()

full_results_df = pd.DataFrame({
    'timestamp': df_full['timestamp'],
    'water_actual': y.values,
    'water_predicted': all_pred_orig
})
full_results_df.to_csv(os.path.join(OUTPUT_DIR, 'water_full_predictions_final.csv'), index=False)
print("✓ water_full_predictions_final.csv")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ WATER MODEL TRAINING COMPLETE")
print("=" * 80)

print(f"\n🎯 FINAL R² SCORE: {best_r2:.4f}")
print(f"   Goal: 85%+")
if best_r2 > 0.85:
    print(f"   Status: ✓ ACHIEVED!")
else:
    print(f"   Status: {best_r2*100:.1f}% (needs {(0.85-best_r2)*100:.1f}% more)")

print(f"\n📊 Model: {model_name}")
print(f"   Features: {len(feature_cols)}")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

print("\n📁 Output Files:")
print(f"   - water_predictions_final.csv")
print(f"   - water_full_predictions_final.csv")
print(f"   - model_water_final.pkl")

print("\n" + "=" * 80 + "\n")
