"""
URIS (Urban Resource Intelligence System) - Smart City ML Pipeline (OPTIMIZED)
Complete end-to-end pipeline with FAST execution by aggregating data to hourly level

✓ Step 1: Load Data - Load all CSV files
✓ Step 2: Data Cleaning - Handle missing values, duplicates
✓ Step 3: Merge Data - Combine all datasets on timestamp (hourly aggregation)
✓ Step 4: Feature Engineering - Time-based, lag, and rolling features
✓ Step 5: Create Waste Feature - Synthetic garbage column
✓ Step 6: Define Features and Targets - Multi-output setup
✓ Step 7: Train-Test Split - 80/20 time-based split
✓ Step 8: Model Training - XGBoost multi-output
✓ Step 9: Evaluation - R², RMSE, MAE metrics
✓ Step 10: Anomaly Detection - Isolation Forest
✓ Step 11: Output - Predictions, anomalies, CSV export
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
OUTPUT_DIR = str(BASE_DIR)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==================== STEP 1: LOAD DATA ====================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

def find_timestamp_column(df):
    """Flexibly find timestamp-like columns"""
    candidates = ['timestamp', 'date', 'datetime', 'time', 'created_at', 'date_time']
    for col in candidates:
        if col.lower() in [c.lower() for c in df.columns]:
            return [c for c in df.columns if c.lower() == col.lower()][0]
    return None

def load_csv_file(filepath, name):
    """Load CSV with error handling"""
    try:
        df = pd.read_csv(filepath)
        print(f"OK: Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"FAIL: {name}: {str(e)}")
        return None

# Load all CSV files
electricity_df = load_csv_file(os.path.join(DATA_DIR, "building_consumption.csv"), "Electricity")
water_df = load_csv_file(os.path.join(DATA_DIR, "water_consumption.csv"), "Water")
weather_df = load_csv_file(os.path.join(DATA_DIR, "weather_data.csv"), "Weather")
calendar_df = load_csv_file(os.path.join(DATA_DIR, "calender.csv"), "Calendar")

if any([electricity_df is None, water_df is None, weather_df is None, calendar_df is None]):
    print("FAIL: Missing critical files. Exiting.")
    exit(1)

print(f"\nData loaded: Electricity {electricity_df.shape} | Water {water_df.shape} | "
      f"Weather {weather_df.shape} | Calendar {calendar_df.shape}\n")

# ==================== STEP 2: DATA CLEANING & HOURLY AGGREGATION ====================
print("=" * 80)
print("STEP 2: DATA CLEANING & HOURLY AGGREGATION")
print("=" * 80)

# Convert timestamps
elec_ts_col = find_timestamp_column(electricity_df)
if elec_ts_col:
    electricity_df[elec_ts_col] = pd.to_datetime(electricity_df[elec_ts_col], errors='coerce')
    electricity_df = electricity_df.rename(columns={elec_ts_col: 'timestamp'})

water_ts_col = find_timestamp_column(water_df)
if water_ts_col:
    water_df[water_ts_col] = pd.to_datetime(water_df[water_ts_col], errors='coerce')
    water_df = water_df.rename(columns={water_ts_col: 'timestamp'})

weather_ts_col = find_timestamp_column(weather_df)
if weather_ts_col:
    weather_df[weather_ts_col] = pd.to_datetime(weather_df[weather_ts_col], errors='coerce')
    weather_df = weather_df.rename(columns={weather_ts_col: 'timestamp'})

# Remove missing values
print("Removing NaN values...")
electricity_df = electricity_df.dropna()
water_df = water_df.dropna()
weather_df = weather_df.dropna()

# Remove duplicates
electricity_df = electricity_df.drop_duplicates()
water_df = water_df.drop_duplicates()
weather_df = weather_df.drop_duplicates()

# Aggregate to HOURLY level for faster processing
print("Aggregating to hourly level...")
electricity_hourly = electricity_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg(
    {'consumption': 'mean', 'campus_id': 'first'}
).reset_index()
electricity_hourly.columns = ['timestamp', 'electricity', 'campus_id']
electricity_hourly = electricity_hourly.dropna()

water_hourly = water_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg(
    {'consumption': 'mean', 'campus_id': 'first'}
).reset_index()
water_hourly.columns = ['timestamp', 'water', 'campus_id']
water_hourly = water_hourly.dropna()

weather_hourly = weather_df.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
    col: 'mean' for col in weather_df.columns if col != 'timestamp' and col not in ['campus_id', 'meter_id']
}).reset_index()
weather_hourly['timestamp'] = weather_df.groupby(pd.Grouper(key='timestamp', freq='H'))['timestamp'].first().values
weather_hourly = weather_hourly.dropna()

print(f"OK: Electricity hourly: {electricity_hourly.shape[0]} records")
print(f"OK: Water hourly: {water_hourly.shape[0]} records")
print(f"OK: Weather hourly: {weather_hourly.shape[0]} records")

# ==================== STEP 3: MERGE DATA ====================
print("\n" + "=" * 80)
print("STEP 3: MERGING DATA")
print("=" * 80)

# Merge electricity + water
df_merged = pd.merge(electricity_hourly, water_hourly, on='timestamp', how='inner')
print(f"OK: Electricity + Water merged: {df_merged.shape[0]} rows")

# Merge with weather
df_merged = pd.merge(df_merged, weather_hourly, on='timestamp', how='left')
print(f"OK: Added Weather: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")

# Merge with calendar
df_merged['date'] = df_merged['timestamp'].dt.date
calendar_df['date'] = pd.to_datetime(calendar_df['date']).dt.date
calendar_df = calendar_df.drop_duplicates(subset=['date'])
df_merged = pd.merge(df_merged, calendar_df[['date', 'is_holiday', 'is_semester', 'is_exam']], 
                     on='date', how='left')
print(f"OK: Added Calendar: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")

# Sort by timestamp
df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
print(f"\nOK: Final merged: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")
print(f"Date range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")

# ==================== STEP 4: FEATURE ENGINEERING ====================
print("\n" + "=" * 80)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 80)

df_features = df_merged.copy()

# Time-based features
df_features['hour'] = df_features['timestamp'].dt.hour
df_features['day'] = df_features['timestamp'].dt.day
df_features['month'] = df_features['timestamp'].dt.month
df_features['weekday'] = df_features['timestamp'].dt.weekday

# Lag features for electricity
df_features['electricity_lag1'] = df_features['electricity'].shift(1)
df_features['electricity_lag24'] = df_features['electricity'].shift(24)

# ** WATER-SPECIFIC LAG FEATURES (Critical for water predictions) **
# Water usage is highly repeatable - same time each day/week has similar consumption
df_features['water_lag1h'] = df_features['water'].shift(1)        # 1 hour ago
df_features['water_lag24h'] = df_features['water'].shift(24)      # Same time yesterday (strong pattern)
df_features['water_lag168h'] = df_features['water'].shift(168)    # Same time last week
df_features['water_rolling7d'] = df_features['water'].rolling(window=7*24, min_periods=1).mean()  # 7-day avg

# Fill NaN from water lags with group mean by hour
for col in ['water_lag1h', 'water_lag24h', 'water_lag168h']:
    df_features[col] = df_features[col].fillna(df_features.groupby('hour')['water'].transform('mean'))

df_features['water_rolling7d'] = df_features['water_rolling7d'].fillna(df_features['water'].mean())

# Rolling features for electricity
df_features['electricity_rolling24'] = df_features['electricity'].rolling(window=24, min_periods=1).mean()

# Fill NaN from electricity lags
df_features['electricity_lag1'] = df_features['electricity_lag1'].fillna(df_features['electricity'].mean())
df_features['electricity_lag24'] = df_features['electricity_lag24'].fillna(df_features['electricity'].mean())

print("OK: Time-based, lag, and rolling features created")

# ==================== STEP 5: SYNTHETIC GARBAGE FEATURE (IMPROVED) ====================
print("\n" + "=" * 80)
print("STEP 5: SYNTHETIC WASTE/GARBAGE FEATURE (ENHANCED)")
print("=" * 80)

# ✅ MORE REALISTIC GARBAGE GENERATION
# Based on real campus/building patterns:
# - Higher garbage during business hours (8-17)
# - Peak times: lunch (12-13), post-work (17-18)
# - Lower on weekends
# - Correlated with electricity (facility activity)
# - Water impact (more people = more water + trash)
# - Temperature effect (summer: more bottles, outdoor waste)

# Hourly business pattern (higher during day)
hourly_factor = df_features['hour'].apply(lambda h: 
    2.0 if 11 <= h <= 13 else      # Lunch peak: 2x factor
    1.8 if 8 <= h <= 18 else       # Business hours: 1.8x
    1.2 if 18 <= h <= 20 else      # Evening: 1.2x
    0.6                             # Night: 0.6x
)

# Weekday vs weekend factor
weekday_factor = df_features['weekday'].apply(lambda d:
    1.5 if d < 5 else              # Weekday: 1.5x (Mon-Fri)
    0.4                             # Weekend: 0.4x (Sat-Sun)
)

# Seasonal/monthly factor (more outdoor garbage in warmer months)
seasonal_factor = df_features['month'].apply(lambda m:
    1.2 if m in [4, 5, 6, 7, 8, 9] else  # Warmer months: 1.2x
    0.9                                   # Cooler months: 0.9x
)

# Day of month effect (higher at month end - billing/accounting)
day_factor = df_features['day'].apply(lambda d:
    1.1 if 25 <= d <= 31 else      # Month-end: 1.1x
    1.0                             # Normal: 1.0x
)

# Holiday effect (much less garbage on holidays)
holiday_factor = df_features['is_holiday'].apply(lambda h:
    0.5 if h == 1 else             # Holiday: 0.5x
    1.0                             # Normal: 1.0x
)

# Exam period effect (more activity, more waste)
exam_factor = df_features['is_exam'].apply(lambda e:
    1.3 if e == 1 else             # Exam period: 1.3x
    1.0                             # Normal: 1.0x
)

# Base garbage correlated with building activity
# Electricity is key indicator of building usage
base_garbage = (
    0.5 * df_features['electricity'] +      # 50% correlation with electricity
    0.2 * df_features['water']              # 20% correlation with water
) + 30  # Base waste level

# Apply all factors with realistic noise
noise = np.random.normal(0, 5, len(df_features))  # Reduced noise

garbage_multiplier = (hourly_factor * weekday_factor * seasonal_factor * 
                     day_factor * holiday_factor * exam_factor)

df_features['garbage'] = (base_garbage * garbage_multiplier + noise).clip(lower=0)

# Normalize garbage to realistic range (kg/hour for campus)
df_features['garbage'] = df_features['garbage'].clip(lower=25, upper=300)

print(f"OK: Garbage feature generated (enhanced)")
print(f"  Mean: {df_features['garbage'].mean():.2f} kg/hr")
print(f"  Min: {df_features['garbage'].min():.2f} kg/hr")
print(f"  Max: {df_features['garbage'].max():.2f} kg/hr")
print(f"  Std: {df_features['garbage'].std():.2f} kg/hr")

# ==================== STEP 6: DEFINE FEATURES & TARGETS ====================
print("\n" + "=" * 80)
print("STEP 6: FEATURE & TARGET DEFINITION")
print("=" * 80)

feature_cols = ['hour', 'day', 'month', 'weekday', 'electricity_lag1', 
                'electricity_lag24', 'electricity_rolling24', 'is_holiday', 
                'is_semester', 'is_exam']

weather_features = [col for col in df_features.columns 
                   if col not in ['timestamp', 'date', 'electricity', 'water', 'garbage',
                                 'hour', 'day', 'month', 'weekday', 'electricity_lag1',
                                 'electricity_lag24', 'electricity_rolling24', 'campus_id']]
feature_cols.extend(weather_features)
feature_cols = [col for col in feature_cols if col in df_features.columns]
# Remove duplicates and preserve order
seen = set()
feature_cols_unique = []
for col in feature_cols:
    if col not in seen:
        feature_cols_unique.append(col)
        seen.add(col)
feature_cols = feature_cols_unique

target_cols = ['electricity', 'water', 'garbage']

X = df_features[feature_cols].copy()
y = df_features[target_cols].copy()

# Remove NaN rows
combined = pd.concat([X, y], axis=1)
initial_rows = len(combined)
combined = combined.dropna()
rows_dropped = initial_rows - len(combined)

X = combined[feature_cols].copy()
y = combined[target_cols].copy()

print(f"OK: Features ({len(feature_cols)}), Targets ({len(target_cols)})")
print(f"OK: Rows after NaN removal: {len(X)} (dropped {rows_dropped})")

# ==================== STEP 7: TRAIN-TEST SPLIT ====================
print("\n" + "=" * 80)
print("STEP 7: TRAIN-TEST SPLIT")
print("=" * 80)

split_idx = int(len(X) * 0.8)
X_train = X[:split_idx].copy()
X_test = X[split_idx:].copy()
y_train = y[:split_idx].copy()
y_test = y[split_idx:].copy()

print(f"OK: Total samples: {len(X)}")
print(f"OK: Training: {len(X_train)} (80%), Test: {len(X_test)} (20%)")

# ==================== STEP 8: MODEL TRAINING (IMPROVED) ====================
print("\n" + "=" * 80)
print("STEP 8: MODEL TRAINING (XGBoost with Garbage Tuning)")
print("=" * 80)

# Main model for electricity and water (standard params)
print("Training XGBoost for: electricity, water")

# ✅ ELECTRICITY-ONLY MODEL (keep as standard - performs well)
xgb_model_electricity = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1
)
xgb_model_electricity.fit(X_train, y_train['electricity'])

# ✅ SPECIALIZED WATER MODEL (improved with lag features & heavy regularization)
print("Training XGBoost for: water (with specialized tuning & lag features)")
xgb_model_water = XGBRegressor(
    n_estimators=80,            # Fewer estimators (sparse data)
    max_depth=4,                # SHALLOW tree (prevent overfitting on 810 records)
    learning_rate=0.05,         # Lower LR for stability
    subsample=0.9,              # 90% of samples per tree
    colsample_bytree=0.95,      # 95% of features per tree
    subsample_freq=1,
    reg_alpha=2.0,              # STRONG L1 regularization
    reg_lambda=3.0,             # STRONG L2 regularization
    gamma=1.0,                  # Min loss reduction
    min_child_weight=10,        # High min weight (prevent small splits)
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1
)

# Add water-specific interaction features
X_water_train = X_train.copy()
X_water_train['water_time_pattern'] = X_water_train['hour'].apply(lambda h: 
    1.3 if 6 <= h <= 9 else        # Morning peak (6-9am)
    1.1 if 11 <= h <= 14 else      # Midday (11am-2pm)
    1.0 if 15 <= h <= 18 else      # Afternoon
    0.7                             # Night/Evening
)
X_water_train['is_weekday'] = X_water_train['weekday'].apply(lambda d: 1 if d < 5 else 0)
X_water_train['water_activity_score'] = (
    X_water_train['water_rolling7d'] / (X_water_train['water_rolling7d'].max() + 1)
)

X_water_test = X_test.copy()
X_water_test['water_time_pattern'] = X_water_test['hour'].apply(lambda h: 
    1.3 if 6 <= h <= 9 else
    1.1 if 11 <= h <= 14 else
    1.0 if 15 <= h <= 18 else
    0.7
)
X_water_test['is_weekday'] = X_water_test['weekday'].apply(lambda d: 1 if d < 5 else 0)
X_water_test['water_activity_score'] = (
    X_water_test['water_rolling7d'] / (X_water_test['water_rolling7d'].max() + 1)
)

xgb_model_water.fit(X_water_train, y_train['water'])

# Separate garbage model with optimized hyperparams
print("Training XGBoost for: garbage (with specialized tuning)")
xgb_model_garbage = XGBRegressor(
    n_estimators=150,           # More boosting rounds for complex pattern
    max_depth=8,                # Deeper tree for garbage patterns
    learning_rate=0.05,         # Lower LR for stability
    subsample=0.85,             # 85% of samples per tree
    colsample_bytree=0.9,       # 90% of features per tree
    subsample_freq=1,           # Apply subsample every iteration
    reg_alpha=0.5,              # L1 regularization
    reg_lambda=2.0,             # L2 regularization
    gamma=2.0,                  # Min loss reduction
    min_child_weight=5,         # Min weight in child node
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1
)

# Add garbage-specific features to help model
X_garbage = X_train.copy()
# Add temporal features specifically for garbage patterns
X_garbage['is_business_hours'] = X_garbage['hour'].apply(lambda h: 1 if 8 <= h <= 18 else 0)
X_garbage['is_peak_lunch'] = X_garbage['hour'].apply(lambda h: 1 if 11 <= h <= 13 else 0)
X_garbage['weekday_score'] = X_garbage['weekday'].apply(lambda d: 1.5 if d < 5 else 0.4)

X_test_garbage = X_test.copy()
X_test_garbage['is_business_hours'] = X_test_garbage['hour'].apply(lambda h: 1 if 8 <= h <= 18 else 0)
X_test_garbage['is_peak_lunch'] = X_test_garbage['hour'].apply(lambda h: 1 if 11 <= h <= 13 else 0)
X_test_garbage['weekday_score'] = X_test_garbage['weekday'].apply(lambda d: 1.5 if d < 5 else 0.4)

y_train_garbage = y_train['garbage'].copy()
y_test_garbage = y_test['garbage'].copy()

xgb_model_garbage.fit(X_garbage, y_train_garbage)
print("OK: Model training completed")

# ==================== STEP 9: EVALUATION ====================
print("\n" + "=" * 80)
print("STEP 9: MODEL EVALUATION")
print("=" * 80)

# Predictions - now with separate models
y_train_pred_electricity = xgb_model_electricity.predict(X_train)
y_test_pred_electricity = xgb_model_electricity.predict(X_test)

y_train_pred_water = xgb_model_water.predict(X_water_train)
y_test_pred_water = xgb_model_water.predict(X_water_test)

y_train_pred_garbage = xgb_model_garbage.predict(X_garbage)
y_test_pred_garbage = xgb_model_garbage.predict(X_test_garbage)

# Combine predictions
y_train_pred = np.column_stack([y_train_pred_electricity, y_train_pred_water, y_train_pred_garbage])
y_test_pred = np.column_stack([y_test_pred_electricity, y_test_pred_water, y_test_pred_garbage])

metrics = {}
for idx, target in enumerate(target_cols):
    r2 = r2_score(y_test[target], y_test_pred[:, idx])
    rmse = np.sqrt(mean_squared_error(y_test[target], y_test_pred[:, idx]))
    mae = mean_absolute_error(y_test[target], y_test_pred[:, idx])
    mape = np.mean(np.abs((y_test[target] - y_test_pred[:, idx]) / y_test[target])) * 100
    
    metrics[target] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    print(f"\n{target.upper()}:")
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

avg_r2 = np.mean([metrics[t]['R2'] for t in target_cols])
print(f"\nAverage R2 Score: {avg_r2:.4f}")

# ==================== STEP 10: ANOMALY DETECTION ====================
print("\n" + "=" * 80)
print("STEP 10: ANOMALY DETECTION")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
anomalies = iso_forest.fit_predict(X_scaled)

# ==================== STEP 11: CREATE RESULTS DATAFRAME & PREDICTIONS ====================
print("\n" + "=" * 80)
print("STEP 11: CREATING RESULTS WITH ALL PREDICTIONS")
print("=" * 80)

# Create results with predictions
# Full predictions for all data - now using specialized models
X_full = X.copy()

# Initialize results dataframe
results_df = X_full.copy()
results_df['electricity_true'] = y['electricity'].values
results_df['water_true'] = y['water'].values
results_df['garbage_true'] = y['garbage'].values

# Electricity predictions (all data)
electricity_pred_all = xgb_model_electricity.predict(X_full)
results_df['electricity_pred'] = electricity_pred_all

# Water predictions (all data) - with water-specific features
X_full_water = X_full.copy()
X_full_water['water_time_pattern'] = X_full_water['hour'].apply(lambda h: 
    1.3 if 6 <= h <= 9 else
    1.1 if 11 <= h <= 14 else
    1.0 if 15 <= h <= 18 else
    0.7
)
X_full_water['is_weekday'] = X_full_water['weekday'].apply(lambda d: 1 if d < 5 else 0)
X_full_water['water_activity_score'] = (
    X_full_water['water_rolling7d'] / (X_full_water['water_rolling7d'].max() + 1)
)
water_pred_all = xgb_model_water.predict(X_full_water)
results_df['water_pred'] = water_pred_all

# Generate garbage predictions using specialized garbage model for ALL data
X_full_garbage = X_full.copy()
X_full_garbage['is_business_hours'] = X_full_garbage['hour'].apply(lambda h: 1 if 8 <= h <= 18 else 0)
X_full_garbage['is_peak_lunch'] = X_full_garbage['hour'].apply(lambda h: 1 if 11 <= h <= 13 else 0)
X_full_garbage['weekday_score'] = X_full_garbage['weekday'].apply(lambda d: 1.5 if d < 5 else 0.4)

garbage_pred_all = xgb_model_garbage.predict(X_full_garbage)
results_df['garbage_pred'] = garbage_pred_all

results_df['is_anomaly'] = (anomalies == -1).astype(int)

# ==================== COST CALCULATION (INDIAN RUPEES) ====================
# Realistic tariff rates for India
ELECTRICITY_RATE = 8.0  # ₹ per kWh (institutional/commercial rate)
WATER_RATE = 45.0  # ₹ per kL (municipal rate)
GARBAGE_RATE = 75.0  # ₹ per unit (waste management charge)

# Calculate actual costs
results_df['electricity_cost_actual'] = results_df['electricity_true'] * ELECTRICITY_RATE
results_df['water_cost_actual'] = results_df['water_true'] * WATER_RATE
results_df['garbage_cost_actual'] = results_df['garbage_true'] * GARBAGE_RATE
results_df['total_cost_actual'] = (results_df['electricity_cost_actual'] + 
                                   results_df['water_cost_actual'] + 
                                   results_df['garbage_cost_actual'])

# Calculate predicted costs
results_df['electricity_cost_pred'] = results_df['electricity_pred'] * ELECTRICITY_RATE
results_df['water_cost_pred'] = results_df['water_pred'] * WATER_RATE
results_df['garbage_cost_pred'] = results_df['garbage_pred'] * GARBAGE_RATE
results_df['total_cost_pred'] = (results_df['electricity_cost_pred'] + 
                                 results_df['water_cost_pred'] + 
                                 results_df['garbage_cost_pred'])

# Cost difference (savings opportunity)
results_df['cost_difference'] = results_df['total_cost_actual'] - results_df['total_cost_pred']

anomaly_count = results_df['is_anomaly'].sum()
anomaly_pct = (anomaly_count / len(results_df)) * 100

print(f"OK: Anomalies detected: {anomaly_count} ({anomaly_pct:.2f}%)")
print(f"OK: Normal records: {len(results_df) - anomaly_count} ({100 - anomaly_pct:.2f}%)")

# ==================== STEP 12: OUTPUT ====================
print("\n" + "=" * 80)
print("STEP 12: SAVING OUTPUTS")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save results
results_output_path = os.path.join(OUTPUT_DIR, "uris_results_fast.csv")
results_df.to_csv(results_output_path, index=False)
print(f"OK: Saved results to {results_output_path}")

# Save sample predictions
sample_df = results_df[split_idx:].head(20)
print(f"\nSample predictions (first 20 test records):")
print(sample_df[['electricity_true', 'electricity_pred', 'water_true', 'water_pred', 'is_anomaly']].to_string())

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"Total records processed: {len(results_df)}")
print(f"Output file: {results_output_path}")
