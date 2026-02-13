"""
Inspect features dataset to understand patterns and completeness.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load features
features_path = Path("data/processed/features_engineered_v01.parquet")
print("=" * 80)
print("LOADING FEATURES DATASET")
print("=" * 80)
print(f"Loading: {features_path}")

df = pd.read_parquet(features_path)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Get feature columns
feat_cols = [c for c in df.columns if c.startswith('feat_')]
print(f"Feature columns: {len(feat_cols)}")

# Display column names by category
print("\n" + "=" * 80)
print("FEATURE CATEGORIES")
print("=" * 80)

case_feats = sorted([c for c in feat_cols if 'cases' in c or 'case' in c])
temp_feats = sorted([c for c in feat_cols if 'temp' in c or 'degree' in c])
rain_feats = sorted([c for c in feat_cols if 'rain' in c])
seasonal_feats = sorted([c for c in feat_cols if any(x in c for x in ['week', 'quarter', 'monsoon'])])
spatial_feats = sorted([c for c in feat_cols if any(x in c for x in ['lat', 'lon'])])
ews_feats = sorted([c for c in feat_cols if any(x in c for x in ['spike', 'acf', 'accel', 'normalized'])])
lai_feats = sorted([c for c in feat_cols if 'lai' in c])

print(f"\n📊 CASE FEATURES ({len(case_feats)}):")
for f in case_feats:
    print(f"  - {f}")

print(f"\n🌡️  TEMPERATURE FEATURES ({len(temp_feats)}):")
for f in temp_feats:
    print(f"  - {f}")

print(f"\n🌧️  RAIN FEATURES ({len(rain_feats)}):")
for f in rain_feats:
    print(f"  - {f}")

print(f"\n📅 SEASONAL FEATURES ({len(seasonal_feats)}):")
for f in seasonal_feats:
    print(f"  - {f}")

print(f"\n🗺️  SPATIAL FEATURES ({len(spatial_feats)}):")
for f in spatial_feats:
    print(f"  - {f}")

print(f"\n⚠️  EWS FEATURES ({len(ews_feats)}):")
for f in ews_feats:
    print(f"  - {f}")

if lai_feats:
    print(f"\n🌿 LAI FEATURES ({len(lai_feats)}):")
    for f in lai_feats:
        print(f"  - {f}")

# TASK 1: Display first 20 rows with key features
print("\n" + "=" * 80)
print("TASK 1: SAMPLE DATA (First 20 rows)")
print("=" * 80)

# Select key columns for display
key_cols = ['state', 'district', 'year', 'week', 'label_outbreak']
case_display = ['feat_cases_lag_1', 'feat_cases_lag_2', 'feat_cases_var_4w', 'feat_cases_growth_rate']
temp_display = ['feat_temp_lag_1', 'feat_temp_anomaly', 'feat_degree_days_above_20']
rain_display = ['feat_rain_lag_1', 'feat_rain_lag_2', 'feat_rain_persist_4w']

display_cols = key_cols + case_display + temp_display + rain_display

# Check which columns exist
display_cols = [c for c in display_cols if c in df.columns]

print(f"\nDisplaying columns: {display_cols}")
print("\nSample rows:")
print(df[display_cols].head(20).to_string(index=False))

# TASK 2: Data Completeness
print("\n" + "=" * 80)
print("TASK 2: DATA COMPLETENESS")
print("=" * 80)

# Count labeled rows
if 'label_outbreak' in df.columns:
    labeled_df = df[df['label_outbreak'].notna()]
    print(f"\nRows with labels: {len(labeled_df):,} ({100*len(labeled_df)/len(df):.1f}%)")
    print(f"  - Outbreak (1): {(labeled_df['label_outbreak']==1).sum():,}")
    print(f"  - Non-outbreak (0): {(labeled_df['label_outbreak']==0).sum():,}")
else:
    labeled_df = df
    print("No label_outbreak column found")

# Define CORE 20 features (based on your specified list)
CORE_20 = [
    'feat_week_sin', 'feat_week_cos',  # Seasonal
    'feat_lat_norm', 'feat_lon_norm',  # Spatial
    'feat_cases_lag_1', 'feat_cases_lag_2', 'feat_cases_lag_4',  # Case lags
    'feat_cases_ma_2w', 'feat_cases_ma_4w',  # Case moving averages
    'feat_temp_lag_1', 'feat_temp_lag_2', 'feat_temp_lag_4',  # Temp lags
    'feat_degree_days_above_20',  # Mechanistic temp
    'feat_temp_anomaly',  # Temp anomaly
    'feat_rain_lag_1', 'feat_rain_lag_2', 'feat_rain_lag_4',  # Rain lags
    'feat_rain_persist_4w',  # Mechanistic rain
    'feat_cases_acf_4w',  # ACF
    'feat_cases_trend_4w'  # Trend
]

# Check completeness for CORE features
print(f"\n🎯 CORE 20 FEATURES COMPLETENESS:")
core_exists = [c for c in CORE_20 if c in df.columns]
core_missing = [c for c in CORE_20 if c not in df.columns]

print(f"  Found: {len(core_exists)}/20")
if core_missing:
    print(f"  Missing: {core_missing}")

# For labeled data, check completeness
if len(labeled_df) > 0 and len(core_exists) > 0:
    completeness = {}
    for col in core_exists:
        non_null = labeled_df[col].notna().sum()
        completeness[col] = (non_null, 100 * non_null / len(labeled_df))
    
    print(f"\n  Completeness in labeled data:")
    for col, (count, pct) in sorted(completeness.items(), key=lambda x: x[1][1], reverse=True):
        print(f"    {col:30s}: {count:6,} ({pct:5.1f}%)")

# Check variable features
VARIABLE_FEATS = ['feat_cases_var_4w', 'feat_cases_growth_rate', 'feat_cases_ma_4w']
print(f"\n📈 VARIABLE FEATURES:")
for feat in VARIABLE_FEATS:
    if feat in df.columns:
        non_null = df[feat].notna().sum()
        pct = 100 * non_null / len(df)
        print(f"  {feat:30s}: {non_null:6,} ({pct:5.1f}%)")

# TASK 3: Inspect outbreak patterns
print("\n" + "=" * 80)
print("TASK 3: OUTBREAK PATTERN INSPECTION")
print("=" * 80)

if 'label_outbreak' in df.columns:
    outbreak_df = df[df['label_outbreak'] == 1].copy()
    print(f"\nAnalyzing {len(outbreak_df):,} outbreak rows...")
    
    if len(outbreak_df) >= 5:
        # Sample 5 outbreak rows
        sample_outbreak = outbreak_df.sample(min(5, len(outbreak_df)), random_state=42)
        
        print("\n🚨 SAMPLE OUTBREAK ROWS:")
        inspect_cols = ['state', 'district', 'year', 'week']
        
        # Case features
        case_inspect = [c for c in ['feat_cases_lag_1', 'feat_cases_lag_2', 'feat_cases_var_4w', 'feat_cases_growth_rate'] if c in df.columns]
        
        # Climate features
        temp_inspect = [c for c in ['feat_temp_anomaly', 'feat_degree_days_above_20', 'feat_temp_lag_1'] if c in df.columns]
        rain_inspect = [c for c in ['feat_rain_lag_1', 'feat_rain_persist_4w'] if c in df.columns]
        
        for idx, row in sample_outbreak.iterrows():
            print(f"\n  📍 {row['state']} - {row['district']} (Week {row['year']}-W{row['week']:02d})")
            
            print("    Case features:")
            for col in case_inspect:
                val = row[col]
                print(f"      {col:30s}: {val:.2f}" if pd.notna(val) else f"      {col:30s}: NaN")
            
            print("    Temperature features:")
            for col in temp_inspect:
                val = row[col]
                print(f"      {col:30s}: {val:.2f}" if pd.notna(val) else f"      {col:30s}: NaN")
            
            print("    Rain features:")
            for col in rain_inspect:
                val = row[col]
                print(f"      {col:30s}: {val:.2f}" if pd.notna(val) else f"      {col:30s}: NaN")
        
        # Compare outbreak vs non-outbreak for instability features
        print("\n" + "=" * 80)
        print("📊 OUTBREAK vs NON-OUTBREAK COMPARISON")
        print("=" * 80)
        
        non_outbreak_df = df[df['label_outbreak'] == 0]
        
        instability_feats = [c for c in ['feat_cases_var_4w', 'feat_cases_growth_rate', 'feat_temp_anomaly'] if c in df.columns]
        
        print("\nMean values:")
        print(f"{'Feature':40s} {'Outbreak':>12s} {'Non-Outbreak':>12s} {'Ratio':>8s}")
        print("-" * 80)
        
        for feat in instability_feats:
            outbreak_mean = outbreak_df[feat].mean()
            non_outbreak_mean = non_outbreak_df[feat].mean()
            ratio = outbreak_mean / (non_outbreak_mean + 1e-6)
            
            print(f"{feat:40s} {outbreak_mean:12.3f} {non_outbreak_mean:12.3f} {ratio:8.2f}x")

# TASK 4: Feature encoding mechanisms
print("\n" + "=" * 80)
print("TASK 4: FEATURE ENCODING MECHANISMS")
print("=" * 80)

print("\n🔬 MECHANISTIC ENCODING ANALYSIS:")

print("\n1. feat_degree_days_above_20:")
print("   ✅ MECHANISTIC: Uses biological threshold (20°C)")
print("   - Formula: sum(max(0, T - 20°C)) over 2 weeks")
print("   - Encodes: Aedes aegypti development threshold")
print("   - Not just raw temp, but accumulated heat above biological minimum")

print("\n2. feat_temp_anomaly:")
print("   ⚠️  SEMI-MECHANISTIC: Deviation from historical mean")
print("   - Formula: current_temp - historical_mean_for_month")
print("   - Encodes: Warmer-than-normal conditions (accelerates mosquito development)")
print("   - Not raw temp, but unexpected thermal conditions")

print("\n3. feat_cases_var_4w:")
print("   ⚠️  STATISTICAL: Raw variance over 4-week window")
print("   - Formula: rolling variance of incidence over 4 weeks")
print("   - Does NOT encode outbreak logic directly")
print("   - Captures instability but not biologically interpreted")

print("\n4. feat_rain_persist_4w:")
print("   ✅ MECHANISTIC: Cumulative rainfall (breeding sites)")
print("   - Formula: sum(rainfall) over 4 weeks")
print("   - Encodes: Accumulated water for mosquito breeding")
print("   - Not just current rain, but persistent moisture")

print("\n5. feat_cases_growth_rate:")
print("   ⚠️  STATISTICAL: Week-over-week growth")
print("   - Formula: (cases_t - cases_t-1) / (cases_t-1 + epsilon)")
print("   - Does NOT encode outbreak logic")
print("   - Raw epidemiological signal")

print("\n📋 SUMMARY:")
print("   ✅ Strongly mechanistic: degree_days, rain_persist")
print("   ⚠️  Semi-mechanistic: temp_anomaly (deviation-based)")
print("   ⚠️  Statistical signals: cases_var, growth_rate (no biological encoding)")

# Missing data summary
print("\n" + "=" * 80)
print("MISSING DATA SUMMARY")
print("=" * 80)

print("\nFeatures with >25% missing in full dataset:")
for col in feat_cols:
    missing_pct = 100 * df[col].isna().sum() / len(df)
    if missing_pct > 25:
        print(f"  {col:40s}: {missing_pct:5.1f}%")

if 'label_outbreak' in df.columns:
    print("\nFeatures with >25% missing in LABELED data:")
    for col in feat_cols:
        missing_pct = 100 * labeled_df[col].isna().sum() / len(labeled_df)
        if missing_pct > 25:
            print(f"  {col:40s}: {missing_pct:5.1f}%")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
