# Feature Dataset Comprehensive Inspection Report
**Date**: February 11, 2026  
**Dataset**: `features_engineered_v01.parquet`  
**Total Rows**: 698 | **Labeled Rows**: 74 (10.6%) | **Outbreaks**: 28

---

## Executive Summary

### ✅ Key Findings
1. **Dataset is SMALL** (only 74 labeled rows, 28 outbreaks) → High risk of overfitting
2. **CORE 20 features**: 18/20 present in labeled data with **90-100% completeness**
3. **Mechanistic encoding**: Only 2/5 key features are truly mechanistic (degree_days, rain_persist)
4. **Instability features** (var, growth_rate) show **opposite pattern** to expectation
5. **High missingness** in full dataset (40-90%) but **low in labeled data** (<10%)

---

## 1. Feature Categories (37 total features)

### 📊 Case Features (11)
- **Lags**: feat_cases_lag_1, lag_2, lag_4, lag_8
- **Moving Averages**: feat_cases_ma_4w, ma_8w
- **Instability**: feat_cases_var_4w, feat_cases_growth_rate
- **EWS**: feat_cases_acf_lag1_4w, feat_cases_trend_4w, feat_cases_skew_4w

### 🌡️ Temperature Features (6)
- **Lags**: feat_temp_lag_1, lag_2, lag_4, lag_8
- **Mechanistic**: feat_degree_days_above_20 ✅
- **Anomaly**: feat_temp_anomaly ⚠️

### 🌧️ Rain Features (5)
- **Lags**: feat_rain_lag_1, lag_2, lag_4, lag_8
- **Mechanistic**: feat_rain_persist_4w ✅

### 📅 Seasonal + 🗺️ Spatial Features (7)
- **Seasonal (4)**: feat_week_sin, week_cos, quarter, is_monsoon
- **Spatial (3)**: feat_lat_norm, lon_norm, lat_lon_interact

### ⚠️ EWS Features (5)
- feat_var_spike_ratio, feat_acf_change, feat_trend_accel
- feat_recent_normalized, feat_cases_acf_lag1_4w

### 🌿 LAI Features (4)
- feat_lai, feat_lai_lag_1, lag_2, lag_4

---

## 2. CORE 20 Features Completeness

### ✅ Complete in Labeled Data (100%)
```
feat_week_sin, feat_week_cos              (Seasonal)
feat_lat_norm, feat_lon_norm              (Spatial)
feat_cases_lag_1, lag_2, lag_4            (Case lags)
feat_cases_ma_4w                          (Case MA)
feat_degree_days_above_20                 (Mechanistic temp) ✅
feat_cases_trend_4w                       (Trend)
```

### ⚠️ High Completeness (91-94%)
```
feat_temp_lag_1, lag_2, lag_4             (93.2%)
feat_temp_anomaly                         (93.2%)
feat_rain_lag_1, lag_2, lag_4             (91.9%)
feat_rain_persist_4w                      (91.9%)
```

### ❌ Missing from CORE 20
```
feat_cases_ma_2w                          (Not computed - only ma_4w and ma_8w exist)
feat_cases_acf_4w                         (Named feat_cases_acf_lag1_4w instead)
```

**Verdict**: 18/20 CORE features present with excellent completeness (90-100%) in labeled data.

---

## 3. Sample Outbreak Patterns

### 🚨 Example: Karnataka - Tumkur (2017-W25)
```
Case Features:
  feat_cases_lag_1:          0.41  (low recent cases)
  feat_cases_lag_2:          1.22  (higher 2 weeks ago)
  feat_cases_var_4w:         0.60  (moderate variance)
  feat_cases_growth_rate:    1.00  (cases doubling)

Temperature:
  feat_temp_anomaly:        -2.22  (cooler than normal)
  feat_degree_days_above_20: 191.43 (high mosquito development)
  feat_temp_lag_1:           36.20  (hot)

Rain:
  feat_rain_lag_1:           0.34  (low recent rain)
  feat_rain_persist_4w:      0.70  (moderate cumulative)
```

### 🚨 Example: Tamil Nadu - Dindigul (2018-W10)
```
Case Features:
  feat_cases_lag_1:          1.39  (elevated cases)
  feat_cases_var_4w:         0.95  (HIGH variance)
  feat_cases_growth_rate:   -0.34  (declining)

Temperature:
  feat_temp_anomaly:         3.43  (MUCH warmer than normal)
  feat_degree_days_above_20: 251.83 (VERY HIGH mosquito development)

Rain:
  feat_rain_lag_1:           0.00  (no recent rain)
  feat_rain_persist_4w:      0.08  (minimal cumulative)
```

### Key Observation
- Outbreaks occur with **diverse patterns** (high/low variance, positive/negative growth)
- **feat_degree_days** consistently elevated during outbreaks
- **feat_temp_anomaly** highly variable (-2.22 to +3.43)
- Rain features show **wide range** (0 to 3+ during outbreaks)

---

## 4. 📊 Outbreak vs Non-Outbreak Comparison

### Instability Features (Mean Values)

| Feature               | Outbreak | Non-Outbreak | Ratio  | Interpretation         |
|-----------------------|----------|--------------|--------|------------------------|
| **feat_cases_var_4w** | 0.370    | 0.572        | 0.65x  | ❌ LOWER during outbreaks! |
| **feat_cases_growth_rate** | 0.136 | 0.067     | 2.04x  | ✅ Higher (expected)   |
| **feat_temp_anomaly** | -0.042   | -0.194       | 0.22x  | ⚠️ Less negative (warmer) |

### 🚨 CRITICAL FINDING: feat_cases_var_4w is LOWER during outbreaks!
- Expected: Higher variance during unstable pre-outbreak periods
- Observed: **0.65x LOWER** variance during labeled outbreaks
- Possible explanation: Labels mark **peak outbreak**, not pre-outbreak instability
- **Implication**: Variance may not be useful as direct outbreak predictor

---

## 5. Feature Encoding Mechanisms

### ✅ Strongly Mechanistic (Encode Biological Processes)

#### **feat_degree_days_above_20**
```python
# Formula: sum(max(0, T - 20°C) * 7) over 2 weeks
# Biological basis: Aedes aegypti development threshold = 20°C
# Encodes: Accumulated heat for mosquito maturation
```
- **NOT** raw temperature
- **IS** biologically meaningful threshold-based accumulation
- ✅ Captures mechanism: mosquito development rate

#### **feat_rain_persist_4w**
```python
# Formula: sum(rainfall_mm) over 4 weeks
# Biological basis: Cumulative water → breeding sites
# Encodes: Habitat availability (standing water persistence)
```
- **NOT** current rain snapshot
- **IS** accumulated moisture for breeding
- ✅ Captures mechanism: habitat formation

---

### ⚠️ Semi-Mechanistic (Deviation-Based)

#### **feat_temp_anomaly**
```python
# Formula: current_temp - historical_mean_by_month
# Basis: Warmer-than-normal → accelerated mosquito development
# Encodes: Unexpected thermal conditions
```
- **NOT** raw temperature
- **IS** deviation from expected
- ⚠️ Semi-mechanistic: captures anomaly but not direct biological threshold

---

### ❌ Statistical Signals (NO Biological Encoding)

#### **feat_cases_var_4w**
```python
# Formula: rolling variance of incidence over 4 weeks
# No threshold, no biological interpretation
# Raw statistical measure of instability
```
- ❌ **NOT** mechanistically encoded
- ❌ Does NOT incorporate outbreak logic
- ❌ Just raw variance (could be high due to data noise, not just outbreaks)

#### **feat_cases_growth_rate**
```python
# Formula: (cases_t - cases_t-1) / (cases_t-1 + epsilon)
# Week-over-week change
# No biological threshold for "dangerous" growth
```
- ❌ **NOT** mechanistically encoded
- ❌ Just raw epidemiological signal
- ⚠️ Shows expected pattern (2x higher in outbreaks) but lacks mechanistic encoding

---

## 6. Missing Data Patterns

### Full Dataset (698 rows)
```
High Missingness (40-90%):
  - Case lags (40-85%)     → Due to need for history
  - Temp lags (32-84%)     → Climate data gaps
  - Rain lags (31-84%)     → Climate data gaps
  - EWS features (69-91%)  → Need long baselines
  - LAI features (25-72%)  → Satellite data gaps
```

### Labeled Dataset (74 rows) - MUCH BETTER
```
Low Missingness (<10%):
  - All CORE 20 features: 0-8% missing
  - Only feat_lai: 25.7% missing

Completeness:
  - 18/20 CORE features: 90-100% complete
  - Variable features: 100% complete in labeled rows
```

**Verdict**: Labeled data has excellent completeness. Missingness is NOT a concern for modeling.

---

## 7. Most Informative Features (Based on Inspection)

### 🥇 Top Tier - Strong Mechanistic + Good Patterns
1. **feat_degree_days_above_20**: Consistently elevated in outbreaks, biologically grounded
2. **feat_rain_persist_4w**: Captures breeding site formation
3. **feat_cases_lag_1, lag_2**: Direct recent case history

### 🥈 Second Tier - Good Signals
4. **feat_temp_anomaly**: Shows variation near outbreaks (warmer-than-normal)
5. **feat_cases_growth_rate**: 2x higher in outbreaks (but noisy)
6. **feat_week_sin, week_cos**: Seasonal patterns (monsoon timing)

### 🥉 Third Tier - Uncertain Utility
7. **feat_cases_var_4w**: Opposite pattern to expectation (lower in outbreaks)
8. **feat_temp_lag_1, lag_2, lag_4**: Redundant with degree_days?
9. **feat_rain_lag_1, lag_2**: Captured by rain_persist?

### ❓ Questionable
- **EWS features** (spike_ratio, acf_change): 69-91% missing, unclear patterns
- **LAI features**: 26-72% missing, minimal inspection evidence

---

## 8. Recommendations for Feature Selection

### ✅ MUST INCLUDE (Mechanistic + Complete)
```python
CORE_MECH = [
    'feat_degree_days_above_20',    # Mechanistic temp
    'feat_rain_persist_4w',         # Mechanistic rain
    'feat_cases_lag_1',             # Direct history
    'feat_cases_lag_2',
    'feat_week_sin', 'feat_week_cos',  # Seasonality
    'feat_lat_norm', 'feat_lon_norm'   # Spatial
]
```

### ⚠️ INCLUDE WITH CAUTION
```python
STATISTICAL = [
    'feat_cases_growth_rate',       # Shows 2x pattern but noisy
    'feat_temp_anomaly',            # Semi-mechanistic deviation
    'feat_cases_ma_4w',             # Smoothed history
]
```

### ❌ CONSIDER EXCLUDING
```python
QUESTIONABLE = [
    'feat_cases_var_4w',     # Opposite pattern (lower in outbreaks)
    'feat_var_spike_ratio',  # 91% missing
    'feat_temp_lag_1',       # Redundant with degree_days?
    'feat_rain_lag_1',       # Redundant with rain_persist?
]
```

---

## 9. Critical Limitations

### 🚨 Sample Size
- **Only 74 labeled rows, 28 outbreaks**
- With 37 features → **2 samples per feature** (severe overfitting risk)
- Even with 18 CORE features → **4 samples per feature** (still risky)

### 🚨 Label Timing Issue
- **feat_cases_var_4w** is LOWER during outbreaks
- Suggests labels mark **peak outbreak**, not **pre-outbreak instability**
- EWS features may need to look **before** labeled outbreak week

### 🚨 Mechanistic Encoding Gaps
- Only 2/5 key features are truly mechanistic (degree_days, rain_persist)
- Most features are **statistical signals** (lags, variance, growth)
- **NO threshold-based encoding** for outbreak risk in case features

---

## 10. Answers to Original Questions

### Q1: Which features exist and their patterns?
- **37 features** across 6 categories (case, temp, rain, seasonal, spatial, EWS)
- **18/20 CORE features** present with 90-100% completeness in labeled data
- Patterns are **diverse** (outbreaks occur with high/low variance, positive/negative growth)

### Q2: Are features complete for labeled data?
- **YES**: 90-100% completeness for 18/20 CORE features in labeled data
- Missing data is NOT a concern for modeling (only 0-8% missing in labeled rows)

### Q3: Do instability features vary near outbreaks?
- **feat_cases_growth_rate**: YES (2x higher in outbreaks) ✅
- **feat_cases_var_4w**: NO (actually LOWER in outbreaks) ❌
- **feat_temp_anomaly**: Weak signal (slightly warmer in outbreaks)

### Q4: Are features mechanistically encoded or just raw signals?
- **Strongly mechanistic** (2): degree_days_above_20, rain_persist_4w
- **Semi-mechanistic** (1): temp_anomaly (deviation-based)
- **Statistical signals** (34): All others (lags, variance, growth, etc.)
- **Verdict**: Mostly RAW SIGNALS with **limited mechanistic encoding**

### Q5: Which features seem most informative?
1. **feat_degree_days_above_20** (mechanistic, consistently elevated)
2. **feat_rain_persist_4w** (mechanistic, breeding habitat)
3. **feat_cases_lag_1, lag_2** (direct history)
4. **feat_cases_growth_rate** (2x higher, but noisy)
5. **feat_temp_anomaly** (semi-mechanistic, variable pattern)

**Avoid**: feat_cases_var_4w (opposite pattern), EWS features (high missingness)

---

## Conclusion

The feature dataset reveals:
1. ✅ **Good completeness** for core features in labeled data
2. ❌ **Limited mechanistic encoding** (only 2/5 key features)
3. ⚠️ **Unstable instability features** (variance shows opposite pattern)
4. 🚨 **Very small sample size** (74 labeled, 28 outbreaks) → overfitting risk
5. 🎯 **Best features**: degree_days, rain_persist, case lags

**Next Steps**: 
- Focus on **8-10 CORE mechanistic features** to avoid overfitting
- Consider **re-labeling** to capture pre-outbreak periods (not just peaks)
- Investigate why **variance is lower** during outbreaks (label timing issue?)
