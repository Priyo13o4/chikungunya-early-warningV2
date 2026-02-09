# Changelog: v6 (Priyo's Thesis-Ready Version)

**Date**: February 5, 2026  
**Source**: Copied from Vishnu-Version-Hist/v5  
**Author**: Priyodip (Researcher)

---

## Purpose of v6

This version applies critical fixes identified during thesis preparation that address:
1. **Config parameter alignment** - Code now respects config settings
2. **Data sparsity root cause** - Imputation instead of aggressive dropna
3. **Methodological rigor** - Aligned percentiles, documented assumptions

---

## Critical Fixes Applied

### 1. **Config-Driven Case Imputation** ✅
**File**: `src/data/loader.py`

**Issue**: Code hardcoded `.fillna(0)` ignoring config `imputation.cases: forward_fill`

**Fix**: 
- Added `imputation_strategy` parameter to `load_epiclim()`
- Supports: `zero_fill`, `forward_fill`, `drop`
- Config updated to `zero_fill` (thesis-defensible for surveillance data)

**Impact**: Clear documentation of missing data handling for thesis methods section

---

### 2. **Percentile Alignment** ✅
**Files**: 
- `src/evaluation/lead_time.py` 
- `config/config_default.yaml`

**Issue**: Training used p75 for labels, but evaluation used p80 for outbreak detection
- This created ~30% fewer detected outbreaks than expected
- Artificially inflated false positives

**Fix**: 
- Changed `DEFAULT_OUTBREAK_PERCENTILE` from **80 → 75**
- Now aligned with `config: outbreak_percentile: 75`

**Expected Impact**: 
- 2-3x more outbreak episodes detected
- More balanced sensitivity/specificity metrics
- Defensible claim: "consistent definition throughout pipeline"

---

### 3. **Imputation Over Dropna** ✅
**File**: `experiments/06_analyze_lead_time.py`

**Issue**: `dropna(subset=feature_cols + [target_col])` removed 94% of data
- Started with 794 raw records → only 44 valid test samples
- High missingness from climate lags, rolling windows

**Fix**:
```python
# OLD (REMOVED):
train_valid = train_df.dropna(subset=feature_cols + [target_col])

# NEW (APPLIED):
train_imputed = impute_missing_features(train_df, feature_cols)
train_valid = train_imputed.dropna(subset=[target_col])  # Only drop if label missing
```

**Expected Impact**:
- Recover ~90% of previously dropped samples
- Increase valid test samples from ~44 → ~400+
- Increase outbreak episodes from 4 → likely 15-25

---

### 4. **Config Documentation** ✅
**File**: `config/config_default.yaml`

**Changes**:
- `imputation.cases: zero_fill` (with justification comment)
- `outbreak_percentile: 75` (documented alignment requirement)
- `degree_day_threshold: 20.0` (added TODO for sensitivity analysis 18/20/22°C)

---

## What Needs Regeneration

After these fixes, you MUST regenerate:

### Data Pipeline (in order):
1. ✅ Config: Already updated
2. ⚠️ `data/processed/panel_chikungunya_v01.parquet` (if using new imputation)
3. ⚠️ `data/processed/features_engineered_v01.parquet`
4. ⚠️ `data/processed/samples_supervised_v01.parquet`

### Model Outputs:
5. ⚠️ All trained models (Bayesian, XGBoost, RF, etc.)
6. ⚠️ `results/metrics/baseline_comparison.json`
7. ⚠️ `results/metrics/bayesian_cv_results.json`

### Phase 7 Analysis:
8. ⚠️ `results/analysis/lead_time_detail_all_folds.csv`
9. ⚠️ `results/analysis/lead_time_summary_overall.csv`
10. ⚠️ All Phase 7 figures in `results/figures/phase7/`

---

## Outstanding Issues (Placeholder Figures)

These were NOT fixed in v6 (require additional implementation):

### A. `case_study_timeseries_placeholder.png`
**Status**: Still placeholder  
**Required**: Create `experiments/08_case_study_plots.py`
- Save district-level predictions during CV
- Plot: cases + outbreak threshold + model warnings

### B. `calibration_curves_placeholder.png`
**Status**: Still placeholder  
**Required**: Modify `experiments/06_analyze_lead_time.py`
- Store `(y_prob, y_true)` pairs per model
- Compute reliability diagram (10 bins)

### C. `decision_state_timeline.png`
**Status**: Illustrative simulation  
**Required**: Implement decision layer per `docs/Version 2/04_decision_layer_minimal_spec_v2.md`
- Define GREEN/YELLOW/RED states
- Apply thresholds to real predictions

---

## Files Modified in v6

```
versions/Priyo-Version-Hist/v6/
├── config/
│   └── config_default.yaml          [MODIFIED - imputation, docs]
├── src/
│   ├── data/
│   │   └── loader.py                [MODIFIED - config-driven imputation]
│   └── evaluation/
│       └── lead_time.py             [MODIFIED - p80 → p75 alignment]
└── experiments/
    └── 06_analyze_lead_time.py      [MODIFIED - imputation over dropna]
```

---

## Validation Checklist

Before running the full pipeline:

- [ ] Verify `impute_missing_features()` function exists in `experiments/06_analyze_lead_time.py` (lines ~260-320)
- [ ] Check all scripts that call `load_epiclim()` pass `imputation_strategy` parameter
- [ ] Confirm `config/config_default.yaml` is being loaded (not using root config)
- [ ] Run sanity check: `python experiments/00_sanity_check.py`
- [ ] Test single fold: `python experiments/06_analyze_lead_time.py --test-years 2019`

---

## Expected Results After Regeneration

| Metric | Before v6 | After v6 (Expected) |
|--------|-----------|---------------------|
| Valid test samples (6 folds) | 44 | ~400-500 |
| Outbreak episodes detected | 4 | ~15-25 |
| Successful CV folds | 3 of 6 | 5-6 of 6 |
| Lead-time plot density | Sparse (1-3 points) | Dense (15-25 points) |
| Percentile alignment | ❌ Mismatched (p75 vs p80) | ✅ Aligned (p75) |
| Config respected | ❌ Hardcoded values | ✅ Config-driven |

---

## Thesis Implications

### Methods Section Updates Required:
1. **Data Imputation**: Document zero-fill for missing cases (Section 3.2)
2. **Outbreak Definition**: Clarify p75 = "elevated transmission risk" not "severe epidemic" (Section 3.3)
3. **Sensitivity Analysis**: Plan robustness checks for:
   - Degree-day threshold: 18°C, 20°C, 22°C
   - Outbreak percentile: p75, p85, p90
   - Probability threshold: ROC-optimal vs fixed 0.5

### Results Section Improvements:
1. More outbreak episodes → statistically valid lead-time comparisons
2. Fold-level variability analysis (now possible with 5-6 successful folds)
3. Calibration curves (once implemented)

### Discussion Points:
1. Impact of imputation strategy on early warning (compare zero-fill vs forward-fill)
2. Sparsity was pipeline artifact, not epidemiological reality
3. Importance of config-code alignment for reproducibility

---

## Next Steps

1. **Immediate**: Run `experiments/00_sanity_check.py` to verify v6 integrity
2. **Short-term**: Regenerate data pipeline (experiments 01-02)
3. **Medium-term**: Re-run Phase 7 analysis (experiments 06-07)
4. **Long-term**: Implement placeholder figures (experiments 08+)

---

## Version History

- **v5** (Vishnu): Original implementation, hardcoded parameters, aggressive dropna
- **v6** (Priyo): Config-driven, aligned percentiles, imputation-based, thesis-ready

---

## Contact

For questions about v6 changes: Priyodip (Researcher)  
For original implementation: Vishnu-Version-Hist/v5
