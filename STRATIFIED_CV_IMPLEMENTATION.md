# Stratified Temporal CV Implementation - Completion Report

## Task Summary
Implemented stratified temporal cross-validation to fix fold imbalance where single-year folds had 0-4 positives, making sensitivity metrics invalid.

## Problem Addressed
**Before:** Current year-based CV created folds with insufficient positives:
- 2019: 4 positives (need 5)
- 2020: 2 positives
- 2021: 0 positives
- 2022: 0 positives

This caused metrics like sensitivity to be undefined (division by zero) or unreliable.

## Solution Implemented
Created `create_stratified_temporal_folds()` function that:
1. Counts positives per year
2. Merges consecutive years with <5 positives into multi-year test windows
3. Maintains temporal ordering (no future data in training)
4. Skips year groups that still don't meet minimum threshold

## Files Modified

### 1. Core CV Module
**File:** `src/evaluation/cv.py`
- Modified `CVFold` dataclass to support multi-year test windows (added `test_years` field)
- Added `create_stratified_temporal_folds()` function (159 lines)
- Kept `create_rolling_origin_splits()` for backward compatibility

### 2. Module Exports
**File:** `src/evaluation/__init__.py`
- Added `create_stratified_temporal_folds` to imports and `__all__` list

### 3. Experiment Scripts Updated
**Files:**
- `experiments/05_evaluate_bayesian.py` - Main Bayesian CV evaluation
- `experiments/06_analyze_lead_time.py` - Lead time analysis with CV
- `experiments/03_train_baselines.py` - Baseline model training
- `experiments/04_train_bayesian.py` - Single-fold Bayesian test

All now use `create_stratified_temporal_folds()` instead of `create_rolling_origin_splits()`.

### 4. Validation Scripts Created
**Files:**
- `test_stratified_cv.py` - Quick test of stratified CV
- `validate_stratified_cv.py` - Comprehensive validation with checks

## Fold Statistics (After Stratification)

```
          Fold Test Years  Train Samples  Test Samples  Positives  Negatives Positive Rate
     fold_2017       2017             15             8          7          1         87.5%
     fold_2018       2018             23            16          6         10         37.5%
fold_2019_2020  2019-2020             39            23          5         18         21.7%
```

### Summary Metrics
- **Total folds:** 3 (reduced from 6)
- **Total test samples:** 47
- **Total positives:** 18
- **Overall positive rate:** 38.3%
- **Min positives per fold:** 5 ✓
- **Max positives per fold:** 7
- **Years skipped:** 2021-2022 (0 positives combined)

## Validation Results

### ✓ All Checks Passed

1. **Minimum positives check:** ✓ All folds have ≥5 positives
   - fold_2017: 7 positives
   - fold_2018: 6 positives
   - fold_2019_2020: 5 positives

2. **Temporal ordering:** ✓ No future data in training
   - fold_2017: train up to 2016, test starts 2017
   - fold_2018: train up to 2017, test starts 2018
   - fold_2019_2020: train up to 2018, test starts 2019

3. **Data leakage:** ✓ No overlap between train/test sets
   - All folds have disjoint train/test indices

4. **Coverage:** ✓ All viable years covered
   - Covered: 2017, 2018, 2019, 2020
   - Skipped: 2021, 2022 (insufficient positives)

## Key Features

### Stratification Strategy
- **Single-year folds:** Used when year has ≥5 positives (2017, 2018)
- **Multi-year folds:** Consecutive years merged when needed (2019-2020)
- **Skipping:** Years with insufficient positives even after merging (2021-2022)

### Backward Compatibility
- Old `create_rolling_origin_splits()` still available
- Same `CVFold` dataclass structure (added optional `test_years` field)
- Works with existing evaluation code

### Verbose Output
When `verbose=True`, the function prints:
- Positives per year
- Warnings for skipped years
- Complete fold statistics table
- Train/test sample counts
- Positive/negative breakdowns

## Usage Example

```python
from src.evaluation.cv import create_stratified_temporal_folds

folds = create_stratified_temporal_folds(
    df=valid_df,
    target_col='label_outbreak',
    year_col='year',
    min_positives=5,
    candidate_test_years=[2017, 2018, 2019, 2020, 2021, 2022],
    verbose=True
)

# Returns 3 folds with ≥5 positives each
for fold in folds:
    print(f"{fold.fold_name}: {fold.test_years}")
# Output:
# fold_2017: [2017]
# fold_2018: [2018]
# fold_2019_2020: [2019, 2020]
```

## Impact on Evaluation

### Before (6 folds)
- 4 folds with <5 positives (2019, 2020, 2021, 2022)
- Sensitivity metrics undefined or unreliable
- Inflated false negative rates

### After (3 folds)
- All folds with ≥5 positives
- Valid sensitivity metrics
- Reliable performance estimates
- Fewer but more robust folds

## Next Steps

As requested, **Phase 05 has NOT been re-run yet.** The implementation is complete and validated, ready for your audit.

### To Re-run Phase 05:
```bash
cd chikungunya-early-warningV2
rm -rf results/bayesian
python experiments/05_evaluate_bayesian.py
```

### Expected Changes:
- 3 folds instead of 6
- All sensitivity metrics will be valid (no division by zero)
- More reliable aggregated metrics
- Runtime reduced by ~50% (fewer folds)

## Conclusion

✓ **Implementation complete and validated**
✓ **All folds have ≥5 positives**
✓ **Temporal ordering preserved**
✓ **No data leakage**
✓ **Ready for Phase 05 evaluation**

The stratified temporal CV now ensures robust evaluation with reliable metrics while maintaining temporal validity for forecasting scenarios.
