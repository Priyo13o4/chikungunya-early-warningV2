# Uncertainty Quantification Visualizations

This folder contains **3 publication-ready plots** for uncertainty quantification and temporal analysis in Bayesian predictions.

## Plots

### 13. **uncertainty_bands_timeseries.png**
- **Purpose**: Show prediction uncertainty over time for one district
- **What to look for**:
  - Shaded band = 95% credible interval
  - Green line = posterior mean probability
  - Red dots = actual outbreak weeks
  - Orange dashed line = alert threshold
- **Interpretation**: 
  - Wider bands = more uncertainty
  - When red dots are within band = good calibration
  - When line crosses threshold = alert triggered
- **Format**: PNG (300 DPI) + PDF

### 14. **prediction_interval_coverage.png**
- **Purpose**: Validate credible interval coverage
- **What to look for**:
  - Bars should be near 95% line (red dashed)
  - Green bars = good coverage (90-100%)
  - Red bars = poor coverage (<85%)
- **Shows**: 
  - Left panel: Coverage by fold
  - Right panel: Summary statistics
- **Ideal**: Mean coverage ≈ 0.95

### 15. **forecast_horizon_uncertainty.png**
- **Purpose**: Show how uncertainty grows with forecast horizon
- **What to look for**:
  - Box plots show distribution of CI widths
  - Error bars show mean ± SD
  - Red dashed line = fitted growth curve
- **Expected**: Uncertainty ∝ √(horizon)
- **Shows**: 1-week, 2-week, 3-week, 4-week ahead forecasts

## Usage

Run the uncertainty visualization script:
```bash
cd chikungunya-early-warningV2/experiments
python viz_uncertainty_plots.py
```

All plots will be saved to this folder.

## Data Sources

- `results/analysis/lead_time_predictions_p75.parquet` - Predictions with uncertainty
- `results/metrics/bayesian_cv_results.json` - Fold information
- Posterior samples from Stan (if available)

## Why Uncertainty Matters

**For Decision Makers:**
1. **Confidence in alerts**: Wide intervals = less certain, may need verification
2. **Risk management**: Can quantify probability of false alarms
3. **Resource allocation**: Prioritize districts with high mean + narrow intervals

**For Model Validation:**
1. **Calibration check**: Do 95% intervals contain truth 95% of time?
2. **Reliability**: Uncertainty should match actual prediction errors
3. **Temporal degradation**: Further ahead = more uncertain (expected)

## Key Insights

Use these plots to answer:
1. **How certain are we about predictions?** → uncertainty_bands_timeseries.png
2. **Are our uncertainty estimates accurate?** → prediction_interval_coverage.png
3. **How far ahead can we reliably forecast?** → forecast_horizon_uncertainty.png

## Interpretation Guide

### Good Uncertainty Quantification:
- ✓ Coverage ≈ 95% (±5%)
- ✓ Uncertainty grows smoothly with horizon
- ✓ Narrow bands when confident, wide when uncertain
- ✓ Bands capture most actual outcomes

### Warning Signs:
- ✗ Coverage << 95% (overconfident)
- ✗ Coverage >> 95% (underconfident)
- ✗ Uniform width (not adapting to context)
- ✗ Unpredictable uncertainty growth

## Technical Notes

**Credible Interval Calculation:**
- Bayesian: Posterior quantiles (2.5%, 97.5%)
- Approximation: Normal in logit space
- z_mean ± 1.96 * z_sd

**Coverage Calculation:**
- For each prediction: check if y_true ∈ [CI_lower, CI_upper]
- Coverage = proportion of predictions where true outcome in interval
- Calculate separately by fold to detect issues
