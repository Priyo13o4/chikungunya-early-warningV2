# Phase 7 Visualization Results

## Directory Structure

```
results/figures/phase7/
├── 01_trackA_performance/    # Baseline model comparison
│   ├── auc_comparison_bar.png
│   └── auc_comparison_bar.txt
├── 02_lead_time_analysis/    # Lead-time comparison (Bayesian vs XGBoost)
│   ├── lead_time_distribution.png
│   ├── differential_lead_histogram.png
│   ├── per_fold_median_lead.png
│   └── case_study_timeseries.png
├── 03_calibration_uncertainty/   # Calibration and uncertainty quantification
│   └── calibration_curves.png
└── README.md                     # This file
```

## Thesis Chapter Mapping

### Chapter: Methods (Track A Baselines)
- `01_trackA_performance/auc_comparison_bar.png`: Figure showing AUC comparison

### Chapter: Results (Lead-Time Analysis)
- `02_lead_time_analysis/lead_time_distribution.png`: Main lead-time comparison
- `02_lead_time_analysis/differential_lead_histogram.png`: ΔL distribution
- `02_lead_time_analysis/per_fold_median_lead.png`: Fold-level variability

### Chapter: Results (Calibration & Uncertainty)
- `03_calibration_uncertainty/calibration_curves.png`: Reliability

## Figure Standards

All figures include:
- Clear title
- Axis labels with units
- Legend (where applicable)
- Saved at 300 DPI
- Accompanying .txt file with interpretation

## Data Sources

- Lead-time results: `results/analysis/lead_time_*_p*.csv`
- Baseline metrics: `results/metrics/baseline_comparison.json`
- Bayesian metrics: `results/metrics/bayesian_cv_results.json`

## Generated

Date: 2026-02-11 12:32:11
Version: v4.2 (Phase 7 Complete)
