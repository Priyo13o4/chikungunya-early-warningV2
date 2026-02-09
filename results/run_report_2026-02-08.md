# Run Report (2026-02-08)

## Scope
- Full pipeline rerun: Experiments 01–11 in V2.
- Snapshot comparison uses [results/_snapshot_before_rerun.txt](results/_snapshot_before_rerun.txt) vs [results/_snapshot_after_rerun.txt](results/_snapshot_after_rerun.txt).
- Diff artifact: [results/_snapshot_diff.json](results/_snapshot_diff.json).

## Key Metrics (Latest Run)
- Comprehensive summary: [results/analysis/comprehensive_metrics.json](results/analysis/comprehensive_metrics.json)
- Bayesian (Phase 5): AUC 0.515 ± 0.209, recall 0.048, F1 0.061, specificity 0.976
- XGBoost: AUC 0.687 ± 0.141, recall 0.557, F1 0.551
- Fusion (Phase 9):
  - Gated decision: AUC 0.678, AUPR 0.607, F1 0.493
  - Weighted ensemble: AUC 0.817, AUPR 0.726, F1 0.566
- Lead-time: total episodes = 3 across 6 folds
  - Metadata: [results/analysis/lead_time_analysis_metadata.json](results/analysis/lead_time_analysis_metadata.json)
- Decision layer: sensitivity 0.071, precision 0.056, net benefit -148.0
  - Details: [results/analysis/decision_simulation_p75.json](results/analysis/decision_simulation_p75.json)

## Output Artifacts (Latest Run)
- Lead-time predictions: [results/analysis/lead_time_predictions_p75.parquet](results/analysis/lead_time_predictions_p75.parquet)
- Lead-time summaries:
  - [results/analysis/lead_time_summary_overall_p75.csv](results/analysis/lead_time_summary_overall_p75.csv)
  - [results/analysis/lead_time_summary_by_fold_p75.csv](results/analysis/lead_time_summary_by_fold_p75.csv)
  - [results/analysis/lead_time_detail_p75.csv](results/analysis/lead_time_detail_p75.csv)
- Fusion results: [results/analysis/fusion_results_p75.json](results/analysis/fusion_results_p75.json)
- Phase 7 figures: [results/figures/phase7/](results/figures/phase7/)
- Sparsity report: [results/analysis/sparsity_report/SPARSITY_REPORT.md](results/analysis/sparsity_report/SPARSITY_REPORT.md)

## Snapshot Comparison (Before vs After)
- Added files: 39
- Removed files: 34
- Changed files: 34

### Notable Changes
- Updated baseline and Bayesian CV metrics:
  - [results/metrics/baseline_comparison.json](results/metrics/baseline_comparison.json)
  - [results/metrics/bayesian_cv_results.json](results/metrics/bayesian_cv_results.json)
- Lead-time prediction artifact grew (more rows):
  - [results/analysis/lead_time_predictions_p75.parquet](results/analysis/lead_time_predictions_p75.parquet)
- Fusion output refreshed (size increased):
  - [results/analysis/fusion_results_p75.json](results/analysis/fusion_results_p75.json)
- Decision simulation refreshed:
  - [results/analysis/decision_simulation_p75.json](results/analysis/decision_simulation_p75.json)
- Phase 7 figure bundle regenerated (all PNG/TXT/README):
  - [results/figures/phase7/README.md](results/figures/phase7/README.md)

## Notes
- Single-class folds (e.g., 2022) can reduce available episode-based metrics; see fold breakdown in [results/analysis/lead_time_analysis_metadata.json](results/analysis/lead_time_analysis_metadata.json).
