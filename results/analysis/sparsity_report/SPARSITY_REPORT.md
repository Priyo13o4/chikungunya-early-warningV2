# Sparsity Report (v6)
Generated: 2026-02-08T22:25:19
## Dataset overview
- Total rows: 731
- Unique states: 21
- Unique districts: 195
- Year range: 2009–2022
- Labeled rows (`label_outbreak` not NaN): 101
- Positives among labeled: 43 (42.6%)
- Engineered feature columns (`feat_*`): 37
## Why folds fail (root cause)
Most fold failures are caused by *complete-case filtering* on engineered features. Many mechanistic/EWS features are undefined early in a district history (rolling windows, 52-week baselines), so dropping any row with any feature NaN collapses the dataset to almost nothing.
- Strict complete-case rows among labeled (require all 37 features): 21
- Strict complete-case rows using CORE feature set (require only 19 features): 77
## Top missing features (overall)
| feature | overall_missing_pct | overall_missing_count |
| --- | --- | --- |
| feat_recent_normalized | 88.92 | 650 |
| feat_var_spike_ratio | 88.92 | 650 |
| label_outbreak | 86.18 | 630 |
| feat_cases_lag_8 | 82.90 | 606 |
| feat_rain_lag_8 | 81.67 | 597 |
| feat_temp_lag_8 | 81.40 | 595 |
| feat_acf_change | 71.96 | 526 |
| feat_lai_lag_4 | 70.73 | 517 |
| feat_cases_acf_lag1_4w | 66.62 | 487 |
| feat_cases_lag_4 | 66.48 | 486 |
## Top missing features (within labeled rows)
| feature | labeled_missing_pct | labeled_missing_count |
| --- | --- | --- |
| feat_lai_lag_1 | 25.74 | 26 |
| feat_lai_lag_4 | 25.74 | 26 |
| feat_lai_lag_2 | 22.77 | 23 |
| feat_lai | 22.77 | 23 |
| feat_var_spike_ratio | 19.80 | 20 |
| feat_recent_normalized | 19.80 | 20 |
| feat_rain_persist_4w | 6.93 | 7 |
| feat_rain_lag_4 | 6.93 | 7 |
| feat_rain_lag_2 | 6.93 | 7 |
| feat_rain_lag_1 | 6.93 | 7 |
## Biggest strict-completeness bottlenecks
`delta_if_drop_feature` tells you how many additional labeled rows you would recover if you *stopped requiring* that single feature to be non-NaN (holding all others fixed).
| feature | labeled_missing | delta_if_drop_feature |
| --- | --- | --- |
| feat_lai_lag_2 | 23 | 5 |
| feat_lai_lag_1 | 26 | 4 |
| feat_lai_lag_4 | 26 | 4 |
| feat_lai | 23 | 3 |
| feat_temp_lag_1 | 6 | 3 |
| feat_temp_lag_2 | 6 | 3 |
| feat_temp_anomaly | 6 | 2 |
| feat_temp_lag_4 | 5 | 2 |
| feat_temp_lag_8 | 4 | 1 |
| feat_var_spike_ratio | 20 | 0 |
## Thesis-consistent core feature set (recommended for sparse panels)
Use this for Track A baselines when you need stable CV. It preserves mechanistic + seasonal signal while avoiding long-baseline EWS features that are mostly undefined in this dataset.
Core set definition (`CORE_FEATURE_SET_V01`):
- feat_week_sin
- feat_week_cos
- feat_quarter
- feat_is_monsoon
- feat_lat_norm
- feat_lon_norm
- feat_lat_lon_interact
- feat_cases_lag_1
- feat_cases_lag_2
- feat_cases_lag_4
- feat_cases_ma_4w
- feat_cases_growth_rate
- feat_cases_var_4w
- feat_temp_lag_1
- feat_temp_lag_2
- feat_rain_lag_1
- feat_rain_lag_2
- feat_temp_anomaly
- feat_rain_persist_4w
- feat_degree_days
### Rationale (what we exclude and why)
- `feat_var_spike_ratio`: requires a long baseline (52 weeks) so it is missing for most district-years in a sparse panel.
- Long lags (e.g., `*_lag_8`) and long-window dynamics become undefined when district-year sequences are short or irregular.
- LAI lags are frequently missing due to upstream data gaps; treat them as optional or impute carefully.
