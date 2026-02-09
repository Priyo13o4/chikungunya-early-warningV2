# Version 5 (v5) — Phase 7 Complete

## Version Metadata

| Field | Value |
|-------|-------|
| **Version** | v5 |
| **Phase** | Phase 7 (Analysis & Usefulness) |
| **Status** | FROZEN |
| **Freeze Date** | 2026-02-03 |
| **Author** | Vishnu |
| **Project** | Chikungunya Early Warning System |

---

## What This Version Represents

Version v5 is the **FIRST and FINAL version** containing:

- ✅ Fully executed Phase 7 lead-time analysis
- ✅ Fair, symmetric lead-time comparison (p75 threshold for BOTH models)
- ✅ All 6 CV folds attempted (3 succeeded, 3 failed due to data limitations)
- ✅ Final CSV outputs with episode-level and summary statistics
- ✅ Final figures (7 total: 4 complete, 2 placeholders, 1 illustrative)
- ✅ Explicit documentation of data sparsity and placeholder status
- ✅ Thesis-ready figure structure and captions

**v5 backs all Phase 7 content in the thesis:**
- Results chapter (lead-time analysis)
- Discussion chapter (model comparison)
- All referenced figures

---

## Contents

```
v5/
├── src/                    # Source code (for reproducibility)
├── experiments/            # Experiment scripts (for reproducibility)
├── config/                 # Configuration files
├── results/
│   ├── analysis/           # Final CSV outputs
│   │   ├── lead_time_detail_all_folds.csv
│   │   ├── lead_time_summary_overall.csv
│   │   ├── lead_time_summary_by_fold.csv
│   │   └── lead_time_analysis_metadata.json
│   └── figures/phase7/     # Final figures
│       ├── 01_trackA_performance/
│       ├── 02_lead_time_analysis/
│       ├── 03_calibration_uncertainty/
│       └── 04_decision_usefulness/
└── metadata/
    └── README.md           # This file
```

### NOT Included (Intentionally)

- `data/` — Raw data not versioned (available in main workspace)
- `notebooks/` — Exploratory notebooks not part of final pipeline
- `__pycache__/` — Python cache files
- Intermediate artifacts

---

## Key Quantitative Facts

### Lead-Time Analysis Results

| Metric | Value |
|--------|-------|
| **Total outbreak episodes analyzed** | 4 |
| **Bayesian model warned** | 3 episodes (75%) |
| **XGBoost model warned** | 1 episode (25%) |
| **Bayesian never-warned rate** | 25% |
| **XGBoost never-warned rate** | 75% |
| **Bayesian mean lead time** | 5.0 weeks |
| **XGBoost mean lead time** | 0.0 weeks |

### CV Fold Status

| Fold | Test Year | Status | Outbreak Episodes |
|------|-----------|--------|-------------------|
| fold_2017 | 2017 | ✅ Success | 1 |
| fold_2018 | 2018 | ❌ Zero outbreaks | 0 |
| fold_2019 | 2019 | ✅ Success | 1 |
| fold_2020 | 2020 | ❌ Zero outbreaks | 0 |
| fold_2021 | 2021 | ✅ Success | 2 |
| fold_2022 | 2022 | ❌ Insufficient data | 0 |

---

## Data Sparsity (Intentional)

The following figures show **sparse visual density**. This is **expected and methodologically correct**:

| Figure | Reason for Sparsity |
|--------|---------------------|
| `lead_time_distribution.png` | n=3 Bayesian / n=1 XGBoost warnings |
| `differential_lead_histogram.png` | n=1 episode where both models warned |
| `per_fold_median_lead.png` | 3 of 6 folds had zero outbreak episodes |

**Root cause:** Chikungunya outbreak episodes are rare events in the surveillance data.

---

## Placeholder Figures (Intentional)

The following figures are **intentionally empty** because they require artifacts not stored during analysis:

| Figure | Missing Artifact |
|--------|------------------|
| `case_study_timeseries_placeholder.png` | Per-district prediction trajectories |
| `calibration_curves_placeholder.png` | Stored (pred_prob, true_label) pairs |

Additionally, `decision_state_timeline.png` shows an **illustrative example** with simulated data.

---

## Freeze Guarantee

> **No further methodological changes are permitted beyond v5.**
> **Subsequent work is writing and interpretation only.**

This version represents the final analytical state of the project. Any future versions (v6+) would only contain:
- Documentation updates
- Thesis text revisions
- No code or result changes

---

## Reproducibility

To reproduce results from v5:

1. Use Python 3.13.0
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure CmdStanPy and Stan are configured
4. Run experiments in order: `00_sanity_check.py` → `06_analyze_lead_time.py`

Note: Results depend on:
- Random seed (42)
- MCMC sampling (1000 warmup, 1000 samples, 4 chains)
- Data file: `data/processed/features_engineered_v01.parquet`

---

## Version History

| Version | Phase | Key Milestone |
|---------|-------|---------------|
| v1 | Phase 1-2 | Data pipeline |
| v2 | Phase 3 | Feature engineering |
| v3 | Phase 4 | Bayesian model implementation |
| v4 | Phase 5 | CV evaluation framework |
| v4.1 | Phase 5 | State-space indexing bug fix |
| **v5** | **Phase 7** | **Lead-time analysis complete (CURRENT)** |

---

*Frozen: 2026-02-03*
*Phase 7 is CLOSED*
