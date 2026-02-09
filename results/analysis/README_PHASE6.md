# Phase 6: Quick Start Guide

## What is Phase 6?

Phase 6 demonstrates that **Bayesian latent risk estimation provides decision-theoretic advantages** over binary classification for outbreak early warning.

**This is NOT about improving AUC.** It's about proving:
1. Earlier warning (lead-time advantage)
2. Better decisions under uncertainty
3. Hybrid ML-Bayesian fusion improves utility

## Prerequisites

✅ Phase 5 must be complete:
- `results/metrics/bayesian_cv_results.json` exists
- `results/metrics/baseline_comparison.json` exists

## Quick Execution

### Option A: Run All Tasks (4-8 hours)

```bash
chmod +x run_phase6.sh
./run_phase6.sh
```

### Option B: Run Individual Tasks

**Fast (no MCMC):**
```bash
python experiments/10_comprehensive_metrics.py
```

**Slow (MCMC required):**
```bash
# Lead-time analysis (~1-2h)
python experiments/06_analyze_lead_time.py

# Visualizations (~30-60min)
python experiments/07_visualize_risk_trajectories.py --n-districts 8

# Decision simulation (~1-2h)
python experiments/08_simulate_decision_layer.py

# Fusion experiments (~2-3h)
python experiments/09_fusion_experiments.py
```

## Outputs

### Analysis Results
- `results/analysis/comprehensive_metrics.json` — Metric comparison
- `results/analysis/lead_time_analysis.json` — Lead-time advantage
- `results/analysis/decision_simulation.json` — Decision quality
- `results/analysis/fusion_results.json` — Hybrid strategies

### Visualizations
- `results/plots/risk_trajectories/*.png` — 8 district examples

## Interpreting Results

### Lead-Time Analysis

**What to look for:**
- Median lead time > 0 (Bayesian warns before outbreak)
- Target: ≥2 weeks
- IQR indicates consistency

**Example:**
```json
{
  "median_lead_time": 2.5,
  "mean_lead_time": 2.8,
  "iqr": [1.0, 4.0],
  "positive_lead_pct": 75.0
}
```

✅ **Good:** 75% of outbreaks had positive lead time (early warning)

### Decision Simulation

**What to look for:**
- Net benefit > 0 (interventions justified)
- Sensitivity (detected outbreaks / total outbreaks)
- False alarm rate < 20%

**Example:**
```json
{
  "sensitivity": 0.68,
  "precision": 0.42,
  "false_alarm_rate": 0.15,
  "net_benefit": 12.5
}
```

✅ **Good:** Caught 68% of outbreaks, 15% false alarms, positive net benefit

### Fusion Results

**What to look for:**
- AUPR improvement over baselines
- Recall improvement (catch more outbreaks)
- Best strategy (feature fusion vs gated vs weighted)

**Example:**
```json
{
  "feature_fusion": {
    "aupr_mean": 0.32,
    "recall_mean": 0.45
  },
  "gated_decision": {
    "aupr_mean": 0.28,
    "recall_mean": 0.38
  }
}
```

✅ **Good:** Feature fusion improved AUPR and recall

## Key Concepts

### Alert Zones

- 🟢 **GREEN:** P(risk) < 0.4 → No action
- 🟡 **YELLOW:** 0.4–0.8 OR high uncertainty → Monitor
- 🔴 **RED:** P(risk) ≥ 0.8 AND low uncertainty → Intervene

### Why Bayesian AUC is Low (0.515)

❌ **Wrong:** "Bayesian model failed"

✅ **Correct:** Bayesian model is a **risk estimator**, not a binary classifier:
- Conservative (high specificity, low false alarms)
- Quantifies uncertainty
- Evaluated on lead time, not AUC

### Fusion Strategies

1. **Feature Fusion:** Add Bayesian outputs as XGBoost features
2. **Gated Decision:** Use Bayesian when risk is high, else ML
3. **Weighted Ensemble:** α * Bayes + (1-α) * XGBoost

## Troubleshooting

### MCMC Convergence Issues

**Problem:** Divergences, high R-hat

**Solution:**
- Increase `adapt_delta` (already 0.95)
- Increase `n_warmup` (currently 1000)
- Check for data issues (NaN, extreme values)

### Memory Issues

**Problem:** OOM during MCMC

**Solution:**
- Reduce `n_chains` from 4 to 2
- Reduce `n_samples` from 1000 to 500
- Run on cluster/cloud

### Slow Execution

**Problem:** Taking > 8 hours

**Solution:**
- Run overnight
- Parallelize folds (modify scripts)
- Use faster machine

## Documentation

**Full Phase 6 Guide:** `docs/09_phase6_decision_fusion.md`

Key sections:
- Conceptual framework (Track A vs B)
- Task definitions
- Literature justification
- Expected outcomes

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Median lead time | ≥ 2 weeks | Check `lead_time_analysis.json` |
| Net benefit | > 0 | Check `decision_simulation.json` |
| Fusion AUPR | ≥ 10% improvement | Check `fusion_results.json` |
| Documentation | Complete | ✅ |

## Next Steps

After Phase 6 completion:
1. Review all JSON outputs
2. Examine risk trajectory plots
3. Summarize findings for faculty
4. Prepare results section for paper/thesis
5. Consider external validation (Brazil data)

## Questions?

See `docs/09_phase6_decision_fusion.md` — Section: "Common Pitfalls & FAQs"
