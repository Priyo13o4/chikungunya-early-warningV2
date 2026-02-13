# MCMC Diagnostic Visualizations

This folder contains **4 publication-ready diagnostic plots** for Bayesian hierarchical model MCMC convergence assessment.

## Plots

### 9. **mcmc_trace_plots.png**
- **Purpose**: Assess chain mixing and convergence
- **What to look for**:
  - Good: chains overlap and explore same space (fuzzy caterpillar)
  - Bad: chains stuck in different regions
- **Metrics**: 
  - R̂ < 1.1 = converged
  - ESS > 400 = sufficient samples
- **Shows**: 4 key parameters across 4 chains

### 10. **posterior_distributions.png**
- **Purpose**: Show learned parameter values with uncertainty
- **What to look for**:
  - Width of distribution = uncertainty
  - Overlap with zero = not significant
  - Prior vs posterior = how much data informed the model
- **Shows**: Beta coefficients for key predictors
- **Format**: PNG (300 DPI) + PDF

### 11. **hierarchical_shrinkage.png**
- **Purpose**: Visualize partial pooling in hierarchical model
- **What to look for**:
  - Points pulled toward red line = shrinkage effect
  - Districts with more data = less shrinkage
  - Wider error bars = more uncertainty
- **Shows**: District-level random effects (α_d)

### 12. **convergence_dashboard.png**
- **Purpose**: Summary table of all parameters
- **What to look for**:
  - Green rows = converged parameters
  - Red rows = problematic parameters
- **Metrics**: Mean, SD, R̂, ESS_bulk, ESS_tail, Status
- **Also saves**: convergence_summary.csv

## Usage

Run the diagnostic script:
```bash
cd chikungunya-early-warningV2/experiments
python viz_diagnostic_plots.py
```

All plots will be saved to this folder.

## Data Sources

- `results/metrics/bayesian_cv_diagnostics.json` - MCMC diagnostics per fold
- `results/metrics/bayesian_cv_results.json` - Overall results
- Stan fit objects (if available) - Actual posterior samples

## Convergence Criteria

**Good convergence:**
- R̂ < 1.1 (preferably < 1.05)
- ESS_bulk > 400
- ESS_tail > 400
- No divergent transitions
- Trace plots show good mixing

**Warning signs:**
- R̂ > 1.1
- ESS < 400
- Divergent transitions
- Chains stuck in different modes

## Key Insights

Use these plots to answer:
1. **Did the model converge?** → convergence_dashboard.png
2. **Are chains mixing well?** → mcmc_trace_plots.png
3. **What did we learn about predictors?** → posterior_distributions.png
4. **How much does the hierarchy help?** → hierarchical_shrinkage.png

## Troubleshooting

If convergence issues:
1. Increase warmup iterations
2. Increase adapt_delta (0.95 → 0.99)
3. Reparameterize model
4. Check for multicollinearity
5. Use stronger priors
