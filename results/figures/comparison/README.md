# Comparison Visualizations

This folder contains **9 publication-ready comparison plots** between Track A (XGBoost baseline) and Track B (Bayesian hierarchical model).

## Plots

### 1. **calibration_curve.png** ⭐ MOST IMPORTANT
- **Purpose**: Shows how well predicted probabilities match observed frequencies
- **What to look for**: Points close to diagonal = well-calibrated model
- **Metric**: Brier score annotated for each model
- **Format**: PNG (300 DPI) + PDF

### 2. **roc_curves.png**
- **Purpose**: Receiver Operating Characteristic curves
- **What to look for**: Higher AUC = better discrimination
- **Metric**: AUC scores in legend
- **Format**: PNG (300 DPI) + PDF

### 3. **precision_recall_curves.png**
- **Purpose**: Precision-Recall trade-off
- **What to look for**: Higher AUPR = better for imbalanced data
- **Metric**: AUPR scores in legend
- **Format**: PNG (300 DPI) + PDF

### 4. **sensitivity_specificity_tradeoff.png**
- **Purpose**: Shows how sensitivity and specificity vary with threshold
- **What to look for**: Optimal operating point (marked at 0.7)
- **Metric**: Both metrics across threshold range

### 5. **performance_heatmap.png**
- **Purpose**: Side-by-side comparison of 6 key metrics
- **What to look for**: Green = better performance, Red = worse
- **Metrics**: AUC, Sensitivity, Specificity, Brier, F1, AUPR

### 6. **decision_cost_analysis.png**
- **Purpose**: Net benefit under different cost assumptions
- **What to look for**: Crossover point where models are equivalent
- **Metric**: Net benefit as function of cost ratio

### 7. **episode_detection_timeline.png**
- **Purpose**: Visual timeline of outbreak episodes and alerts
- **What to look for**: 
  - Green markers = early alerts (≥2 weeks)
  - Yellow markers = late alerts (1 week)
  - Red markers = missed/on-time
- **Shows**: When Track B alerted relative to actual outbreaks

### 8. **feature_importance_comparison.png**
- **Purpose**: Compare which features drive each model
- **What to look for**: 
  - XGBoost: relative importance scores
  - Bayesian: posterior coefficients with uncertainty
- **Note**: Using placeholder data (load actual model objects for production)

## Usage

Run the visualization script:
```bash
cd chikungunya-early-warningV2/experiments
python viz_comparison_plots.py
```

All plots will be saved to this folder.

## Data Sources

- `results/metrics/bayesian_cv_results.json` - Track B metrics
- `results/metrics/baseline_comparison.json` - Track A metrics
- `results/analysis/lead_time_predictions_p75.parquet` - Predictions
- `results/analysis/lead_time_detail_p75.csv` - Episode details

## Key Insights

Use these plots to answer:
1. **Which model is better calibrated?** → calibration_curve.png
2. **Which has better discrimination?** → roc_curves.png
3. **Which is better for rare events?** → precision_recall_curves.png
4. **How early can we detect outbreaks?** → episode_detection_timeline.png
5. **What's the cost-benefit trade-off?** → decision_cost_analysis.png
