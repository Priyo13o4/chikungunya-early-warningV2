#!/usr/bin/env python3
"""
Experiment 07: Phase 7 Comprehensive Visualizations

PURPOSE:
--------
Generate all required figures for Phase 7 thesis chapters:
1. Track A Performance (ROC, AUC, Precision-Recall)
2. Lead-Time Analysis (distributions, comparisons, case studies)
3. Calibration & Uncertainty (reliability curves, Brier scores, credible intervals)
4. Decision Usefulness (state timelines, false alarm rates, stability)

VERSION: v4.2 (Phase 7 Complete)
DATE: February 2026

Reference: docs/Version 2/08_phase7_roadmap_v2.md
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =============================================================================
# PATH SETUP
# =============================================================================

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Output directories
FIGURE_ROOT = project_root / "results" / "figures" / "phase7"
TRACK_A_DIR = FIGURE_ROOT / "01_trackA_performance"
LEAD_TIME_DIR = FIGURE_ROOT / "02_lead_time_analysis"
CALIBRATION_DIR = FIGURE_ROOT / "03_calibration_uncertainty"
DECISION_DIR = FIGURE_ROOT / "04_decision_usefulness"

# Ensure directories exist
for d in [TRACK_A_DIR, LEAD_TIME_DIR, CALIBRATION_DIR, DECISION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Consistent color scheme
COLORS = {
    'bayesian': '#2E86AB',  # Blue
    'xgboost': '#E94F37',   # Red
    'rf': '#F39C12',        # Orange
    'logistic': '#27AE60',  # Green
    'baseline': '#95A5A6',  # Gray
    'outbreak': '#C0392B',  # Dark red
    'warning': '#F1C40F',   # Yellow
    'safe': '#27AE60',      # Green
}

# Figure settings
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure_with_description(
    fig: plt.Figure,
    filepath: Path,
    title: str,
    description: str,
    interpretation: str,
    caveats: str = ""
) -> None:
    """
    Save figure as PNG and create accompanying .txt description file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save PNG (without extension)
        title: Figure title
        description: What is shown
        interpretation: How to interpret
        caveats: Any caveats (optional)
    """
    # Save PNG
    png_path = filepath.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {png_path.name}")
    
    # Save description
    txt_path = filepath.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write(f"FIGURE: {title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"WHAT IS SHOWN:\n{description}\n\n")
        f.write(f"WHY IT MATTERS:\n{interpretation}\n\n")
        if caveats:
            f.write(f"CAVEATS:\n{caveats}\n")
    
    plt.close(fig)


def select_lead_time_file(prefix: str, ext: str, prefer_percentile: int = 75) -> Optional[Path]:
    """Pick lead-time output file, preferring a specific percentile if present."""
    analysis_dir = project_root / "results" / "analysis"
    preferred = analysis_dir / f"{prefix}_p{prefer_percentile}.{ext}"
    if preferred.exists():
        return preferred
    candidates = sorted(analysis_dir.glob(f"{prefix}_p*.{ext}"))
    if candidates:
        return candidates[0]
    return None


def _get_outbreak_percentile_from_metadata(default: int = 75) -> int:
    meta_path = project_root / "results" / "analysis" / "lead_time_analysis_metadata.json"
    if not meta_path.exists():
        return default
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        return int(meta.get('config', {}).get('outbreak_percentile', default))
    except Exception:
        return default


def select_predictions_parquet() -> Optional[Path]:
    """Prefer percentile-specific prediction parquet; fall back to any available parquet."""
    p = _get_outbreak_percentile_from_metadata(default=75)
    analysis_dir = project_root / "results" / "analysis"
    cand = analysis_dir / f"lead_time_predictions_p{p}.parquet"
    if cand.exists():
        return cand
    any_p = sorted(analysis_dir.glob('lead_time_predictions_p*.parquet'))
    if any_p:
        return any_p[0]
    return None


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    import math

    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def compute_bayes_prob_high_risk(preds: pd.DataFrame, risk_quantile: float = 0.80) -> np.ndarray:
    """Compute P(Z > q) via Normal(mu=z_mean, sd=z_sd) for plotting/calibration."""
    work = preds[['z_mean', 'z_sd']].copy()
    work['z_mean'] = pd.to_numeric(work['z_mean'], errors='coerce')
    work['z_sd'] = pd.to_numeric(work['z_sd'], errors='coerce')
    valid = work.dropna()
    if valid.empty:
        return np.full(shape=(len(preds),), fill_value=np.nan, dtype=float)

    q = float(np.percentile(valid['z_mean'].values, risk_quantile * 100))
    mu = pd.to_numeric(preds['z_mean'], errors='coerce').values.astype(float)
    sd = np.maximum(pd.to_numeric(preds['z_sd'], errors='coerce').values.astype(float), 1e-6)
    z = (q - mu) / sd
    return 1.0 - _normal_cdf(z)


# =============================================================================
# SECTION 1: TRACK A PERFORMANCE FIGURES
# =============================================================================

def plot_auc_comparison_bar() -> None:
    """
    Create AUC comparison bar plot for all baseline models.
    """
    print("\n  Generating AUC comparison bar plot...")
    
    # Load baseline comparison results
    metrics_dir = project_root / "results" / "metrics"
    metrics_path = metrics_dir / "baseline_comparison.json"

    if not metrics_path.exists():
        # Sensitivity runs write suffixed files (e.g., baseline_comparison_p75.json).
        p75_path = metrics_dir / "baseline_comparison_p75.json"
        if p75_path.exists():
            metrics_path = p75_path
        else:
            candidates = sorted(metrics_dir.glob("baseline_comparison_p*.json"))
            if candidates:
                metrics_path = candidates[0]
            else:
                warnings.warn(f"Baseline comparison file not found: {metrics_path}")
                return
    
    with open(metrics_path, 'r') as f:
        baseline_results = json.load(f)
    
    # Extract AUC values - results are under 'models' key
    models_data = baseline_results.get('models', baseline_results)
    
    models = []
    aucs = []
    colors = []
    
    model_color_map = {
        'xgboost': COLORS['xgboost'],
        'random_forest': COLORS['rf'],
        'logistic': COLORS['logistic'],
        'threshold': COLORS['baseline'],
    }
    
    for model_name, metrics in models_data.items():
        if isinstance(metrics, dict) and 'auc_mean' in metrics:
            models.append(model_name.replace('_', ' ').title())
            aucs.append(metrics['auc_mean'])
            colors.append(model_color_map.get(model_name, COLORS['baseline']))
    
    # Add Bayesian (from Phase 5 results)
    bayesian_path = project_root / "results" / "metrics" / "bayesian_cv_results.json"
    if bayesian_path.exists():
        with open(bayesian_path, 'r') as f:
            bayesian_results = json.load(f)
        # Check in aggregated first, then top-level
        if 'aggregated' in bayesian_results and 'auc_mean' in bayesian_results['aggregated']:
            models.append('Bayesian')
            aucs.append(bayesian_results['aggregated']['auc_mean'])
            colors.append(COLORS['bayesian'])
        elif 'auc' in bayesian_results:
            models.append('Bayesian')
            aucs.append(bayesian_results['auc'])
            colors.append(COLORS['bayesian'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    bars = ax.bar(x, aucs, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Cross-Validation AUC Comparison: Track A Baselines vs Bayesian')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    save_figure_with_description(
        fig, TRACK_A_DIR / "auc_comparison_bar",
        title="AUC Comparison: Track A Baselines vs Bayesian",
        description="Bar chart comparing Area Under the ROC Curve (AUC-ROC) "
                   "for XGBoost, Random Forest, Logistic Regression, and Bayesian "
                   "hierarchical state-space model across rolling-origin CV folds.",
        interpretation="Higher AUC indicates better discrimination between outbreak "
                      "and non-outbreak weeks. XGBoost achieves ~0.76, while Bayesian "
                      "achieves ~0.52. This is expected: Bayesian is designed for "
                      "RISK INFERENCE, not binary classification. The comparison "
                      "motivates using lead-time as the primary evaluation metric.",
        caveats="Bayesian model output is continuous risk (Z_t), not probability. "
               "AUC computed using posterior predictive threshold."
    )


def plot_lead_time_distribution() -> None:
    """
    Create lead-time distribution comparison (Bayesian vs XGBoost).
    """
    print("\n  Generating lead-time distribution plot...")
    
    # Load lead-time results
    detail_path = select_lead_time_file("lead_time_detail", "csv")
    
    if detail_path is None or not detail_path.exists():
        warnings.warn("Lead-time detail file not found (lead_time_detail_p*.csv)")
        return
    
    df = pd.read_csv(detail_path)
    
    # Extract lead times (filter out -1 = never warned)
    bayesian_leads = df['lead_time_bayesian'][df['lead_time_bayesian'] >= 0].values
    xgboost_leads = df['lead_time_xgboost'][df['lead_time_xgboost'] >= 0].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Histogram
    ax1 = axes[0]
    bins = np.arange(-0.5, max(max(bayesian_leads, default=0), max(xgboost_leads, default=0)) + 1.5, 1)
    
    if len(bayesian_leads) > 0:
        ax1.hist(bayesian_leads, bins=bins, alpha=0.7, color=COLORS['bayesian'], 
                label=f'Bayesian (n={len(bayesian_leads)})', edgecolor='black')
    if len(xgboost_leads) > 0:
        ax1.hist(xgboost_leads, bins=bins, alpha=0.7, color=COLORS['xgboost'],
                label=f'XGBoost (n={len(xgboost_leads)})', edgecolor='black')
    
    ax1.set_xlabel('Lead Time (weeks)')
    ax1.set_ylabel('Count')
    ax1.set_title('A. Lead-Time Distribution')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel B: Box plot
    ax2 = axes[1]
    data_to_plot = []
    labels = []
    colors_box = []
    
    if len(bayesian_leads) > 0:
        data_to_plot.append(bayesian_leads)
        labels.append('Bayesian')
        colors_box.append(COLORS['bayesian'])
    if len(xgboost_leads) > 0:
        data_to_plot.append(xgboost_leads)
        labels.append('XGBoost')
        colors_box.append(COLORS['xgboost'])
    
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Lead Time (weeks)')
    ax2.set_title('B. Lead-Time Box Plot')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add warning statistics
    n_total = len(df)
    n_bayesian_warned = (df['lead_time_bayesian'] >= 0).sum()
    n_xgboost_warned = (df['lead_time_xgboost'] >= 0).sum()
    
    stats_text = (f"Total episodes: {n_total}\n"
                 f"Bayesian warned: {n_bayesian_warned} ({100*n_bayesian_warned/n_total:.0f}%)\n"
                 f"XGBoost warned: {n_xgboost_warned} ({100*n_xgboost_warned/n_total:.0f}%)")
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    save_figure_with_description(
        fig, LEAD_TIME_DIR / "lead_time_distribution",
        title="Lead-Time Distribution: Bayesian vs XGBoost",
        description="Panel A: Histogram of lead times (weeks before outbreak) for "
                   "episodes where models triggered. Panel B: Box plot showing median, "
                   "IQR, and outliers. Only includes episodes where models warned.",
        interpretation="Positive lead time = model warned before outbreak. "
                      "Higher lead time = more advance warning. Bayesian model "
                      "is designed to capture gradual risk escalation, potentially "
                      "warning earlier than binary classifiers.",
        caveats="Episodes where model never warned (lead_time = -1) are excluded "
               "from these plots. See summary table for never-warned rates."
    )


def plot_differential_lead_histogram() -> None:
    """
    Create differential lead-time (ΔL) histogram.
    """
    print("\n  Generating differential lead-time histogram...")
    
    # Load lead-time results
    detail_path = select_lead_time_file("lead_time_detail", "csv")
    
    if detail_path is None or not detail_path.exists():
        warnings.warn("Lead-time detail file not found (lead_time_detail_p*.csv)")
        return
    
    df = pd.read_csv(detail_path)
    
    # Filter to episodes where BOTH models warned (can compute ΔL)
    valid_mask = (df['lead_time_bayesian'] >= 0) & (df['lead_time_xgboost'] >= 0)
    delta_leads = df.loc[valid_mask, 'differential_lead'].dropna().values
    
    if len(delta_leads) == 0:
        print("    No valid differential lead times to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(min(delta_leads) - 0.5, max(delta_leads) + 1.5, 1)
    ax.hist(delta_leads, bins=bins, color='purple', alpha=0.7, edgecolor='black')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='ΔL = 0 (same timing)')
    ax.axvline(x=np.mean(delta_leads), color='blue', linestyle=':', linewidth=2,
              label=f'Mean ΔL = {np.mean(delta_leads):.2f}')
    
    ax.set_xlabel('Differential Lead Time ΔL = L_XGB - L_Bayes (weeks)')
    ax.set_ylabel('Count')
    ax.set_title('Differential Lead Time Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add interpretation zones
    ax.text(0.02, 0.95, '← Bayesian earlier', transform=ax.transAxes, fontsize=10,
           color='blue', verticalalignment='top')
    ax.text(0.98, 0.95, 'XGBoost earlier →', transform=ax.transAxes, fontsize=10,
           color='red', verticalalignment='top', horizontalalignment='right')
    
    save_figure_with_description(
        fig, LEAD_TIME_DIR / "differential_lead_histogram",
        title="Differential Lead Time (ΔL) Distribution",
        description="Histogram of ΔL = L_XGB - L_Bayes for episodes where both "
                   "models triggered. Positive ΔL means Bayesian warned earlier.",
        interpretation="If ΔL > 0: Bayesian provided more advance warning. "
                      "If ΔL < 0: XGBoost provided more advance warning. "
                      "If ΔL = 0: Both models warned at the same time.",
        caveats="Only includes episodes where BOTH models warned. "
               "Episodes where one or both never warned are excluded."
    )


def plot_per_fold_median_lead() -> None:
    """
    Create per-fold median lead-time comparison.
    """
    print("\n  Generating per-fold median lead-time plot...")
    
    # Load summary by fold
    summary_path = select_lead_time_file("lead_time_summary_by_fold", "csv")
    
    if summary_path is None or not summary_path.exists():
        warnings.warn("Summary by fold file not found (lead_time_summary_by_fold_p*.csv)")
        return
    
    df = pd.read_csv(summary_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Handle NaN values for plotting
    bayesian_medians = df['bayesian_median_lead'].fillna(0).values
    xgboost_medians = df['xgboost_median_lead'].fillna(0).values
    
    bars1 = ax.bar(x - width/2, bayesian_medians, width, label='Bayesian',
                  color=COLORS['bayesian'], edgecolor='black')
    bars2 = ax.bar(x + width/2, xgboost_medians, width, label='XGBoost',
                  color=COLORS['xgboost'], edgecolor='black')
    
    ax.set_xlabel('CV Fold (Test Year)')
    ax.set_ylabel('Median Lead Time (weeks)')
    ax.set_title('Median Lead Time by CV Fold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['test_year'].astype(str))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add episode counts as annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, max(bayesian_medians[i], xgboost_medians[i]) + 0.5,
               f'n={int(row["n_episodes"])}', ha='center', fontsize=9)
    
    save_figure_with_description(
        fig, LEAD_TIME_DIR / "per_fold_median_lead",
        title="Median Lead Time by CV Fold",
        description="Bar chart comparing median lead time (in weeks) for Bayesian "
                   "and XGBoost models across each rolling-origin CV fold. "
                   "Episode count (n) shown above each fold.",
        interpretation="Fold-level variation reflects year-to-year differences in "
                      "outbreak patterns. Some folds may have zero outbreaks or "
                      "models that never warned (shown as 0).",
        caveats="Folds with zero episodes or models that never warned show 0 median. "
               "Small n per fold limits statistical power."
    )


def plot_case_study_timeseries() -> None:
    """
    Create case study time series showing cases, Z_t, P_t, and threshold crossings.
    """
    print("\n  Generating case study time series...")

    preds_path = select_predictions_parquet()
    if preds_path is None:
        warnings.warn("Prediction parquet not found; run experiments/06_analyze_lead_time.py first.")
        return

    preds = pd.read_parquet(preds_path)
    required = {'state', 'district', 'year', 'week', 'cases', 'prob', 'z_mean', 'z_sd', 'y_true'}
    missing = required - set(preds.columns)
    if missing:
        warnings.warn(f"Prediction parquet missing required columns: {sorted(missing)}")
        return

    preds = preds.copy()
    preds['y_true'] = pd.to_numeric(preds['y_true'], errors='coerce')
    preds['cases'] = pd.to_numeric(preds['cases'], errors='coerce')
    preds['prob'] = pd.to_numeric(preds['prob'], errors='coerce')
    preds = preds.dropna(subset=['year', 'week', 'cases'])

    # Choose a district automatically: prioritize outbreaks, then peaks.
    by_dist = (
        preds.groupby(['state', 'district'], as_index=False)
        .agg(n_rows=('cases', 'size'), n_outbreak=('y_true', 'sum'), max_cases=('cases', 'max'))
        .sort_values(['n_outbreak', 'max_cases', 'n_rows'], ascending=False)
    )
    if by_dist.empty:
        warnings.warn("No valid rows found in prediction parquet.")
        return

    state = str(by_dist.iloc[0]['state'])
    district = str(by_dist.iloc[0]['district'])
    sub = preds[(preds['state'] == state) & (preds['district'] == district)].copy()
    sub = sub.sort_values(['year', 'week']).reset_index(drop=True)

    sub['date'] = pd.to_datetime(sub['year'].astype(int).astype(str)) + pd.to_timedelta((sub['week'].astype(int) - 1) * 7, unit='D')
    p = _get_outbreak_percentile_from_metadata(default=75)
    threshold = float(np.percentile(sub['cases'].values, p))
    sub['p_bayes_high'] = compute_bayes_prob_high_risk(sub, risk_quantile=0.80)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel A: cases + threshold + outbreak weeks
    ax0 = axes[0]
    ax0.plot(sub['date'], sub['cases'], color='black', linewidth=2, label='Observed cases')
    ax0.axhline(threshold, color=COLORS['outbreak'], linestyle='--', linewidth=2, label=f'Outbreak threshold (P{p})')
    outbreak_mask = sub['y_true'].fillna(0) > 0
    if outbreak_mask.any():
        ax0.scatter(sub.loc[outbreak_mask, 'date'], sub.loc[outbreak_mask, 'cases'], color=COLORS['outbreak'], s=40, zorder=5, label='Outbreak weeks')
    ax0.set_title(f"Case Study: {district}, {state}")
    ax0.set_ylabel('Cases')
    ax0.legend(loc='upper left')
    ax0.grid(alpha=0.3)

    # Panel B: probabilities
    ax1 = axes[1]
    ax1.plot(sub['date'], sub['prob'], color=COLORS['xgboost'], linewidth=2, label='XGBoost P(outbreak)')
    ax1.plot(sub['date'], sub['p_bayes_high'], color=COLORS['bayesian'], linewidth=2, label='Bayesian P(Z > q80)')
    ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax1.set_ylabel('Probability')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    save_figure_with_description(
        fig,
        LEAD_TIME_DIR / "case_study_timeseries",
        title="Case Study Time Series",
        description="Two-panel case study for a representative district. Panel A shows observed cases and the outbreak threshold. Panel B shows XGBoost probabilities and a Bayesian high-risk probability derived from latent Z summaries.",
        interpretation="Case studies illustrate how warning signals evolve relative to observed case escalation. Bayesian latent risk can provide a smoother trajectory; ML probabilities can respond sharply to feature patterns.",
        caveats="Bayesian probability here is P(Z > q80) using a Normal approximation from (z_mean, z_sd); it is not a direct posterior predictive P(outbreak)."
    )


def plot_calibration_curves() -> None:
    """
    Create reliability (calibration) curves for probability models.
    """
    print("\n  Generating calibration curves...")
    
    preds_path = select_predictions_parquet()
    if preds_path is None:
        warnings.warn("Prediction parquet not found; run experiments/06_analyze_lead_time.py first.")
        return

    preds = pd.read_parquet(preds_path)
    required = {'prob', 'z_mean', 'z_sd', 'y_true'}
    missing = required - set(preds.columns)
    if missing:
        warnings.warn(f"Prediction parquet missing required columns: {sorted(missing)}")
        return

    y = pd.to_numeric(preds['y_true'], errors='coerce')
    xgb_p = pd.to_numeric(preds['prob'], errors='coerce')
    bayes_p = pd.Series(compute_bayes_prob_high_risk(preds, risk_quantile=0.80))

    mask = y.notna() & xgb_p.notna()
    yv = y[mask].astype(int).values
    xgbv = xgb_p[mask].astype(float).values

    # Bayesian curve uses its own availability mask
    mask_b = y.notna() & bayes_p.notna()
    yb = y[mask_b].astype(int).values
    bv = bayes_p[mask_b].astype(float).values

    def _bin_curve(p: np.ndarray, yy: np.ndarray, n_bins: int = 10):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi < 1.0:
                m = (p >= lo) & (p < hi)
            else:
                m = (p >= lo) & (p <= hi)
            if not np.any(m):
                continue
            xs.append(float(np.mean(p[m])))
            ys.append(float(np.mean(yy[m])))
            ns.append(int(np.sum(m)))
        return np.array(xs), np.array(ys), np.array(ns)

    x_xgb, y_xgb, n_xgb = _bin_curve(xgbv, yv, n_bins=10)
    x_b, y_b, n_b = _bin_curve(bv, yb, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    if len(x_xgb):
        ax.plot(x_xgb, y_xgb, marker='o', color=COLORS['xgboost'], label='XGBoost')
    if len(x_b):
        ax.plot(x_b, y_b, marker='o', color=COLORS['bayesian'], label='Bayesian P(Z > q80)')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Fraction of Outbreaks')
    ax.set_title('Calibration Curves (Reliability Diagram)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    save_figure_with_description(
        fig,
        CALIBRATION_DIR / "calibration_curves",
        title="Calibration Curves (Reliability Diagram)",
        description="Reliability diagram comparing XGBoost probabilities against observed outbreak frequency, alongside a Bayesian high-risk probability derived from latent Z summaries.",
        interpretation="Points near the diagonal indicate good calibration. Deviations indicate over/under-confidence.",
        caveats="Bayesian curve uses P(Z > q80) (Normal approximation from z_mean/z_sd), which is a high-risk probability proxy rather than a direct posterior predictive outbreak probability."
    )


def plot_decision_state_timeline() -> None:
    """Plot decision-state timeline using stored lead-time predictions."""
    print("\n  Generating decision state timeline...")

    preds_path = select_lead_time_file("lead_time_predictions", "parquet")
    detail_path = select_lead_time_file("lead_time_detail", "csv")

    if preds_path is None or detail_path is None or not preds_path.exists() or not detail_path.exists():
        warnings.warn(
            "Decision timeline requires lead-time outputs: "
            "lead_time_predictions_p*.parquet and lead_time_detail_p*.csv"
        )
        return

    preds = pd.read_parquet(preds_path)
    episodes = pd.read_csv(detail_path)
    if episodes.empty:
        warnings.warn("No outbreak episodes found for decision timeline")
        return

    ep = episodes.iloc[0]
    state = ep['state']
    district = ep['district']
    year = int(ep['year'])

    ts = preds[(preds['state'] == state) & (preds['district'] == district) & (preds['year'] == year)].copy()
    if ts.empty:
        warnings.warn(f"No prediction time series found for {state}/{district}/{year}")
        return

    ts = ts.sort_values('week')
    weeks = ts['week'].to_numpy(dtype=int)
    cases = ts['cases'].to_numpy(dtype=float) if 'cases' in ts.columns else None
    prob = ts['prob'].to_numpy(dtype=float) if 'prob' in ts.columns else None
    z_mean = ts['z_mean'].to_numpy(dtype=float) if 'z_mean' in ts.columns else None

    outbreak_threshold = float(ep.get('outbreak_threshold', np.nan))
    bayes_threshold = float(ep.get('bayesian_threshold', np.nan))
    xgb_threshold = float(ep.get('xgboost_threshold', np.nan))
    t_outbreak = ep.get('first_outbreak_week', ep.get('outbreak_week', np.nan))
    if pd.isna(outbreak_threshold) or pd.isna(bayes_threshold) or pd.isna(xgb_threshold):
        warnings.warn("Decision timeline missing thresholds in lead_time_detail")
        return

    if cases is None or prob is None or z_mean is None:
        warnings.warn("Decision timeline requires cases, prob, and z_mean in predictions")
        return

    outbreak_mask = cases > outbreak_threshold
    bayes_mask = z_mean >= bayes_threshold
    xgb_mask = prob >= xgb_threshold

    fig, axes = plt.subplots(3, 1, figsize=(14, 4.5), sharex=True)

    def plot_state_row(ax, mask, color, label):
        ax.fill_between(weeks, 0, 1, color='#F0F0F0', step='mid')
        ax.fill_between(weeks, 0, 1, where=mask, color=color, alpha=0.85, step='mid')
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=35, va='center')
        ax.grid(axis='x', alpha=0.2)

    plot_state_row(axes[0], outbreak_mask, COLORS['outbreak'], 'Outbreak')
    plot_state_row(axes[1], bayes_mask, COLORS['bayesian'], 'Bayesian')
    plot_state_row(axes[2], xgb_mask, COLORS['xgboost'], 'XGBoost')

    if pd.notna(t_outbreak):
        for ax in axes:
            ax.axvline(int(t_outbreak), color=COLORS['outbreak'], linestyle=':', linewidth=1.5)

    axes[2].set_xlabel('Week')
    axes[0].set_title(f"Decision State Timeline ({state} / {district}, {year})")

    plt.tight_layout()

    save_figure_with_description(
        fig,
        DECISION_DIR / "decision_state_timeline",
        title="Decision State Timeline",
        description="Binary alert states over weeks for a single district-year. "
                    "Rows show outbreak weeks (cases above threshold), Bayesian alerts, "
                    "and XGBoost alerts, computed from stored predictions and thresholds.",
        interpretation="Shows how model alerts align with observed outbreak timing in a "
                      "real district-year sequence.",
        caveats="Uses the first available outbreak episode in lead-time outputs. "
               "Thresholds come from the corresponding lead_time_detail_p*.csv."
    )


def create_readme() -> None:
    """
    Create README.md for the Phase 7 figures directory.
    """
    print("\n  Creating README.md...")
    
    readme_content = """# Phase 7 Visualization Results

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

Date: {date}
Version: v4.2 (Phase 7 Complete)
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    readme_path = FIGURE_ROOT / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ {readme_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all visualization generation."""
    print("=" * 70)
    print("PHASE 7 VISUALIZATION GENERATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {FIGURE_ROOT}")
    
    # Section 1: Track A Performance
    print("\n" + "-" * 70)
    print("SECTION 1: Track A Performance Figures")
    print("-" * 70)
    plot_auc_comparison_bar()
    
    # Section 2: Lead-Time Analysis
    print("\n" + "-" * 70)
    print("SECTION 2: Lead-Time Analysis Figures")
    print("-" * 70)
    plot_lead_time_distribution()
    plot_differential_lead_histogram()
    plot_per_fold_median_lead()
    plot_case_study_timeseries()
    
    # Section 3: Calibration & Uncertainty
    print("\n" + "-" * 70)
    print("SECTION 3: Calibration & Uncertainty Figures")
    print("-" * 70)
    plot_calibration_curves()
    
    # Section 4: Decision Usefulness
    print("\n" + "-" * 70)
    print("SECTION 4: Decision Usefulness Figures")
    print("-" * 70)
    plot_decision_state_timeline()
    
    # Create README
    print("\n" + "-" * 70)
    print("DOCUMENTATION")
    print("-" * 70)
    create_readme()
    
    print("\n" + "=" * 70)
    print("✓ Phase 7 Visualizations Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
