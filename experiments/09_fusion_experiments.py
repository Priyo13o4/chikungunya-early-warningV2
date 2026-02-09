#!/usr/bin/env python3
"""Experiment 09: Fusion Experiments (Phase 7+ aligned)

This script evaluates fusion rules on top of the *stored out-of-sample*
predictions produced by Experiment 06:
  - XGBoost probability: `prob`
  - Bayesian latent risk summary: `z_mean`, `z_sd`
  - True label: `y_true`

This avoids refitting Bayesian models or using district-level proxies for test
weeks (both were causing placeholder-ish / misaligned behavior).

Fusion Strategies
- Gated decision fusion: if Bayesian risk is high, trust Bayes; else trust ML.
- Weighted ensemble: α * P_bayes + (1-α) * P_ml, grid-search α.

Output
- results/analysis/fusion_results_p{p}.json (by default)
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    cohen_kappa_score,
    brier_score_loss,
)

def get_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / 'src').exists() and (candidate / 'config').exists() and (candidate / 'results').exists():
            return candidate
    return start


project_root = get_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(project_root))

from src.config import load_config


def _preds_path(percentile: int) -> Path:
    analysis_dir = project_root / 'results' / 'analysis'
    path = analysis_dir / f'lead_time_predictions_p{percentile}.parquet'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Run experiments/06_analyze_lead_time.py first.'
        )
    return path


def load_predictions(percentile: int) -> pd.DataFrame:
    df = pd.read_parquet(_preds_path(percentile))
    needed = {'fold', 'prob', 'z_mean', 'z_sd', 'y_true'}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f'Predictions parquet missing required columns: {sorted(missing)}')
    return df


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def compute_bayes_prob_high_risk(df: pd.DataFrame, risk_quantile: float = 0.80) -> np.ndarray:
    """Compute P(Z > q) via Normal(mu=z_mean, sd=z_sd)."""
    work = df[['z_mean', 'z_sd']].copy()
    work = work.dropna()
    if work.empty:
        return np.full(shape=(len(df),), fill_value=np.nan, dtype=float)

    q = float(np.percentile(work['z_mean'].astype(float).values, risk_quantile * 100))
    mu = df['z_mean'].astype(float).values
    sd = np.maximum(df['z_sd'].astype(float).values, 1e-6)
    z = (q - mu) / sd
    return 1.0 - _normal_cdf(z)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        out['auc'] = float(roc_auc_score(y_true, y_prob))
        out['aupr'] = float(average_precision_score(y_true, y_prob))
    else:
        out['auc'] = 0.5
        out['aupr'] = 0.0

    out['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    out['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    out['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    out['kappa'] = float(cohen_kappa_score(y_true, y_pred))
    out['brier'] = float(brier_score_loss(y_true, y_prob))
    return out


def fusion_strategy_b_gated_decision(
    fold_df: pd.DataFrame,
    prob_threshold: float,
    gate_threshold: float = 0.8,
    risk_quantile: float = 0.80,
) -> Dict[str, Any]:
    """Gated fusion: when Bayes is high-risk, trust Bayes; else trust XGB."""
    valid_mask = pd.to_numeric(fold_df['y_true'], errors='coerce').notna()
    df = fold_df.loc[valid_mask].copy()
    if df.empty:
        raise RuntimeError('No valid y_true rows in this fold')

    y_true = pd.to_numeric(df['y_true'], errors='coerce').astype(int).values
    xgb = df['prob'].astype(float).values
    bayes = compute_bayes_prob_high_risk(df, risk_quantile=risk_quantile)

    fused = np.where(bayes >= gate_threshold, bayes, xgb)
    metrics = compute_metrics(y_true, fused, threshold=prob_threshold)
    bayes_usage = float((bayes >= gate_threshold).mean()) if len(bayes) else 0.0

    return {
        'strategy': 'gated_decision',
        'gate_threshold': gate_threshold,
        'risk_quantile': risk_quantile,
        'bayes_usage_rate': bayes_usage,
        'metrics': metrics,
    }


def fusion_strategy_c_weighted_ensemble(
    fold_df: pd.DataFrame,
    prob_threshold: float,
    risk_quantile: float = 0.80,
    alpha_values: List[float] | None = None,
) -> Dict[str, Any]:
    """
    Strategy C: Weighted Ensemble
    weighted_prob = α * P_bayes + (1-α) * P_xgboost
    Try different α values.
    """
    valid_mask = pd.to_numeric(fold_df['y_true'], errors='coerce').notna()
    df = fold_df.loc[valid_mask].copy()
    if df.empty:
        raise RuntimeError('No valid y_true rows in this fold')

    y_true = pd.to_numeric(df['y_true'], errors='coerce').astype(int).values
    y_pred_proba_xgb = df['prob'].astype(float).values
    y_pred_proba_bayes = compute_bayes_prob_high_risk(df, risk_quantile=risk_quantile)

    alpha_values = alpha_values or [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = []
    
    for alpha in alpha_values:
        y_pred_proba_weighted = alpha * y_pred_proba_bayes + (1 - alpha) * y_pred_proba_xgb
        metrics = compute_metrics(y_true, y_pred_proba_weighted, threshold=prob_threshold)
        
        results.append({
            'alpha': alpha,
            'metrics': metrics
        })
    
    best_result = max(results, key=lambda x: x['metrics']['aupr']) if results else None
    
    return {
        'strategy': 'weighted_ensemble',
        'alpha_grid': alpha_values,
        'results': results,
        'best_alpha': best_result['alpha'] if best_result else None,
        'best_metrics': best_result['metrics'] if best_result else None,
        'risk_quantile': risk_quantile,
    }


# =============================================================================
# MAIN FUSION EXPERIMENTS
# =============================================================================

def run_fusion_experiments_for_fold(
    fold_df: pd.DataFrame,
    fold_name: str,
    prob_threshold: float,
) -> Dict[str, Any]:
    """
    Run all fusion strategies for one CV fold.
    """
    print(f"\n{'='*60}")
    print(f"Fusion Experiments: {fold_name}")
    print(f"{'='*60}")
    
    results = {
        'fold': fold_name,
        'n_rows': int(len(fold_df)),
        'strategies': {}
    }
    
    try:
        results['strategies']['gated_decision'] = fusion_strategy_b_gated_decision(
            fold_df,
            prob_threshold=prob_threshold,
        )
    except Exception as e:
        print(f"    ✗ Gated decision failed: {e}")
        results['strategies']['gated_decision'] = {'error': str(e)}
    
    try:
        results['strategies']['weighted_ensemble'] = fusion_strategy_c_weighted_ensemble(
            fold_df,
            prob_threshold=prob_threshold,
        )
    except Exception as e:
        print(f"    ✗ Weighted ensemble failed: {e}")
        results['strategies']['weighted_ensemble'] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 6 Task 4: Fusion Experiments')
    parser.add_argument('--config', type=str, default='config/config_default.yaml',
                       help='Path to config file')
    parser.add_argument('--outbreak-percentile', type=int, default=75,
                       help='Outbreak percentile used in Experiment 06 outputs')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: results/analysis/fusion_results_p{p}.json)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("Loading stored predictions from Experiment 06...")
    preds = load_predictions(args.outbreak_percentile)
    print(f"Loaded {len(preds)} prediction rows")

    prob_threshold = float(config.get('evaluation', {}).get('probability_threshold', 0.5))
    
    all_results = []
    for fold_name, fold_df in preds.groupby('fold', sort=False):
        fold_result = run_fusion_experiments_for_fold(
            fold_df=fold_df,
            fold_name=str(fold_name),
            prob_threshold=prob_threshold,
        )
        all_results.append(fold_result)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATED FUSION RESULTS")
    print(f"{'='*60}")
    
    # Collect metrics across folds for each strategy
    strategy_names = ['gated_decision', 'weighted_ensemble']
    aggregated = {}
    
    for strategy in strategy_names:
        strategy_metrics = []
        
        for fold_result in all_results:
            if strategy not in fold_result['strategies']:
                continue
            strat = fold_result['strategies'][strategy]
            metrics = strat.get('metrics')
            if strategy == 'weighted_ensemble':
                metrics = strat.get('best_metrics')
            if metrics:
                strategy_metrics.append(metrics)
        
        if len(strategy_metrics) > 0:
            # Average metrics
            avg_metrics = {}
            for key in strategy_metrics[0].keys():
                values = [m[key] for m in strategy_metrics if m[key] is not None]
                if len(values) > 0:
                    avg_metrics[f'{key}_mean'] = float(np.mean(values))
                    avg_metrics[f'{key}_std'] = float(np.std(values))
            
            aggregated[strategy] = avg_metrics
    
    for strategy, metrics in aggregated.items():
        print(f"\n{strategy.upper()}:")
        if 'auc_mean' in metrics:
            print(f"  AUC: {metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
        if 'aupr_mean' in metrics:
            print(f"  AUPR: {metrics['aupr_mean']:.3f} ± {metrics['aupr_std']:.3f}")
        if 'f1_mean' in metrics:
            print(f"  F1: {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Experiment 09 - Fusion Experiments (prediction-level)',
        'input': {
            'predictions_parquet': str(_preds_path(args.outbreak_percentile)),
            'probability_threshold': prob_threshold,
        },
        'strategies': {
            'gated_decision': 'Use Bayesian P(Z>q) when high risk, else XGBoost',
            'weighted_ensemble': 'Weighted combination of probabilities',
        },
        'aggregated': aggregated,
        'fold_results': all_results,
    }

    if args.output:
        output_path = project_root / args.output
    else:
        output_path = project_root / 'results' / 'analysis' / f'fusion_results_p{args.outbreak_percentile}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
