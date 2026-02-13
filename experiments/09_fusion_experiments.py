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
    """Compute Bayesian risk scores from forecast predictions.
    
    In forecast mode, z_mean IS the forecast (point estimate of latent risk)
    and z_sd is the forecast uncertainty. We apply sigmoid transformation
    to convert log-risk to probability [0,1] for fusion compatibility.
    """
    work = df[['z_mean', 'z_sd']].copy()
    work = work.dropna()
    if work.empty:
        return np.full(shape=(len(df),), fill_value=np.nan, dtype=float)

    # Apply sigmoid transformation to convert log-risk to probability [0,1]
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    
    mu = df['z_mean'].astype(float).values
    return _sigmoid(mu)


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


def weighted_ensemble_safe(
    bayes_prob: np.ndarray,
    xgb_prob: np.ndarray,
    bayes_auc: float,
    xgb_auc: float,
    fallback: str = 'xgboost'
) -> tuple:
    """
    NaN-safe weighted ensemble combining Bayesian and XGBoost predictions.
    
    Args:
        bayes_prob: Bayesian outbreak probabilities
        xgb_prob: XGBoost outbreak probabilities
        bayes_auc: Bayesian model AUC (can be NaN)
        xgb_auc: XGBoost model AUC (can be NaN)
        fallback: Which model to use when both AUCs are NaN ('bayesian' or 'xgboost')
    
    Returns:
        tuple: (combined predictions, fusion method description)
    
    Fallback Strategy:
        - Both AUCs valid → AUC-weighted average
        - One AUC is NaN → Use the valid model only
        - Both AUCs NaN → Use fallback model (default: xgboost)
    """
    # Check for NaN cases
    bayes_valid = not np.isnan(bayes_auc) and bayes_auc > 0
    xgb_valid = not np.isnan(xgb_auc) and xgb_auc > 0
    
    if bayes_valid and xgb_valid:
        # Both models valid: AUC-weighted average
        total_auc = bayes_auc + xgb_auc
        weight_bayes = bayes_auc / total_auc
        weight_xgb = xgb_auc / total_auc
        
        combined = weight_bayes * bayes_prob + weight_xgb * xgb_prob
        method = f"weighted (Bayes {weight_bayes:.2f}, XGB {weight_xgb:.2f})"
        
    elif bayes_valid and not xgb_valid:
        # Only Bayesian valid: use Bayesian
        combined = bayes_prob
        method = "bayesian_only (XGB AUC=NaN)"
        
    elif xgb_valid and not bayes_valid:
        # Only XGBoost valid: use XGBoost
        combined = xgb_prob
        method = "xgboost_only (Bayes AUC=NaN)"
        
    else:
        # Both invalid: use fallback
        combined = xgb_prob if fallback == 'xgboost' else bayes_prob
        method = f"{fallback}_fallback (both AUC=NaN)"
    
    return combined, method


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
    Strategy C: Weighted Ensemble (NaN-safe version)
    weighted_prob = α * P_bayes + (1-α) * P_xgboost
    
    Handles single-class folds where AUC cannot be computed.
    Uses AUC-based weighting when both models are valid, falls back
    to best individual model otherwise.
    """
    valid_mask = pd.to_numeric(fold_df['y_true'], errors='coerce').notna()
    df = fold_df.loc[valid_mask].copy()
    if df.empty:
        raise RuntimeError('No valid y_true rows in this fold')

    y_true = pd.to_numeric(df['y_true'], errors='coerce').astype(int).values
    y_pred_proba_xgb = df['prob'].astype(float).values
    y_pred_proba_bayes = compute_bayes_prob_high_risk(df, risk_quantile=risk_quantile)

    # Calculate individual model AUCs (may be NaN for single-class folds)
    try:
        if len(np.unique(y_true)) > 1:
            bayes_auc = roc_auc_score(y_true, y_pred_proba_bayes)
        else:
            bayes_auc = np.nan
    except (ValueError, RuntimeError):
        bayes_auc = np.nan
    
    try:
        if len(np.unique(y_true)) > 1:
            xgb_auc = roc_auc_score(y_true, y_pred_proba_xgb)
        else:
            xgb_auc = np.nan
    except (ValueError, RuntimeError):
        xgb_auc = np.nan
    
    # Use NaN-safe weighted ensemble
    y_pred_proba_weighted, fusion_method = weighted_ensemble_safe(
        bayes_prob=y_pred_proba_bayes,
        xgb_prob=y_pred_proba_xgb,
        bayes_auc=bayes_auc,
        xgb_auc=xgb_auc,
        fallback='xgboost'
    )
    
    # Compute metrics for weighted ensemble
    try:
        metrics = compute_metrics(y_true, y_pred_proba_weighted, threshold=prob_threshold)
    except Exception as e:
        print(f"    ⚠️  Metrics computation failed: {e}")
        metrics = {'error': str(e)}
    
    # Also try grid search over alpha values (for comparison)
    alpha_values = alpha_values or [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    alpha_results = []
    
    for alpha in alpha_values:
        try:
            y_pred_alpha = alpha * y_pred_proba_bayes + (1 - alpha) * y_pred_proba_xgb
            # Skip if produces NaN
            if not np.any(np.isnan(y_pred_alpha)):
                alpha_metrics = compute_metrics(y_true, y_pred_alpha, threshold=prob_threshold)
                alpha_results.append({
                    'alpha': alpha,
                    'metrics': alpha_metrics
                })
        except Exception:
            pass  # Skip invalid alpha values
    
    best_alpha_result = max(alpha_results, key=lambda x: x['metrics'].get('aupr', 0.0)) if alpha_results else None
    
    return {
        'strategy': 'weighted_ensemble',
        'fusion_method': fusion_method,
        'bayes_auc': float(bayes_auc) if not np.isnan(bayes_auc) else None,
        'xgb_auc': float(xgb_auc) if not np.isnan(xgb_auc) else None,
        'metrics': metrics,
        'alpha_grid_search': {
            'alpha_values': alpha_values,
            'results': alpha_results,
            'best_alpha': best_alpha_result['alpha'] if best_alpha_result else None,
            'best_metrics': best_alpha_result['metrics'] if best_alpha_result else None,
        },
        'risk_quantile': risk_quantile,
    }


# =============================================================================
# FUSION DIAGNOSTICS
# =============================================================================

def print_fusion_summary(all_results: List[Dict[str, Any]]) -> None:
    """
    Print summary of fusion strategies used per fold.
    """
    print(f"\n{'='*60}")
    print("FUSION STRATEGY SUMMARY")
    print(f"{'='*60}")
    
    print("\nWeighted Ensemble Methods Used:")
    for fold_result in all_results:
        fold_name = fold_result['fold']
        if 'weighted_ensemble' not in fold_result['strategies']:
            print(f"  {fold_name}: ❌ Not executed")
            continue
        
        strat = fold_result['strategies']['weighted_ensemble']
        if 'error' in strat:
            print(f"  {fold_name}: ❌ Failed - {strat['error']}")
        elif 'fusion_method' in strat:
            method = strat['fusion_method']
            auc = strat.get('metrics', {}).get('auc', 'N/A')
            if isinstance(auc, float):
                print(f"  {fold_name}: {method} (AUC={auc:.3f})")
            else:
                print(f"  {fold_name}: {method}")
        else:
            print(f"  {fold_name}: ✓ Success")
    
    # Count fusion method types
    method_counts = {}
    valid_folds = 0
    for fold_result in all_results:
        if 'weighted_ensemble' not in fold_result['strategies']:
            continue
        strat = fold_result['strategies']['weighted_ensemble']
        if 'error' in strat or 'fusion_method' not in strat:
            continue
        
        valid_folds += 1
        method = strat['fusion_method']
        # Extract method type (e.g., "weighted", "xgboost_only", "fallback")
        method_type = method.split('(')[0].strip()
        method_counts[method_type] = method_counts.get(method_type, 0) + 1
    
    if method_counts:
        print(f"\nFusion Method Distribution ({valid_folds} valid folds):")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"  {method}: {count} folds")


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
        print(f"    ✓ Gated decision completed")
    except Exception as e:
        print(f"    ✗ Gated decision failed: {e}")
        results['strategies']['gated_decision'] = {'error': str(e)}
    
    try:
        result = fusion_strategy_c_weighted_ensemble(
            fold_df,
            prob_threshold=prob_threshold,
        )
        results['strategies']['weighted_ensemble'] = result
        
        # Print fusion method used
        fusion_method = result.get('fusion_method', 'unknown')
        print(f"    ✓ Weighted ensemble: {fusion_method}")
        
        # Print AUC if available
        if 'metrics' in result and 'auc' in result['metrics']:
            auc = result['metrics']['auc']
            print(f"      AUC: {auc:.3f}")
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
    
    # Print fusion summary
    print_fusion_summary(all_results)
    
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
            
            # Get metrics based on strategy
            if strategy == 'weighted_ensemble':
                metrics = strat.get('metrics')
            else:
                metrics = strat.get('metrics')
            
            if metrics and 'error' not in metrics:
                strategy_metrics.append(metrics)
        
        if len(strategy_metrics) > 0:
            # Average metrics
            avg_metrics = {}
            for key in strategy_metrics[0].keys():
                values = [m[key] for m in strategy_metrics if key in m and m[key] is not None]
                if len(values) > 0:
                    avg_metrics[f'{key}_mean'] = float(np.mean(values))
                    avg_metrics[f'{key}_std'] = float(np.std(values))
            
            avg_metrics['n_valid_folds'] = len(strategy_metrics)
            aggregated[strategy] = avg_metrics
    
    for strategy, metrics in aggregated.items():
        print(f"\n{strategy.upper()}:")
        n_folds = metrics.get('n_valid_folds', 0)
        print(f"  Valid folds: {n_folds}")
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


def test_nan_safe_fusion():
    """
    Validation test for NaN-safe weighted ensemble.
    Tests all 4 scenarios: both valid, one NaN, both NaN.
    """
    print(f"\n{'='*60}")
    print("VALIDATION: NaN-Safe Fusion Test Cases")
    print(f"{'='*60}\n")
    
    # Test data
    bayes_prob = np.array([0.3, 0.7, 0.5, 0.6])
    xgb_prob = np.array([0.4, 0.8, 0.6, 0.7])
    
    test_cases = [
        {
            'name': 'Both valid',
            'bayes_auc': 0.6,
            'xgb_auc': 0.75,
            'expected_method': 'weighted'
        },
        {
            'name': 'Bayes NaN',
            'bayes_auc': np.nan,
            'xgb_auc': 0.75,
            'expected_method': 'xgboost_only'
        },
        {
            'name': 'XGB NaN',
            'bayes_auc': 0.6,
            'xgb_auc': np.nan,
            'expected_method': 'bayesian_only'
        },
        {
            'name': 'Both NaN',
            'bayes_auc': np.nan,
            'xgb_auc': np.nan,
            'expected_method': 'fallback'
        }
    ]
    
    all_passed = True
    for i, case in enumerate(test_cases, 1):
        combined, method = weighted_ensemble_safe(
            bayes_prob, xgb_prob,
            case['bayes_auc'], case['xgb_auc']
        )
        
        # Check no NaN in output
        has_nan = np.any(np.isnan(combined))
        expected_in_method = case['expected_method'] in method
        
        status = "\u2713" if (not has_nan and expected_in_method) else "\u2717"
        if has_nan or not expected_in_method:
            all_passed = False
        
        print(f"Test {i}: {case['name']}")
        print(f"  Bayes AUC: {case['bayes_auc']}")
        print(f"  XGB AUC: {case['xgb_auc']}")
        print(f"  Result: {method}")
        print(f"  NaN-free: {not has_nan} {status}")
        print(f"  Expected method type: {case['expected_method']} {'[\u2713]' if expected_in_method else '[\u2717]'}")
        print()
    
    print(f"{'='*60}")
    if all_passed:
        print("\u2713 All validation tests PASSED")
    else:
        print("\u2717 Some validation tests FAILED")
    print(f"{'='*60}\n")
    
    return all_passed


if __name__ == '__main__':
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_nan_safe_fusion()
    else:
        main()
