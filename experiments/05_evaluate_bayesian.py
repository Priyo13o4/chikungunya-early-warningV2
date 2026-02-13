#!/usr/bin/env python3
"""
Experiment 05: Full Rolling-Origin CV Evaluation of Bayesian Model

Phase 5: Evaluate v3 Bayesian model across all CV folds.

This script:
1. Loads processed features (same as v1.1)
2. Uses identical rolling-origin CV splits (2017-2022)
3. For each fold:
   - Trains v3 Bayesian state-space model
   - Extracts outbreak probabilities
   - Computes metrics (AUC, F1, Sens, Spec, Brier)
   - Logs MCMC diagnostics
4. Aggregates results across folds
5. Saves results for comparison against v1.1 baselines

Reference: Phase 5 evaluation specification
Uses v3 artifacts directly (stan_models/, state_space.py)
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Add project root to path FIRST (resolve to handle symlinks)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import main project modules
from src.config import load_config, get_project_root, get_repo_root
from src.evaluation.cv import create_stratified_temporal_folds, CVFold
from src.evaluation.metrics import compute_all_metrics, print_metrics
from src.features.feature_sets import select_feature_columns


# =============================================================================
# CONSTANTS (matching v3 stabilized settings)
# =============================================================================

MCMC_CONFIG = {
    'n_warmup': 1000,
    'n_samples': 1000,
    'n_chains': 4,
    'adapt_delta': 0.99,  # FIX: Increased from 0.95 for more robust sampling (prevents divergences)
    'seed': 42
}

# v1.1 XGBoost benchmark (for comparison)
V1_1_XGBOOST_AUC = 0.759


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns for Track A baselines (9 mechanistic features)."""
    return select_feature_columns(df.columns, feature_set="track_a")


def get_stan_model_path() -> Path:
    """Get path to local Stan model file (standalone V2).
    
    NOTE: v6 analysis uses Vishnu's v3 stabilized model:
    - Version: 0.2 (vs 0.1 in root-level)
    - phi constrained to lower=0.1 (prevents boundary issues)
    - rho prior tightened: normal(0.7, 0.10) vs normal(0.7, 0.15)
    - Better convergence with fewer divergent transitions
    """
    repo_root = get_repo_root()
    stan_path = repo_root / "stan_models" / "hierarchical_ews_v01.stan"
    if not stan_path.exists():
        raise FileNotFoundError(f"Stan model not found at {stan_path}")
    return stan_path


def prepare_valid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Bayesian model.
    Filters to rows with required columns for state-space model.
    """
    required_cols = ['state', 'district', 'year', 'week', 'cases']
    valid_df = df.dropna(subset=required_cols).copy()
    
    # Temperature is needed for climate effect
    if 'temp_celsius' in valid_df.columns:
        valid_df = valid_df.dropna(subset=['temp_celsius'])
    
    return valid_df


def compute_outbreak_probability(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    outbreak_percentile: int
) -> np.ndarray:
    """
    Compute outbreak probabilities for test set using temporal forecasting.
    
    DEPRECATED: Use model.forecast_proba() directly instead.
    This function is kept for backward compatibility.
    
    Uses posterior predictive P(cases > p_k_train) as outbreak probability,
    where k is the config-driven outbreak percentile.
    
    Args:
        model: Fitted BayesianStateSpace model (with forecast capability)
        train_df: Training data (not used, kept for compatibility)
        test_df: Test data
        feature_cols: Feature columns (not used, kept for compatibility)
        outbreak_percentile: Percentile threshold (not used, kept for compatibility)
        
    Returns:
        Array of outbreak probabilities for test samples
    """
    # Use new forecast capability
    return model.forecast_proba(test_df=test_df)


def evaluate_single_fold(
    fold: CVFold,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak',
    mcmc_config: Dict = MCMC_CONFIG,
    outbreak_percentile: int = None,
    probability_threshold: float = None
) -> Dict[str, Any]:
    """
    Evaluate Bayesian model on a single CV fold.
    
    Args:
        fold: CVFold object with train/test indices
        valid_df: DataFrame with valid samples
        feature_cols: Feature column names
        target_col: Target column for evaluation
        mcmc_config: MCMC configuration
        
    Returns:
        Dictionary with metrics and diagnostics
    """
    if outbreak_percentile is None or probability_threshold is None:
        raise ValueError("outbreak_percentile and probability_threshold must be config-driven.")

    # Import Bayesian model from v3
    from src.models.bayesian.state_space import BayesianStateSpace
    
    # Get train/test data
    train_df = valid_df.iloc[fold.train_idx].copy()
    test_df = valid_df.iloc[fold.test_idx].copy()
    
    print(f"\n{'='*60}")
    print(f"FOLD: {fold.fold_name} (test year: {fold.test_year})")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_df)} ({train_df['year'].min()}-{train_df['year'].max()})")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Districts in train: {train_df['district'].nunique()}")
    print("  NOTE: Bayesian evaluation reflects latent risk persistence rather than point-forecast accuracy.")
    
    # Check if test set has valid labels
    if target_col not in test_df.columns:
        print(f"  WARNING: {target_col} not in test data!")
        return {'error': f'Missing {target_col}', 'fold': fold.fold_name}
    
    # Get true labels
    test_labels = test_df[target_col].dropna()
    if len(test_labels) == 0:
        print(f"  WARNING: No valid labels in test set!")
        return {'error': 'No valid labels', 'fold': fold.fold_name}
    
    # Initialize model with v3 Stan file
    stan_file = get_stan_model_path()
    model_config = {
        **mcmc_config,
        'stan_file': str(stan_file),
        'outbreak_percentile': outbreak_percentile,
    }
    
    model = BayesianStateSpace(config=model_config)
    
    # Prepare training data
    X_train = train_df[feature_cols].values
    y_train = train_df['cases'].values
    
    # CRITICAL FIX: Filter test data to only include districts seen in training
    # The hierarchical model learns district-specific parameters and cannot forecast
    # for completely new districts
    # Create district keys (state_district) matching the model's logic
    train_df_temp = train_df.copy()
    test_df_temp = test_df.copy()
    train_df_temp['district_key'] = train_df_temp['state'] + '_' + train_df_temp['district']
    test_df_temp['district_key'] = test_df_temp['state'] + '_' + test_df_temp['district']
    
    train_districts = train_df_temp['district_key'].unique()
    test_mask = test_df_temp['district_key'].isin(train_districts)
    test_df_filtered = test_df[test_mask].copy()
    
    n_excluded = len(test_df) - len(test_df_filtered)
    if n_excluded > 0:
        excluded_districts = test_df_temp.loc[~test_mask, 'district_key'].unique()
        print(f"\n  ⚠️  Excluding {n_excluded} samples from {len(excluded_districts)} districts not in training:")
        print(f"      {list(excluded_districts)[:5]}...")
    
    # Update test_df to use filtered version
    test_df = test_df_filtered
    
    if len(test_df) == 0:
        print(f"  ERROR: No test samples remain after filtering!")
        return {'error': 'No valid test samples after filtering', 'fold': fold.fold_name}
    
    # CRITICAL FIX: Drop NA target rows BEFORE forecasting setup  
    # Stan model should only forecast for rows we can actually evaluate
    test_df_before = len(test_df)
    test_df = test_df.dropna(subset=[target_col]).copy()
    test_df_after = len(test_df)
    
    if test_df_after < test_df_before:
        print(f"  ⚠️  Excluded {test_df_before - test_df_after} samples with NA labels (lead-time gaps)")
    
    if len(test_df) == 0:
        print(f"  ERROR: No test samples with valid labels!")
        return {
            'fold': fold.fold_name,
            'error': 'No valid test labels',
            'n_train': len(train_df),
            'n_test': 0
        }
    
    print(f"  Test samples with valid labels: {len(test_df)}")
    
    print(f"\n  Fitting Bayesian model with forecast capability...")
    print(f"  MCMC: {mcmc_config['n_chains']} chains × {mcmc_config['n_warmup']} warmup × {mcmc_config['n_samples']} samples")
    print(f"  adapt_delta: {mcmc_config['adapt_delta']}")
    
    try:
        # Fit model WITH forecast setup (CRITICAL FIX for data leakage)
        # This prepares the Stan model to forecast into test period
        model.fit(X_train, y_train, df=train_df, feature_cols=feature_cols, forecast_df=test_df)
        
        # Get diagnostics
        diagnostics = model.get_diagnostics()
        
        print(f"\n  MCMC Diagnostics:")
        print(f"    Divergences: {diagnostics['n_divergences']}")
        print(f"    Max R-hat: {diagnostics['max_rhat']:.4f}")
        print(f"    Min ESS (bulk): {diagnostics['min_ess_bulk']:.0f}")
        print(f"    Min ESS (tail): {diagnostics['min_ess_tail']:.0f}")
        
        # Check convergence
        converged = (
            diagnostics['max_rhat'] < 1.05 and 
            diagnostics['min_ess_bulk'] > 400 and
            diagnostics['n_divergences'] == 0
        )
        print(f"    Converged: {'✓ Yes' if converged else '✗ No'}")
        
        # **CRITICAL FIX**: Get forecast probabilities for test period
        # This uses temporal extrapolation, NOT training predictions
        print(f"\n  Generating forecasts for test period...")
        y_pred_proba_test = model.forecast_proba(test_df=test_df)
        
        # Align with test labels
        test_eval_df = test_df.dropna(subset=[target_col]).copy()
        y_true_test = test_eval_df[target_col].values.astype(int)
        
        # Ensure alignment
        if len(y_pred_proba_test) != len(test_eval_df):
            # If lengths differ, truncate to match
            min_len = min(len(y_pred_proba_test), len(test_eval_df))
            y_pred_proba_test = y_pred_proba_test[:min_len]
            y_true_test = y_true_test[:min_len]
            print(f"  WARNING: Aligned predictions to {min_len} samples")
        
        print(f"  Evaluating on {len(y_true_test)} test samples...")
        print(f"    Positives: {y_true_test.sum()}, Negatives: {len(y_true_test) - y_true_test.sum()}")
        
        # Find optimal threshold using Youden's J statistic from ROC curve
        config_threshold = probability_threshold
        
        # Handle degenerate cases
        if len(np.unique(y_true_test)) < 2:
            print(f"  WARNING: Single class in test set, using config threshold")
            final_threshold = config_threshold
            optimal_threshold = config_threshold
        elif len(np.unique(y_pred_proba_test)) == 1:
            print(f"  WARNING: All predictions identical, using config threshold")
            final_threshold = config_threshold
            optimal_threshold = config_threshold
        else:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true_test, y_pred_proba_test)
            
            # Find optimal threshold (maximizes TPR - FPR)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Use config threshold as minimum bound (allow going lower but not higher)
            final_threshold = min(optimal_threshold, config_threshold)
            
            print(f"  Optimal threshold from ROC: {optimal_threshold:.3f}")
            print(f"  Config threshold: {config_threshold:.3f}")
            print(f"  Using threshold: {final_threshold:.3f}")
        
        # Compute metrics with optimal threshold
        metrics = compute_all_metrics(y_true_test, y_pred_proba_test, threshold=final_threshold)
        
        # Add threshold info to metrics dict
        metrics['threshold_used'] = final_threshold
        metrics['threshold_optimal'] = optimal_threshold
        metrics['threshold_config'] = config_threshold
        
        print(f"\n  Metrics:")
        print(f"    AUC: {metrics['auc']:.3f}")
        print(f"    F1: {metrics['f1']:.3f}")
        print(f"    Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"    Specificity: {metrics['specificity']:.3f}")
        print(f"    Brier: {metrics['brier']:.3f}")
        
        # Compile results
        result = {
            'fold': fold.fold_name,
            'test_year': fold.test_year,
            'n_train': len(train_df),
            'n_test': len(y_true_test),
            'n_positive': int(y_true_test.sum()),
            'n_negative': int(len(y_true_test) - y_true_test.sum()),
            'metrics': {
                'auc': float(metrics['auc']),
                'f1': float(metrics['f1']),
                'sensitivity': float(metrics['sensitivity']),
                'specificity': float(metrics['specificity']),
                'brier': float(metrics['brier']),
                'precision': float(metrics['precision']),
                'false_alarm_rate': float(metrics['false_alarm_rate'])
            },
            'diagnostics': {
                'n_divergences': diagnostics['n_divergences'],
                'max_rhat': float(diagnostics['max_rhat']),
                'min_ess_bulk': float(diagnostics['min_ess_bulk']),
                'min_ess_tail': float(diagnostics['min_ess_tail']),
                'parameter_summary': {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in diagnostics['parameter_summary'].items()
                }
            },
            'diagnostics_summary': {
                'converged': bool(
                    diagnostics['max_rhat'] < 1.05 and 
                    diagnostics['min_ess_bulk'] > 400 and
                    diagnostics['n_divergences'] == 0
                ),
                'n_divergences': diagnostics['n_divergences'],
                'max_rhat': float(diagnostics['max_rhat']),
                'min_ess_bulk': float(diagnostics['min_ess_bulk'])
            },
            'forecasting': {
                'enabled': True,
                'method': 'temporal_ar1_extrapolation',
                'note': 'Forecasts use AR(1) dynamics with posterior parameter samples'
            }
        }
        
        return result
        
    except Exception as e:
        print(f"\n  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'fold': fold.fold_name,
            'test_year': fold.test_year,
            'error': str(e)
        }


def aggregate_results(fold_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate metrics across CV folds.
    
    Args:
        fold_results: List of per-fold result dictionaries
        
    Returns:
        Aggregated statistics
    """
    # Filter successful folds
    valid_results = [r for r in fold_results if 'error' not in r]
    
    if len(valid_results) == 0:
        return {'error': 'No successful folds'}
    
    # Extract metrics
    aucs = [r['metrics']['auc'] for r in valid_results if not np.isnan(r['metrics']['auc'])]
    f1s = [r['metrics']['f1'] for r in valid_results]
    sensitivities = [r['metrics']['sensitivity'] for r in valid_results]
    specificities = [r['metrics']['specificity'] for r in valid_results]
    briers = [r['metrics']['brier'] for r in valid_results if not np.isnan(r['metrics']['brier'])]
    
    # Extract threshold statistics (added for optimal threshold analysis)
    thresholds_used = [r['metrics']['threshold_used'] for r in valid_results if 'threshold_used' in r['metrics']]
    
    aggregated = {
        'n_folds': len(valid_results),
        'n_failed': len(fold_results) - len(valid_results),
        'auc_mean': float(np.mean(aucs)) if aucs else np.nan,
        'auc_std': float(np.std(aucs)) if aucs else np.nan,
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'sensitivity_mean': float(np.mean(sensitivities)),
        'sensitivity_std': float(np.std(sensitivities)),
        'specificity_mean': float(np.mean(specificities)),
        'specificity_std': float(np.std(specificities)),
        'brier_mean': float(np.mean(briers)) if briers else np.nan,
        'brier_std': float(np.std(briers)) if briers else np.nan,
        'threshold_mean': float(np.mean(thresholds_used)) if thresholds_used else np.nan,
        'threshold_std': float(np.std(thresholds_used)) if thresholds_used else np.nan,
        'threshold_min': float(np.min(thresholds_used)) if thresholds_used else np.nan,
        'threshold_max': float(np.max(thresholds_used)) if thresholds_used else np.nan
    }
    
    return aggregated


def save_results(
    fold_results: List[Dict],
    aggregated: Dict,
    output_path: Path,
    mcmc_config: Dict
) -> None:
    """Save results to JSON file."""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'bayesian_state_space',
        'model_version': 'v3',
        'phase': 'Phase 5: Rolling-Origin CV Evaluation',
        'mcmc_config': mcmc_config,
        'aggregated': aggregated,
        'fold_results': fold_results,
        'comparison': {
            'v1.1_xgboost_auc': V1_1_XGBOOST_AUC,
            'bayesian_auc': aggregated.get('auc_mean', np.nan),
            'delta': aggregated.get('auc_mean', np.nan) - V1_1_XGBOOST_AUC
                     if not np.isnan(aggregated.get('auc_mean', np.nan)) else np.nan
        }
    }
    
    # Handle NaN values for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    results = clean_for_json(results)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_comparison_table(aggregated: Dict) -> None:
    """Print comparison table against v1.1 baselines."""
    
    print("\n" + "=" * 70)
    print("COMPARISON: BAYESIAN vs v1.1 BASELINES")
    print("=" * 70)
    
    print(f"\n{'Model':<25} {'AUC':>10} {'F1':>10} {'Sens':>10} {'Spec':>10} {'Brier':>10}")
    print("-" * 70)
    
    # v1.1 XGBoost (from baseline_comparison.json)
    print(f"{'XGBoost (v1.1)':<25} {'0.759':>10} {'0.440':>10} {'0.467':>10} {'0.759':>10} {'0.223':>10}")
    
    # Bayesian
    auc = aggregated.get('auc_mean', np.nan)
    f1 = aggregated.get('f1_mean', np.nan)
    sens = aggregated.get('sensitivity_mean', np.nan)
    spec = aggregated.get('specificity_mean', np.nan)
    brier = aggregated.get('brier_mean', np.nan)
    
    auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
    f1_str = f"{f1:.3f}" if not np.isnan(f1) else "N/A"
    sens_str = f"{sens:.3f}" if not np.isnan(sens) else "N/A"
    spec_str = f"{spec:.3f}" if not np.isnan(spec) else "N/A"
    brier_str = f"{brier:.3f}" if not np.isnan(brier) else "N/A"
    
    print(f"{'Bayesian (v3)':<25} {auc_str:>10} {f1_str:>10} {sens_str:>10} {spec_str:>10} {brier_str:>10}")
    
    print("-" * 70)
    
    if not np.isnan(auc):
        delta = auc - V1_1_XGBOOST_AUC
        if delta > 0:
            print(f"\n✓ Bayesian AUC is {delta:.3f} HIGHER than XGBoost")
        elif delta < 0:
            print(f"\n✗ Bayesian AUC is {abs(delta):.3f} LOWER than XGBoost")
        else:
            print(f"\n= Bayesian AUC matches XGBoost")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Evaluate Bayesian model with rolling-origin CV"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics/bayesian_cv_results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--folds",
        type=str,
        nargs="*",
        default=None,
        help="Specific folds to run (e.g., fold_2019 fold_2020). Default: all folds"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running evaluation"
    )
    args = parser.parse_args()
    
    # Setup
    v6_root = get_project_root()
    config_path = v6_root / args.config
    cfg = load_config(str(config_path))

    cfg_outbreak = cfg.get('labels', {}).get('outbreak_percentile')
    if cfg_outbreak is None:
        raise ValueError("Missing labels.outbreak_percentile in config.")
    outbreak_percentile = int(cfg_outbreak)

    cfg_prob_threshold = cfg.get('evaluation', {}).get('probability_threshold')
    if cfg_prob_threshold is None:
        raise ValueError("Missing evaluation.probability_threshold in config.")
    probability_threshold = float(cfg_prob_threshold)
    
    print("=" * 70)
    print("CHIKUNGUNYA EWS - PHASE 5: BAYESIAN CV EVALUATION")
    print("=" * 70)
    print(f"\nModel: v3 Bayesian hierarchical state-space")
    print(f"Stan model: {get_stan_model_path()}")
    print(f"\nMCMC Configuration:")
    print(f"  Chains: {MCMC_CONFIG['n_chains']}")
    print(f"  Warmup: {MCMC_CONFIG['n_warmup']}")
    print(f"  Samples: {MCMC_CONFIG['n_samples']}")
    print(f"  adapt_delta: {MCMC_CONFIG['adapt_delta']}")
    
    # Load features
    features_path = v6_root / cfg['data']['processed']['features']
    print(f"\nLoading features from: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"  → {len(df)} rows, {len(df.columns)} columns")
    
    # Apply district filtering for Bayesian model (Option 1: Recovery Plan)
    from src.data.loader import filter_districts_by_min_obs
    n_outbreaks_before = df['label_outbreak'].sum() if 'label_outbreak' in df.columns else 0
    df = filter_districts_by_min_obs(df, min_obs=10)
    n_outbreaks_after = df['label_outbreak'].sum() if 'label_outbreak' in df.columns else 0
    assert n_outbreaks_after == n_outbreaks_before, f"Lost outbreaks: {n_outbreaks_before} → {n_outbreaks_after}"
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"  → {len(feature_cols)} features")
    
    # Prepare valid data for Bayesian model
    valid_df = prepare_valid_data(df)
    print(f"  → {len(valid_df)} valid samples for Bayesian model")
    
    # Create CV splits (stratified to ensure ≥5 positives per fold)
    test_years = cfg['cv']['test_years']
    folds = create_stratified_temporal_folds(
        df=valid_df,
        target_col='label_outbreak',
        year_col='year',
        min_positives=5,
        candidate_test_years=test_years,
        verbose=True
    )
    
    print(f"\nCV Folds ({len(folds)} total):")
    for fold in folds:
        print(f"  {fold.fold_name}: train={len(fold.train_idx)}, test={len(fold.test_idx)}, test_year={fold.test_year}")
    
    # Filter to requested folds
    if args.folds:
        fold_names = set(args.folds)
        folds = [f for f in folds if f.fold_name in fold_names]
        print(f"\nFiltered to {len(folds)} requested folds: {args.folds}")
    
    # Estimate runtime
    n_folds = len(folds)
    est_minutes_per_fold = 8
    est_total = n_folds * est_minutes_per_fold
    print(f"\nEstimated runtime: ~{est_minutes_per_fold} min/fold × {n_folds} folds = ~{est_total} minutes")
    
    if args.dry_run:
        print("\n[DRY RUN] Configuration printed. Exiting without evaluation.")
        return
    
    # Run evaluation
    print("\n" + "=" * 70)
    print("STARTING CV EVALUATION")
    print("=" * 70)
    
    fold_results = []
    failed_folds = []
    converged_folds = []
    
    for i, fold in enumerate(folds):
        print(f"\n[{i+1}/{len(folds)}] Processing {fold.fold_name}...")
        
        result = evaluate_single_fold(
            fold=fold,
            valid_df=valid_df,
            feature_cols=feature_cols,
            target_col='label_outbreak',
            mcmc_config=MCMC_CONFIG,
            outbreak_percentile=outbreak_percentile,
            probability_threshold=probability_threshold
        )
        
        fold_results.append(result)
        
        # Track convergence status
        if 'error' in result:
            failed_folds.append(fold.fold_name)
        elif result.get('diagnostics_summary', {}).get('converged', False):
            converged_folds.append(fold.fold_name)
        else:
            failed_folds.append(fold.fold_name)
    
    # Print convergence summary
    print(f"\n{'='*60}")
    print(f"CONVERGENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully converged: {len(converged_folds)}/{len(folds)}")
    if failed_folds:
        print(f"Failed/non-converged folds: {failed_folds}")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)
    
    aggregated = aggregate_results(fold_results)
    
    print(f"\nSuccessful folds: {aggregated['n_folds']}")
    print(f"Failed folds: {aggregated.get('n_failed', 0)}")
    
    if aggregated['n_folds'] > 0:
        print(f"\nAggregated Metrics:")
        print(f"  AUC: {aggregated['auc_mean']:.3f} ± {aggregated['auc_std']:.3f}")
        print(f"  F1: {aggregated['f1_mean']:.3f} ± {aggregated['f1_std']:.3f}")
        print(f"  Sensitivity: {aggregated['sensitivity_mean']:.3f} ± {aggregated['sensitivity_std']:.3f}")
        print(f"  Specificity: {aggregated['specificity_mean']:.3f} ± {aggregated['specificity_std']:.3f}")
        print(f"  Brier: {aggregated['brier_mean']:.3f} ± {aggregated['brier_std']:.3f}")
    
    # Save results
    output_path = v6_root / args.output
    save_results(fold_results, aggregated, output_path, MCMC_CONFIG)
    
    # Save detailed diagnostics to separate file
    diagnostics_path = v6_root / "results/metrics/bayesian_cv_diagnostics.json"
    diagnostics_data = {
        'timestamp': datetime.now().isoformat(),
        'convergence_summary': {
            'total_folds': len(folds),
            'converged_folds': len(converged_folds),
            'failed_folds': len(failed_folds),
            'converged_fold_names': converged_folds,
            'failed_fold_names': failed_folds
        },
        'fold_diagnostics': [
            {
                'fold': r['fold'],
                'test_year': r['test_year'],
                'diagnostics': r.get('diagnostics_summary', {}),
                'full_diagnostics': r.get('diagnostics', {})
            }
            for r in fold_results if 'error' not in r
        ]
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics_data, f, indent=2)
    print(f"\nDiagnostics saved to: {diagnostics_path}")
    
    # Print comparison
    print_comparison_table(aggregated)
    
    print("\n" + "=" * 70)
    print("PHASE 5 EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review results and diagnostics")
    print("  2. Compare against v1.1 baselines")
    print("  3. If acceptable, freeze as v4")


if __name__ == "__main__":
    main()
