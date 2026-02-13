#!/usr/bin/env python3
"""
Experiment 04: Train Bayesian State-Space Model

Phase 4.1: Single-fold test to validate model compilation and diagnostics.

This script:
1. Loads processed features
2. Selects ONE CV fold (fold_2019)
3. Fits the Bayesian model
4. Reports MCMC diagnostics
5. Generates one posterior predictive example

Reference: Phase 4 design specification
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, get_project_root, get_repo_root
from src.evaluation.cv import create_stratified_temporal_folds
from src.features.feature_sets import select_feature_columns


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns for Track A baselines (9 mechanistic features)."""
    return select_feature_columns(df.columns, feature_set="track_a")


def main():
    parser = argparse.ArgumentParser(description="Train Bayesian state-space model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold_2019",
        help="Which CV fold to use for single-fold test"
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=300,
        help="MCMC warmup iterations"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="MCMC sampling iterations"
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=2,
        help="Number of MCMC chains"
    )
    args = parser.parse_args()
    
    # Load config (from v6 snapshot)
    root = get_project_root()
    repo_root = get_repo_root()
    config_path = root / args.config
    cfg = load_config(str(config_path))
    
    print("=" * 60)
    print("CHIKUNGUNYA EWS - BAYESIAN MODEL (Phase 4.1)")
    print("=" * 60)
    print(f"Mode: Single-fold diagnostic test")
    print(f"Target fold: {args.fold}")
    print(f"MCMC: {args.n_chains} chains, {args.n_warmup} warmup, {args.n_samples} samples")
    
    # Load features (from repo-level processed data)
    features_path = repo_root / cfg['data']['processed']['features']
    print(f"\nLoading features from {features_path}...")
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
    
    # Filter to rows with valid data for Bayesian model
    # We need: cases, district info, time info, temperature
    required_cols = ['state', 'district', 'year', 'week', 'cases']
    valid_df = df.dropna(subset=required_cols).copy()
    
    # Also need temperature for climate effect
    if 'temp_celsius' in valid_df.columns:
        valid_df = valid_df.dropna(subset=['temp_celsius'])
    
    print(f"  → {len(valid_df)} valid samples for Bayesian model")
    
    # Create stratified CV splits
    test_years = cfg['cv']['test_years']
    folds = create_stratified_temporal_folds(
        df=valid_df,
        target_col='label_outbreak',
        year_col='year',
        min_positives=5,
        candidate_test_years=test_years,
        verbose=False
    )
    
    print(f"\nCV Folds available: {len(folds)} (stratified)")
    for fold in folds:
        marker = " ← TARGET" if fold.fold_name == args.fold else ""
        test_years_str = "-".join(map(str, fold.test_years))
        print(f"  {fold.fold_name}: train={len(fold.train_idx)}, test={len(fold.test_idx)}, test_years={test_years_str}{marker}")
    
    # Find target fold
    target_fold = None
    for fold in folds:
        if fold.fold_name == args.fold:
            target_fold = fold
            break
    
    if target_fold is None:
        print(f"\nERROR: Fold '{args.fold}' not found!")
        return
    
    # Get train data for this fold
    train_df = valid_df.iloc[target_fold.train_idx].copy()
    test_df = valid_df.iloc[target_fold.test_idx].copy()
    
    print(f"\nSelected fold: {args.fold}")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Districts in train: {train_df['district'].nunique()}")
    print(f"  Year range (train): {train_df['year'].min()}-{train_df['year'].max()}")
    
    # Import and instantiate Bayesian model
    print("\n" + "=" * 60)
    print("INITIALIZING BAYESIAN MODEL")
    print("=" * 60)
    
    from src.models.bayesian.state_space import BayesianStateSpace
    
    outbreak_percentile = cfg.get('labels', {}).get('outbreak_percentile')
    if outbreak_percentile is None:
        raise ValueError("Missing labels.outbreak_percentile in config.")

    model_config = {
        'n_warmup': args.n_warmup,
        'n_samples': args.n_samples,
        'n_chains': args.n_chains,
        'seed': 42,
        'outbreak_percentile': int(outbreak_percentile),
    }
    
    model = BayesianStateSpace(config=model_config)
    
    # Fit model
    print("\n" + "=" * 60)
    print("FITTING MODEL")
    print("=" * 60)
    
    # Prepare dummy X, y for API compatibility
    X_train = train_df[feature_cols].values
    y_train = train_df['cases'].values
    
    try:
        model.fit(X_train, y_train, df=train_df, feature_cols=feature_cols)
        
        # Print diagnostics
        model.print_diagnostics()
        
        # Get posterior predictive samples
        print("\n" + "=" * 60)
        print("POSTERIOR PREDICTIVE CHECK")
        print("=" * 60)
        
        y_rep = model.get_posterior_predictive()
        y_obs = model.data_['y']
        
        print(f"\nPosterior predictive samples shape: {y_rep.shape}")
        print(f"Observed data shape: {y_obs.shape}")
        
        # Compare observed vs predicted (mean)
        y_pred_mean = y_rep.mean(axis=0)
        
        print(f"\nObserved cases: min={y_obs.min()}, max={y_obs.max()}, mean={y_obs.mean():.2f}")
        print(f"Predicted (mean): min={y_pred_mean.min():.2f}, max={y_pred_mean.max():.2f}, mean={y_pred_mean.mean():.2f}")
        
        # Correlation between observed and predicted
        if len(y_obs) > 1:
            corr = np.corrcoef(y_obs, y_pred_mean)[0, 1]
            print(f"Correlation (obs vs pred mean): {corr:.3f}")
        
        # Coverage check: what fraction of observations fall within 90% CI?
        y_lower = np.percentile(y_rep, 5, axis=0)
        y_upper = np.percentile(y_rep, 95, axis=0)
        coverage = np.mean((y_obs >= y_lower) & (y_obs <= y_upper))
        print(f"90% CI coverage: {coverage:.1%}")
        
        print("\n" + "=" * 60)
        print("PHASE 4.1 COMPLETE")
        print("=" * 60)
        print("✓ Stan model compiled successfully")
        print("✓ MCMC sampling completed")
        print("✓ Diagnostics reported")
        print("✓ Posterior predictive generated")
        print("\nReady for review before proceeding to full CV evaluation.")
        
    except Exception as e:
        print(f"\nERROR during model fitting: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
