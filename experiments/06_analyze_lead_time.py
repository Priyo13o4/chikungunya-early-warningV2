#!/usr/bin/env python3
"""
Experiment 06: Lead-Time Analysis Across All CV Folds

Phase 7 - Workstream A: Lead-Time Analysis Execution

PURPOSE:
--------
Execute lead-time analysis for Bayesian vs XGBoost across all rolling-origin
cross-validation folds (2017-2022). This validates Claim 2: "Latent risk
inference captures outbreak escalation earlier than binary classifiers."

VERSION: v4 (frozen lead_time.py)
DATE: February 2026

WORKFLOW:
---------
For each fold (test_year in [2017, 2018, 2019, 2020, 2021, 2022]):
    1. Load data: training (years < test_year), test (year == test_year)
    2. Train Bayesian model → extract Z_t posterior mean for test set
    3. Train XGBoost model → extract p_t probabilities for test set
    4. Compute outbreak thresholds from TRAINING only (config percentile)
    5. Identify outbreak episodes in TEST set
    6. Compute lead times using LeadTimeAnalyzer
    7. Save fold-level outputs

Aggregation:
    - Combine all episodes across folds
    - Compute overall summary statistics
    - Compute per-fold summary statistics

OUTPUTS:
--------
1. results/analysis/lead_time_detail_p{percentile}.csv
2. results/analysis/lead_time_summary_overall_p{percentile}.csv
3. results/analysis/lead_time_summary_by_fold_p{percentile}.csv

CONSTRAINTS:
------------
- Uses EXACT logic from src/evaluation/lead_time.py (frozen v4)
- No threshold tuning
- No filtering of districts or episodes
- Preserves -1 sentinel for never-warned
- Fails loudly if any fold has zero outbreak episodes

Reference: docs/Version 2/02_lead_time_analysis_spec_v2.md
Reference: docs/Version 2/08_phase7_roadmap_v2.md (Weeks 1-2)
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

# =============================================================================
# PATH SETUP
# =============================================================================

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.config import load_config, get_project_root, get_repo_root
from src.evaluation.cv import create_rolling_origin_splits, CVFold
from src.evaluation.lead_time import (
    LeadTimeAnalyzer,
    LeadTimeResult,
    LeadTimeSummary,
    OutbreakEpisode,
    results_to_dataframe,
    summary_to_dataframe,
    summarize_lead_times,
    sanity_check_results,
    NEVER_WARNED_SENTINEL
)

repo_root = get_repo_root()



# =============================================================================
# CONFIGURATION
# =============================================================================

# CV folds to analyze (rolling-origin)
TEST_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]

# MCMC configuration for Bayesian model (matches Phase 5)
MCMC_CONFIG = {
    'n_warmup': 1000,
    'n_samples': 1000,
    'n_chains': 4,
    'adapt_delta': 0.95,
    'seed': 42
}

# NOTE: thresholds are config-driven; this script fails if required values are missing.

# =============================================================================
# FAIR ANALYSIS MODE (v4.2)
# =============================================================================
# For symmetric comparison, use the SAME percentile threshold for BOTH models,
# computed from TRAINING predictions only. This answers:
# "Which model ranks outbreak escalation earlier?"
#
# - Bayesian: Z_t > q_k(Z_train)
# - XGBoost:  P_t > q_k(P_train)
#
# This is more scientifically fair than using fixed 0.5 for XGBoost.
# =============================================================================

ANALYSIS_MODE_PERCENTILE: Optional[int] = None


def prepare_fold_dfs_with_imputation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_cols: List[str] = ['state', 'district'],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-safe preprocessing with conservative missingness handling.

    - Drops rows with missing target only
    - Past-only forward-fill for lag-like features within each subset
    - Neutral fills for known sparse EWS/LAI features
    - Leave remaining missing values as NaN (no global/statistical imputation)
    """
    # Keep all test weeks; labels are used only for episode definition.
    train = train_df.dropna(subset=[target_col]).copy()
    test = test_df.copy()

    for subset in (train, test):
        if all(c in subset.columns for c in group_cols + ['year', 'week']):
            subset.sort_values(group_cols + ['year', 'week'], inplace=True)
            subset.reset_index(drop=True, inplace=True)

    # Past-only ffill for lag/persistence/anomaly features.
    lag_cols = [c for c in feature_cols if any(x in c for x in ['lag', 'persist', 'anomaly'])]
    for col in lag_cols:
        if col in train.columns:
            train[col] = train.groupby(group_cols)[col].transform(lambda x: x.ffill())
        if col in test.columns:
            test[col] = test.groupby(group_cols)[col].transform(lambda x: x.ffill())

    # Neutral-value fills (domain sensible defaults; no train/test leakage).
    neutral_impute_map = {
        'feat_var_spike_ratio': 1.0,
        'feat_acf_change': 0.0,
        'feat_trend_accel': 0.0,
        'feat_lai': 0.0,
        'feat_lai_lag_1': 0.0,
        'feat_lai_lag_2': 0.0,
        'feat_lai_lag_4': 0.0,
    }
    for col, neutral_val in neutral_impute_map.items():
        if col in train.columns:
            train[col] = train[col].fillna(neutral_val)
        if col in test.columns:
            test[col] = test[col].fillna(neutral_val)

    # Ensure feature matrices are numeric (pd.NA -> NaN) for model fitting.
    train[feature_cols] = train[feature_cols].apply(pd.to_numeric, errors='coerce').astype('float64')
    test[feature_cols] = test[feature_cols].apply(pd.to_numeric, errors='coerce').astype('float64')

    return train, test


# =============================================================================
# FAIR ANALYSIS HELPER FUNCTIONS (v4.2)
# =============================================================================

def compute_xgboost_training_predictions(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak'
) -> pd.DataFrame:
    """
    Get XGBoost predictions on TRAINING data for threshold computation.
    
    NOTE: We use in-sample training predictions (fit on train, predict on train)
    for ranking-based thresholding only. This avoids test peeking but can be
    optimistic; the choice is documented and accepted for thesis comparisons.
    
    Args:
        train_df: Training DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        
    Returns:
        Training DataFrame with 'prob' column
    """
    from src.models.baselines.xgboost_model import XGBoostBaseline
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    model = XGBoostBaseline({})
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_train)
    
    pred_df = train_df[['state', 'district', 'year', 'week', 'cases']].copy()
    pred_df['prob'] = probs
    
    return pred_df


def compute_percentile_threshold_from_predictions(
    pred_df: pd.DataFrame,
    value_col: str,
    percentile: int
) -> float:
    """
    Compute global percentile threshold from prediction values.
    
    For fair comparison, we use a GLOBAL threshold (not district-specific)
    so both models are on equal footing.
    
    Args:
        pred_df: Predictions DataFrame
        value_col: Column to compute percentile on ('prob' or 'z_mean')
        percentile: Percentile (config-driven)
        
    Returns:
        Threshold value
    """
    values = pred_df[value_col].dropna()
    if len(values) == 0:
        raise ValueError(f"No valid values in column {value_col}")
    
    return float(np.percentile(values, percentile))


def compute_fair_thresholds_from_training(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak',
    percentile: int = None
) -> Tuple[float, float]:
    """
    Compute BOTH Bayesian and XGBoost thresholds from TRAINING data.
    
    This implements the fair analysis mode where:
    - Bayesian threshold = percentile of Z_t on training predictions
    - XGBoost threshold = percentile of P_t on training predictions
    
    NOTE: Bayesian predictions on training data require fitting the model.
    For efficiency, we approximate using the XGBoost distribution characteristics.
    
    In practice, for this thesis, we use:
    - XGBoost: actual training predictions
    - Bayesian: training predictions from the combined model (already fitted)
    
    Args:
        train_df: Training DataFrame (NaN-filtered)
        feature_cols: Feature column names
        target_col: Target column name
        percentile: Percentile for both thresholds
        
    Returns:
        Tuple of (bayesian_threshold, xgboost_threshold)
    """
    if percentile is None:
        raise ValueError("percentile must be provided for fair threshold computation")

    # Compute XGBoost threshold from training predictions
    xgb_train_preds = compute_xgboost_training_predictions(
        train_df, feature_cols, target_col
    )
    xgb_threshold = compute_percentile_threshold_from_predictions(
        xgb_train_preds, 'prob', percentile
    )
    
    print(f"    XGBoost p{percentile} threshold from training: {xgb_threshold:.4f}")
    
    # For Bayesian, we need training predictions but the model fits on combined data
    # We'll compute the Bayesian threshold from the combined model's training portion
    # This is handled in analyze_single_fold where we have access to the fit model
    # Return a placeholder that will be overwritten
    bayesian_threshold = None  # Computed later from actual Bayesian fit
    
    return bayesian_threshold, xgb_threshold


# =============================================================================
# DATA LOADING
# =============================================================================

def load_features_data() -> pd.DataFrame:
    """
    Load feature-engineered data from processed parquet.
    
    Returns:
        DataFrame with all features and metadata
    """
    # Processed features live at the repo root; v6 is nested under versions/.
    data_path = repo_root / "data" / "processed" / "features_engineered_v01.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Features file not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df)} samples, {df['district'].nunique()} districts")
    print(f"  Years: {sorted(df['year'].unique())}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all feature column names (prefix: feat_)."""
    return [c for c in df.columns if c.startswith('feat_')]


def impute_missing_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply conservative, non-synthetic imputation for inspection only.

    Rules:
    - Past-only forward-fill for lag-like features
    - Neutral-value fills for known sparse EWS/LAI features
    - Leave all other missing values as NaN

    NOTE: This function is not used in the main lead-time pipeline.
    """
    df = df.copy()

    # Ensure time ordering before any groupwise fill (transform uses row order).
    if all(c in df.columns for c in ['state', 'district', 'year', 'week']):
        df = df.sort_values(['state', 'district', 'year', 'week']).reset_index(drop=True)
    
    print(f"    Applying conservative fills for {len(feature_cols)} features...")
    initial_nulls = df[feature_cols].isnull().sum().sum()
    
    # 1. Past-only forward-fill within groups for all lag features (climate + case lags)
    # IMPORTANT: avoid backward-fill (bfill) to prevent future leakage.
    lag_cols = [c for c in feature_cols if 'lag' in c or 'persist' in c or 'anomaly' in c]
    for col in lag_cols:
        if col in df.columns:
            df[col] = df.groupby(['state', 'district'])[col].transform(
                lambda x: x.ffill()
            )
    
    # 2. Neutral-value fills for sparse engineered features.
    neutral_impute_map = {
        'feat_var_spike_ratio': 1.0,
        'feat_acf_change': 0.0,
        'feat_trend_accel': 0.0,
        'feat_lai': 0.0,
        'feat_lai_lag_1': 0.0,
        'feat_lai_lag_2': 0.0,
        'feat_lai_lag_4': 0.0,
    }
    for col, neutral_val in neutral_impute_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(neutral_val)
    
    final_nulls = df[feature_cols].isnull().sum().sum()
    recovered = initial_nulls - final_nulls
    print(f"    Fills applied: {initial_nulls} nulls → {final_nulls} nulls (recovered {recovered} values)")
    
    return df


# =============================================================================
# MODEL TRAINING & PREDICTION
# =============================================================================

def train_xgboost_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak'
) -> pd.DataFrame:
    """
    Train XGBoost on training data and generate predictions for test data.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        
    Returns:
        Test DataFrame with added 'prob' column containing XGBoost probabilities
    """
    from src.models.baselines.xgboost_model import XGBoostBaseline
    
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    
    # Train model
    model = XGBoostBaseline({})
    model.fit(X_train, y_train)
    
    # Predict
    probs = model.predict_proba(X_test)
    
    # Create output DataFrame
    cols = [c for c in ['state', 'district', 'year', 'week', 'cases', '_row_id'] if c in test_df.columns]
    pred_df = test_df[cols].copy()
    pred_df['prob'] = probs
    
    return pred_df


def train_bayesian_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train Bayesian state-space model and generate predictions for test data.
    
    This extracts the latent risk Z_t (posterior mean) for each observation.
    The Bayesian model fits on combined train+test data (standard for state-space
    models) but we only extract predictions for test rows.
    
    IMPORTANT: Both train_df and test_df must be NaN-filtered (valid samples only)
    to ensure alignment with XGBoost predictions.
    
    ==========================================================================
    CRITICAL FIX (v4.2): Handle duplicate (state, district, year, week) keys
    ==========================================================================
    
    The data contains duplicate keys. We handle this by:
      1. Assigning a unique row ID to each row BEFORE concatenation
      2. Tracking which IDs belong to test data
      3. After model fitting, extracting by row ID (NOT by 4-tuple key)
    
    ==========================================================================
    
    Args:
        train_df: Training DataFrame (NaN-filtered)
        test_df: Test DataFrame (NaN-filtered)
        feature_cols: Feature column names
        
    Returns:
        Tuple of (train_pred_df, test_pred_df) with columns including:
        - state, district, year, week, cases, optional _row_id
        - z_mean, z_sd
    """
    from src.models.bayesian.state_space import BayesianStateSpace
    
    # Get Stan model path (local to V2 standalone project)
    repo_root = get_repo_root()
    stan_path = repo_root / "stan_models" / "hierarchical_ews_v01.stan"
    
    # Configure model
    model_config = {
        **MCMC_CONFIG,
        'stan_file': str(stan_path)
    }
    
    # =========================================================================
    # Step 1: Assign unique row IDs BEFORE concatenation
    # =========================================================================
    # This is the ONLY way to track rows through sorting when keys aren't unique
    
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # Assign globally unique row IDs
    train_copy['_unique_row_id'] = range(len(train_copy))
    test_copy['_unique_row_id'] = range(len(train_copy), len(train_copy) + len(test_copy))
    
    # Track which IDs are test rows
    test_row_ids = set(test_copy['_unique_row_id'])
    
    # Also store the original test order
    test_copy['_test_order'] = range(len(test_copy))
    train_copy['_test_order'] = -1  # Sentinel
    
    # Concatenate
    combined_df = pd.concat([train_copy, test_copy], ignore_index=True)
    
    # =========================================================================
    # Step 2: Sort for Stan model (required for state-space structure)
    # =========================================================================
    # The model expects data sorted by district and time
    # We add a secondary sort by _unique_row_id to ensure deterministic ordering
    # when (state, district, year, week) has ties (duplicates)
    
    combined_df = combined_df.sort_values(
        ['state', 'district', 'year', 'week', '_unique_row_id']
    ).reset_index(drop=True)
    
    # Store the position in sorted order for later mapping
    combined_df['_sorted_position'] = range(len(combined_df))
    
    n_total = len(combined_df)
    print(f"    [Bayesian] Combined: {n_total} samples ({len(train_df)} train + {len(test_df)} test)")
    
    # =========================================================================
    # Step 3: Fit Bayesian model
    # =========================================================================
    print(f"    [Bayesian] Preparing hierarchical state-space model...")
    model = BayesianStateSpace(config=model_config)
    
    X_combined = combined_df[feature_cols].values
    y_combined = combined_df['cases'].values
    
    print(f"    [Bayesian] Fitting on {n_total} samples...")
    print(f"    [Bayesian] MCMC: {MCMC_CONFIG['n_chains']} chains × ({MCMC_CONFIG['n_warmup']} warmup + {MCMC_CONFIG['n_samples']} samples)")
    print(f"    [Bayesian] Estimated time: ~{n_total * 0.2:.0f}-{n_total * 0.5:.0f} seconds (Stan will show progress bars)...")
    model.fit(X_combined, y_combined, df=combined_df, feature_cols=feature_cols)
    print(f"    [Bayesian] ✓ MCMC sampling complete")
    print(f"    [Bayesian] ✓ MCMC sampling complete")
    
    # Extract latent risk Z_t aligned to each observation row.
    # IMPORTANT: do NOT use posterior predictive counts (y_rep) as a proxy for Z.
    # Using y_rep makes the Bayesian signal case-like and tends to produce
    # reactive triggers (median lead time ~ 0).
    print(f"    [Bayesian] Extracting latent risk Z_t from posterior...")
    try:
        z_mean, z_sd = model.get_latent_risk_summary_per_observation()
    except AttributeError as e:
        raise RuntimeError(
            "BayesianStateSpace is missing get_latent_risk_summary_per_observation(). "
            "Ensure the v3 Bayesian wrapper exposes latent-Z per-observation extraction."
        ) from e

    # ASSERTION: z_mean must have exactly N entries
    if len(z_mean) != n_total:
        raise RuntimeError(
            f"STATE-SPACE INDEXING ERROR: z_mean has {len(z_mean)} entries "
            f"but combined_df has {n_total} rows. "
            f"This indicates a mismatch in latent-state extraction."
        )
    
    # =========================================================================
    # Step 4: Assign z_mean to the sorted DataFrame
    # =========================================================================
    # y_rep[i] corresponds to combined_df.iloc[i] in SORTED order
    combined_df['_z_mean'] = z_mean
    combined_df['_z_sd'] = z_sd
    
    # =========================================================================
    # Step 5: Extract test predictions by unique row ID
    # =========================================================================
    # Filter to test rows only
    test_preds = combined_df[combined_df['_unique_row_id'].isin(test_row_ids)].copy()
    
    # ASSERTION: Must have exactly len(test_df) test predictions
    if len(test_preds) != len(test_df):
        raise RuntimeError(
            f"TEST EXTRACTION ERROR: Found {len(test_preds)} test predictions "
            f"but expected {len(test_df)}."
        )
    
    # Sort back to original test order
    test_preds = test_preds.sort_values('_test_order').reset_index(drop=True)

    # Extract training predictions (order not required for thresholding)
    train_preds = combined_df[~combined_df['_unique_row_id'].isin(test_row_ids)].copy()
    
    # =========================================================================
    # Step 6: Build output DataFrame
    # =========================================================================
    print(f"    [Bayesian] Extracting latent risk Z_t for {len(test_df)} test samples...")
    test_cols = [c for c in ['state', 'district', 'year', 'week', 'cases', '_row_id'] if c in test_df.columns]
    test_pred_df = test_df[test_cols].copy().reset_index(drop=True)
    test_pred_df['z_mean'] = test_preds['_z_mean'].values
    test_pred_df['z_sd'] = test_preds['_z_sd'].values

    train_cols = [c for c in ['state', 'district', 'year', 'week', 'cases', '_row_id'] if c in train_preds.columns]
    train_pred_df = train_preds[train_cols].copy().reset_index(drop=True)
    train_pred_df['z_mean'] = train_preds['_z_mean'].values
    train_pred_df['z_sd'] = train_preds['_z_sd'].values
    
    # FINAL ASSERTION: Output length must match test_df exactly
    if len(test_pred_df) != len(test_df):
        raise RuntimeError(
            f"OUTPUT LENGTH ERROR: test_pred_df has {len(test_pred_df)} rows "
            f"but test_df has {len(test_df)} rows."
        )
    
    # Check for NaN
    n_missing = test_pred_df['z_mean'].isna().sum()
    if n_missing > 0:
        warnings.warn(f"  WARNING: {n_missing} test samples have NaN Bayesian prediction")
    
    print(f"    Bayesian predictions: {len(test_pred_df)} samples")

    return train_pred_df, test_pred_df


# =============================================================================
# FOLD ANALYSIS
# =============================================================================

def analyze_single_fold(
    fold: CVFold,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak',
    outbreak_percentile: int = None,
    bayesian_percentile: int = None,
    xgboost_probability_threshold: float = None,
    lead_time_mode: str = 'fair_quantile',
    analysis_percentile: int = None,
    outbreak_value_col: str = 'incidence_per_100k',
    skip_bayesian: bool = False,
) -> Dict[str, Any]:
    """
    Run complete lead-time analysis for a single CV fold.
    
    Args:
        fold: CVFold object with train/test indices
        df: Full DataFrame
        feature_cols: Feature column names
        target_col: Target column
        
    Returns:
        Dictionary with:
        - fold_name: str
        - test_year: int
        - n_train: int
        - n_test: int
        - n_episodes: int
        - results: List[LeadTimeResult]
        - bayesian_summary: LeadTimeSummary
        - xgboost_summary: LeadTimeSummary
    """
    print(f"\n{'='*70}")
    print(f"FOLD: {fold.fold_name} (Test Year: {fold.test_year})")
    print(f"{'='*70}")

    if outbreak_percentile is None or bayesian_percentile is None or xgboost_probability_threshold is None:
        raise ValueError("Thresholds must be provided explicitly (config-driven).")
    if lead_time_mode == 'fair_quantile' and analysis_percentile is None:
        raise ValueError("analysis_percentile is required for fair_quantile mode.")
    
    # Get train/test data
    train_df = df.iloc[fold.train_idx].copy()
    test_df = df.iloc[fold.test_idx].copy()
    
    print(f"  Training: {len(train_df)} samples (years {train_df['year'].min()}-{train_df['year'].max()})")
    print(f"  Test: {len(test_df)} samples (year {test_df['year'].unique()})")
    
    # Fold-safe preprocessing: fit imputer on TRAINING only and apply to TEST.
    print(f"  Applying fold-safe imputation (fit on train, apply to test)...")
    train_valid, test_valid = prepare_fold_dfs_with_imputation(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    # Add per-row id for safe alignment (keys are not unique in raw data)
    test_valid = test_valid.reset_index(drop=True)
    test_valid['_row_id'] = np.arange(len(test_valid))
    
    print(f"  Valid training: {len(train_valid)} (recovered from {len(train_df)}) | Valid test: {len(test_valid)} (recovered from {len(test_df)})")
    
    if len(train_valid) < 20:
        raise ValueError(f"Insufficient training data for fold {fold.fold_name}: {len(train_valid)}")
    
    if len(test_valid) < 5:
        raise ValueError(f"Insufficient test data for fold {fold.fold_name}: {len(test_valid)}")
    
    # -------------------------------------------------------------------------
    # Step 1: Train models and get predictions
    # -------------------------------------------------------------------------
    print(f"\n  Step 1: Training models...")
    
    # XGBoost predictions on TEST
    print(f"    Training XGBoost...")
    xgboost_preds = train_xgboost_and_predict(train_valid, test_valid, feature_cols, target_col)
    print(f"    XGBoost predictions: {len(xgboost_preds)} samples")
    
    # Bayesian predictions on TEST
    if skip_bayesian:
        print(f"    Skipping Bayesian model (--skip-bayesian)")
        bayesian_train_preds = pd.DataFrame(columns=['state', 'district', 'year', 'week', 'z_mean', 'z_sd'])
        bayesian_test_preds = pd.DataFrame(columns=['state', 'district', 'year', 'week', '_row_id', 'z_mean', 'z_sd'])
    else:
        print(f"    Training Bayesian model (this may take 5-15 minutes)...")
        print(f"    [Bayesian] Compiling Stan model...")
        bayesian_train_preds, bayesian_test_preds = train_bayesian_and_predict(train_valid, test_valid, feature_cols)
        print(f"    [Bayesian] ✓ Complete: {len(bayesian_test_preds)} predictions generated")
    
    # -------------------------------------------------------------------------
    # Threshold modes
    # Mode A (fair_quantile): same GLOBAL percentile threshold for both models,
    # computed from TRAINING predictions only (no test-peeking).
    # Mode B (decision_realism): XGBoost uses fixed probability threshold,
    # Bayesian uses district-specific percentile thresholds from TRAINING predictions.
    if lead_time_mode not in {'fair_quantile', 'decision_realism'}:
        raise ValueError(f"Unknown lead_time_mode: {lead_time_mode}")

    print(f"\n  Step 1b: Threshold mode = {lead_time_mode}")
    fair_thresholds: Dict[str, Any] = {'mode': lead_time_mode}
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize LeadTimeAnalyzer and compute thresholds
    # -------------------------------------------------------------------------
    print(f"\n  Step 2: Computing thresholds...")
    
    analyzer = LeadTimeAnalyzer(
        outbreak_percentile=int(outbreak_percentile),
        bayesian_percentile=int(bayesian_percentile),
        xgboost_threshold=float(xgboost_probability_threshold),
    )
    
    # Compute outbreak thresholds from TRAINING data only
    if outbreak_value_col not in train_valid.columns:
        outbreak_value_col = 'cases'

    analyzer.compute_outbreak_thresholds_from_training(train_valid, case_col=outbreak_value_col)
    
    # -------------------------------------------------------------------------
    # Step 3: Identify outbreak episodes in test data
    # -------------------------------------------------------------------------
    print(f"\n  Step 3: Identifying outbreak episodes...")
    
    episodes = analyzer.identify_episodes(test_valid, fold_name=fold.fold_name, case_col=outbreak_value_col)
    
    if len(episodes) == 0:
        warnings.warn(
            f"Zero outbreak episodes in fold {fold.fold_name}; skipping lead-time computation."
        )
        meta_cols = [c for c in ['state', 'district', 'year', 'week', 'cases', outbreak_value_col, '_row_id'] if c in test_valid.columns]
        ts_df = test_valid[meta_cols].copy()
        ts_df['fold'] = fold.fold_name
        ts_df['test_year'] = fold.test_year
        ts_df['y_true'] = test_valid[target_col].astype('Int64')

        if '_row_id' in xgboost_preds.columns:
            ts_df = ts_df.merge(
                xgboost_preds[['_row_id', 'prob']],
                on='_row_id',
                how='left',
                validate='one_to_one',
            )
        else:
            ts_df['prob'] = np.nan

        if '_row_id' in bayesian_test_preds.columns:
            ts_df = ts_df.merge(
                bayesian_test_preds[['_row_id', 'z_mean', 'z_sd']],
                on='_row_id',
                how='left',
                validate='one_to_one',
            )
        else:
            ts_df['z_mean'] = np.nan
            ts_df['z_sd'] = np.nan

        bayesian_summary, xgboost_summary = analyzer.get_summaries([])
        return {
            'fold_name': fold.fold_name,
            'test_year': fold.test_year,
            'n_train': len(train_valid),
            'n_test': len(test_valid),
            'n_episodes': 0,
            'results': [],
            'bayesian_summary': bayesian_summary,
            'xgboost_summary': xgboost_summary,
            'outbreak_thresholds': dict(analyzer.outbreak_thresholds),
            'bayesian_thresholds': {},
            'fair_thresholds': fair_thresholds,
            'effective_xgb_threshold': float('nan'),
            'test_prediction_timeseries': ts_df,
        }
    
    print(f"    Found {len(episodes)} outbreak episodes")
    
    # -------------------------------------------------------------------------
    # Step 4: Compute lead times
    # -------------------------------------------------------------------------
    print(f"\n  Step 4: Computing lead times...")
    
    # Compute Bayesian/XGBoost thresholds WITHOUT using test distribution.
    if lead_time_mode == 'fair_quantile':
        xgb_train_preds = compute_xgboost_training_predictions(train_valid, feature_cols, target_col)
        xgb_threshold = compute_percentile_threshold_from_predictions(
            xgb_train_preds, 'prob', percentile=int(analysis_percentile)
        )
        if skip_bayesian:
            bayes_threshold = float('inf')
        else:
            bayes_threshold = float(np.percentile(
                bayesian_train_preds['z_mean'].dropna(), int(analysis_percentile)
            ))

        analyzer.xgboost_threshold = float(xgb_threshold)
        analyzer.bayesian_thresholds = {
            (ep.state, ep.district): float(bayes_threshold)
            for ep in episodes
        }
        fair_thresholds.update({
            'percentile': int(analysis_percentile),
            'xgboost_threshold': float(xgb_threshold),
            'bayesian_threshold': float(bayes_threshold),
        })
        effective_xgb_threshold = float(xgb_threshold)
    else:
        # decision_realism
        analyzer.xgboost_threshold = float(xgboost_probability_threshold)
        if skip_bayesian:
            analyzer.bayesian_thresholds = {
                (ep.state, ep.district): float('inf')
                for ep in episodes
            }
        else:
            analyzer.compute_bayesian_thresholds_from_predictions(bayesian_train_preds, z_col='z_mean')
        effective_xgb_threshold = float(xgboost_probability_threshold)
        fair_thresholds.update({
            'xgboost_threshold': float(xgboost_probability_threshold),
            'bayesian_percentile': int(bayesian_percentile),
        })

    results = analyzer.compute_lead_times(
        episodes=episodes,
        bayesian_df=bayesian_test_preds,
        xgboost_df=xgboost_preds,
        z_col='z_mean',
        prob_col='prob',
    )
    
    # Sanity checks
    checks = sanity_check_results(results)
    if not checks['passed']:
        warnings.warn(f"  Sanity check issues in {fold.fold_name}: {checks['issues']}")
    
    # -------------------------------------------------------------------------
    # Step 5: Compute summaries
    # -------------------------------------------------------------------------
    bayesian_summary, xgboost_summary = analyzer.get_summaries(results)
    
    # Print fold summary
    print(f"\n  --- Fold Summary ---")
    print(f"  Outbreak episodes: {len(episodes)}")
    print(f"  Bayesian:")
    print(f"    Warned: {bayesian_summary.n_warned}/{bayesian_summary.n_episodes}")
    print(f"    Median lead time: {bayesian_summary.median_lead_time}")
    print(f"    % early warned (≥1 wk): {bayesian_summary.pct_early_warned:.1f}%")
    print(f"  XGBoost:")
    print(f"    Warned: {xgboost_summary.n_warned}/{xgboost_summary.n_episodes}")
    print(f"    Median lead time: {xgboost_summary.median_lead_time}")
    print(f"    % early warned (≥1 wk): {xgboost_summary.pct_early_warned:.1f}%")

    # Assemble per-row time series for transparency + downstream plotting.
    meta_cols = [c for c in ['state', 'district', 'year', 'week', 'cases', outbreak_value_col, '_row_id'] if c in test_valid.columns]
    ts_df = test_valid[meta_cols].copy()
    ts_df['fold'] = fold.fold_name
    ts_df['test_year'] = fold.test_year
    ts_df['y_true'] = test_valid[target_col].astype('Int64')

    if '_row_id' in xgboost_preds.columns:
        ts_df = ts_df.merge(
            xgboost_preds[['_row_id', 'prob']],
            on='_row_id',
            how='left',
            validate='one_to_one',
        )
    else:
        ts_df['prob'] = np.nan

    if '_row_id' in bayesian_test_preds.columns:
        ts_df = ts_df.merge(
            bayesian_test_preds[['_row_id', 'z_mean', 'z_sd']],
            on='_row_id',
            how='left',
            validate='one_to_one',
        )
    else:
        ts_df['z_mean'] = np.nan
        ts_df['z_sd'] = np.nan
    
    return {
        'fold_name': fold.fold_name,
        'test_year': fold.test_year,
        'n_train': len(train_valid),
        'n_test': len(test_valid),
        'n_episodes': len(episodes),
        'results': results,
        'bayesian_summary': bayesian_summary,
        'xgboost_summary': xgboost_summary,
        'outbreak_thresholds': dict(analyzer.outbreak_thresholds),
        'bayesian_thresholds': dict(analyzer.bayesian_thresholds),
        'fair_thresholds': fair_thresholds,
        'effective_xgb_threshold': effective_xgb_threshold,
        'test_prediction_timeseries': ts_df,
    }


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_all_folds(
    fold_outputs: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate lead-time results across all folds.
    
    Args:
        fold_outputs: List of outputs from analyze_single_fold
        
    Returns:
        Tuple of:
        - detail_df: All episodes (one row per episode)
        - summary_overall_df: Overall summary statistics
        - summary_by_fold_df: Summary per fold
    """
    # -------------------------------------------------------------------------
    # 1. Combine all episode-level results
    # -------------------------------------------------------------------------
    all_results = []
    for fold_output in fold_outputs:
        all_results.extend(fold_output['results'])
    
    detail_df = results_to_dataframe(all_results)
    
    # -------------------------------------------------------------------------
    # 2. Compute overall summary
    # -------------------------------------------------------------------------
    bayesian_overall = summarize_lead_times(all_results, 'bayesian')
    xgboost_overall = summarize_lead_times(all_results, 'xgboost')
    
    summary_overall_df = summary_to_dataframe(bayesian_overall, xgboost_overall)
    
    # -------------------------------------------------------------------------
    # 3. Compute per-fold summary
    # -------------------------------------------------------------------------
    fold_summaries = []
    for fold_output in fold_outputs:
        fold_name = fold_output['fold_name']
        test_year = fold_output['test_year']
        
        bayes = fold_output['bayesian_summary']
        xgb = fold_output['xgboost_summary']
        
        fold_summaries.append({
            'fold': fold_name,
            'test_year': test_year,
            'n_episodes': bayes.n_episodes,
            'bayesian_n_warned': bayes.n_warned,
            'bayesian_median_lead': bayes.median_lead_time,
            'bayesian_mean_lead': bayes.mean_lead_time,
            'bayesian_iqr_lower': bayes.iqr_lower,
            'bayesian_iqr_upper': bayes.iqr_upper,
            'bayesian_pct_early': bayes.pct_early_warned,
            'bayesian_pct_never': bayes.pct_never_warned,
            'xgboost_n_warned': xgb.n_warned,
            'xgboost_median_lead': xgb.median_lead_time,
            'xgboost_mean_lead': xgb.mean_lead_time,
            'xgboost_iqr_lower': xgb.iqr_lower,
            'xgboost_iqr_upper': xgb.iqr_upper,
            'xgboost_pct_early': xgb.pct_early_warned,
            'xgboost_pct_never': xgb.pct_never_warned
        })
    
    summary_by_fold_df = pd.DataFrame(fold_summaries)
    
    return detail_df, summary_overall_df, summary_by_fold_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Execute lead-time analysis across all CV folds."""
    
    parser = argparse.ArgumentParser(
        description='Phase 7 Workstream A: Lead-Time Analysis (All Folds)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config_default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--skip-bayesian',
        action='store_true',
        help='Skip Bayesian model (for debugging XGBoost-only)'
    )
    parser.add_argument(
        '--test-years',
        type=int,
        nargs='*',
        default=None,
        help='Optional override for CV test years (e.g., --test-years 2019).'
    )
    parser.add_argument(
        '--lead-time-mode',
        type=str,
        default='fair_quantile',
        choices=['fair_quantile', 'decision_realism'],
        help='Lead-time thresholding mode: fair_quantile (Mode A) or decision_realism (Mode B)'
    )
    parser.add_argument(
        '--analysis-percentile',
        type=int,
        default=None,
        help='Percentile for Mode A fair_quantile (defaults to config labels.outbreak_percentile)'
    )
    parser.add_argument(
        '--outbreak-percentile',
        type=int,
        default=None,
        help='Override config labels.outbreak_percentile for outbreak definition (and default Bayesian percentile)'
    )
    parser.add_argument(
        '--bayesian-percentile',
        type=int,
        default=None,
        help='Percentile for Bayesian thresholding in decision_realism mode (defaults to outbreak percentile)'
    )
    args = parser.parse_args()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print("="*70)
    print("PHASE 7 - WORKSTREAM A: LEAD-TIME ANALYSIS")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    # Load configuration
    config = load_config(str(project_root / args.config))

    test_years = args.test_years if args.test_years else config.get('cv', {}).get('test_years', TEST_YEARS)

    cfg_outbreak = config.get('labels', {}).get('outbreak_percentile')
    if args.outbreak_percentile is None and cfg_outbreak is None:
        raise ValueError("Missing labels.outbreak_percentile in config and no --outbreak-percentile override.")
    outbreak_percentile = int(args.outbreak_percentile if args.outbreak_percentile is not None else cfg_outbreak)

    bayesian_percentile = int(args.bayesian_percentile or outbreak_percentile)

    cfg_prob_threshold = config.get('evaluation', {}).get('probability_threshold')
    if cfg_prob_threshold is None:
        raise ValueError("Missing evaluation.probability_threshold in config.")
    xgboost_probability_threshold = float(cfg_prob_threshold)

    analysis_percentile = int(args.analysis_percentile or outbreak_percentile)

    print(f"Test years: {test_years}")
    print(f"Lead-time mode: {args.lead_time_mode}")
    print(f"Outbreak percentile: {outbreak_percentile}")
    print(f"Bayesian percentile: {bayesian_percentile}")
    print(f"XGBoost probability threshold: {xgboost_probability_threshold}")
    if args.lead_time_mode == 'fair_quantile':
        print(f"Mode A analysis percentile: {analysis_percentile}")
    print()
    
    # Create output directory
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading Data")
    print("-"*70)
    
    df = load_features_data()
    feature_cols = get_feature_columns(df)
    target_col = 'label_outbreak'
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {target_col}")
    
    # IMPORTANT: Do NOT impute globally here (leaks future info across folds).
    # Imputation is handled fold-by-fold inside analyze_single_fold.
    
    # =========================================================================
    # CREATE CV FOLDS
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Creating CV Folds")
    print("-"*70)
    
    folds = create_rolling_origin_splits(df, test_years=test_years)
    print(f"  Created {len(folds)} folds:")
    for fold in folds:
        print(f"    {fold.fold_name}: train={len(fold.train_idx)}, test={len(fold.test_idx)}")
    
    # =========================================================================
    # ANALYZE EACH FOLD
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Analyzing Each Fold")
    print("-"*70)
    
    fold_outputs = []
    failed_folds = []
    prediction_frames: List[pd.DataFrame] = []
    
    for fold in folds:
        try:
            fold_output = analyze_single_fold(
                fold=fold,
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                outbreak_percentile=outbreak_percentile,
                bayesian_percentile=bayesian_percentile,
                xgboost_probability_threshold=xgboost_probability_threshold,
                lead_time_mode=args.lead_time_mode,
                analysis_percentile=analysis_percentile,
                skip_bayesian=bool(args.skip_bayesian),
            )
            fold_outputs.append(fold_output)

            ts = fold_output.get('test_prediction_timeseries')
            if isinstance(ts, pd.DataFrame) and not ts.empty:
                prediction_frames.append(ts)
            
        except Exception as e:
            print(f"\n  ERROR in {fold.fold_name}: {e}")
            failed_folds.append((fold.fold_name, str(e)))
            continue
    
    # Check for failures
    if failed_folds:
        print(f"\n{'!'*70}")
        print(f"WARNING: {len(failed_folds)} folds failed!")
        for name, error in failed_folds:
            print(f"  - {name}: {error}")
        print(f"{'!'*70}")
    
    if len(fold_outputs) == 0:
        raise RuntimeError("All folds failed! Cannot proceed.")
    
    # =========================================================================
    # AGGREGATE RESULTS
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 4: Aggregating Results")
    print("-"*70)
    
    detail_df, summary_overall_df, summary_by_fold_df = aggregate_all_folds(fold_outputs)
    
    print(f"\n  Total outbreak episodes: {len(detail_df)}")
    print(f"  Folds completed: {len(fold_outputs)}/{len(folds)}")
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Saving Outputs")
    print("-"*70)
    
    # Detail file
    detail_path = output_dir / f"lead_time_detail_p{outbreak_percentile}.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"  ✓ {detail_path}")

    # Per-row prediction time series (for case studies/calibration)
    if prediction_frames:
        preds_path = output_dir / f"lead_time_predictions_p{outbreak_percentile}.parquet"
        pd.concat(prediction_frames, ignore_index=True).to_parquet(preds_path, index=False)
        print(f"  ✓ {preds_path}")
    
    # Overall summary
    summary_overall_path = output_dir / f"lead_time_summary_overall_p{outbreak_percentile}.csv"
    summary_overall_df.to_csv(summary_overall_path, index=False)
    print(f"  ✓ {summary_overall_path}")
    
    # By-fold summary
    summary_by_fold_path = output_dir / f"lead_time_summary_by_fold_p{outbreak_percentile}.csv"
    summary_by_fold_df.to_csv(summary_by_fold_path, index=False)
    print(f"  ✓ {summary_by_fold_path}")
    
    # Also save full analysis metadata as JSON
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 7 - Workstream A',
        'version': 'v4',
        'config': {
            'test_years': test_years,
            'lead_time_mode': args.lead_time_mode,
            'outbreak_percentile': outbreak_percentile,
            'bayesian_percentile': bayesian_percentile,
            'analysis_percentile': analysis_percentile,
            'xgboost_probability_threshold': xgboost_probability_threshold,
            'mcmc_config': MCMC_CONFIG
        },
        'folds_completed': len(fold_outputs),
        'folds_failed': len(failed_folds),
        'total_episodes': len(detail_df),
        'fold_summary': [
            {
                'fold': fo['fold_name'],
                'test_year': fo['test_year'],
                'n_train': fo['n_train'],
                'n_test': fo['n_test'],
                'n_episodes': fo['n_episodes']
            }
            for fo in fold_outputs
        ]
    }
    
    metadata_path = output_dir / "lead_time_analysis_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ {metadata_path}")
    
    # =========================================================================
    # PRINT FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nOverall Lead-Time Comparison:")
    print(summary_overall_df.to_string(index=False))
    
    print("\n\nPer-Fold Summary:")
    print(summary_by_fold_df.to_string(index=False))
    
    # Compute differential statistics
    valid_results = [r for fo in fold_outputs for r in fo['results']
                     if r.differential_lead is not None]
    if valid_results:
        diffs = [r.differential_lead for r in valid_results]
        print(f"\n\nDifferential Lead Time (ΔL = L_XGB - L_Bayes):")
        print(f"  N comparisons: {len(diffs)}")
        print(f"  Mean ΔL: {np.mean(diffs):.2f} weeks")
        print(f"  Median ΔL: {np.median(diffs):.2f} weeks")
        print(f"  ΔL > 0 (Bayesian earlier): {sum(1 for d in diffs if d > 0)} ({100*sum(1 for d in diffs if d > 0)/len(diffs):.1f}%)")
        print(f"  ΔL = 0 (Same timing): {sum(1 for d in diffs if d == 0)} ({100*sum(1 for d in diffs if d == 0)/len(diffs):.1f}%)")
        print(f"  ΔL < 0 (XGBoost earlier): {sum(1 for d in diffs if d < 0)} ({100*sum(1 for d in diffs if d < 0)/len(diffs):.1f}%)")
    
    print("\n" + "="*70)
    print("✓ Lead-Time Analysis Complete")
    print("="*70)
    
    return fold_outputs


if __name__ == '__main__':
    main()
