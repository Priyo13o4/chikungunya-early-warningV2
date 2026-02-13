#!/usr/bin/env python3
"""
Experiment 03: Train Baseline Models

Trains all Track A baseline models with temporal CV:
- Threshold rule
- Logistic Regression
- Random Forest
- XGBoost

Output: results/metrics/baseline_comparison.json

Usage:
    python experiments/03_train_baselines.py
    python experiments/03_train_baselines.py --config config/config_baseline.yaml
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.config import load_config, get_project_root, get_repo_root
from src.features.feature_sets import select_feature_columns
from src.evaluation.cv import create_stratified_temporal_folds, prepare_train_test
from src.evaluation.metrics import compute_all_metrics, print_metrics
from src.labels.outbreak_labels import create_outbreak_labels

# Import models
from src.models.baselines.threshold import ThresholdBaseline
from src.models.baselines.logistic import LogisticBaseline
from src.models.baselines.random_forest import RandomForestBaseline

# XGBoost is optional (requires libomp on Mac)
try:
    from src.models.baselines.xgboost_model import XGBoostBaseline
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Warning: XGBoost not available ({e})")
    XGBOOST_AVAILABLE = False


def get_baseline_models(config: dict) -> dict:
    """Create all baseline model instances."""
    model_cfg = config.get('models', {}).get('baselines', {})
    
    models = {
        'threshold': ThresholdBaseline(model_cfg.get('threshold', {})),
        'logistic': LogisticBaseline(model_cfg.get('logistic', {})),
        'random_forest': RandomForestBaseline(model_cfg.get('random_forest', {})),
    }
    
    # Only add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBoostBaseline(model_cfg.get('xgboost', {}))
    
    return models


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float
) -> dict:
    """Train model and compute metrics."""
    # Fit
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    # NOTE: threshold is config-driven for alignment across pipeline.
    # Callers should supply the desired probability threshold.
    metrics = compute_all_metrics(y_test, y_pred_proba, threshold=threshold)
    
    return metrics


def prepare_fold_arrays_with_imputation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str] = ['state', 'district'],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Fold-safe preprocessing with conservative missingness handling.

    Steps:
    - Drop rows with missing target only
    - Apply past-only forward-fill within each subset for lag-like columns
    - Apply neutral-value fills for specific EWS/LAI sparse features
    - Leave remaining missing values as NaN (no global/statistical imputation)
    """

    train = train_df.dropna(subset=[target_col]).copy()
    test = test_df.dropna(subset=[target_col]).copy()

    # Ensure deterministic temporal ordering before groupwise transforms.
    if all(c in train.columns for c in group_cols + ['year', 'week']):
        train = train.sort_values(group_cols + ['year', 'week']).reset_index(drop=True)
    if all(c in test.columns for c in group_cols + ['year', 'week']):
        test = test.sort_values(group_cols + ['year', 'week']).reset_index(drop=True)

    # Past-only forward fill for lag/persistence/anomaly features.
    lag_cols = [
        c for c in feature_cols
        if any(x in c for x in ['lag', 'persist', 'anomaly'])
    ]
    for col in lag_cols:
        if col in train.columns:
            train[col] = train.groupby(group_cols)[col].transform(lambda x: x.ffill())
        if col in test.columns:
            test[col] = test.groupby(group_cols)[col].transform(lambda x: x.ffill())

    # Neutral-value fills for sparse engineered features.
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

    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train[target_col].to_numpy(dtype=float)
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    meta_cols = [c for c in ['state', 'district', 'year', 'week'] if c in test.columns]
    test_meta = test[meta_cols].copy().reset_index(drop=True)

    return X_train, y_train, X_test, y_test, test_meta


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="track_a",
        choices=["full", "core", "track_a"],
        help="Which feature set to use: full (all 37), core (20 sparse-robust), track_a (9 mechanistic)"
    )
    parser.add_argument(
        "--outbreak-percentiles",
        type=int,
        nargs='*',
        default=None,
        help="Optional sensitivity run(s): recompute labels and train baselines for these percentiles (e.g. 70 75 80)."
    )
    args = parser.parse_args()
    
    # Load config (from v6 snapshot)
    v6_root = get_project_root()
    repo_root = get_repo_root()
    config_path = v6_root / args.config
    cfg = load_config(str(config_path))
    
    # Load features (from repo-level processed data)
    features_path = repo_root / cfg['data']['processed']['features']
    print("=" * 60)
    print("CHIKUNGUNYA EWS - TRAIN BASELINES")
    print("=" * 60)
    
    print(f"\nLoading features from {features_path}...")
    df = pd.read_parquet(features_path)
    print(f"  → {len(df)} rows, {len(df.columns)} columns")

    # Ensure deterministic temporal ordering before any groupwise transforms.
    # (GroupBy.transform respects the current row order.)
    if all(c in df.columns for c in ['state', 'district', 'year', 'week']):
        df = df.sort_values(['state', 'district', 'year', 'week']).reset_index(drop=True)
    
    # Get feature columns
    feature_cols = select_feature_columns(df.columns, feature_set=args.feature_set)
    target_col = 'label_outbreak'
    
    print(f"  → Feature set: {args.feature_set}")
    print(f"  → {len(feature_cols)} features")
    print(f"  → Target: {target_col}")
    
    percentiles = args.outbreak_percentiles
    if not percentiles:
        percentiles = [int(cfg.get('labels', {}).get('outbreak_percentile', 75))]

    # Get models (shared config)
    models = get_baseline_models(cfg)

    for percentile in percentiles:
        print("\n" + "=" * 60)
        print(f"LABEL SENSITIVITY RUN: p{percentile}")
        print("=" * 60)

        df_run = df.copy()
        if 'incidence_per_100k' in df_run.columns:
            df_run = create_outbreak_labels(
                df_run,
                percentile=int(percentile),
                horizon=int(cfg.get('labels', {}).get('horizon', 3)),
                value_col='incidence_per_100k',
            )
        else:
            raise ValueError("Expected incidence_per_100k column for label recomputation")

        # Filter to valid samples (target only)
        valid_df = df_run.dropna(subset=[target_col]).copy()
        print(f"  → {len(valid_df)} labeled samples")

        # Create stratified CV splits
        test_years = cfg['cv']['test_years']
        folds = create_stratified_temporal_folds(
            df=valid_df,
            target_col=target_col,
            year_col='year',
            min_positives=5,
            candidate_test_years=test_years,
            verbose=False
        )
        print(f"\nCV Folds: {len(folds)} (stratified to ensure ≥5 positives/fold)")
        for fold in folds:
            test_years_str = "-".join(map(str, fold.test_years))
            print(f"  {fold.fold_name}: train={len(fold.train_idx)}, test={len(fold.test_idx)}, test_years={test_years_str}")

        print(f"\nModels: {list(models.keys())}")

        # Results storage
        all_results: dict[str, dict] = {}

        # Train and evaluate each model
        print("\n" + "=" * 60)
        print("TRAINING & EVALUATION")
        print("=" * 60)

        preds_out_dir = v6_root / cfg.get('output', {}).get('predictions_dir', 'results/predictions')
        preds_out_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = v6_root / cfg.get('output', {}).get('metrics_dir', 'results/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)

        prob_threshold = cfg.get('evaluation', {}).get('probability_threshold')
        if prob_threshold is None:
            raise ValueError("Missing evaluation.probability_threshold in config.")
        prob_threshold = float(prob_threshold)

        for model_name in models.keys():
            print(f"\n{'='*40}")
            print(f"Model: {model_name.upper()}")
            print(f"{'='*40}")

            fold_metrics: list[dict] = []
            fold_predictions: list[pd.DataFrame] = []
            skipped_folds: list[dict] = []

            for fold in folds:
                train_df = valid_df.iloc[fold.train_idx]
                test_df = valid_df.iloc[fold.test_idx]

                X_train, y_train, X_test, y_test, test_meta = prepare_fold_arrays_with_imputation(
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                )

                # Skip if insufficient data
                if len(X_train) < 10 or len(X_test) < 5:
                    print(f"  {fold.fold_name}: Skipping (insufficient data)")
                    skipped_folds.append({'fold': fold.fold_name, 'reason': 'insufficient_data'})
                    continue

                if len(np.unique(y_train)) < 2:
                    print(f"  {fold.fold_name}: Skipping (single class in train)")
                    skipped_folds.append({'fold': fold.fold_name, 'reason': 'single_class_train'})
                    continue

                try:
                    # Re-instantiate model for fresh training
                    if model_name == 'threshold':
                        fold_model = ThresholdBaseline(cfg.get('models', {}).get('baselines', {}).get('threshold', {}))
                    elif model_name == 'logistic':
                        fold_model = LogisticBaseline(cfg.get('models', {}).get('baselines', {}).get('logistic', {}))
                    elif model_name == 'random_forest':
                        fold_model = RandomForestBaseline(cfg.get('models', {}).get('baselines', {}).get('random_forest', {}))
                    else:
                        fold_model = XGBoostBaseline(cfg.get('models', {}).get('baselines', {}).get('xgboost', {}))

                    fold_model.fit(X_train, y_train)
                    y_pred_proba = fold_model.predict_proba(X_test)
                    metrics = compute_all_metrics(y_test, y_pred_proba, threshold=prob_threshold)
                    metrics['fold'] = fold.fold_name
                    metrics['test_year'] = fold.test_year
                    fold_metrics.append(metrics)

                    # Persist per-row predictions for transparency/calibration plots.
                    pred_df = test_meta.copy()
                    pred_df['fold'] = fold.fold_name
                    pred_df['test_year'] = fold.test_year
                    pred_df['model'] = model_name
                    pred_df['outbreak_percentile'] = int(percentile)
                    pred_df['y_true'] = y_test.astype(int)
                    pred_df['y_pred_proba'] = np.asarray(y_pred_proba, dtype=float)
                    pred_df['threshold'] = prob_threshold
                    fold_predictions.append(pred_df)

                    print(
                        f"  {fold.fold_name}: AUC={metrics['auc']:.3f}, "
                        f"F1={metrics['f1']:.3f}, n={metrics['n_samples']}"
                    )
                except Exception as e:
                    print(f"  {fold.fold_name}: ERROR - {e}")
                    skipped_folds.append({'fold': fold.fold_name, 'reason': f'error: {type(e).__name__}'})
                    continue

            if fold_predictions:
                preds_name = (
                    f"baseline_cv_predictions_{model_name}_p{percentile}.parquet"
                    if len(percentiles) > 1
                    else f"baseline_cv_predictions_{model_name}.parquet"
                )
                preds_path = preds_out_dir / preds_name
                pd.concat(fold_predictions, ignore_index=True).to_parquet(preds_path, index=False)
                print(f"  → Saved predictions: {preds_path}")

            # Aggregate metrics
            if fold_metrics:
                auc_vals = [m['auc'] for m in fold_metrics if not np.isnan(m.get('auc', np.nan))]
                brier_vals = [m['brier'] for m in fold_metrics if not np.isnan(m.get('brier', np.nan))]
                agg = {
                    'auc_mean': float(np.mean(auc_vals)) if auc_vals else float('nan'),
                    'auc_std': float(np.std(auc_vals)) if auc_vals else float('nan'),
                    'f1_mean': float(np.mean([m['f1'] for m in fold_metrics])),
                    'f1_std': float(np.std([m['f1'] for m in fold_metrics])),
                    'sensitivity_mean': float(np.mean([m['sensitivity'] for m in fold_metrics])),
                    'specificity_mean': float(np.mean([m['specificity'] for m in fold_metrics])),
                    'brier_mean': float(np.mean(brier_vals)) if brier_vals else float('nan'),
                    'n_folds': int(len(fold_metrics)),
                    'fold_results': fold_metrics,
                    'skipped_folds': skipped_folds,
                }
                all_results[model_name] = agg

                print(
                    f"\n  AGGREGATE: AUC={agg['auc_mean']:.3f}±{agg['auc_std']:.3f}, "
                    f"F1={agg['f1_mean']:.3f}±{agg['f1_std']:.3f}"
                )
            else:
                all_results[model_name] = {
                    'auc_mean': float('nan'),
                    'auc_std': float('nan'),
                    'f1_mean': float('nan'),
                    'f1_std': float('nan'),
                    'sensitivity_mean': float('nan'),
                    'specificity_mean': float('nan'),
                    'brier_mean': float('nan'),
                    'n_folds': 0,
                    'fold_results': [],
                    'skipped_folds': skipped_folds,
                }

        # Summary comparison
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\n{'Model':<15} {'AUC':>10} {'F1':>10} {'Sens':>10} {'Spec':>10} {'Brier':>10}")
        print("-" * 65)
        for model_name, results in all_results.items():
            print(
                f"{model_name:<15} "
                f"{results['auc_mean']:>10.3f} {results['f1_mean']:>10.3f} "
                f"{results['sensitivity_mean']:>10.3f} {results['specificity_mean']:>10.3f} "
                f"{results['brier_mean']:>10.3f}"
            )

        # Save results for this percentile
        out_name = (
            f"baseline_comparison_p{percentile}.json"
            if len(percentiles) > 1
            else "baseline_comparison.json"
        )
        output_path = metrics_dir / out_name

        serializable_results = {
            'timestamp': datetime.now().isoformat(),
            'config_path': str(config_path),
            'feature_set': args.feature_set,
            'outbreak_percentile': int(percentile),
            'n_features': int(len(feature_cols)),
            'n_samples': int(len(valid_df)),
            'cv_folds': int(len(folds)),
            'probability_threshold': float(prob_threshold),
            'models': all_results,
        }

        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=convert_to_serializable)
        print(f"\n✓ Saved results to {output_path}")

    return


if __name__ == "__main__":
    main()
