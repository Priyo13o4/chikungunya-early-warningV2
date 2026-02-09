#!/usr/bin/env python3
"""
Experiment 02: Build Features

Computes all engineered features from the panel dataset:
- Case-based features (lags, MA, growth, variance, ACF, trend, skewness)
- Climate features (degree-days, rainfall persistence, temp anomaly)
- EWS features (variance spike, ACF change, trend acceleration)
- Seasonal and spatial features

Output: data/processed/features_engineered_v01.parquet

Usage:
    python experiments/02_build_features.py
    python experiments/02_build_features.py --config config/config_default.yaml
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, get_project_root, get_repo_root
from src.features.engineering import compute_all_features, get_feature_columns
from src.labels.outbreak_labels import add_labels_to_features


def main():
    parser = argparse.ArgumentParser(description="Build engineered features")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--add-labels",
        action="store_true",
        default=True,
        help="Add outbreak labels"
    )
    args = parser.parse_args()
    
    # Load config
    root = get_project_root()
    repo_root = get_repo_root()
    config_path = root / args.config
    cfg = load_config(str(config_path))
    
    # Paths
    # v6 is nested; canonical artifacts live at the repo root.
    panel_path = repo_root / cfg['data']['processed']['panel']
    features_path = repo_root / cfg['data']['processed']['features']
    
    print("=" * 60)
    print("CHIKUNGUNYA EWS - BUILD FEATURES")
    print("=" * 60)
    
    # Compute features
    features_df = compute_all_features(
        panel_path=str(panel_path),
        config=cfg,
        output_path=None  # Don't save yet
    )
    
    # Add labels
    if args.add_labels:
        print("\n" + "=" * 60)
        print("ADDING OUTBREAK LABELS")
        print("=" * 60)
        
        features_df = add_labels_to_features(
            features_df,
            percentile=cfg['labels']['outbreak_percentile'],
            horizon=cfg['labels']['horizon'],
            use_dynamic=True
        )
    
    # Save
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_path, index=False)
    print(f"\n✓ Saved to {features_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(features_df)}")
    print(f"Total columns: {len(features_df.columns)}")
    
    feature_cols = get_feature_columns(features_df)
    print(f"Feature columns: {len(feature_cols)}")
    
    # Check for labels
    if 'label_outbreak' in features_df.columns:
        valid = features_df['label_outbreak'].notna().sum()
        pos = (features_df['label_outbreak'] == 1).sum()
        print(f"Labeled samples: {valid} ({pos} positive, {100*pos/valid:.1f}%)")
    
    # Show feature names
    print("\nFeature columns:")
    for i, col in enumerate(sorted(feature_cols)):
        print(f"  {i+1}. {col}")
    
    return features_df


if __name__ == "__main__":
    main()
