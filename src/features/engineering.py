"""
Main Feature Engineering Orchestrator for Chikungunya EWS

This module coordinates all feature computation and produces
the final feature matrix ready for modeling.

Reference: 03_tdd.md Section 3.3
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

from .case_features import compute_all_case_features
from .climate_features import compute_all_climate_features
from .ews_features import compute_all_ews_features


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonal encoding features.
    
    Uses cyclical encoding for week (sin/cos) to capture
    annual seasonality without discontinuity at year boundaries.
    
    Args:
        df: Panel DataFrame with 'week' column
        
    Returns:
        DataFrame with seasonal features
    """
    df = df.copy()
    
    # Cyclical encoding for week of year
    df['feat_week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['feat_week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    
    # Quarter indicators
    df['feat_quarter'] = ((df['week'] - 1) // 13) + 1
    
    # Monsoon season indicator (roughly June-September = weeks 22-39)
    df['feat_is_monsoon'] = ((df['week'] >= 22) & (df['week'] <= 39)).astype(int)
    
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic spatial features from coordinates.
    
    Args:
        df: Panel DataFrame with latitude, longitude
        
    Returns:
        DataFrame with spatial features
    """
    df = df.copy()
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Normalized coordinates (India roughly 8-37°N, 68-97°E)
        df['feat_lat_norm'] = (df['latitude'] - 8) / (37 - 8)
        df['feat_lon_norm'] = (df['longitude'] - 68) / (97 - 68)
        
        # Interaction (captures regional patterns)
        df['feat_lat_lon_interact'] = df['feat_lat_norm'] * df['feat_lon_norm']
    
    return df


def compute_all_features(
    panel_path: str,
    config: Optional[dict] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute all features from panel dataset.
    
    Features computed:
    1. Case-based features (lags, MA, growth, variance, ACF, trend, skewness)
    2. Climate mechanistic features (degree-days, rainfall persistence, temp anomaly)
    3. EWS features (variance spike, ACF change, trend acceleration)
    4. Seasonal features (cyclical week encoding, monsoon indicator)
    5. Spatial features (normalized coordinates)
    
    Args:
        panel_path: Path to panel parquet file
        config: Optional config dict
        output_path: If provided, save features to this path
        
    Returns:
        DataFrame with all features
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load panel
    print(f"Loading panel from {panel_path}...")
    df = pd.read_parquet(panel_path)
    print(f"  → {len(df)} rows, {len(df.columns)} columns")
    
    # Sort by district and time
    df = df.sort_values(['state', 'district', 'year', 'week']).reset_index(drop=True)
    
    # 1. Case-based features
    df = compute_all_case_features(df, config)
    
    # 2. Climate features
    df = compute_all_climate_features(df, config)
    
    # 3. EWS features
    df = compute_all_ews_features(df, config)
    
    # 4. Seasonal features
    print("Computing seasonal features...")
    df = add_seasonal_features(df)
    print("  ✓ Week sin/cos, quarter, monsoon indicator")
    
    # 5. Spatial features
    print("Computing spatial features...")
    df = add_spatial_features(df)
    print("  ✓ Normalized coordinates")
    
    # Summary
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    print(f"\n{'=' * 60}")
    print(f"FEATURE SUMMARY: {len(feature_cols)} features computed")
    print(f"{'=' * 60}")
    
    # Group features by type
    case_feats = [c for c in feature_cols if 'cases' in c or c.startswith('feat_cases')]
    climate_feats = [c for c in feature_cols if any(x in c for x in ['temp', 'rain', 'degree', 'lai'])]
    ews_feats = [c for c in feature_cols if any(x in c for x in ['spike', 'acf_change', 'accel', 'normalized'])]
    other_feats = [c for c in feature_cols if c not in case_feats + climate_feats + ews_feats]
    
    print(f"  Case features: {len(case_feats)}")
    print(f"  Climate features: {len(climate_feats)}")
    print(f"  EWS features: {len(ews_feats)}")
    print(f"  Other features: {len(other_feats)}")
    
    # Missing data report
    print(f"\nFeatures with >50% missing:")
    for col in feature_cols:
        missing_pct = 100 * df[col].isna().sum() / len(df)
        if missing_pct > 50:
            print(f"  {col}: {missing_pct:.1f}%")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\n✓ Saved to {output_path}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature column names."""
    return [c for c in df.columns if c.startswith('feat_')]


def get_target_column() -> str:
    """Get the standard target column name."""
    return 'label_outbreak'
