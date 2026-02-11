"""
Outbreak Label Generation for Chikungunya EWS

Creates binary outbreak labels for supervised learning.
Uses a config-driven percentile threshold that adapts
to each district's baseline.

Reference: 05_experiments.md Section 5.4.1
"""
import pandas as pd
import numpy as np
from typing import Optional
from src.common.thresholds import enforce_minimum_threshold


def create_outbreak_labels(
    df: pd.DataFrame,
    percentile: int,
    horizon: int = 3,
    value_col: str = 'incidence_per_100k',
    group_cols: list = ['state', 'district']
) -> pd.DataFrame:
    """
    Create binary outbreak labels.
    
    Definition:
    Y_t = 1 if incidence in [t+1, t+H] exceeds p-th percentile of 
          historical incidence for that district
    Y_t = 0 otherwise
    
    The label is forward-looking: we want to predict if an outbreak
    will occur in the NEXT H weeks (t+1 to t+H, exclusive of current week t).
    FIX #4: Corrected to prevent label leakage by excluding current week t.
    
    Args:
        df: Panel DataFrame with incidence column
        percentile: Threshold percentile (config-driven)
        horizon: Prediction horizon (weeks ahead)
        value_col: Column for incidence
        group_cols: Grouping columns (district-level)
        
    Returns:
        DataFrame with label_outbreak column
    """
    if percentile is None:
        raise ValueError("percentile must be provided via config or caller")

    df = df.copy()
    
    # Compute district-specific threshold (p-th percentile)
    # Using only data BEFORE current point to prevent leakage
    def compute_dynamic_threshold(group):
        """Compute expanding percentile threshold (no future data)."""
        thresholds = []
        for i in range(len(group)):
            if i < 10:  # Need minimum history
                thresholds.append(np.nan)
            else:
                historical = group.iloc[:i][value_col].values
                valid_historical = historical[~np.isnan(historical)]
                if len(valid_historical) >= 5:  # Need at least 5 valid points
                    threshold = np.percentile(valid_historical, percentile)
                    # Enforce minimum threshold to avoid classifying <1 case as outbreak
                    threshold = enforce_minimum_threshold(threshold)
                    thresholds.append(threshold)
                else:
                    thresholds.append(np.nan)
        return pd.Series(thresholds, index=group.index)
    
    print(f"Computing outbreak labels (p{percentile}, horizon={horizon} weeks)...")
    
    # Sort first
    df = df.sort_values(group_cols + ['year', 'week']).reset_index(drop=True)
    
    # Compute dynamic threshold per district
    df['_threshold'] = df.groupby(group_cols, group_keys=False).apply(
        compute_dynamic_threshold
    ).values
    
    # Check if ANY of the next H weeks exceeds threshold
    # This is our forward-looking label
    def check_future_outbreak(group):
        """Check if outbreak occurs in next H weeks."""
        labels = []
        values = group[value_col].values
        thresholds = group['_threshold'].values
        
        for i in range(len(group)):
            if pd.isna(thresholds[i]):
                labels.append(np.nan)
                continue
                
            # FIX #4: Look at NEXT H weeks (t+1 to t+H), excluding current week t
            # This prevents label leakage - we predict future outbreaks, not current state
            future_start = i + 1
            future_end = min(i + 1 + horizon, len(group))
            
            if future_end <= future_start:
                labels.append(np.nan)
                continue
            
            future_values = values[future_start:future_end]
            threshold = thresholds[i]
            
            # Label = 1 if any future week exceeds threshold
            if any(v > threshold for v in future_values if not np.isnan(v)):
                labels.append(1)
            else:
                labels.append(0)
        
        return pd.Series(labels, index=group.index)
    
    df['label_outbreak'] = df.groupby(group_cols, group_keys=False).apply(
        check_future_outbreak
    ).values
    
    # Also store threshold for reference
    df['label_threshold'] = df['_threshold']
    
    # Cleanup
    df = df.drop(columns=['_threshold'])
    
    # Stats
    valid = df['label_outbreak'].notna().sum()
    positive = (df['label_outbreak'] == 1).sum()
    print(f"  ✓ {valid} labeled samples ({positive} positive, {valid-positive} negative)")
    print(f"  ✓ Positive rate: {100*positive/valid:.1f}%")
    
    return df


def create_labels_static_threshold(
    df: pd.DataFrame,
    percentile: int,
    horizon: int = 3,
    value_col: str = 'incidence_per_100k',
    group_cols: list = ['state', 'district']
) -> pd.DataFrame:
    """
    Alternative: static threshold per district (simpler, faster).
    
    Uses the entire history to compute threshold (not expanding).
    Faster but has minor look-ahead bias (acceptable for baseline).
    
    Args:
        df: Panel DataFrame
        percentile: Threshold percentile (config-driven)
        horizon: Prediction horizon
        value_col: Incidence column
        group_cols: Grouping columns
        
    Returns:
        DataFrame with label_outbreak column
    """
    if percentile is None:
        raise ValueError("percentile must be provided via config or caller")

    df = df.copy()
    
    # Compute static threshold per district (entire history)
    thresholds = df.groupby(group_cols)[value_col].transform(
        lambda x: np.percentile(x.dropna(), percentile)
    )
    df['_threshold'] = thresholds
    
    # Forward maximum over horizon
    df['_future_max'] = df.groupby(group_cols)[value_col].transform(
        lambda x: x.rolling(window=horizon, min_periods=1).max().shift(-(horizon-1))
    )
    
    # Label
    df['label_outbreak'] = (df['_future_max'] > df['_threshold']).astype(float)
    df['label_threshold'] = df['_threshold']
    
    # Cleanup
    df = df.drop(columns=['_threshold', '_future_max'])
    
    return df


def add_labels_to_features(
    features_df: pd.DataFrame,
    percentile: int,
    horizon: int = 3,
    use_dynamic: bool = True
) -> pd.DataFrame:
    """
    Add outbreak labels to feature DataFrame.
    
    Args:
        features_df: DataFrame with features computed
        percentile: Outbreak threshold percentile (config-driven)
        horizon: Prediction horizon (weeks)
        use_dynamic: Use dynamic (expanding) threshold vs static
        
    Returns:
        DataFrame with labels added
    """
    if percentile is None:
        raise ValueError("percentile must be provided via config or caller")

    if use_dynamic:
        return create_outbreak_labels(
            features_df, 
            percentile=percentile, 
            horizon=horizon
        )
    else:
        return create_labels_static_threshold(
            features_df,
            percentile=percentile,
            horizon=horizon
        )
