"""
Climate-based Feature Engineering for Chikungunya EWS

Mechanistic Features:
- Temperature lags (1, 2, 4, 8 weeks)
- Degree-days above threshold (Aedes development)
- Rainfall lags
- Rainfall persistence (4w cumulative)
- Temperature anomaly (deviation from historical mean)

Reference: 03_tdd.md Section 3.3.1 (Climate Mechanistic Features)
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def compute_climate_lags(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 4, 8],
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute lagged climate features for temperature and precipitation.
    
    Args:
        df: Panel DataFrame with temp_celsius, precipitation_mm
        lags: List of lag values (weeks back)
        group_cols: Grouping columns
        
    Returns:
        DataFrame with climate lag columns
    """
    df = df.copy()
    
    # Temperature lags
    if 'temp_celsius' in df.columns:
        for lag in lags:
            df[f'feat_temp_lag_{lag}'] = df.groupby(group_cols)['temp_celsius'].shift(lag)
    
    # Precipitation lags
    if 'precipitation_mm' in df.columns:
        for lag in lags:
            df[f'feat_rain_lag_{lag}'] = df.groupby(group_cols)['precipitation_mm'].shift(lag)
    
    return df


def compute_degree_days(
    df: pd.DataFrame,
    threshold: float = 20.0,
    window: int = 2,
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute degree-days above threshold (mechanistic feature).
    
    Degree-days capture accumulated heat for Aedes mosquito development.
    Aedes aegypti development threshold is approximately 20°C.
    
    Formula: sum(max(0, T - threshold)) over window
    
    Args:
        df: Panel DataFrame with temp_celsius
        threshold: Temperature threshold (Celsius)
        window: Number of weeks to accumulate
        group_cols: Grouping columns
        
    Returns:
        DataFrame with degree-days column
    """
    df = df.copy()
    
    if 'temp_celsius' not in df.columns:
        print("  ⚠ temp_celsius not found, skipping degree-days")
        return df
    
    # Compute daily excess degree (assuming temp_celsius is weekly mean)
    # Multiply by 7 to approximate degree-days for the week
    df['_temp_excess'] = (df['temp_celsius'] - threshold).clip(lower=0) * 7
    
    # Rolling sum over window weeks
    df[f'feat_degree_days_above_{int(threshold)}'] = df.groupby(group_cols)['_temp_excess'].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum()
    )
    
    # Cleanup temp column
    df = df.drop(columns=['_temp_excess'])
    
    return df


def compute_rainfall_persistence(
    df: pd.DataFrame,
    window: int = 4,
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute cumulative rainfall over window (habitat formation).
    
    Aedes mosquitoes breed in standing water. Accumulated rainfall
    over several weeks creates more breeding sites.
    
    Args:
        df: Panel DataFrame with precipitation_mm
        window: Number of weeks to accumulate
        group_cols: Grouping columns
        
    Returns:
        DataFrame with rainfall persistence column
    """
    df = df.copy()
    
    if 'precipitation_mm' not in df.columns:
        print("  ⚠ precipitation_mm not found, skipping rainfall persistence")
        return df
    
    df[f'feat_rain_persist_{window}w'] = df.groupby(group_cols)['precipitation_mm'].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum()
    )
    
    return df


def compute_temperature_anomaly(
    df: pd.DataFrame,
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute temperature anomaly (deviation from historical month mean).
    
    Warmer-than-normal temperatures can accelerate mosquito development.
    
    Args:
        df: Panel DataFrame with temp_celsius, year, week
        group_cols: Grouping columns
        
    Returns:
        DataFrame with temperature anomaly column
    """
    df = df.copy()
    
    if 'temp_celsius' not in df.columns:
        print("  ⚠ temp_celsius not found, skipping temperature anomaly")
        return df
    
    # Approximate month from week (week 1-4 = Jan, 5-8 = Feb, etc.)
    df['_month'] = ((df['week'] - 1) // 4) + 1
    df['_month'] = df['_month'].clip(upper=12)
    
    # Compute historical mean by district-month (across all years)
    historical_mean = df.groupby(group_cols + ['_month'])['temp_celsius'].transform('mean')
    
    # Anomaly = current - historical mean
    df['feat_temp_anomaly'] = df['temp_celsius'] - historical_mean
    
    # Cleanup
    df = df.drop(columns=['_month'])
    
    return df


def compute_lai_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 4],
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute Leaf Area Index (LAI) features.
    
    LAI is a proxy for vegetation which can indicate mosquito habitat.
    
    Args:
        df: Panel DataFrame with lai column
        lags: Lag values
        group_cols: Grouping columns
        
    Returns:
        DataFrame with LAI features
    """
    df = df.copy()
    
    if 'lai' not in df.columns:
        print("  ⚠ lai not found, skipping LAI features")
        return df
    
    # Current LAI
    df['feat_lai'] = df['lai']
    
    # LAI lags
    for lag in lags:
        df[f'feat_lai_lag_{lag}'] = df.groupby(group_cols)['lai'].shift(lag)
    
    return df


def compute_all_climate_features(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute all climate-based mechanistic features.
    
    Args:
        df: Panel DataFrame with climate columns
        config: Optional config with feature_engineering settings
        
    Returns:
        DataFrame with all climate features added
    """
    # Default settings
    lags = [1, 2, 4, 8]
    dd_threshold = 20.0
    rain_window = 4
    
    if config:
        fe = config.get('feature_engineering', {})
        lags = fe.get('climate_lags', lags)
        dd_threshold = fe.get('degree_day_threshold', dd_threshold)
    
    print("Computing climate-based features...")
    
    # Ensure sorted
    df = df.sort_values(['state', 'district', 'year', 'week']).reset_index(drop=True)
    
    # Compute features
    df = compute_climate_lags(df, lags=lags)
    print(f"  ✓ Climate lags: {lags}")
    
    df = compute_degree_days(df, threshold=dd_threshold, window=2)
    print(f"  ✓ Degree-days above {dd_threshold}°C")
    
    df = compute_rainfall_persistence(df, window=rain_window)
    print(f"  ✓ Rainfall persistence: {rain_window}w")
    
    df = compute_temperature_anomaly(df)
    print("  ✓ Temperature anomaly")
    
    df = compute_lai_features(df, lags=[1, 2, 4])
    print("  ✓ LAI features")
    
    return df
