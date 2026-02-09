"""
Case-based Feature Engineering for Chikungunya EWS

Features:
- Case lags (1, 2, 4, 8 weeks)
- Moving averages (2w, 4w)
- Growth rate
- Rolling variance (4w)
- Autocorrelation (lag-1, 4w window)
- Trend (4w slope)
- Skewness (4w)

Reference: 03_tdd.md Section 3.3.1
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from scipy import stats


def compute_case_lags(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 4, 8],
    value_col: str = 'incidence_per_100k',
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute lagged case features.
    
    Args:
        df: Panel DataFrame sorted by group + time
        lags: List of lag values (weeks back)
        value_col: Column to lag
        group_cols: Grouping columns (district-level)
        
    Returns:
        DataFrame with added lag columns
    """
    df = df.copy()
    
    for lag in lags:
        col_name = f'feat_cases_lag_{lag}'
        df[col_name] = df.groupby(group_cols)[value_col].shift(lag)
    
    return df


def compute_moving_averages(
    df: pd.DataFrame,
    windows: List[int] = [2, 4],
    value_col: str = 'incidence_per_100k',
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute moving average features.
    
    Args:
        df: Panel DataFrame
        windows: Window sizes in weeks
        value_col: Column to average
        group_cols: Grouping columns
        
    Returns:
        DataFrame with added MA columns
    """
    df = df.copy()
    
    for window in windows:
        col_name = f'feat_cases_ma_{window}w'
        df[col_name] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return df


def compute_growth_rate(
    df: pd.DataFrame,
    value_col: str = 'incidence_per_100k',
    group_cols: List[str] = ['state', 'district'],
    clip_range: tuple = (-1.0, 1.0)
) -> pd.DataFrame:
    """
    Compute week-over-week growth rate.
    
    Formula: (c_t - c_{t-1}) / (c_{t-1} + epsilon)
    Clipped to prevent extreme values.
    
    Args:
        df: Panel DataFrame
        value_col: Column for growth rate
        group_cols: Grouping columns
        clip_range: Min/max bounds for clipping
        
    Returns:
        DataFrame with growth rate column
    """
    df = df.copy()
    epsilon = 0.1  # Prevent division by zero
    
    prev_value = df.groupby(group_cols)[value_col].shift(1)
    df['feat_cases_growth_rate'] = (df[value_col] - prev_value) / (prev_value + epsilon)
    df['feat_cases_growth_rate'] = df['feat_cases_growth_rate'].clip(*clip_range)
    
    return df


def compute_rolling_variance(
    df: pd.DataFrame,
    window: int = 4,
    value_col: str = 'incidence_per_100k',
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute rolling variance (early warning indicator).
    
    Rising variance can signal approaching transition/outbreak.
    
    Args:
        df: Panel DataFrame
        window: Window size in weeks
        value_col: Column for variance
        group_cols: Grouping columns
        
    Returns:
        DataFrame with variance column
    """
    df = df.copy()
    
    df[f'feat_cases_var_{window}w'] = df.groupby(group_cols)[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=2).var()
    )
    
    return df


def compute_autocorrelation(
    df: pd.DataFrame,
    lag: int = 1,
    window: int = 4,
    value_col: str = 'incidence_per_100k',
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute rolling lag-1 autocorrelation (early warning indicator).
    
    Loss of autocorrelation can signal system instability.
    
    Args:
        df: Panel DataFrame
        lag: Autocorrelation lag
        window: Rolling window size
        value_col: Column for autocorrelation
        group_cols: Grouping columns
        
    Returns:
        DataFrame with ACF column
    """
    df = df.copy()
    
    def rolling_acf(series, lag=1, window=4):
        """Compute rolling autocorrelation."""
        result = []
        for i in range(len(series)):
            if i < window:
                result.append(np.nan)
            else:
                window_data = series.iloc[i-window:i].dropna()
                std_val = window_data.std()
                if len(window_data) >= window - 1 and not pd.isna(std_val) and std_val > 0:
                    # Lag-1 correlation
                    try:
                        acf = window_data.autocorr(lag=lag)
                        result.append(acf if not pd.isna(acf) else np.nan)
                    except:
                        result.append(np.nan)
                else:
                    result.append(np.nan)
        return pd.Series(result, index=series.index)
    
    df[f'feat_cases_acf_lag{lag}_{window}w'] = df.groupby(group_cols)[value_col].transform(
        lambda x: rolling_acf(x, lag=lag, window=window)
    )
    
    return df


def compute_trend(
    df: pd.DataFrame,
    window: int = 4,
    value_col: str = 'incidence_per_100k',
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute rolling trend (linear slope over window).
    
    Positive slope indicates increasing cases.
    
    Args:
        df: Panel DataFrame
        window: Window size for trend
        value_col: Column for trend
        group_cols: Grouping columns
        
    Returns:
        DataFrame with trend column
    """
    df = df.copy()
    
    def rolling_slope(series, window=4):
        """Compute rolling linear slope."""
        result = []
        for i in range(len(series)):
            if i < window - 1:
                result.append(np.nan)
            else:
                window_data = series.iloc[i-window+1:i+1].values
                if len(window_data) == window and not np.isnan(window_data).any():
                    x = np.arange(window)
                    slope, _, _, _, _ = stats.linregress(x, window_data)
                    result.append(slope)
                else:
                    result.append(np.nan)
        return pd.Series(result, index=series.index)
    
    df[f'feat_cases_trend_{window}w'] = df.groupby(group_cols)[value_col].transform(
        lambda x: rolling_slope(x, window=window)
    )
    
    return df


def compute_skewness(
    df: pd.DataFrame,
    window: int = 4,
    value_col: str = 'incidence_per_100k',
    group_cols: List[str] = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute rolling skewness (early warning indicator).
    
    Right-skewed distribution can indicate emerging outbreak.
    
    Args:
        df: Panel DataFrame
        window: Window size
        value_col: Column for skewness
        group_cols: Grouping columns
        
    Returns:
        DataFrame with skewness column
    """
    df = df.copy()
    
    df[f'feat_cases_skew_{window}w'] = df.groupby(group_cols)[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=3).skew()
    )
    
    return df


def compute_all_case_features(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute all case-based features.
    
    Args:
        df: Panel DataFrame with incidence_per_100k column
        config: Optional config dict with feature_engineering settings
        
    Returns:
        DataFrame with all case features added
    """
    # Default settings
    lags = [1, 2, 4, 8]
    ma_windows = [2, 4]
    var_window = 4
    
    if config:
        fe = config.get('feature_engineering', {})
        lags = fe.get('case_lags', lags)
        ma_windows = fe.get('rolling_windows', ma_windows)
    
    print("Computing case-based features...")
    
    # Ensure sorted
    df = df.sort_values(['state', 'district', 'year', 'week']).reset_index(drop=True)
    
    # Compute features
    df = compute_case_lags(df, lags=lags)
    print(f"  ✓ Case lags: {lags}")
    
    df = compute_moving_averages(df, windows=ma_windows)
    print(f"  ✓ Moving averages: {ma_windows}w")
    
    df = compute_growth_rate(df)
    print("  ✓ Growth rate")
    
    df = compute_rolling_variance(df, window=var_window)
    print(f"  ✓ Rolling variance: {var_window}w")
    
    df = compute_autocorrelation(df, lag=1, window=var_window)
    print(f"  ✓ Autocorrelation: lag-1, {var_window}w")
    
    df = compute_trend(df, window=var_window)
    print(f"  ✓ Trend: {var_window}w")
    
    df = compute_skewness(df, window=var_window)
    print(f"  ✓ Skewness: {var_window}w")
    
    return df
