"""
Early Warning System (EWS) Features for Chikungunya

Unique EWS indicators that detect transitions before outbreaks:
- Variance spike ratio (current vs baseline)
- ACF change (loss of autocorrelation)
- Trend acceleration

Reference: 03_tdd.md Section 3.3.1 (Early-Warning Indicators)
These features are the RESEARCH CONTRIBUTION (unique to this project).
"""
import pandas as pd
import numpy as np
from typing import Optional


def compute_variance_spike_ratio(
    df: pd.DataFrame,
    current_window: int = 4,
    baseline_window: int = 52,
    value_col: str = 'incidence_per_100k',
    group_cols: list = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute variance spike ratio (current vs baseline variance).
    
    Formula: var_4w / mean(var_52w_prior)
    
    A high ratio indicates recent volatility is elevated compared to
    typical historical variance — potential early warning signal.
    
    Args:
        df: Panel DataFrame
        current_window: Window for current variance
        baseline_window: Window for baseline variance
        value_col: Column to compute variance on
        group_cols: Grouping columns
        
    Returns:
        DataFrame with variance spike ratio column
    """
    df = df.copy()
    
    # Current variance (short window)
    df['_var_current'] = df.groupby(group_cols)[value_col].transform(
        lambda x: x.rolling(window=current_window, min_periods=2).var()
    )
    
    # Baseline variance (causal expanding window).
    # IMPORTANT: fixed 52-week baselines are too sparse for many districts;
    # use expanding variance with a minimum history requirement.
    # Also shift(1) to avoid using current-week value in the baseline.
    baseline_min_periods = 12
    df['_var_baseline'] = df.groupby(group_cols)[value_col].transform(
        lambda x: x.shift(1).expanding(min_periods=baseline_min_periods).var()
    )

    # Spike ratio
    epsilon = 0.01  # Prevent division by zero
    df['feat_var_spike_ratio'] = df['_var_current'] / (df['_var_baseline'] + epsilon)

    # Cleanup
    df = df.drop(columns=['_var_current', '_var_baseline'])
    
    return df


def compute_acf_change(
    df: pd.DataFrame,
    lag: int = 1,
    window: int = 4,
    value_col: str = 'incidence_per_100k',
    group_cols: list = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute change in autocorrelation (ACF shift).
    
    Formula: acf_lag1_4w - acf_lag1_4w_prior
    
    Sudden loss of autocorrelation can indicate system entering
    unstable state before transition.
    
    Args:
        df: Panel DataFrame
        lag: Autocorrelation lag
        window: Rolling window for ACF
        value_col: Column for ACF
        group_cols: Grouping columns
        
    Returns:
        DataFrame with ACF change column
    """
    df = df.copy()
    
    # Check if ACF was already computed
    acf_col = f'feat_cases_acf_lag{lag}_{window}w'
    
    if acf_col not in df.columns:
        # Compute ACF if not present
        def rolling_acf(series, lag=1, window=4):
            result = []
            for i in range(len(series)):
                if i < window:
                    result.append(np.nan)
                else:
                    window_data = series.iloc[i-window:i]
                    if len(window_data) >= window and window_data.std() > 0:
                        acf = window_data.autocorr(lag=lag)
                        result.append(acf)
                    else:
                        result.append(np.nan)
            return pd.Series(result, index=series.index)
        
        df[acf_col] = df.groupby(group_cols)[value_col].transform(
            lambda x: rolling_acf(x, lag=lag, window=window)
        )
    
    # Compute change (current - previous week's ACF)
    df['feat_acf_change'] = df.groupby(group_cols)[acf_col].diff()
    
    return df


def compute_trend_acceleration(
    df: pd.DataFrame,
    window: int = 4,
    value_col: str = 'incidence_per_100k',
    group_cols: list = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute trend acceleration (rate of change of trend).
    
    Formula: d/dt of 4-week slope
    
    Accelerating trend (positive second derivative) indicates
    exponential-like growth approaching outbreak.
    
    Args:
        df: Panel DataFrame
        window: Window for trend computation
        value_col: Column for trend
        group_cols: Grouping columns
        
    Returns:
        DataFrame with trend acceleration column
    """
    df = df.copy()
    
    # Check if trend was already computed
    trend_col = f'feat_cases_trend_{window}w'
    
    if trend_col not in df.columns:
        from scipy import stats
        
        def rolling_slope(series, window=4):
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
        
        df[trend_col] = df.groupby(group_cols)[value_col].transform(
            lambda x: rolling_slope(x, window=window)
        )
    
    # Acceleration = change in trend
    df['feat_trend_accel'] = df.groupby(group_cols)[trend_col].diff()
    
    return df


def compute_recent_normalized_incidence(
    df: pd.DataFrame,
    window: int = 2,
    value_col: str = 'incidence_per_100k',
    group_cols: list = ['state', 'district']
) -> pd.DataFrame:
    """
    Compute recent average incidence normalized by historical baseline.
    
    Formula: c_ma_2w / mean(c_historical)
    
    Values > 1 indicate current incidence is above historical average.
    
    Args:
        df: Panel DataFrame
        window: Short window for recent average
        value_col: Column for incidence
        group_cols: Grouping columns
        
    Returns:
        DataFrame with normalized incidence column
    """
    df = df.copy()
    
    # Recent average
    df['_recent_avg'] = df.groupby(group_cols)[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    # Historical mean (expanding window, shifted)
    df['_historical_mean'] = df.groupby(group_cols)[value_col].transform(
        lambda x: x.shift(1).expanding(min_periods=12).mean()
    )
    
    # Normalized
    epsilon = 0.01
    df['feat_recent_normalized'] = df['_recent_avg'] / (df['_historical_mean'] + epsilon)
    
    # Cleanup
    df = df.drop(columns=['_recent_avg', '_historical_mean'])
    
    return df


def compute_all_ews_features(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute all EWS (Early Warning System) features.
    
    These features are UNIQUE/NOVEL contributions of this project.
    
    Args:
        df: Panel DataFrame with case features computed
        config: Optional config dict
        
    Returns:
        DataFrame with all EWS features added
    """
    print("Computing EWS (early warning) features...")
    
    # Ensure sorted
    df = df.sort_values(['state', 'district', 'year', 'week']).reset_index(drop=True)
    
    df = compute_variance_spike_ratio(df, current_window=4, baseline_window=52)
    print("  ✓ Variance spike ratio")
    
    df = compute_acf_change(df, lag=1, window=4)
    print("  ✓ ACF change")
    
    df = compute_trend_acceleration(df, window=4)
    print("  ✓ Trend acceleration")
    
    df = compute_recent_normalized_incidence(df, window=2)
    print("  ✓ Recent normalized incidence")
    
    return df
