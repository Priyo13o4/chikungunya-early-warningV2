"""
Temporal Cross-Validation for Chikungunya EWS

Implements rolling-origin (expanding window) CV to prevent data leakage.
Train on past, test on future — mimics real deployment.

Reference: 05_experiments.md Section 5.2
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass


@dataclass
class CVFold:
    """Represents a single CV fold."""
    fold_name: str
    train_years: List[int]
    test_year: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def create_rolling_origin_splits(
    df: pd.DataFrame,
    test_years: List[int] = [2017, 2018, 2019, 2020, 2021, 2022],
    year_col: str = 'year'
) -> List[CVFold]:
    """
    Create rolling-origin CV splits.
    
    For each test year Y:
    - Training: all data from years < Y
    - Test: all data from year == Y
    
    Args:
        df: DataFrame with year column
        test_years: Years to use as test sets
        year_col: Column name for year
        
    Returns:
        List of CVFold objects
    """
    folds = []
    min_year = df[year_col].min()
    
    for test_year in test_years:
        # Training: all years before test year
        train_mask = df[year_col] < test_year
        test_mask = df[year_col] == test_year
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"  WARNING: Skipping test year {test_year}: no data available")
            continue
        
        train_years = list(range(min_year, test_year))
        
        fold = CVFold(
            fold_name=f"fold_{test_year}",
            train_years=train_years,
            test_year=test_year,
            train_idx=train_idx,
            test_idx=test_idx
        )
        folds.append(fold)
    
    if len(folds) == 0:
        raise ValueError(f"No valid CV folds created from test_years {test_years}. Check data availability.")
    
    return folds


def cv_split_generator(
    df: pd.DataFrame,
    test_years: List[int] = [2017, 2018, 2019, 2020, 2021, 2022]
) -> Generator[Tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generator that yields (fold_name, train_df, test_df) tuples.
    
    Args:
        df: Full DataFrame
        test_years: Years to use as test sets
        
    Yields:
        Tuples of (fold_name, train_df, test_df)
    """
    folds = create_rolling_origin_splits(df, test_years)
    
    for fold in folds:
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        yield fold.fold_name, train_df, test_df


def prepare_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak',
    drop_na: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare X, y arrays for training and testing.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        drop_na: If True, drop rows with NaN in features or target
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    if drop_na:
        raise ValueError("Feature-level drop_na is disabled in v6. Use target-only filtering.")

    # Filter to required columns
    required_cols = feature_cols + [target_col]

    train_subset = train_df[required_cols].copy()
    test_subset = test_df[required_cols].copy()

    # Target-only filtering (features may remain NaN).
    train_subset = train_subset.dropna(subset=[target_col])
    test_subset = test_subset.dropna(subset=[target_col])
    
    # Use numpy conversion with dtype=float to avoid object arrays caused by
    # pandas nullable dtypes (e.g., Int64 with pd.NA). This preserves missing
    # values as np.nan so downstream models can handle them consistently.
    X_train = train_subset[feature_cols].to_numpy(dtype=float)
    y_train = train_subset[target_col].to_numpy(dtype=float)
    X_test = test_subset[feature_cols].to_numpy(dtype=float)
    y_test = test_subset[target_col].to_numpy(dtype=float)
    
    return X_train, y_train, X_test, y_test


def get_valid_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'label_outbreak'
) -> pd.DataFrame:
    """
    Get DataFrame with only rows that have valid features and target.
    
    Args:
        df: Input DataFrame
        feature_cols: Feature columns to check
        target_col: Target column
        
    Returns:
        DataFrame with valid rows only
    """
    # Only require target presence; feature NaNs are permitted.
    return df.dropna(subset=[target_col])
