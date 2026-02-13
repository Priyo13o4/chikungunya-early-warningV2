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
    test_year: int  # For single-year folds; primary test year for multi-year folds
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_years: Optional[List[int]] = None  # For multi-year test windows


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


def create_stratified_temporal_folds(
    df: pd.DataFrame,
    target_col: str = 'label_outbreak',
    year_col: str = 'year',
    min_positives: int = 5,
    candidate_test_years: Optional[List[int]] = None,
    verbose: bool = True
) -> List[CVFold]:
    """
    Create stratified temporal CV folds ensuring min_positives per fold.
    
    Strategy:
    - Count positives per year
    - Group consecutive years with <min_positives into multi-year test windows
    - Respect temporal ordering (no future data in training)
    - Skip years/groups with insufficient positives after grouping
    
    Example: For 2017-2022 data with sparse positives:
    - Fold 1 test: 2017 (8 positives) → single year
    - Fold 2 test: 2018 (7 positives) → single year
    - Fold 3 test: 2019-2020 (4+2=6 positives) → merged years
    - Skip 2021-2022 (0+0=0 positives) → insufficient
    
    Args:
        df: DataFrame with year and target columns
        target_col: Column name for binary target (outbreak labels)
        year_col: Column name for year
        min_positives: Minimum positive samples required per fold
        candidate_test_years: Years to consider for test sets (default: auto-detect from data)
        verbose: If True, print fold statistics
        
    Returns:
        List of CVFold objects with ≥min_positives per fold
    """
    # Count positives per year
    if candidate_test_years is None:
        all_years = sorted(df[year_col].dropna().unique())
        # Use years with actual data for testing (skip early sparse years)
        year_counts = df.groupby(year_col).size()
        candidate_test_years = [y for y in all_years if year_counts.get(y, 0) > 0]
    
    if verbose:
        print("\n=== STRATIFIED TEMPORAL CV FOLD CREATION ===")
        print(f"Target: {target_col}, min_positives: {min_positives}")
    
    # Count positives and total samples per year
    yearly_stats = df.groupby(year_col)[target_col].agg(['sum', 'count']).to_dict('index')
    
    if verbose:
        print("\nPositives per year:")
        for year in sorted(candidate_test_years):
            pos = yearly_stats.get(year, {}).get('sum', 0)
            total = yearly_stats.get(year, {}).get('count', 0)
            print(f"  {year}: {int(pos):3d} positives / {total:4d} total")
    
    # Create folds by grouping years to meet min_positives threshold
    folds = []
    min_year = df[year_col].min()
    
    i = 0
    while i < len(candidate_test_years):
        # Start a new test window with current year
        test_years_group = [candidate_test_years[i]]
        positives_count = yearly_stats.get(candidate_test_years[i], {}).get('sum', 0)
        total_count = yearly_stats.get(candidate_test_years[i], {}).get('count', 0)
        
        # Try to merge with next consecutive years if needed
        j = i + 1
        while positives_count < min_positives and j < len(candidate_test_years):
            next_year = candidate_test_years[j]
            # Only merge consecutive years
            if next_year == test_years_group[-1] + 1:
                test_years_group.append(next_year)
                positives_count += yearly_stats.get(next_year, {}).get('sum', 0)
                total_count += yearly_stats.get(next_year, {}).get('count', 0)
                j += 1
            else:
                break
        
        # Check if we have enough positives
        if positives_count < min_positives:
            if verbose:
                years_str = "-".join(map(str, test_years_group))
                print(f"\n  WARNING: Skipping test years {years_str}: "
                      f"only {int(positives_count)} positives (need {min_positives})")
            i = j  # Skip to next ungrouped year
            continue
        
        # Create fold with this test window
        test_year_min = min(test_years_group)
        test_year_max = max(test_years_group)
        
        # Training: all data from years < minimum test year
        train_mask = df[year_col] < test_year_min
        
        # Test: all data from years in test window
        test_mask = df[year_col].isin(test_years_group)
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        if len(train_idx) == 0:
            if verbose:
                years_str = "-".join(map(str, test_years_group))
                print(f"\n  WARNING: Skipping test years {years_str}: no training data available")
            i = j
            continue
        
        # Build fold name
        if len(test_years_group) == 1:
            fold_name = f"fold_{test_years_group[0]}"
        else:
            fold_name = f"fold_{test_year_min}_{test_year_max}"
        
        train_years = list(range(int(min_year), int(test_year_min)))
        
        fold = CVFold(
            fold_name=fold_name,
            train_years=train_years,
            test_year=test_year_max,  # Use max year as primary test year
            train_idx=train_idx,
            test_idx=test_idx,
            test_years=test_years_group
        )
        folds.append(fold)
        
        # Move to next ungrouped year
        i = j
    
    if len(folds) == 0:
        raise ValueError(
            f"No valid CV folds created with min_positives={min_positives}. "
            f"Insufficient positive samples in test years {candidate_test_years}."
        )
    
    if verbose:
        print(f"\n=== CREATED {len(folds)} STRATIFIED FOLDS ===")
        print(f"{'Fold':<15} {'Test Years':<15} {'Train Samples':<15} {'Test Samples':<15} {'Positives':<12} {'Negatives':<12}")
        print("-" * 94)
        
        for fold in folds:
            test_years_str = "-".join(map(str, fold.test_years))
            train_samples = len(fold.train_idx)
            test_samples = len(fold.test_idx)
            
            # Count positives/negatives in test set
            test_df = df.iloc[fold.test_idx]
            positives = int(test_df[target_col].sum())
            negatives = test_samples - positives
            
            print(f"{fold.fold_name:<15} {test_years_str:<15} {train_samples:<15} "
                  f"{test_samples:<15} {positives:<12} {negatives:<12}")
    
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
