#!/usr/bin/env python3
"""Test stratified temporal CV implementation."""
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.cv import create_stratified_temporal_folds

def main():
    # Load config and data
    with open('config/config_default.yaml') as f:
        cfg = yaml.safe_load(f)
    
    features_path = cfg['data']['processed']['features']
    df = pd.read_parquet(features_path)
    
    print('Testing stratified temporal CV...')
    print('='*94)
    
    # Test with default settings (min_positives=5)
    folds = create_stratified_temporal_folds(
        df=df,
        target_col='label_outbreak',
        year_col='year',
        min_positives=5,
        candidate_test_years=[2017, 2018, 2019, 2020, 2021, 2022],
        verbose=True
    )
    
    print('\n' + '='*94)
    print('VALIDATION COMPLETE')
    print(f'Created {len(folds)} folds')
    print('\nAll folds have ≥5 positives: ', end='')
    
    all_valid = True
    for fold in folds:
        test_df = df.iloc[fold.test_idx]
        positives = int(test_df['label_outbreak'].sum())
        if positives < 5:
            all_valid = False
            print(f'✗ ({fold.fold_name} has {positives} positives)')
            break
    
    if all_valid:
        print('✓ YES')
    
    # Verify temporal ordering
    print('\nTemporal ordering check: ', end='')
    ordering_valid = True
    for fold in folds:
        max_train_year = max(fold.train_years) if fold.train_years else 0
        min_test_year = min(fold.test_years)
        if max_train_year >= min_test_year:
            ordering_valid = False
            print(f'✗ ({fold.fold_name}: train up to {max_train_year}, test starts {min_test_year})')
            break
    
    if ordering_valid:
        print('✓ YES (no future data in training)')
    
    print('\n' + '='*94)
    return 0

if __name__ == "__main__":
    sys.exit(main())
