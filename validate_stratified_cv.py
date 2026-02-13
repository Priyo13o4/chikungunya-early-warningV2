#!/usr/bin/env python3
"""
Comprehensive validation of stratified temporal CV implementation.
Generates fold statistics report for audit.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.cv import create_stratified_temporal_folds

def main():
    print("=" * 94)
    print("STRATIFIED TEMPORAL CV VALIDATION REPORT")
    print("=" * 94)
    
    # Load config and data
    with open('config/config_default.yaml') as f:
        cfg = yaml.safe_load(f)
    
    features_path = cfg['data']['processed']['features']
    df = pd.read_parquet(features_path)
    
    print(f"\nDataset: {features_path}")
    print(f"Total samples: {len(df)}")
    print(f"Years: {sorted(df['year'].unique())}")
    
    # Apply the same filtering as experiment 05 (Bayesian model)
    from src.data.loader import filter_districts_by_min_obs
    df_filtered = filter_districts_by_min_obs(df, min_obs=10)
    
    # Prepare valid data (same as Phase 05)
    required_cols = ['state', 'district', 'year', 'week', 'cases', 'label_outbreak']
    valid_df = df_filtered.dropna(subset=required_cols).copy()
    
    if 'temp_celsius' in valid_df.columns:
        valid_df = valid_df.dropna(subset=['temp_celsius'])
    
    print(f"Valid samples after filtering: {len(valid_df)}")
    
    # Create stratified folds
    test_years = cfg['cv']['test_years']
    print(f"\nCandidate test years from config: {test_years}")
    
    folds = create_stratified_temporal_folds(
        df=valid_df,
        target_col='label_outbreak',
        year_col='year',
        min_positives=5,
        candidate_test_years=test_years,
        verbose=True
    )
    
    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================
    print("\n" + "=" * 94)
    print("VALIDATION CHECKS")
    print("=" * 94)
    
    # Check 1: All folds have ≥5 positives
    print("\n1. Minimum positives check (≥5 required):")
    all_valid = True
    for fold in folds:
        test_df = valid_df.iloc[fold.test_idx]
        positives = int(test_df['label_outbreak'].sum())
        status = "✓" if positives >= 5 else "✗"
        print(f"   {status} {fold.fold_name}: {positives} positives")
        if positives < 5:
            all_valid = False
    
    if all_valid:
        print("   RESULT: ✓ ALL FOLDS HAVE ≥5 POSITIVES")
    else:
        print("   RESULT: ✗ SOME FOLDS HAVE <5 POSITIVES")
    
    # Check 2: Temporal ordering
    print("\n2. Temporal ordering check (no future data in training):")
    ordering_valid = True
    for fold in folds:
        max_train_year = max(fold.train_years) if fold.train_years else 0
        min_test_year = min(fold.test_years)
        
        if max_train_year >= min_test_year:
            print(f"   ✗ {fold.fold_name}: train up to {max_train_year}, test starts {min_test_year}")
            ordering_valid = False
        else:
            print(f"   ✓ {fold.fold_name}: train up to {max_train_year}, test starts {min_test_year}")
    
    if ordering_valid:
        print("   RESULT: ✓ TEMPORAL ORDERING PRESERVED")
    else:
        print("   RESULT: ✗ TEMPORAL ORDERING VIOLATED")
    
    # Check 3: No data leakage (train and test indices don't overlap)
    print("\n3. Data leakage check (train/test disjoint):")
    leakage_found = False
    for fold in folds:
        train_set = set(fold.train_idx)
        test_set = set(fold.test_idx)
        overlap = train_set & test_set
        
        if len(overlap) > 0:
            print(f"   ✗ {fold.fold_name}: {len(overlap)} samples in both train and test")
            leakage_found = True
        else:
            print(f"   ✓ {fold.fold_name}: no overlap")
    
    if not leakage_found:
        print("   RESULT: ✓ NO DATA LEAKAGE")
    else:
        print("   RESULT: ✗ DATA LEAKAGE DETECTED")
    
    # Check 4: Coverage (all test years covered or explicitly skipped)
    print("\n4. Coverage check:")
    covered_years = set()
    for fold in folds:
        covered_years.update(fold.test_years)
    
    skipped_years = set(test_years) - covered_years
    print(f"   Covered years: {sorted(covered_years)}")
    print(f"   Skipped years: {sorted(skipped_years) if skipped_years else 'None'}")
    
    if skipped_years:
        # Check if skipped years had insufficient positives
        print(f"   Skipped years analysis:")
        for year in sorted(skipped_years):
            year_df = valid_df[valid_df['year'] == year]
            pos = int(year_df['label_outbreak'].sum())
            print(f"     {year}: {pos} positives (insufficient)")
    
    # =========================================================================
    # FOLD STATISTICS TABLE (for thesis/report)
    # =========================================================================
    print("\n" + "=" * 94)
    print("FOLD STATISTICS TABLE (ready for thesis)")
    print("=" * 94)
    print()
    
    # Create DataFrame for easy formatting
    stats_data = []
    for fold in folds:
        test_df = valid_df.iloc[fold.test_idx]
        train_df = valid_df.iloc[fold.train_idx]
        
        test_years_str = "-".join(map(str, fold.test_years))
        train_samples = len(fold.train_idx)
        test_samples = len(fold.test_idx)
        positives = int(test_df['label_outbreak'].sum())
        negatives = test_samples - positives
        
        # Additional metrics
        pos_rate = positives / test_samples if test_samples > 0 else 0
        
        stats_data.append({
            'Fold': fold.fold_name,
            'Test Years': test_years_str,
            'Train Samples': train_samples,
            'Test Samples': test_samples,
            'Positives': positives,
            'Negatives': negatives,
            'Positive Rate': f"{pos_rate:.1%}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "-" * 94)
    print("SUMMARY:")
    total_positives = sum(row['Positives'] for row in stats_data)
    total_test_samples = sum(row['Test Samples'] for row in stats_data)
    print(f"  Total folds: {len(folds)}")
    print(f"  Total test samples: {total_test_samples}")
    print(f"  Total positives: {total_positives}")
    print(f"  Overall positive rate: {total_positives/total_test_samples:.1%}")
    print(f"  Min positives per fold: {min(row['Positives'] for row in stats_data)}")
    print(f"  Max positives per fold: {max(row['Positives'] for row in stats_data)}")
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "=" * 94)
    print("FINAL VERDICT")
    print("=" * 94)
    
    all_checks_passed = all_valid and ordering_valid and not leakage_found
    
    if all_checks_passed:
        print("\n✓ ALL VALIDATION CHECKS PASSED")
        print("✓ Implementation is ready for Phase 05 evaluation")
        print(f"✓ Created {len(folds)} stratified folds with ≥5 positives each")
        print("✓ Temporal ordering preserved (no future data leakage)")
        return 0
    else:
        print("\n✗ SOME VALIDATION CHECKS FAILED")
        print("✗ Review issues above before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())
