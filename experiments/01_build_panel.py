#!/usr/bin/env python3
"""
Experiment 01: Build Panel Dataset

This script builds the canonical panel dataset from raw sources:
- EpiClim (disease + climate data)
- Census 2011 (population data)

Output: data/processed/panel_chikungunya_v01.parquet

Usage:
    python experiments/01_build_panel.py
    python experiments/01_build_panel.py --config config/config_default.yaml
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, get_project_root, get_repo_root
from src.data.loader import build_panel


def main():
    parser = argparse.ArgumentParser(description="Build panel dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config_default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    # Load config
    root = get_project_root()
    repo_root = get_repo_root()
    config_path = root / args.config
    cfg = load_config(str(config_path))
    
    print("=" * 60)
    print("CHIKUNGUNYA EWS - BUILD PANEL DATASET")
    print("=" * 60)
    
    # v6 is nested; raw inputs and canonical processed outputs live at the repo root.
    epiclim_path = repo_root / cfg['data']['raw']['epiclim']
    census_path = repo_root / cfg['data']['raw']['census']
    output_path = repo_root / cfg['data']['processed']['panel']

    score_threshold = cfg.get('processing', {}).get('score_threshold')
    if score_threshold is None:
        raise ValueError("Missing processing.score_threshold in config.")

    cases_imputation = cfg.get('processing', {}).get('imputation', {}).get('cases')
    if cases_imputation is None:
        raise ValueError("Missing processing.imputation.cases in config.")

    # Build panel
    panel = build_panel(
        epiclim_path=epiclim_path,
        census_path=census_path,
        disease=cfg['processing']['disease_filter'],
        cases_imputation_strategy=cases_imputation,
        output_path=output_path,
        score_threshold=int(score_threshold),
    )
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("PANEL SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(panel)}")
    print(f"Unique states: {panel['state'].nunique()}")
    print(f"Unique districts: {panel['district'].nunique()}")
    print(f"Year range: {panel['year'].min()} - {panel['year'].max()}")
    print(f"Week range: {panel['week'].min()} - {panel['week'].max()}")
    print(f"\nColumns: {list(panel.columns)}")
    
    # Missing data report
    print("\nMissing data:")
    for col in panel.columns:
        missing = panel[col].isna().sum()
        if missing > 0:
            pct = 100 * missing / len(panel)
            print(f"  {col}: {missing} ({pct:.1f}%)")
    
    # Population matching report
    pop_matched = panel['population'].notna().sum()
    pop_total = len(panel)
    print(f"\nPopulation matched: {pop_matched}/{pop_total} ({100*pop_matched/pop_total:.1f}%)")
    
    print("\n✓ Panel build complete!")
    return panel


if __name__ == "__main__":
    main()
