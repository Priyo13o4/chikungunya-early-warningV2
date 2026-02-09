#!/usr/bin/env python3
"""
Experiment 00: Sanity Check

Quick verification that the project is set up correctly:
1. Config loads
2. Data files exist
3. Basic imports work
4. Environment is correct

Usage:
    python experiments/00_sanity_check.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_config():
    """Test config loading."""
    print("Checking config...", end=" ")
    try:
        from src.config import load_config, get_project_root
        cfg = load_config()
        assert 'data' in cfg
        assert 'models' in cfg
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def check_data_files():
    """Test data file existence."""
    print("Checking data files...", end=" ")
    try:
        from src.config import load_config, get_repo_root
        cfg = load_config()
        root = get_repo_root()
        
        epiclim = root / cfg['data']['raw']['epiclim']
        census = root / cfg['data']['raw']['census']
        
        assert epiclim.exists(), f"EpiClim not found: {epiclim}"
        assert census.exists(), f"Census not found: {census}"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def check_imports():
    """Test key imports."""
    print("Checking imports...", end=" ")
    try:
        import pandas as pd
        import numpy as np
        import yaml
        from rapidfuzz import fuzz
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ (Missing: {e})")
        return False


def check_data_loader():
    """Test data loader functions."""
    print("Checking data loader...", end=" ")
    try:
        from src.data.loader import load_epiclim, load_census
        from src.config import load_config, get_repo_root
        
        cfg = load_config()
        root = get_repo_root()
        
        # Quick load test
        epiclim = load_epiclim(
            root / cfg['data']['raw']['epiclim'],
            disease_filter="Chikungunya"
        )
        assert len(epiclim) > 0, "No data loaded"
        assert 'cases' in epiclim.columns
        assert 'temp_celsius' in epiclim.columns
        
        census = load_census(root / cfg['data']['raw']['census'])
        assert len(census) > 0
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def main():
    print("=" * 60)
    print("CHIKUNGUNYA EWS - SANITY CHECK")
    print("=" * 60)
    
    checks = [
        ("Config", check_config),
        ("Data Files", check_data_files),
        ("Imports", check_imports),
        ("Data Loader", check_data_loader),
    ]
    
    results = []
    for name, check_fn in checks:
        results.append(check_fn())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"ALL CHECKS PASSED ({passed}/{total}) ✓")
        print("Ready to proceed with experiments!")
    else:
        print(f"CHECKS FAILED ({passed}/{total}) ✗")
        print("Please fix issues before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
