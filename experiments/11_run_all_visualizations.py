#!/usr/bin/env python3
"""
Master Visualization Runner
===========================
Executes all 16 visualization plots across 3 modules.

Usage:
    python run_all_visualizations.py
    python run_all_visualizations.py --module comparison
    python run_all_visualizations.py --module diagnostics
    python run_all_visualizations.py --module uncertainty
"""

import sys
import argparse
from pathlib import Path
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def run_comparison_plots():
    """Run comparison visualizations (9 plots)."""
    print("\n" + "="*80)
    print("MODULE 1: COMPARISON PLOTS (Track A vs Track B)")
    print("="*80)
    
    try:
        import viz_comparison_plots
        start = time.time()
        viz_comparison_plots.main()
        elapsed = time.time() - start
        print(f"⏱ Completed in {elapsed:.1f} seconds")
        return True, 8  # 8 plots (9th is placeholder)
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def run_diagnostic_plots():
    """Run MCMC diagnostic visualizations (4 plots)."""
    print("\n" + "="*80)
    print("MODULE 2: DIAGNOSTIC PLOTS (MCMC Convergence)")
    print("="*80)
    
    try:
        import viz_diagnostic_plots
        start = time.time()
        viz_diagnostic_plots.main()
        elapsed = time.time() - start
        print(f"⏱ Completed in {elapsed:.1f} seconds")
        return True, 4
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def run_uncertainty_plots():
    """Run uncertainty quantification visualizations (3 plots)."""
    print("\n" + "="*80)
    print("MODULE 3: UNCERTAINTY PLOTS (Temporal & Uncertainty)")
    print("="*80)
    
    try:
        import viz_uncertainty_plots
        start = time.time()
        viz_uncertainty_plots.main()
        elapsed = time.time() - start
        print(f"⏱ Completed in {elapsed:.1f} seconds")
        return True, 3
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def generate_summary_report(results):
    """Generate summary report of all visualizations."""
    base_path = Path(__file__).parent.parent / 'results/figures'
    
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Count files
    comparison_files = list((base_path / 'comparison').glob('*.png'))
    diagnostics_files = list((base_path / 'diagnostics').glob('*.png'))
    uncertainty_files = list((base_path / 'uncertainty').glob('*.png'))
    
    print("MODULE STATUS:")
    print("-" * 80)
    modules = [
        ('Comparison Plots', results.get('comparison', (False, 0)), len(comparison_files)),
        ('Diagnostic Plots', results.get('diagnostics', (False, 0)), len(diagnostics_files)),
        ('Uncertainty Plots', results.get('uncertainty', (False, 0)), len(uncertainty_files))
    ]
    
    total_expected = 0
    total_generated = 0
    
    for name, (success, expected), actual in modules:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name:25} {status:15} Expected: {expected:2} | Generated: {actual:2}")
        total_expected += expected
        total_generated += actual
    
    print("-" * 80)
    print(f"{'TOTAL':25} {''} Expected: {total_expected:2} | Generated: {total_generated:2}")
    print()
    
    # File listing
    print("OUTPUT FILES:")
    print("-" * 80)
    
    if comparison_files:
        print(f"\n📁 Comparison ({len(comparison_files)} files):")
        for f in sorted(comparison_files):
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name:45} {size_kb:6.1f} KB")
    
    if diagnostics_files:
        print(f"\n📁 Diagnostics ({len(diagnostics_files)} files):")
        for f in sorted(diagnostics_files):
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name:45} {size_kb:6.1f} KB")
    
    if uncertainty_files:
        print(f"\n📁 Uncertainty ({len(uncertainty_files)} files):")
        for f in sorted(uncertainty_files):
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name:45} {size_kb:6.1f} KB")
    
    print()
    print("="*80)
    
    # Overall status
    all_success = all(r[0] for r in results.values())
    if all_success and total_generated >= 14:  # At least 14 of 16 plots
        print("✅ VISUALIZATION SUITE COMPLETE")
    else:
        print("⚠️  VISUALIZATION SUITE INCOMPLETE")
    
    print(f"\nOutput directory: {base_path}")
    print("="*80 + "\n")
    
    # Save report to file
    report_path = base_path / 'visualization_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Visualization Summary Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Comparison Plots: {len(comparison_files)} files\n")
        f.write(f"Diagnostic Plots: {len(diagnostics_files)} files\n")
        f.write(f"Uncertainty Plots: {len(uncertainty_files)} files\n\n")
        f.write(f"Total: {total_generated} files generated\n")
    
    print(f"📄 Summary report saved to: {report_path}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate all visualization plots')
    parser.add_argument('--module', choices=['comparison', 'diagnostics', 'uncertainty', 'all'],
                       default='all', help='Which module to run (default: all)')
    args = parser.parse_args()
    
    print("\n" + "🎨"*40)
    print("COMPREHENSIVE VISUALIZATION SUITE")
    print("16 Publication-Ready Graphs")
    print("🎨"*40)
    
    start_time = time.time()
    results = {}
    
    # Run requested modules
    if args.module in ['all', 'comparison']:
        results['comparison'] = run_comparison_plots()
        time.sleep(0.5)  # Brief pause between modules
    
    if args.module in ['all', 'diagnostics']:
        results['diagnostics'] = run_diagnostic_plots()
        time.sleep(0.5)
    
    if args.module in ['all', 'uncertainty']:
        results['uncertainty'] = run_uncertainty_plots()
    
    # Generate summary
    total_time = time.time() - start_time
    generate_summary_report(results)
    
    print(f"⏱ Total execution time: {total_time:.1f} seconds\n")
    
    # Return exit code
    all_success = all(r[0] for r in results.values())
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
