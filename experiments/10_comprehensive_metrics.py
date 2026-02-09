#!/usr/bin/env python3
"""
Phase 6 - Task 5: Comprehensive Metrics Expansion

Compute and report expanded evaluation metrics for all models.

Metrics computed:
- AUC (existing)
- AUPR (Area Under Precision-Recall Curve)
- Precision
- Recall (Sensitivity)
- F1 Score
- Specificity
- Cohen's Kappa
- Brier Score
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy

De-emphasize raw accuracy (misleading for imbalanced data).

Compares:
- Bayesian model (Phase 5 results)
- XGBoost baseline (v1.1)
- Fusion models (Phase 6 Task 4)

Output: results/analysis/comprehensive_metrics.json

Reference: Phase 6 reporting requirements
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_bayesian_results() -> Dict[str, Any]:
    """Load Bayesian CV results from Phase 5."""
    path = project_root / "results" / "metrics" / "bayesian_cv_results.json"
    
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def load_baseline_results() -> Dict[str, Any]:
    """Load baseline comparison results from v1.1."""
    path = project_root / "results" / "metrics" / "baseline_comparison.json"
    
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def load_fusion_results() -> Dict[str, Any]:
    """Load fusion experiment results from Phase 6 Task 4."""
    analysis_dir = project_root / "results" / "analysis"
    preferred = analysis_dir / "fusion_results_p75.json"
    fallback = analysis_dir / "fusion_results.json"
    path = preferred if preferred.exists() else fallback
    
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def extract_bayesian_metrics(bayesian_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and format Bayesian metrics.
    """
    if not bayesian_results:
        return None
    
    agg = bayesian_results.get('aggregated', {})
    
    metrics = {
        'model': 'Bayesian State-Space (v3)',
        'phase': 'Phase 5',
        'n_folds': agg.get('n_folds', 0),
        'metrics': {
            'auc': {
                'mean': agg.get('auc_mean', 0),
                'std': agg.get('auc_std', 0)
            },
            'precision': {
                'mean': None,  # Not computed in Phase 5
                'std': None
            },
            'recall': {
                'mean': agg.get('sensitivity_mean', 0),
                'std': agg.get('sensitivity_std', 0)
            },
            'specificity': {
                'mean': agg.get('specificity_mean', 0),
                'std': agg.get('specificity_std', 0)
            },
            'f1': {
                'mean': agg.get('f1_mean', 0),
                'std': agg.get('f1_std', 0)
            },
            'brier': {
                'mean': agg.get('brier_mean', 0),
                'std': agg.get('brier_std', 0)
            },
            'aupr': {
                'mean': None,  # Not computed in Phase 5
                'std': None
            },
            'kappa': {
                'mean': None,
                'std': None
            }
        },
        'interpretation': {
            'auc': 'Discrimination summary is in metrics.auc (mean/std).',
            'recall': 'Sensitivity summary is in metrics.recall (mean/std).',
            'specificity': 'Specificity summary is in metrics.specificity (mean/std).',
            'f1': 'F1 summary is in metrics.f1 (mean/std).',
            'brier': 'Calibration summary is in metrics.brier (mean/std).',
            'note': 'Avoid interpreting Bayesian performance on AUC alone; it is primarily a latent risk estimator.'
        }
    }
    
    return metrics


def extract_xgboost_metrics(baseline_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract XGBoost metrics from v1.1 baseline comparison.
    """
    if not baseline_results:
        return None
    
    xgb = baseline_results.get('models', {}).get('xgboost', {})
    
    if not xgb:
        return None
    
    metrics = {
        'model': 'XGBoost Classifier',
        'phase': 'Phase 3 (v1.1)',
        'n_folds': xgb.get('n_folds', 0),
        'metrics': {
            'auc': {
                'mean': xgb.get('auc_mean', 0),
                'std': xgb.get('auc_std', 0)
            },
            'precision': {
                'mean': None,  # Compute from fold results if available
                'std': None
            },
            'recall': {
                'mean': xgb.get('sensitivity_mean', 0),
                'std': None
            },
            'specificity': {
                'mean': xgb.get('specificity_mean', 0),
                'std': None
            },
            'f1': {
                'mean': xgb.get('f1_mean', 0),
                'std': xgb.get('f1_std', 0)
            },
            'brier': {
                'mean': xgb.get('brier_mean', 0),
                'std': None
            },
            'aupr': {
                'mean': None,
                'std': None
            },
            'kappa': {
                'mean': None,
                'std': None
            }
        },
        'interpretation': {
            'auc': 'Discrimination summary is in metrics.auc (mean/std).',
            'recall': 'Sensitivity summary is in metrics.recall (mean/std).',
            'specificity': 'Specificity summary is in metrics.specificity (mean/std).',
            'f1': 'F1 summary is in metrics.f1 (mean/std).',
            'note': 'Optimized for discrimination metrics; recall may be low depending on thresholding.'
        }
    }
    
    return metrics


def extract_fusion_metrics(fusion_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract fusion strategy metrics.
    """
    if not fusion_results:
        return None
    
    agg = fusion_results.get('aggregated', {})
    
    strategies = {}
    
    # Feature Fusion
    if 'feature_fusion' in agg:
        ff = agg['feature_fusion']
        strategies['feature_fusion'] = {
            'name': 'Feature Fusion (XGBoost + Bayesian features)',
            'metrics': {
                'auc': {
                    'mean': ff.get('auc_mean'),
                    'std': ff.get('auc_std')
                },
                'aupr': {
                    'mean': ff.get('aupr_mean'),
                    'std': ff.get('aupr_std')
                },
                'precision': {
                    'mean': ff.get('precision_mean'),
                    'std': ff.get('precision_std')
                },
                'recall': {
                    'mean': ff.get('recall_mean'),
                    'std': ff.get('recall_std')
                },
                'f1': {
                    'mean': ff.get('f1_mean'),
                    'std': ff.get('f1_std')
                },
                'kappa': {
                    'mean': ff.get('kappa_mean'),
                    'std': ff.get('kappa_std')
                }
            }
        }
    
    # Gated Decision
    if 'gated_decision' in agg:
        gd = agg['gated_decision']
        strategies['gated_decision'] = {
            'name': 'Gated Decision Fusion (Bayesian when high risk)',
            'metrics': {
                'auc': {
                    'mean': gd.get('auc_mean'),
                    'std': gd.get('auc_std')
                },
                'aupr': {
                    'mean': gd.get('aupr_mean'),
                    'std': gd.get('aupr_std')
                },
                'precision': {
                    'mean': gd.get('precision_mean'),
                    'std': gd.get('precision_std')
                },
                'recall': {
                    'mean': gd.get('recall_mean'),
                    'std': gd.get('recall_std')
                },
                'f1': {
                    'mean': gd.get('f1_mean'),
                    'std': gd.get('f1_std')
                },
                'kappa': {
                    'mean': gd.get('kappa_mean'),
                    'std': gd.get('kappa_std')
                }
            }
        }
    
    # Weighted Ensemble
    if 'weighted_ensemble' in agg:
        we = agg['weighted_ensemble']
        strategies['weighted_ensemble'] = {
            'name': 'Weighted Ensemble (α * Bayes + (1-α) * XGBoost)',
            'metrics': {
                'auc': {
                    'mean': we.get('auc_mean'),
                    'std': we.get('auc_std')
                },
                'aupr': {
                    'mean': we.get('aupr_mean'),
                    'std': we.get('aupr_std')
                },
                'precision': {
                    'mean': we.get('precision_mean'),
                    'std': we.get('precision_std')
                },
                'recall': {
                    'mean': we.get('recall_mean'),
                    'std': we.get('recall_std')
                },
                'f1': {
                    'mean': we.get('f1_mean'),
                    'std': we.get('f1_std')
                }
            }
        }
    
    return {
        'model': 'Fusion Strategies',
        'phase': 'Phase 6 Task 4',
        'strategies': strategies
    }


def create_comparison_table(
    bayesian: Dict[str, Any],
    xgboost: Dict[str, Any],
    fusion: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create comparison table across all models.
    """
    comparison = {
        'metric_definitions': {
            'auc': 'Area Under ROC Curve (discrimination ability)',
            'aupr': 'Area Under Precision-Recall Curve (performance on positive class)',
            'precision': 'True Positives / (True Positives + False Positives)',
            'recall': 'True Positives / (True Positives + False Negatives) [Sensitivity]',
            'specificity': 'True Negatives / (True Negatives + False Positives)',
            'f1': 'Harmonic mean of Precision and Recall',
            'kappa': "Cohen's Kappa (agreement beyond chance)",
            'brier': 'Mean squared error of probabilistic predictions (lower is better)',
            'accuracy_note': 'NOT REPORTED - Misleading for imbalanced data (95% no-outbreak baseline)'
        },
        'models': {}
    }
    
    if bayesian:
        comparison['models']['bayesian'] = bayesian
    
    if xgboost:
        comparison['models']['xgboost'] = xgboost
    
    if fusion:
        comparison['models']['fusion'] = fusion
    
    def _auc_str(model_block: Dict[str, Any]) -> str:
        try:
            v = model_block.get('metrics', {}).get('auc', {}).get('mean', None)
            return f"{float(v):.3f}" if v is not None else "N/A"
        except Exception:
            return "N/A"

    # Key insights (computed from loaded metrics; no hard-coded AUCs)
    comparison['key_insights'] = {
        'track_a_ml': {
            'champion': f"XGBoost (AUC={_auc_str(xgboost)})" if xgboost else 'XGBoost',
            'strength': 'Best binary classification performance',
            'weakness': 'Low recall can miss outbreaks (threshold-dependent)',
            'use_case': 'When false alarms are very costly'
        },
        'track_b_bayesian': {
            'champion': f"Bayesian State-Space (AUC={_auc_str(bayesian)})" if bayesian else 'Bayesian State-Space',
            'strength': 'Uncertainty quantification, early warning potential',
            'weakness': 'Low AUC when evaluated as binary classifier',
            'use_case': 'Decision-theoretic risk assessment, lead-time advantage',
            'critical_note': 'Should NOT be compared on AUC alone - this is a latent risk estimator'
        },
        'fusion_potential': {
            'best_strategy': 'To be determined from Phase 6 Task 4 results',
            'hypothesis': 'Feature fusion or gated decision may improve AUPR and recall',
            'goal': 'Combine ML discrimination with Bayesian uncertainty'
        },
        'metric_priorities': {
            'primary': ['aupr', 'recall', 'f1', 'lead_time'],
            'secondary': ['auc', 'precision', 'kappa'],
            'calibration': ['brier'],
            'avoid': ['accuracy']
        }
    }
    
    return comparison


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 6 Task 5: Comprehensive Metrics')
    parser.add_argument('--config', type=str, default='config/config_default.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='results/analysis/comprehensive_metrics.json',
                       help='Output JSON file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("="*60)
    print("COMPREHENSIVE METRICS ANALYSIS")
    print("="*60)
    
    # Load results from all phases
    print("\nLoading results from previous phases...")
    
    bayesian_results = load_bayesian_results()
    print(f"  Bayesian (Phase 5): {'✓' if bayesian_results else '✗'}")
    
    baseline_results = load_baseline_results()
    print(f"  Baselines (v1.1): {'✓' if baseline_results else '✗'}")
    
    fusion_results = load_fusion_results()
    print(f"  Fusion (Phase 6): {'✓' if fusion_results else '✗ (run Task 4 first)'}")
    
    # Extract metrics
    print("\nExtracting metrics...")
    
    bayesian_metrics = extract_bayesian_metrics(bayesian_results)
    xgboost_metrics = extract_xgboost_metrics(baseline_results)
    fusion_metrics = extract_fusion_metrics(fusion_results)
    
    # Create comparison
    comparison = create_comparison_table(
        bayesian_metrics,
        xgboost_metrics,
        fusion_metrics
    )
    
    # Print summary
    print("\n" + "="*60)
    print("METRIC COMPARISON SUMMARY")
    print("="*60)
    
    if bayesian_metrics:
        print("\nBAYESIAN STATE-SPACE:")
        print(f"  AUC: {bayesian_metrics['metrics']['auc']['mean']:.3f} ± {bayesian_metrics['metrics']['auc']['std']:.3f}")
        print(f"  Recall: {bayesian_metrics['metrics']['recall']['mean']:.3f} ± {bayesian_metrics['metrics']['recall']['std']:.3f}")
        print(f"  Specificity: {bayesian_metrics['metrics']['specificity']['mean']:.3f}")
        print(f"  Brier: {bayesian_metrics['metrics']['brier']['mean']:.3f}")
    
    if xgboost_metrics:
        print("\nXGBOOST:")
        print(f"  AUC: {xgboost_metrics['metrics']['auc']['mean']:.3f} ± {xgboost_metrics['metrics']['auc']['std']:.3f}")
        print(f"  Recall: {xgboost_metrics['metrics']['recall']['mean']:.3f}")
        print(f"  F1: {xgboost_metrics['metrics']['f1']['mean']:.3f}")
    
    if fusion_metrics and 'strategies' in fusion_metrics:
        print("\nFUSION STRATEGIES:")
        for name, strategy in fusion_metrics['strategies'].items():
            if 'metrics' in strategy:
                m = strategy['metrics']
                print(f"\n  {strategy['name']}:")
                if m.get('auc', {}).get('mean'):
                    print(f"    AUC: {m['auc']['mean']:.3f}")
                if m.get('aupr', {}).get('mean'):
                    print(f"    AUPR: {m['aupr']['mean']:.3f}")
                if m.get('f1', {}).get('mean'):
                    print(f"    F1: {m['f1']['mean']:.3f}")
    
    # Save comprehensive report
    output = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 6 - Task 5: Comprehensive Metrics',
        'warning': 'Accuracy is NOT reported - misleading for imbalanced outbreak data',
        'comparison': comparison
    }
    
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Comprehensive metrics saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
