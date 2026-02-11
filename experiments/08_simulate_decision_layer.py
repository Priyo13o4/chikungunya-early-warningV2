#!/usr/bin/env python3
"""Experiment 08: Decision-Layer Simulation (Phase 7+ aligned)

This script implements uncertainty-aware decision rules on top of the Bayesian
latent-risk outputs produced in Experiment 06.

IMPORTANT ALIGNMENT CHANGES (Feb 2026):
- Do NOT refit the Bayesian model here.
- Consume `results/analysis/lead_time_predictions_p{p}.parquet` from Experiment 06.
    That file already contains out-of-sample, per-row test predictions for each fold:
    - `z_mean`, `z_sd` (latent Z summary)
    - `prob` (XGBoost probability)
    - `y_true` (label_outbreak)

Decision Zones:
- GREEN (No Action): P(Z_t > q) < 0.4
- YELLOW (Monitor): 0.4 ≤ P(Z_t > q) < 0.8 OR high uncertainty
- RED (Intervene): P(Z_t > q) ≥ 0.8 AND low uncertainty

Outputs:
- results/analysis/decision_simulation_p{p}.json
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config


# =============================================================================
# DECISION FRAMEWORK
# =============================================================================

class AlertZone(Enum):
    """Decision zones for outbreak early warning."""
    GREEN = "no_action"
    YELLOW = "monitor"
    RED = "intervene"


@dataclass
class DecisionThresholds:
    """Thresholds for decision zones."""
    risk_quantile: float = 0.80  # 80th percentile threshold for P(Z_t > q)
    prob_low: float = 0.40       # P(Z_t > q) < 0.4 → GREEN
    prob_high: float = 0.80      # P(Z_t > q) ≥ 0.8 → RED (if low uncertainty)
    uncertainty_threshold: float = 0.5  # Coefficient of variation threshold


@dataclass
class DecisionCosts:
    """Cost-loss parameters for decision evaluation."""
    cost_intervention: float = 1.0      # Cost of intervention (normalized)
    loss_missed_outbreak: float = 10.0  # Loss from missed outbreak
    cost_false_alarm: float = 0.5       # Cost of false alarm
    benefit_early_warning: float = 5.0  # Benefit of early intervention


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _select_predictions_parquet(prefix: str, percentile: int) -> Path:
    analysis_dir = project_root / "results" / "analysis"
    preferred = analysis_dir / f"{prefix}_p{percentile}.parquet"
    if preferred.exists():
        return preferred
    raise FileNotFoundError(
        f"Expected predictions parquet not found: {preferred}. Run experiments/06_analyze_lead_time.py first."
    )


def load_predictions(percentile: int) -> pd.DataFrame:
    """Load per-row, per-fold test predictions saved by Experiment 06."""
    path = _select_predictions_parquet("lead_time_predictions", percentile)
    df = pd.read_parquet(path)
    needed = {'fold', 'state', 'district', 'year', 'week', 'cases', 'y_true', 'z_mean', 'z_sd'}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Predictions parquet missing required columns: {sorted(missing)}")
    return df


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Normal CDF without scipy dependency."""
    import math

    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/scalar types for JSON serialization."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def compute_bayesian_risk_scores(
    preds_df: pd.DataFrame,
    thresholds: DecisionThresholds,
) -> pd.DataFrame:
    """Compute decision-layer risk scores from stored latent-Z forecasts.

    In forecast mode, z_mean IS the forecast (point estimate of latent risk)
    and z_sd is the forecast uncertainty. We use z_mean directly as the risk score
    without applying Normal CDF transformation.
    """
    work = preds_df.copy()
    work = work.dropna(subset=['z_mean', 'z_sd']).copy()

    # Use z_mean directly as the forecast risk score (no CDF transformation needed)
    mu = work['z_mean'].astype(float).values
    sd = work['z_sd'].astype(float).values
    sd = np.maximum(sd, 1e-6)

    work['latent_risk_mean'] = mu
    work['latent_risk_std'] = sd
    work['prob_high_risk'] = mu  # Direct use of forecast as risk score
    work['uncertainty_cv'] = work['latent_risk_std'] / (np.abs(work['latent_risk_mean']) + 1e-6)
    work['alert_zone'] = [
        assign_alert_zone(p, u, thresholds)
        for p, u in zip(work['prob_high_risk'].values, work['uncertainty_cv'].values)
    ]

    return work


def assign_alert_zone(
    prob_high_risk: float,
    uncertainty_cv: float,
    thresholds: DecisionThresholds
) -> str:
    """
    Assign alert zone based on probability and uncertainty.
    
    Rules:
    1. GREEN: Low probability of high risk
    2. YELLOW: Moderate probability OR high uncertainty
    3. RED: High probability AND low uncertainty
    """
    if prob_high_risk < thresholds.prob_low:
        return AlertZone.GREEN.value
    
    elif prob_high_risk >= thresholds.prob_high and uncertainty_cv < thresholds.uncertainty_threshold:
        return AlertZone.RED.value
    
    else:
        return AlertZone.YELLOW.value


def evaluate_decision_performance(
    risk_df: pd.DataFrame,
    thresholds: DecisionThresholds,
    costs: DecisionCosts
) -> Dict[str, Any]:
    """
    Evaluate decision layer performance.
    
    Computes:
    - Lead time to RED intervention before outbreaks
    - False alarms (RED without outbreak)
    - Missed outbreaks (never RED)
    - Decision stability (transition frequency)
    - Expected cost
    """
    results = {
        'interventions': {'total': 0, 'true_positive': 0, 'false_positive': 0},
        'outbreaks': {'total': 0, 'detected': 0, 'missed': 0},
        'lead_times': [],
        'zone_transitions': 0,
        'zone_distribution': {},
        'cost_analysis': {}
    }
    
    # Count zone distribution
    zone_counts = risk_df['alert_zone'].value_counts().to_dict()
    total = len(risk_df)
    results['zone_distribution'] = {
        zone: {'count': count, 'percentage': count / total * 100}
        for zone, count in zone_counts.items()
    }
    
    # Analyze by district-year episodes
    # Use y_true (label_outbreak) from Experiment 06 outputs.
    for (state, district, year), group in risk_df.groupby(['state', 'district', 'year']):
        group = group.sort_values('week')
        
        # Check if this year had an outbreak
        if 'y_true' not in group.columns:
            continue
        y = pd.to_numeric(group['y_true'], errors='coerce')
        has_outbreak = (y == 1).sum() > 0
        
        # Find first RED zone week
        red_weeks = group[group['alert_zone'] == AlertZone.RED.value]
        has_intervention = len(red_weeks) > 0
        
        if has_outbreak:
            results['outbreaks']['total'] += 1
            
            # Find first outbreak week
            outbreak_weeks = group[pd.to_numeric(group['y_true'], errors='coerce') == 1]
            if len(outbreak_weeks) > 0:
                first_outbreak_week = outbreak_weeks.iloc[0]['week']
                
                if has_intervention:
                    first_red_week = red_weeks.iloc[0]['week']
                    lead_time = first_outbreak_week - first_red_week
                    
                    if lead_time > 0:  # Intervention before outbreak
                        results['outbreaks']['detected'] += 1
                        results['interventions']['true_positive'] += 1
                        results['lead_times'].append(lead_time)
                    else:
                        results['outbreaks']['missed'] += 1
                else:
                    results['outbreaks']['missed'] += 1
        
        elif has_intervention:
            # Intervention without outbreak = false alarm
            results['interventions']['false_positive'] += 1
        
        if has_intervention:
            results['interventions']['total'] += 1
        
        # Count zone transitions
        zones = group['alert_zone'].values
        transitions = np.sum(zones[1:] != zones[:-1])
        results['zone_transitions'] += transitions
    
    # Compute metrics
    n_interventions = results['interventions']['total']
    n_outbreaks = results['outbreaks']['total']
    
    metrics = {}
    
    if n_interventions > 0:
        metrics['precision'] = results['interventions']['true_positive'] / n_interventions
        metrics['false_alarm_rate'] = results['interventions']['false_positive'] / n_interventions
    else:
        metrics['precision'] = 0.0
        metrics['false_alarm_rate'] = 0.0
    
    if n_outbreaks > 0:
        metrics['sensitivity'] = results['outbreaks']['detected'] / n_outbreaks
        metrics['miss_rate'] = results['outbreaks']['missed'] / n_outbreaks
    else:
        metrics['sensitivity'] = 0.0
        metrics['miss_rate'] = 0.0
    
    if len(results['lead_times']) > 0:
        metrics['median_lead_time'] = float(np.median(results['lead_times']))
        metrics['mean_lead_time'] = float(np.mean(results['lead_times']))
        metrics['lead_time_std'] = float(np.std(results['lead_times']))
    else:
        metrics['median_lead_time'] = None
        metrics['mean_lead_time'] = None
        metrics['lead_time_std'] = None
    
    # Decision stability (transitions per district-year)
    n_episodes = risk_df.groupby(['state', 'district', 'year']).ngroups
    metrics['avg_transitions_per_episode'] = results['zone_transitions'] / n_episodes if n_episodes > 0 else 0
    
    # Cost-loss analysis
    total_cost = (
        results['interventions']['total'] * costs.cost_intervention +
        results['interventions']['false_positive'] * costs.cost_false_alarm +
        results['outbreaks']['missed'] * costs.loss_missed_outbreak -
        results['outbreaks']['detected'] * costs.benefit_early_warning
    )
    
    cost_analysis = {
        'total_cost': total_cost,
        'intervention_cost': results['interventions']['total'] * costs.cost_intervention,
        'false_alarm_cost': results['interventions']['false_positive'] * costs.cost_false_alarm,
        'missed_outbreak_loss': results['outbreaks']['missed'] * costs.loss_missed_outbreak,
        'early_warning_benefit': results['outbreaks']['detected'] * costs.benefit_early_warning,
        'net_benefit': -total_cost  # Negative cost = positive benefit
    }
    
    results['metrics'] = metrics
    results['cost_analysis'] = cost_analysis
    
    return results


def simulate_decision_layer_for_fold(
    fold_df: pd.DataFrame,
    fold_name: str,
    thresholds: DecisionThresholds,
    costs: DecisionCosts,
) -> Dict[str, Any]:
    """Simulate decision layer for one fold using stored predictions."""
    print(f"\n{'='*60}")
    print(f"Simulating fold: {fold_name}")
    print(f"{'='*60}")

    print("\nStep 1: Computing Bayesian risk scores from stored latent Z...")
    risk_df = compute_bayesian_risk_scores(fold_df, thresholds)

    print("\nStep 2: Evaluating decision performance...")
    results = evaluate_decision_performance(risk_df, thresholds, costs)
    
    # Print summary
    print("\n--- Decision Layer Results ---")
    print(f"Total interventions (RED): {results['interventions']['total']}")
    print(f"  True positives: {results['interventions']['true_positive']}")
    print(f"  False positives: {results['interventions']['false_positive']}")
    print(f"Total outbreaks: {results['outbreaks']['total']}")
    print(f"  Detected: {results['outbreaks']['detected']}")
    print(f"  Missed: {results['outbreaks']['missed']}")
    
    if results['metrics']['median_lead_time'] is not None:
        print(f"Median lead time: {results['metrics']['median_lead_time']:.1f} weeks")
    print(f"Sensitivity: {results['metrics']['sensitivity']:.3f}")
    print(f"Precision: {results['metrics']['precision']:.3f}")
    print(f"False alarm rate: {results['metrics']['false_alarm_rate']:.3f}")
    print(f"\nNet benefit: {results['cost_analysis']['net_benefit']:.2f}")
    
    return {
        'fold': fold_name,
        'thresholds': {
            'risk_quantile': thresholds.risk_quantile,
            'prob_low': thresholds.prob_low,
            'prob_high': thresholds.prob_high,
            'uncertainty_threshold': thresholds.uncertainty_threshold
        },
        'results': results
    }


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 6 Task 3: Decision-Layer Simulation')
    parser.add_argument('--config', type=str, default='config/config_default.yaml',
                       help='Path to config file')
    parser.add_argument('--outbreak-percentile', type=int, default=75,
                       help='Outbreak percentile used in Experiment 06 outputs (selects lead_time_predictions_p{p}.parquet)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize thresholds and costs
    thresholds = DecisionThresholds()
    costs = DecisionCosts()
    
    print("Loading stored predictions from Experiment 06...")
    preds = load_predictions(args.outbreak_percentile)
    print(f"Loaded {len(preds)} prediction rows across folds")

    # Simulate each fold (preds are already per-fold test sets)
    all_results = []
    for fold_name, fold_df in preds.groupby('fold', sort=False):
        fold_result = simulate_decision_layer_for_fold(
            fold_df=fold_df,
            fold_name=str(fold_name),
            thresholds=thresholds,
            costs=costs,
        )
        all_results.append(fold_result)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATED DECISION PERFORMANCE")
    print(f"{'='*60}")
    
    total_interventions = sum([r['results']['interventions']['total'] for r in all_results])
    total_tp = sum([r['results']['interventions']['true_positive'] for r in all_results])
    total_fp = sum([r['results']['interventions']['false_positive'] for r in all_results])
    total_outbreaks = sum([r['results']['outbreaks']['total'] for r in all_results])
    total_detected = sum([r['results']['outbreaks']['detected'] for r in all_results])
    total_missed = sum([r['results']['outbreaks']['missed'] for r in all_results])
    
    all_lead_times = []
    for r in all_results:
        all_lead_times.extend(r['results']['lead_times'])
    
    aggregated = {
        'total_interventions': total_interventions,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'total_outbreaks': total_outbreaks,
        'detected': total_detected,
        'missed': total_missed,
        'sensitivity': total_detected / total_outbreaks if total_outbreaks > 0 else 0,
        'precision': total_tp / total_interventions if total_interventions > 0 else 0,
        'false_alarm_rate': total_fp / total_interventions if total_interventions > 0 else 0
    }
    
    if len(all_lead_times) > 0:
        aggregated['median_lead_time'] = float(np.median(all_lead_times))
        aggregated['mean_lead_time'] = float(np.mean(all_lead_times))
        aggregated['std_lead_time'] = float(np.std(all_lead_times))
    
    # Calculate total cost
    total_cost = sum([r['results']['cost_analysis']['total_cost'] for r in all_results])
    total_benefit = sum([r['results']['cost_analysis']['net_benefit'] for r in all_results])
    
    aggregated['total_cost'] = total_cost
    aggregated['net_benefit'] = total_benefit
    
    print(f"\nTotal interventions: {aggregated['total_interventions']}")
    print(f"Total outbreaks: {aggregated['total_outbreaks']}")
    print(f"Sensitivity: {aggregated['sensitivity']:.3f}")
    print(f"Precision: {aggregated['precision']:.3f}")
    if 'median_lead_time' in aggregated:
        print(f"Median lead time: {aggregated['median_lead_time']:.1f} weeks")
    print(f"Net benefit: {aggregated['net_benefit']:.2f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 6 - Task 3: Decision-Layer Simulation',
        'decision_framework': {
            'thresholds': {
                'risk_quantile': thresholds.risk_quantile,
                'prob_low': thresholds.prob_low,
                'prob_high': thresholds.prob_high,
                'uncertainty_threshold': thresholds.uncertainty_threshold
            },
            'costs': {
                'intervention': costs.cost_intervention,
                'missed_outbreak_loss': costs.loss_missed_outbreak,
                'false_alarm': costs.cost_false_alarm,
                'early_warning_benefit': costs.benefit_early_warning
            }
        },
        'aggregated': aggregated,
        'fold_results': all_results
    }
    
    if args.output:
        output_path = project_root / args.output
    else:
        output_path = project_root / "results" / "analysis" / f"decision_simulation_p{args.outbreak_percentile}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(_json_safe(output), f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
