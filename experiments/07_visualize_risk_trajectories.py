#!/usr/bin/env python3
"""Experiment 07: Risk Trajectory Visualization (Phase 7+ aligned)

This script generates district-level plots using the canonical per-row
out-of-sample prediction artifact from Experiment 06:
  - results/analysis/lead_time_predictions_p{p}.parquet

It intentionally does NOT refit Bayesian models. Instead it plots:
  - Observed cases
  - District-specific outbreak threshold (cases p{p}) computed from available rows
  - Bayesian forecast latent risk summary: z_mean with approximate 50%/90% intervals using z_sd

Outputs:
  - results/plots/risk_trajectories/*.png
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


plt.style.use('seaborn-v0_8-darkgrid')
FIGSIZE = (14, 6)
DPI = 150


def _preds_path(percentile: int) -> Path:
    path = project_root / 'results' / 'analysis' / f'lead_time_predictions_p{percentile}.parquet'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Run experiments/06_analyze_lead_time.py first.'
        )
    return path


def load_predictions(percentile: int) -> pd.DataFrame:
    df = pd.read_parquet(_preds_path(percentile))
    needed = {'state', 'district', 'year', 'week', 'cases', 'fold', 'prob', 'z_mean', 'z_sd', 'y_true'}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f'Predictions parquet missing required columns: {sorted(missing)}')
    return df


def create_week_date(year: int, week: int) -> datetime:
    """Convert year-week to an approximate week start date (ISO-like)."""
    jan1 = datetime(int(year), 1, 1)
    start_of_year = jan1 - timedelta(days=jan1.weekday())
    return start_of_year + timedelta(weeks=int(week) - 1)


def select_representative_districts(df: pd.DataFrame, n_districts: int) -> List[Tuple[str, str]]:
    stats = []
    for (state, district), g in df.groupby(['state', 'district']):
        y_true = pd.to_numeric(g['y_true'], errors='coerce')
        y_valid = y_true.dropna()
        stats.append(
            {
                'state': state,
                'district': district,
                'n_rows': len(g),
                'n_outbreak_weeks': int(np.nansum(y_true.values)),
                'outbreak_rate': float(y_valid.mean()) if len(y_valid) else 0.0,
                'max_cases': float(np.nanmax(g['cases'].values)) if len(g) else 0.0,
                'total_cases': float(np.nansum(g['cases'].values)) if len(g) else 0.0,
            }
        )

    stats_df = pd.DataFrame(stats)
    if stats_df.empty:
        return []

    selected: List[Tuple[str, str]] = []

    # 1) Most outbreak weeks
    high_freq = stats_df.sort_values('n_outbreak_weeks', ascending=False).head(max(1, n_districts // 4))
    selected.extend(list(zip(high_freq['state'], high_freq['district'])))

    # 2) Highest peaks
    high_peak = stats_df.sort_values('max_cases', ascending=False).head(max(1, n_districts // 4))
    for s, d in zip(high_peak['state'], high_peak['district']):
        if (s, d) not in selected:
            selected.append((s, d))

    # 3) Fill remaining with diversity (stable seed)
    remaining = stats_df[~stats_df.apply(lambda r: (r['state'], r['district']) in selected, axis=1)]
    if not remaining.empty and len(selected) < n_districts:
        fill = remaining.sample(n=min(n_districts - len(selected), len(remaining)), random_state=42)
        selected.extend(list(zip(fill['state'], fill['district'])))

    return selected[:n_districts]


def _add_intervals(plot_df: pd.DataFrame) -> pd.DataFrame:
    z = plot_df['z_mean'].astype(float).values
    sd = np.maximum(plot_df['z_sd'].astype(float).values, 1e-6)

    # Normal approx quantiles
    z_q05 = z - 1.645 * sd
    z_q95 = z + 1.645 * sd
    z_q25 = z - 0.674 * sd
    z_q75 = z + 0.674 * sd

    plot_df = plot_df.copy()
    plot_df['z_q05'] = z_q05
    plot_df['z_q95'] = z_q95
    plot_df['z_q25'] = z_q25
    plot_df['z_q75'] = z_q75
    return plot_df


def plot_risk_trajectory(plot_df: pd.DataFrame, state: str, district: str, percentile: int, out_path: Path) -> None:
    plot_df = plot_df.copy()
    plot_df['date'] = plot_df.apply(lambda r: create_week_date(r['year'], r['week']), axis=1)
    plot_df = plot_df.sort_values('date')
    plot_df = _add_intervals(plot_df)

    # District-specific outbreak threshold for reference
    threshold = float(np.percentile(plot_df['cases'].astype(float).values, percentile))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.plot(
        plot_df['date'],
        plot_df['cases'],
        'o-',
        color='black',
        linewidth=2,
        markersize=4,
        label='Observed Cases',
        alpha=0.7,
        zorder=3,
    )

    ax.axhline(
        y=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Outbreak Threshold (P{percentile} = {threshold:.1f})',
        zorder=2,
    )

    ax.plot(
        plot_df['date'],
        plot_df['z_mean'],
        '-',
        color='#1f77b4',
        linewidth=2.5,
        label='Bayesian Forecast Risk (z_mean)',
        zorder=4,
    )

    ax.fill_between(
        plot_df['date'],
        plot_df['z_q05'],
        plot_df['z_q95'],
        color='#1f77b4',
        alpha=0.15,
        label='90% Interval (Normal approx)',
        zorder=1,
    )
    ax.fill_between(
        plot_df['date'],
        plot_df['z_q25'],
        plot_df['z_q75'],
        color='#1f77b4',
        alpha=0.25,
        label='50% Interval (Normal approx)',
        zorder=1,
    )

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cases / Latent Risk', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk Trajectory: {district}, {state}', fontsize=14, fontweight='bold', pad=15)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha='right')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Experiment 07: Risk Trajectory Visualization (artifact-driven)')
    parser.add_argument('--outbreak-percentile', type=int, default=75)
    parser.add_argument('--output-dir', type=str, default='results/plots/risk_trajectories')
    parser.add_argument('--n-districts', type=int, default=8)
    args = parser.parse_args()

    df = load_predictions(args.outbreak_percentile)
    selected = select_representative_districts(df, n_districts=args.n_districts)
    if not selected:
        raise RuntimeError('No districts available in prediction parquet.')

    out_dir = project_root / args.output_dir
    print(f"Loaded {len(df)} prediction rows from {_preds_path(args.outbreak_percentile).name}")
    print(f"Selected {len(selected)} districts")

    for i, (state, district) in enumerate(selected, 1):
        sub = df[(df['state'] == state) & (df['district'] == district)].copy()
        if sub.empty:
            continue

        safe = f"{district.replace(' ', '_')}_{state.replace(' ', '_')}.png"
        out_path = out_dir / safe
        plot_risk_trajectory(sub, state, district, args.outbreak_percentile, out_path)
        print(f"[{i}/{len(selected)}] Saved {out_path}")


if __name__ == '__main__':
    main()
