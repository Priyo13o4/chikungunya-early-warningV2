#!/usr/bin/env python3
"""Analyze district-level sparsity to evaluate filtering options."""

import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/features_engineered_v01.parquet')

# Analyze by district
district_stats = df.groupby(['state', 'district']).agg({
    'cases': ['count', 'sum', 'mean', 'max'],
    'label_outbreak': 'sum'
}).reset_index()

district_stats.columns = ['state', 'district', 'n_obs', 'total_cases', 'mean_cases', 'max_cases', 'n_outbreaks']

# Sort by observations
district_stats = district_stats.sort_values('n_obs', ascending=False)

print('=' * 80)
print('DISTRICT-LEVEL SPARSITY ANALYSIS')
print('=' * 80)
print(f'Total districts: {len(district_stats)}')
print(f'Total observations: {len(df)}')
print(f'Mean obs per district: {district_stats["n_obs"].mean():.1f}')
print(f'Median obs per district: {district_stats["n_obs"].median():.1f}')
print()

# Quantiles
print('Observation distribution:')
for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
    val = district_stats['n_obs'].quantile(q)
    print(f'  {int(q*100)}th percentile: {val:.0f} obs')
print()

# Show top districts
print('Top 20 districts by observations:')
print(district_stats.head(20)[['state', 'district', 'n_obs', 'n_outbreaks', 'total_cases']].to_string(index=False))
print()

# Filtering scenarios
print('=' * 80)
print('FILTERING SCENARIOS')
print('=' * 80)
for min_obs in [10, 20, 30, 50]:
    filtered = district_stats[district_stats['n_obs'] >= min_obs]
    total_obs = filtered['n_obs'].sum()
    total_outbreaks = filtered['n_outbreaks'].sum()
    pct_retained = total_obs/len(df)*100
    print(f'Filter: >= {min_obs} obs/district')
    print(f'  → {len(filtered)} districts ({len(filtered)/len(district_stats)*100:.1f}%)')
    print(f'  → {total_obs} samples ({pct_retained:.1f}%)')
    print(f'  → {int(total_outbreaks)} labeled outbreaks')
    print(f'  → Mean obs/district: {total_obs/len(filtered):.1f}')
    print()
