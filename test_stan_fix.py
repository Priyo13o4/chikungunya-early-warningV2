#!/usr/bin/env python3
"""Quick test to verify Z_forecast is accessible from Stan model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from cmdstanpy import CmdStanModel
import numpy as np

print("=" * 60)
print("Testing Stan Model Z_forecast Fix")
print("=" * 60)

# Compile the model
print("\n1. Compiling Stan model...")
model = CmdStanModel(stan_file='stan_models/hierarchical_ews_v01.stan')
print("   ✓ Stan model compiled successfully")

# Test with minimal data
print("\n2. Preparing minimal test data...")
data = {
    'N': 10,
    'D': 2,
    'T_max': 5,
    'district': np.array([1,1,1,1,1,2,2,2,2,2], dtype=int),
    'time': np.array([1,2,3,4,5,1,2,3,4,5], dtype=int),
    'y': np.array([0,1,2,1,0,0,0,1,0,0], dtype=int),
    'temp_anomaly': np.zeros(10),
    'N_forecast': 2,
    'district_forecast': np.array([1,2], dtype=int),
    'time_forecast': np.array([6,6], dtype=int),
    'temp_anomaly_forecast': np.zeros(2)
}
print("   ✓ Test data prepared")

# Sample (quick test with very few iterations)
print("\n3. Running MCMC (1 chain, 50 warmup + 50 samples)...")
fit = model.sample(data=data, chains=1, iter_warmup=50, iter_sampling=50, show_console=False)
print("   ✓ Stan sampling completed")

# Check if Z_forecast is available
print("\n4. Checking if Z_forecast is accessible...")
try:
    z_forecast = fit.stan_variable('Z_forecast')
    print(f"   ✓ Z_forecast extracted successfully!")
    print(f"   Shape: {z_forecast.shape}")
    print(f"   Expected: (50, 2, 1) [draws × districts × forecast_time_steps]")
    
    # Also check other variables
    y_forecast = fit.stan_variable('y_forecast')
    print(f"\n   ✓ y_forecast also accessible, shape: {y_forecast.shape}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Stan model fix is working correctly!")
    print("=" * 60)
except Exception as e:
    print(f"   ✗ FAILED to extract Z_forecast: {e}")
    print("\n" + "=" * 60)
    print("FAILURE: Stan model still has issues")
    print("=" * 60)
    sys.exit(1)
