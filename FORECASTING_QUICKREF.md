# Quick Reference: Using Temporal Forecasting

## Basic Usage

### 1. Import the Model
```python
from src.models.bayesian.state_space import BayesianStateSpace
```

### 2. Setup Configuration
```python
model_config = {
    'n_warmup': 1000,
    'n_samples': 1000,
    'n_chains': 4,
    'adapt_delta': 0.95,
    'seed': 42,
    'outbreak_percentile': 90  # For threshold calculation
}

model = BayesianStateSpace(config=model_config)
```

### 3. Fit with Forecasting Enabled
```python
# Fit on training data and prepare forecast for test period
model.fit(
    X_train, 
    y_train, 
    df=train_df,              # Training DataFrame
    feature_cols=feature_cols,
    forecast_df=test_df        # ← KEY: Enables forecasting
)
```

### 4. Get Forecast Predictions
```python
# Option A: Get outbreak probabilities
proba = model.forecast_proba(test_df=test_df)

# Option B: Get raw forecast samples
y_forecast = model.forecast(test_df=test_df)  # Shape: (n_draws, N_test)
```

### 5. Evaluate
```python
from src.evaluation.metrics import compute_all_metrics

metrics = compute_all_metrics(
    y_true=test_df['label_outbreak'].values,
    y_pred_proba=proba,
    threshold=0.5  # Probability threshold
)

print(f"AUC: {metrics['auc']:.3f}")
print(f"F1:  {metrics['f1']:.3f}")
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Example: Temporal forecasting with Bayesian model"""
import pandas as pd
import numpy as np
from src.models.bayesian.state_space import BayesianStateSpace
from src.evaluation.metrics import compute_all_metrics

# Load data
df = pd.read_csv('data/processed/features_panel.csv')

# Split train/test
train_df = df[df['year'] < 2019].copy()
test_df = df[df['year'] == 2019].copy()

# Get features
feature_cols = [c for c in df.columns if c.startswith('feat_')]
X_train = train_df[feature_cols].values
y_train = train_df['cases'].values

# Initialize model
model = BayesianStateSpace(config={
    'n_warmup': 1000,
    'n_samples': 1000,
    'n_chains': 4,
    'adapt_delta': 0.95,
    'outbreak_percentile': 90
})

# Fit with forecast capability
print("Fitting model...")
model.fit(
    X_train, y_train,
    df=train_df,
    feature_cols=feature_cols,
    forecast_df=test_df  # Enable forecasting
)

# Get forecast probabilities
print("Generating forecasts...")
proba = model.forecast_proba(test_df=test_df)

# Evaluate
y_true = test_df['label_outbreak'].values
metrics = compute_all_metrics(y_true, proba, threshold=0.5)

print(f"\nResults:")
print(f"  AUC:         {metrics['auc']:.3f}")
print(f"  F1 Score:    {metrics['f1']:.3f}")
print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
print(f"  Specificity: {metrics['specificity']:.3f}")
```

---

## API Reference

### `BayesianStateSpace.fit(X, y, df, feature_cols, forecast_df=None)`

Fit the model on training data.

**Parameters:**
- `X`: Feature matrix (for API compatibility)
- `y`: Target vector (for API compatibility)
- `df`: Training DataFrame with metadata
- `feature_cols`: List of feature column names
- `forecast_df`: Optional test DataFrame to enable forecasting

**Returns:** `self`

---

### `BayesianStateSpace.forecast(test_df=None, n_draws=None)`

Get raw forecast samples.

**Parameters:**
- `test_df`: Test DataFrame (for validation)
- `n_draws`: Number of posterior draws (default: all)

**Returns:** `np.ndarray` of shape `(n_draws, N_forecast)` with predicted case counts

**Raises:** `ValueError` if model not fitted with `forecast_df`

---

### `BayesianStateSpace.forecast_proba(test_df=None)`

Get outbreak probabilities for forecast period.

**Parameters:**
- `test_df`: Test DataFrame (for validation)

**Returns:** `np.ndarray` of length `N_forecast` with outbreak probabilities

**Raises:** `ValueError` if model not fitted with `forecast_df`

---

### `BayesianStateSpace.predict_proba(X, df=None, use_forecast=False)`

Get outbreak probabilities (training or forecast).

**Parameters:**
- `X`: Feature matrix (for API compatibility)
- `df`: DataFrame with metadata
- `use_forecast`: If True, use forecast predictions; if False, use training predictions

**Returns:** `np.ndarray` with outbreak probabilities

---

## Key Differences

### OLD (Wrong - Data Leakage)
```python
# Fit on training only
model.fit(X_train, y_train, df=train_df, feature_cols=feature_cols)

# Get predictions (ONLY for training period!)
proba = model.predict_proba(X_train, df=train_df)

# ❌ Using training predictions to evaluate test set
# This is DATA LEAKAGE
```

### NEW (Correct - Temporal Forecasting)
```python
# Fit on training WITH forecast setup
model.fit(
    X_train, y_train,
    df=train_df,
    feature_cols=feature_cols,
    forecast_df=test_df  # ← Enable forecasting
)

# Get FORECAST predictions for test period
proba = model.forecast_proba(test_df=test_df)

# ✅ Using true temporal forecasts
# NO DATA LEAKAGE
```

---

## Validation Checks

The implementation includes several validation checks:

1. **District Alignment:** Forecast districts must exist in training data
2. **Time Continuity:** Forecast time indices must be > T_max
3. **Forecast Availability:** Methods verify forecast was prepared during fit
4. **Data Consistency:** Validates test_df structure matches forecast setup

---

## Troubleshooting

### Error: "Model was not fitted with forecast capability"
**Solution:** Add `forecast_df=test_df` to the `fit()` call

### Error: "Forecast data contains districts not in training"
**Solution:** Ensure test set only includes districts present in training set

### Error: "outbreak_percentile must be provided via config"
**Solution:** Add `'outbreak_percentile': 90` to model config

---

## Performance Notes

- **Compilation:** Stan model compiles once (~5 seconds)
- **Sampling:** 4 chains × 1000 warmup × 1000 samples ≈ 10-30 minutes
- **Forecasting:** Generated during sampling (no additional time)
- **Memory:** Forecast samples stored in memory (~50MB for 1000 draws × 5000 tests)

---

## References

- [Stan User's Guide: Generated Quantities](https://mc-stan.org/docs/stan-users-guide/generated-quantities.html)
- [FORECASTING_IMPLEMENTATION.md](FORECASTING_IMPLEMENTATION.md) - Full technical details
- [experiments/test_forecasting_capability.py](experiments/test_forecasting_capability.py) - Working example
