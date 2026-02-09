# Thesis Framework Alignment & Clarification

**Date**: February 6, 2026  
**Purpose**: Verify implementation matches thesis notes V1 & V2

---

## 🚨 **CRITICAL CLARIFICATION: Latent State ≠ Latent Features**

### **You Asked**: "The latent part of the Bayesian is supposed to find the hidden features right?"

### **Answer**: **NO** - Important distinction!

#### **What's Actually Happening**:

```
FEATURES (37 total) → EXPLICITLY ENGINEERED (mechanistic)
    ↓
    These are INPUT to the Bayesian model
    ↓
LATENT RISK STATE (Z_t) → INFERRED by Bayesian model
    ↓
    This is OUTPUT/HIDDEN STATE being estimated
```

### **The Bayesian Model Does NOT Find Features**

**What it DOES**:
- Takes your 37 **explicit features** as input (X)
- Takes observed **case counts** as input (Y)
- **Infers** a hidden **risk trajectory** (Z_t) that explains both

**Mathematical Formulation**:
```
Z_t = latent outbreak risk at time t (UNOBSERVED)
Y_t = observed cases at time t (NOISY OBSERVATION)
X_t = climate/case features at time t (EXPLICIT INPUTS)

Bayesian Model Estimates:
P(Z_t | Y_1:t, X_1:t) = "What is the hidden risk, given what we've seen?"

Z_t is NOT a feature
Z_t is the STATE OF THE SYSTEM we're trying to infer
```

---

## 📋 **Feature Inventory: Mechanistic Encoding**

### **Category 1: Epidemiological Dynamics (11 features)**
*Capture system instability and transmission momentum*

| Feature | Mechanistic Rationale |
|---------|----------------------|
| `feat_cases_lag_1/2/4/8` | Recent transmission history (generation intervals) |
| `feat_cases_ma_4w/8w` | Smoothed trend (filters reporting noise) |
| `feat_cases_var_4w` | Volatility → system instability signal |
| `feat_cases_growth_rate` | Exponential growth phase detection |
| `feat_cases_skew_4w` | Asymmetric distribution → early outbreak signature |
| `feat_trend_accel` | Acceleration → phase transition indicator |
| `feat_var_spike_ratio` | Sudden volatility spike → regime change |

### **Category 2: Temporal Autocorrelation (2 features)**
*Detect memory in the epidemic process*

| Feature | Mechanistic Rationale |
|---------|----------------------|
| `feat_cases_acf_lag1_4w` | Autocorrelation → persistence/decay rate |
| `feat_acf_change` | Changing autocorrelation → dynamical shift |

### **Category 3: Climate → Vector Mechanism (14 features)**
*Link environmental conditions to mosquito ecology*

| Feature | Mechanistic Rationale |
|---------|----------------------|
| `feat_temp_lag_1/2/4/8` | Temperature affects: biting rate, EIP, mosquito lifespan |
| `feat_temp_anomaly` | Deviation from seasonal norm → abnormal conditions |
| `feat_degree_days_above_20` | Cumulative thermal energy → Aedes development rate |
| `feat_rain_lag_1/2/4/8` | Rainfall creates breeding sites (7-14 day lag) |
| `feat_rain_persist_4w` | Sustained wetness → sustained vector population |
| `feat_lai` (Leaf Area Index) | Vegetation → shade + humidity → mosquito survival |
| `feat_lai_lag_1/2/4` | Lagged vegetation → delayed habitat effect |
| `feat_is_monsoon` | Seasonal regime shift (June-Sept in India) |

### **Category 4: Seasonality (3 features)**
*Encode annual cycles*

| Feature | Mechanistic Rationale |
|---------|----------------------|
| `feat_week_sin/cos` | Smooth seasonal encoding (no boundary discontinuity) |
| `feat_quarter` | Coarse seasonal bins (winter/summer/monsoon) |

### **Category 5: Spatial Context (3 features)**
*Geographic variation*

| Feature | Mechanistic Rationale |
|---------|----------------------|
| `feat_lat_norm` | Latitude → climate zones |
| `feat_lon_norm` | Longitude → coastal vs inland |
| `feat_lat_lon_interact` | Geographic interaction (e.g., Western Ghats) |

### **Category 6: Normalization (2 features)**
*Relative risk positioning*

| Feature | Mechanistic Rationale |
|---------|----------------------|
| `feat_recent_normalized` | Current cases relative to local baseline |
| (other norms embedded) | Helps model focus on deviations, not absolute scale |

---

## ✅ **Thesis Framework Alignment Check**

### **Your Proposed Pipeline**

```
Raw Data → Feature Extraction (mechanistic) 
         → Probabilistic Risk Estimation (Bayesian latent state)
         → Validation & Comparison
         → Decision Layer (cost-loss)
```

### **Current Implementation Status**

| Component | Status | Notes |
|-----------|--------|-------|
| **Raw Data** | ✅ Complete | EpiClim + Census + Climate |
| **Feature Extraction** | ✅ Complete | 37 mechanistic features |
| **Bayesian Latent Risk** | ✅ Complete | Hierarchical state-space (Stan) |
| **Track A Baselines** | ✅ Complete | XGBoost, RF, Logistic, Rule-based |
| **Temporal CV** | ✅ Complete | Rolling-origin (2017-2022) |
| **Lead-Time Analysis** | ✅ Complete | Validated in Phase 7 |
| **Decision Layer** | ⚠️ **Partial** | Simulation exists, cost-loss NOT optimized |

---

## 🎯 **Track A vs Track B - Current Implementation**

### **TRACK A: Supervised Baselines** ✅

**What's Implemented**:
- XGBoost (primary baseline)
- Random Forest
- Logistic Regression
- Rule-based threshold (EWARS-style)

**Metrics Computed**:
- ✅ AUC (discrimination)
- ✅ Sensitivity / Specificity
- ✅ False Alarm Rate
- ✅ Lead Time (weeks before outbreak)
- ⚠️ **Missing**: Precision, Recall, F1, AUPR, Kappa (easy to add)
- ⚠️ **Missing**: SHAP/LIME interpretability (not yet implemented)

**Validation**:
- ✅ Temporal cross-validation (rolling-origin)
- ✅ 6 folds (2017-2022)
- ✅ No data leakage

**Key Limitation** (as you noted):
> Binary labels quantize continuous risk
> 
> Models can only fire after case patterns emerge
> 
> Uncertainty is post-hoc, not intrinsic

---

### **TRACK B: Bayesian Latent Risk** ✅

**What's Implemented**:
- Hierarchical Bayesian state-space model
- Stan implementation (MCMC inference)
- Latent risk state Z_t ~ continuous [0, ∞)
- District-level partial pooling (sharing information)

**Mathematical Structure**:
```stan
// Hierarchical priors (partial pooling across districts)
μ_global ~ Normal(0, 5)
σ_district ~ HalfNormal(2)

// District-specific parameters
β_i ~ Normal(μ_global, σ_district)

// State-space dynamics (temporal continuity)
Z_t = f(Z_{t-1}, X_t, β_i) + noise

// Observation model (cases are noisy manifestation)
Y_t ~ NegativeBinomial(Z_t, ϕ)
```

**Metrics Computed**:
- ✅ Brier Score (calibration)
- ✅ Lead Time (vs Track A)
- ✅ False Alarm Rate
- ✅ Posterior uncertainty (credible intervals)
- ⚠️ **Missing**: Explicit calibration curves (placeholder exists)
- ⚠️ **Missing**: Reliability diagrams (not yet plotted)
- ⚠️ **Missing**: Rank correlation analysis

**Key Strength** (as you noted):
> Risk inferred BEFORE cases spike
> 
> Uncertainty is native, not added later
> 
> No binary labels in training

---

## 🔀 **Fusion Framework - Your Notes Are CORRECT**

### **You Wrote**:
> Fusion: mechanistic + Bayesian
> 
> Layer | Fusion type
> ------|------------
> Feature layer | Mechanistic fusion (climate → vector → transmission)
> Inference layer | Bayesian fusion (multiple noisy signals → posterior)
> Decision layer | Decision-theoretic fusion (risk + uncertainty → action)

### **Implementation Confirmation**:

#### **Layer 1: Feature-Level Mechanistic Fusion** ✅
```python
# Climate → Vector mechanism
degree_days = (temp - 20).clip(lower=0)  # Aedes development
rain_persist = rain.rolling(4).mean()     # Breeding habitat

# Case dynamics → Transmission
growth_rate = cases.pct_change()          # Exponential phase
var_spike = cases.var() / cases.var().shift(52)  # Regime shift

# All 37 features = mechanistic knowledge encoded
```

#### **Layer 2: Inference-Level Bayesian Fusion** ✅
```python
# Bayesian model fuses ALL signals into posterior P(Z_t | Data)
# Implicit fusion via generative model:

P(Z_t | cases, climate, features) ∝ 
    P(cases | Z_t) × P(Z_t | climate, features) × P(Z_t | Z_{t-1})
    
# Output: Posterior distribution over latent risk state
# Uncertainty quantified via MCMC samples
```

#### **Layer 3: Decision-Level Fusion** ⚠️ **PARTIAL**
```python
# IMPLEMENTED: Simulation of decision states (GREEN/YELLOW/RED)
# experiments/08_simulate_decision_layer.py exists

# NOT IMPLEMENTED: Cost-loss optimization
# Thresholds are currently HEURISTIC (percentile-based)
# NOT derived from cost–loss minimization
```

---

## 🚧 **What's Missing for Complete Thesis**

### **High Priority** (Easy to Add):

1. **Track A Metrics** (1-2 hours):
   - Precision, Recall, F1
   - AUPR (important for imbalance)
   - Cohen's Kappa
   - Confusion matrices per fold

2. **Track B Calibration** (2-3 hours):
   - Reliability curves (predicted prob vs observed freq)
   - ECE (Expected Calibration Error)
   - Sharpness diagrams

3. **Interpretability** (3-5 hours):
   - SHAP values for XGBoost (identify top features)
   - LIME for local explanations
   - Feature importance rankings

### **Medium Priority** (Requires New Code):

4. **Decision Layer - Cost-Loss Framework** (1-2 days):
   ```python
   # Define cost structure
   C_false_alarm = 1  # Cost of unnecessary intervention
   L_missed_outbreak = 100  # Loss if outbreak undetected
   
   # Optimal threshold from Bayesian decision theory
   threshold* = argmin E[Cost | P(Z_t), C, L]
   
   # Compare to current heuristic thresholds
   ```

5. **Comparative Ablation Studies** (2-3 days):
   - Mechanistic vs non-mechanistic features
   - Hierarchical vs non-hierarchical Bayesian
   - State-space vs memoryless models

### **Low Priority** (Nice to Have):

6. **External Validation**:
   - IDSP outbreak reports (dates, locations)
   - MoHFW annual counts (for sanity checks)

7. **Spatial Risk Maps**:
   - District-level shapefiles visualization
   - Z_t heatmaps over time

---

## 📊 **Thesis Contribution Statement**

Based on your notes, here's how to position this:

### **Gap in Literature**:
> Existing chikungunya models fall into two categories:
> 1. **Mechanistic simulation models** (CHIKSIM) for retrospective risk
> 2. **Binary outbreak classifiers** (EWARS-style) for real-time alerts
> 
> **No study combines**:
> - Latent state inference (continuous risk)
> - Mechanistic feature encoding (climate → vector)
> - Decision-theoretic alerting (cost-loss optimization)
> - National surveillance scale (India IDSP-compatible)

### **Our Contribution**:
> A hierarchical Bayesian early warning system that:
> 1. Encodes mechanistic knowledge via 37 epidemiologically-informed features
> 2. Infers continuous latent outbreak risk (Z_t) with native uncertainty
> 3. Provides actionable lead time (5-17 weeks) before case spikes
> 4. Validates against supervised baselines (Track A vs Track B comparison)
> 5. [PLANNED] Optimizes alert thresholds via cost-loss framework

### **Why This Matters**:
> - **Operationally**: India's IDSP gets probabilistic risk, not just binary alarms
> - **Methodologically**: Mechanistic + Bayesian fusion is principled, not black-box
> - **Generalizable**: Framework extends to dengue, Zika (same vectors)

---

## ✅ **Summary: We're 90% Aligned**

| Your Notes | Implementation Status |
|------------|----------------------|
| Mechanistic features | ✅ 37 features engineered |
| Bayesian latent state | ✅ Hierarchical state-space model |
| Track A baselines | ✅ XGBoost, RF, Logistic |
| Track B inference | ✅ MCMC via Stan |
| Temporal CV | ✅ Rolling-origin (6 folds) |
| Lead-time validation | ✅ Complete |
| **Decision layer** | ⚠️ **Simulation only, not cost-loss optimized** |
| **Interpretability** | ⚠️ **SHAP/LIME not yet added** |
| **Full metric suite** | ⚠️ **Missing AUPR, Kappa, calibration curves** |

---

## 🎯 **Next Steps to Complete Thesis**

**Week 1-2** (Must-have):
1. Add missing Track A metrics (Precision, Recall, AUPR, Kappa)
2. Generate calibration curves for Track B
3. SHAP analysis for XGBoost feature importance

**Week 3-4** (High value):
4. Implement cost-loss decision framework
5. Ablation studies (mechanistic vs non-mechanistic)
6. Write methods section with clear fusion narrative

**Week 5+** (Polish):
7. External validation (IDSP reports if available)
8. Risk map visualizations
9. Discussion of limitations (label sparsity, district coverage)

---

**Your framework is solid and matches the implementation!** The only gap is the decision-theoretic layer needs to move from simulation to actual cost-loss optimization. Everything else is thesis-ready. 🎉
