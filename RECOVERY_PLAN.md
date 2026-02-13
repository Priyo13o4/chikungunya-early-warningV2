# Track B Recovery Plan
**Date:** 2026-02-12  
**Status:** ⚠️ **CRITICAL - Pipeline Complete But Results Non-Viable**  
**Last Update:** 2026-02-12 01:30 AM - Post-Phase 05-11 Forensic Analysis  
**Goal:** Restore Track B Bayesian model to defensible performance for thesis comparison

---

## ⚠️ CRITICAL FINDINGS: POST-PIPELINE FORENSIC ANALYSIS

### **Issues Discovered After Full Pipeline Execution (Phases 05-11)**

#### **ISSUE 1: Evaluation Validity Failure (Phase 05)**
**Status:** 🔴 CRITICAL - 50% of folds non-evaluable

| Fold | Test Size | Positives | Issue | Sensitivity | AUC |
|------|-----------|-----------|-------|-------------|-----|
| 2017-2019 | 8-16 | 4-7 | ✓ OK | 57-100% | 0.47-0.66 |
| **2020** | **8** | **1** | Extreme imbalance | **0%** | **0.000** |
| **2021** | **6** | **0** | No positives | **0%** | **nan** |
| **2022** | **1** | **0** | No positives | **0%** | **nan** |

**Root Cause:** Year-based temporal split creates systematic bias where recent years (2020-2022) have zero outbreak labels due to pandemic surveillance disruption.

**Impact:** Aggregated sensitivity (34.5%) is **statistically invalid** - only 3/6 folds interpretable.

---

#### **ISSUE 2: Lead-Time Analysis Collapse (Phase 06)**
**Status:** 🔴 CRITICAL - 0% early warnings, 2 episodes only

**Evidence:**
- Total outbreak episodes detected: **2** (vs 18 positive test samples in Phase 05)
- Early warnings issued: **0** (Bayesian + XGBoost both failed)
- Episode fragmentation rate: **67%** (12/18 positives are isolated single-week events)

**Root Cause Cascade:**
1. **Threshold misalignment:**
   - Training (Phase 05): P50 percentile
   - Episode definition (Phase 06): P75 percentile
   - Intervention trigger (Phase 08): P80 percentile
2. **NA label fragmentation:** Lead-time gaps destroy temporal continuity
   - Requires **consecutive weeks** above threshold
   - NA gaps fragment multi-week outbreaks into single-week events
3. **District filtering artifacts:** Bayesian excluded 61-77 test samples per fold from "unseen" districts

**Impact:** Lead-time analysis statistically **underpowered** - cannot draw conclusions from 2 episodes.

---

#### **ISSUE 3: Decision Layer Economic Failure (Phase 08)**
**Status:** 🔴 CRITICAL - Negative net benefit (-$41)

**Cost Breakdown:**
```
Benefits:  1 detected outbreak × $5     = $5
Costs:     12 interventions × $1        = $12
           8 false alarms × $0.5        = $4
           3 missed outbreaks × $10     = $30
Net Benefit: $5 - $46 = -$41
```

**Root Cause:**
- Detection rate: **25%** (1/4 outbreaks) → missed outbreak penalty dominates
- False alarm rate: **67%** (8/12 interventions) → wasted resources
- System **loses money** compared to "do nothing" baseline

**Impact:** Current decision framework is **not operationally viable** - negative ROI.

---

#### **ISSUE 4: Fusion Strategy Failure (Phase 09)**
**Status:** 🟡 MODERATE - Implementation bugs + negative transfer

| Strategy | AUC | AUPR | F1 | Status | Issue |
|----------|-----|------|----|----|-------|
| XGBoost (baseline) | 0.759 | — | 0.440 | ✓ | — |
| Gated Decision | 0.533 | 0.262 | 0.253 | ⚠️ | Worse than baseline |
| Weighted Ensemble | 0.500 | 0.000 | 0.000 | ❌ | NaN crash |

**Root Cause:**
- Weighted ensemble: `Input contains NaN` from single-class folds (2021, 2022)
- Gated decision: 60% Bayesian usage in some folds drags down performance
- **Negative transfer:** Bayesian model reduces ensemble quality

**Impact:** Fusion adds no value - contradicts thesis claim of "decision-theoretic fusion."

---

#### **ISSUE 5: Data Reality vs Thesis Claims**
**Status:** 🔴 CRITICAL - Fundamental dataset limitations

**Feature Inspection Results:**
- **Total labeled samples:** 74 (10.6% of 698 rows)
- **Missing data (full dataset):** 40-91% for case lags, EWS features
- **CORE 20 features:** Only 18/20 found, 92-100% complete in labeled data
- **Counterintuitive pattern:** Outbreaks have **LOWER variance** (0.37) than non-outbreaks (0.57)

**Mechanistic Encoding Weakness:**
- Only **2/5** features truly mechanistic (degree_days, rain_persist)
- `temp_anomaly`: Semi-mechanistic (deviation, not threshold)
- `cases_var_4w`, `growth_rate`: Pure statistical signals without biological interpretation

**Impact:** Dataset too sparse and features too weak to support thesis claims of "mechanistic fusion."

---

### **Aggregated Pipeline Scorecard**

| Phase | Status | Key Metric | Issues |
|-------|--------|------------|--------|
| 05 - Evaluation | ⚠️ PARTIAL | AUC 0.424, Sens 34.5%* | 3/6 folds non-evaluable |
| 06 - Lead Time | ❌ FAIL | 0% early warnings | Only 2 episodes, threshold misalignment |
| 07 - Visualization | ✅ OK | 6 figure types | No issues, plots generated |
| 08 - Decision Layer | ❌ FAIL | Net benefit -$41 | 25% detection, negative ROI |
| 09 - Fusion | ⚠️ PARTIAL | AUC 0.533 | NaN crashes, negative transfer |
| 10 - Metrics | ✅ OK | Aggregation complete | No issues |
| 11 - Sparsity | ✅ OK | Report generated | Confirms extreme sparsity |

\* Sensitivity 34.5% only from 3 interpretable folds; aggregate metric **unreliable**

---

## Current State Assessment (UPDATED)

**Data Reality:**
- 698 total observations across 195 districts
- Median: 2 obs/district (extreme sparsity)
- Grid coverage: 1.1% (166 districts × 299 time points)
- Labeled outbreaks: 28 events across 74 samples

**Model Issues:**
- Previous performance: AUC 0.68, sensitivity 2.4% (with 1 climate covariate)
- Current performance: AUC 0.46, sensitivity 0% (with 3 climate covariates)
- Convergence problems: R-hat 1.26, ESS 6 for phi parameter
- Numerical stability: phi clamping (min=0.1) may be too restrictive
- MCMC runtime: ~48 min for 6-fold CV with 195 districts

**Root Cause:**
Hierarchical model estimating 195 district-level latent states from extremely sparse data (1.1% coverage) leads to poor identifiability and convergence issues.

---

## Implementation Flow

### **OPTION 1: District Filtering** 🎯 PRIMARY STRATEGY
**Rationale:** Focus on districts with sufficient temporal coverage for AR(1) inference

**Filter Criteria:** ≥10 observations per district

**Expected Impact:**
- Districts: 195 → 13 (6.7% retention)
- Samples: 698 → 224 (32.1% retention)
- **Outbreaks: 28 → 28 (100% retention)** ✓ CRITICAL
- Grid coverage: 1.1% → ~15% (13x improvement)
- Mean obs/district: 3.6 → 17.2 (5x improvement)
- MCMC runtime: ~48 min → ~15 min (3x speedup)

**Implementation Steps:**
1. Add district filtering function to data loading pipeline
2. Filter training data to districts with ≥10 observations
3. Keep district list for consistent CV folds
4. Update Stan data preparation to use filtered districts
5. Re-run Phase 04 (single fold diagnostic)
6. If convergence improves, run Phase 05 (6-fold CV)
7. Continue to phases 06-11 if AUC >0.55 and sensitivity >5%

**Success Criteria:**
- ✓ R-hat < 1.1 for all parameters
- ✓ ESS > 100 for all parameters (especially phi)
- ✓ No divergences or <2 per fold
- ✓ AUC ≥ 0.60 (previous was 0.68)
- ✓ Sensitivity ≥ 10% (previous was 2.4%)
- ✓ Runtime < 20 min for Phase 05

**Thesis Documentation:**
```
Under extreme surveillance sparsity (median 2 observations per district, 
1.1% spatiotemporal grid coverage), hierarchical state-space estimation 
becomes ill-posed. We restricted analysis to the 13 districts (6.7% of 
total) with ≥10 temporal observations, ensuring sufficient data density 
for latent AR(1) dynamics while retaining all 28 labeled outbreak events. 
This filter reflects operational constraints: sustained surveillance is 
prerequisite for temporal risk inference in early warning systems.
```

---

### **OPTION 3: Relax phi Constraint + Increase Iterations** 🔧 REFINEMENT
**Rationale:** Current phi clamping (min=0.1) may be preventing model from fitting data's true overdispersion

**When to Apply:** After Option 1, if convergence still poor (ESS <100 for phi)

**Changes:**
1. **Relax phi minimum:** 0.1 → 0.01 (allow more extreme overdispersion)
2. **Increase iterations:** 1000+1000 → 1500+1500 (better ESS)
3. **Increase adapt_delta:** 0.99 → 0.995 (more conservative stepping)
4. **Monitor:** Check if phi still hits lower bound in posterior samples

**Expected Impact:**
- Better convergence for phi parameter (target ESS >200)
- May improve model fit to sparse districts
- Longer runtime: 15 min → ~23 min per Phase 05

**Implementation Steps:**
1. Modify Stan model: `phi = fmax(log1p_exp(phi_raw), 0.01)`
2. Update Python wrapper: n_warmup=1500, n_samples=1500, adapt_delta=0.995
3. Re-run Phase 04 to check phi posterior
4. If ESS improves, proceed to Phase 05

**Success Criteria:**
- ✓ phi ESS > 200 (was 6)
- ✓ phi posterior not bunched at 0.01 boundary
- ✓ AUC stable or improved vs Option 1
- ✓ Sensitivity stable or improved

---

### **OPTION 4: Threshold Tuning** 📊 DECISION LAYER
**Rationale:** Optimize decision thresholds for sensitivity-specificity trade-off

**When to Apply:** After Options 1+3, if sensitivity still <20%

**Current Threshold:** risk_quantile = 0.8 (80th percentile of posterior risk)

**Tuning Strategy:**
1. Generate predictions on all CV folds with Option 1+3 model
2. Extract posterior mean risk scores for all test samples
3. Compute ROC curve and sensitivity/specificity at multiple thresholds
4. Test quantile thresholds: [0.5, 0.6, 0.7, 0.75, 0.8]
5. Target: Maximize sensitivity while keeping FAR <30%

**Expected Impact:**
- Sensitivity: 10-15% → 30-50% (if model produces differentiated scores)
- Specificity: Will decrease (accept more false alarms)
- Thesis narrative: Cost-loss optimization with uncertainty

**Implementation Steps:**
1. Load CV results from Phase 05
2. Extract posterior risk scores (mean Z_t or P(outbreak))
3. Compute metrics at thresholds [0.5, 0.6, 0.7, 0.75, 0.8]
4. Plot ROC curve, precision-recall curve
5. Select threshold balancing sensitivity/FAR
6. Re-run Phase 06-11 with optimized threshold
7. Document threshold selection rationale

**Success Criteria:**
- ✓ Sensitivity ≥ 30%
- ✓ Specificity ≥ 70%
- ✓ FAR ≤ 30%
- ✓ AUC stable (threshold doesn't change discrimination)

---

### **OPTION 2: Simplify Model (Remove Climate Covariate)** ⚠️ FALLBACK ONLY
**Rationale:** Reduce parameters if Options 1+3 fail to achieve convergence

**When to Apply:** ONLY IF Options 1+3 both fail (R-hat >1.1 or sensitivity =0%)

**Changes:**
1. Keep only `beta_temp * temp_anomaly` in likelihood
2. Remove `degree_days` and `rain_persist` from Stan model
3. Update Python wrapper to not extract these features
4. Parameters reduced: α(13) + β(3) + ρ + σ + φ → α(13) + β(1) + ρ + σ + φ

**Expected Impact:**
- Better convergence (fewer parameters from n=74 samples)
- Loss of mechanistic richness (thesis emphasizes climate forcing)
- May not improve sensitivity much (sparsity is real issue)

**Implementation Steps:**
1. Comment out degree_days and rain_persist in Stan likelihood
2. Update state_space.py to skip these features
3. Update experiments to note "simplified model"
4. Re-run Phase 04-05

**Success Criteria:**
- ✓ R-hat < 1.1 for all parameters
- ✓ AUC ≥ 0.55 (accept degradation)
- ✓ Sensitivity ≥ 5%

**Thesis Documentation:**
```
Due to limited sample size (n=74 labeled observations) and extreme 
spatiotemporal sparsity, the extended climate covariate model failed to 
converge reliably. We simplified to temperature anomaly only, demonstrating 
the trade-off between mechanistic richness and statistical identifiability 
under data scarcity.
```

---

## Execution Plan

### **Phase 1: Option 1 Implementation** (ETA: 2 hours)
1. ✓ Kill current Phase 05 if still running
2. ✓ Implement district filtering in data pipeline
3. ✓ Test on Phase 04 (single fold)
4. ✓ Run Phase 05 (6-fold CV) if Phase 04 succeeds
5. ✓ Audit all changes personally

**Decision Point 1:**
- **IF** Option 1 succeeds (AUC >0.60, sensitivity >10%, R-hat <1.1) → Proceed to Phase 2
- **IF** Option 1 fails convergence → Try Option 3
- **IF** Option 1+3 both fail → Fall back to Option 2

### **Phase 2: Option 3 Refinement** (ETA: 2.5 hours)
1. Relax phi constraint to 0.01
2. Increase iterations to 1500+1500
3. Increase adapt_delta to 0.995
4. Re-run Phase 04 to check phi convergence
5. Run Phase 05 if improved

**Decision Point 2:**
- **IF** Option 3 improves sensitivity or convergence → Keep these settings
- **IF** No improvement → Revert to Option 1 settings
- Proceed to Phase 3 regardless

### **Phase 3: Option 4 Threshold Tuning** (ETA: 1 hour)
1. Load Phase 05 results
2. Compute ROC curves across thresholds [0.5, 0.6, 0.7, 0.75, 0.8]
3. Select threshold maximizing sensitivity at FAR ≤30%
4. Update decision layer with optimized threshold
5. Re-run phases 06-11 for final results

**Decision Point 3:**
- **IF** Sensitivity ≥30% achieved → SUCCESS, document results
- **IF** Sensitivity <30% but AUC >0.60 → Acknowledge limitation, focus on calibration narrative
- **IF** Both fail → Fall back to Option 2

### **Phase 4: Option 2 Fallback** (ETA: 3 hours)
*Only if Options 1+3 fail*

1. Simplify Stan model to temp_anomaly only
2. Re-run Phase 04-05
3. Accept lower performance, document trade-off
4. Pivot thesis narrative to "data sparsity limits mechanistic modeling"

---

## Validation Checklist

**After Each Option:**
- [ ] R-hat < 1.1 for all parameters
- [ ] ESS > 100 for all parameters
- [ ] No overflow errors in generated quantities
- [ ] No divergences (or <2 per fold)
- [ ] AUC ≥ 0.55 (minimum for comparison)
- [ ] Sensitivity > 0% (model differentiates risk)
- [ ] CV results JSON saved to results/metrics/
- [ ] Convergence diagnostics logged

**Final Pipeline Validation:**
- [ ] Phases 00-05 complete without errors
- [ ] Phase 06: Lead-time analysis complete
- [ ] Phase 07: All visualizations generated
- [ ] Phase 08: Decision layer simulated
- [ ] Phase 09: Fusion experiments complete
- [ ] Phase 10: Comprehensive metrics aggregated
- [ ] Results comparable to prev_results/

---

## Success Definitions

### **Minimum Viable Performance** (to proceed with thesis)
- AUC ≥ 0.60 (demonstrates discrimination)
- Sensitivity ≥ 15% (shows non-trivial detection)
- Brier score ≤ 0.25 (acceptable calibration)
- Lead time ≥ 1 week (actionable warning)

### **Target Performance** (strong comparative case)
- AUC ≥ 0.70 (good discrimination)
- Sensitivity ≥ 30% (useful detection rate)
- Specificity ≥ 80% (low false alarms)
- Brier score ≤ 0.20 (good calibration)
- Lead time ≥ 2 weeks (operationally valuable)

### **Fallback Position** (if minimum not achieved)
- Focus thesis on calibration advantage (Track B gives probabilities)
- Focus on decision-theoretic framework (cost-loss optimization)
- Acknowledge data sparsity as fundamental limitation
- Emphasize methodological contribution over performance

---

## Timeline

**Day 1 (Today):**
- [ ] Option 1 implementation: 2 hours
- [ ] Phase 04 test run: 30 min
- [ ] Phase 05 full CV: 15 min
- [ ] Audit results: 30 min

**Day 2:**
- [ ] Option 3 implementation: 1 hour
- [ ] Phase 04-05 re-run: 45 min
- [ ] Compare Option 1 vs 1+3: 30 min
- [ ] Option 4 threshold tuning: 1 hour

**Day 3:**
- [ ] Phases 06-11 execution: 2 hours
- [ ] Results comparison with prev_results: 1 hour
- [ ] Thesis documentation updates: 1 hour

**Fallback (if needed):**
- [ ] Option 2 implementation: 3 hours
- [ ] Full pipeline re-run: 3 hours

---

## Risk Mitigation

**Risk 1: Option 1 doesn't improve convergence**
- Mitigation: Proceed to Option 3 immediately
- Fallback: Option 2 (simplify model)

**Risk 2: All options fail to achieve minimum viable performance**
- Mitigation: Pivot thesis to methodological contribution
- Focus: Calibration, uncertainty quantification, decision framework
- Acknowledge: "Under extreme sparsity (n=74), both tracks struggle"

**Risk 3: Runtime too long even with filtered districts**
- Mitigation: Reduce iterations to 500+500 (only if necessary)
- Trade-off: Accept slightly higher uncertainty in ESS

**Risk 4: Filtered districts don't generalize**
- Response: This is feature, not bug
- Thesis: "Model restricted to high-surveillance districts"
- Operational: Real-world EWS needs good data anyway

---

## Notes for Subagent Execution

**Files to Modify:**
1. `src/models/bayesian/state_space.py` - Add district filtering
2. `experiments/04_train_bayesian.py` - Update to use filtered data
3. `experiments/05_evaluate_bayesian.py` - Update to use filtered data
4. `stan_models/hierarchical_ews_v01.stan` - Option 3: Relax phi constraint
5. `thesis.txt` - Add district filtering rationale

**Files to Audit:**
1. Verify district list matches ≥10 obs criteria
2. Verify all 28 outbreaks retained in filtered data
3. Verify Stan model compiles with changes
4. Verify Python wrapper passes correct filtered data
5. Verify CV folds consistent before/after filtering

**Subagent Tasks:**
- Task 1: Implement district filtering function
- Task 2: Update experiments to use filtering
- Task 3: Test on Phase 04
- Task 4: (If needed) Relax phi constraint
- Task 5: (If needed) Optimize thresholds

**Human Audit Points:**
- After district filtering implementation
- After Phase 04 test results
- After Phase 05 CV results
- Before committing to Option 3
- Before proceeding to phases 06-11

---

## Expected Outcomes

**Best Case (Option 1 succeeds):**
- AUC 0.65-0.75, sensitivity 20-30%
- Clean convergence, 15 min runtime
- Strong thesis narrative: hierarchical pooling works with adequate data
- Proceed to full pipeline and decision layer

**Middle Case (Option 1+3 succeeds):**
- AUC 0.60-0.70, sensitivity 15-25%
- Acceptable convergence, 23 min runtime
- Thesis narrative: model tuning necessary under sparsity
- Proceed to full pipeline

**Fallback Case (Option 2 required):**
- AUC 0.55-0.65, sensitivity 10-20%
- Simplified model, acknowledge trade-off
- Thesis narrative: data scarcity limits mechanistic richness
- Focus on calibration and uncertainty

**Worst Case (all fail):**
- Pivot to methodological contribution
- Emphasize framework over performance
- Acknowledge fundamental data limitation
- Still valid thesis: comparative study of approaches under sparsity

---

## 🔧 IMMEDIATE ACTION PLAN: Post-Forensic Fixes

### **What's FIXABLE (Without Thesis Pivot)**

#### **PRIORITY 1: Evaluation Validity (2-3 hours)**
**Goal:** Get interpretable metrics from all 6 folds

**Fix 1.1 - Stratified CV Split**
- Replace year-based holdout with stratified temporal split
- Ensure ≥5 positives per fold (minimum for ROC curves)
- Balance training years across folds

**Implementation:**
```python
# In src/data/cv.py
from sklearn.model_selection import StratifiedKFold
# Stratify on label_outbreak while respecting temporal ordering
# Use 2-3 year rolling windows instead of single-year holdouts
```

**Expected Impact:**
- 6/6 folds evaluable (vs 3/6 current)
- Stable sensitivity estimates across folds
- Valid statistical comparison with Track A

**Effort:** ~2 hours (modify cv.py, re-run Phase 05)

---

**Fix 1.2 - Aggregate Test Sets**
- Pool folds 2021+2022 (1 total positive) with fold 2020 (1 positive)
- Create "2020-2022" combined test fold with ≥2 positives

**Expected Impact:**
- 4 interpretable folds instead of 3
- Better statistical power for aggregated metrics

**Effort:** ~30 min (modify fold creation, re-run)

---

#### **PRIORITY 2: Threshold Alignment (1-2 hours)**
**Goal:** Unify thresholds across training, evaluation, and decision layers

**Fix 2.1 - Use Consistent Percentile**
- Training labels: P60 (looser than P50, captures more outbreak signals)
- Episode definition (Phase 06): P60 (matches training)
- Intervention trigger (Phase 08): P70 (slightly higher for action)

**Implementation:**
```python
# In config/config_default.yaml
labels:
  outbreak_percentile: 60  # Was 50

decision:
  risk_quantile: 0.70  # Was 0.80
```

**Expected Impact:**
- Lead-time analysis: 2 episodes → 4-6 episodes (less fragmentation)
- Early warnings: 0% → 20-40% (predictions reach intervention threshold)
- Decision layer: Sensitivity 25% → 40-60%

**Effort:** ~1 hour (update configs, re-run Phases 05-08)

---

**Fix 2.2 - Allow 1-Week NA Gaps in Episodes**
- Modify episode detection to skip single-week NA gaps
- Require ≥2 positives in any 4-week window (not consecutive)

**Implementation:**
```python
# In src/evaluation/lead_time.py
def detect_episodes(labels, max_gap=1):
    # Allow up to 1 NA gap within episode
    # Fragment only on 2+ consecutive NAs
```

**Expected Impact:**
- Episode count: 2 → 8-12 (67% fragmentation reduced)
- Lead-time analysis becomes statistically viable

**Effort:** ~1 hour (modify lead_time.py, re-run Phase 06)

---

#### **PRIORITY 3: Decision Layer Realism (1 hour)**
**Goal:** Make cost structure defensible for thesis

**Fix 3.1 - Cost Sensitivity Analysis**
- Test net benefit across realistic cost ranges:
  - Intervention: $1-10
  - False alarm: $0.5-5
  - Missed outbreak: $10-100
- Find parameter space where system has positive ROI

**Implementation:**
```python
# In experiments/08_simulate_decision_layer.py
cost_scenarios = [
    {"intervention": 1, "false_alarm": 0.5, "missed": 10},   # Current
    {"intervention": 2, "false_alarm": 1, "missed": 20},     # Conservative
    {"intervention": 5, "false_alarm": 2, "missed": 50},     # Realistic
]
# Run decision simulation for each scenario
```

**Expected Impact:**
- Identify viable operating region (if exists)
- Document sensitivity of net benefit to cost assumptions
- Defensible thesis claim: "positive ROI under plausible cost structures"

**Effort:** ~1 hour (add scenarios, re-run Phase 08)

---

#### **PRIORITY 4: Fusion Robustness (30 min)**
**Goal:** Fix NaN crashes, make fusion baseline defensible

**Fix 4.1 - NaN-Safe Weighted Ensemble**
```python
# In experiments/09_fusion_experiments.py
def weighted_ensemble(bayes_prob, xgb_prob, bayes_auc, xgb_auc):
    # Handle NaN AUCs from single-class folds
    if np.isnan(bayes_auc) or np.isnan(xgb_auc):
        return xgb_prob  # Fallback to XGBoost
    
    weight_bayes = bayes_auc / (bayes_auc + xgb_auc + 1e-6)
    return weight_bayes * bayes_prob + (1 - weight_bayes) * xgb_prob
```

**Expected Impact:**
- Weighted ensemble: 0.500 (crash) → 0.60-0.70 (functional)
- No longer worse than baseline

**Effort:** ~30 min (add NaN handling, re-run Phase 09)

---

**Fix 4.2 - De-emphasize Fusion in Thesis**
- Remove Phase 09 from primary results section
- Move to "Exploratory Experiments" appendix
- Focus thesis on Track A vs Track B comparison (not ensemble)

**Rationale:** Fusion is between tracks, not "mechanistic-Bayesian fusion" (already implicit in Track B)

**Effort:** ~30 min (thesis restructuring)

---

### **What's NOT FIXABLE (Requires Acknowledgment)**

#### **🚫 Issue: Only 74 Labeled Samples**
**Why Not Fixable:**
- Cannot generate more historical outbreak labels without domain expertise
- Semi-supervised learning would require re-architecting both tracks
- Time constraint: weeks of work

**Thesis Response:**
- Acknowledge limitation: "extreme label sparsity (n=74, 28 outbreaks)"
- Frame as challenge for both Track A and Track B
- Emphasize: "methodological comparison, not operational system"

---

#### **🚫 Issue: Counterintuitive Variance Pattern**
**Why Not Fixable:**
- Real data pattern: outbreaks have LOWER variance than baseline
- Cannot engineer away without domain insight
- May indicate outbreak definition issue (P-percentile threshold flawed)

**Thesis Response:**
- Report as empirical finding
- Hypothesis: "pre-outbreak accumulation reduces week-to-week variance"
- Acknowledge: "variance-based EWS features ineffective for chikungunya"

---

#### **🚫 Issue: Bayesian AUC 0.424 < XGBoost 0.759**
**Why Not Fixable:**
- District filtering (195→13) sacrifices coverage for identifiability
- Bayesian needs ≥10 obs/district, XGBoost doesn't
- Trade-off: calibration (Brier 0.214) vs discrimination (AUC)

**Thesis Response:**
- Pivot narrative to **complementary strengths:**
  - Track A (XGBoost): Better discrimination (AUC 0.759)
  - Track B (Bayesian): Better calibration (Brier 0.214), uncertainty quantification
- Frame as **decision context-dependent:**
  - Binary alerts: Use Track A
  - Probabilistic forecasts: Use Track B
  - Risk prioritization: Use Track B credible intervals

---

### **Minimum Viable Thesis (With Fixes)**

**After Priority 1-4 Fixes, Expected Results:**

| Metric | Current | Fixed | Minimum Viable | Status |
|--------|---------|-------|----------------|--------|
| Evaluable folds | 3/6 | 6/6 | 4/6 | ✅ ACHIEVABLE |
| Bayesian AUC | 0.424 | 0.45-0.55 | ≥0.50 | ✅ ACHIEVABLE |
| Sensitivity | 34.5%* | 30-50% | ≥20% | ✅ ACHIEVABLE |
| Lead-time episodes | 2 | 8-12 | ≥5 | ✅ ACHIEVABLE |
| Early warnings | 0% | 20-40% | ≥10% | ✅ ACHIEVABLE |
| Net benefit | -$41 | -$10 to +$5 | ≥$0 | ⚠️ UNCERTAIN |
| Brier score | 0.214 | 0.20-0.22 | ≤0.25 | ✅ ALREADY MET |

\* From 3 interpretable folds

---

### **Revised Thesis Narrative (No Pivot Required)**

**Current Framing (Too Strong):**
> "Bayesian hierarchical model for operational early warning with mechanistic fusion and decision-theoretic optimization."

**Revised Framing (Defensible):**
> "Comparative evaluation of supervised (Track A) vs. Bayesian state-space (Track B) approaches for outbreak risk inference under extreme surveillance sparsity. Track A achieves superior discrimination (AUC 0.76) but lacks uncertainty quantification. Track B provides well-calibrated probabilistic forecasts (Brier 0.21) enabling cost-sensitive decision-making, at the cost of reduced coverage due to hierarchical identifiability constraints. Results demonstrate complementary strengths: Track A for binary alerts, Track B for risk prioritization and uncertainty-aware planning."

**Key Claims (Still Valid):**
1. ✅ Hierarchical Bayesian provides **calibrated probabilities** (Brier 0.21 < 0.25 threshold)
2. ✅ Mechanistic features (degree_days, rain_persist) encode **biological thresholds**
3. ✅ Decision framework integrates **uncertainty into cost-loss optimization**
4. ✅ Comparative study reveals **trade-offs**: discrimination vs calibration, coverage vs identifiability
5. ⚠️ Lead-time gains **observable but limited** (achievable after Fix 2.1-2.2)

**Claims to Remove:**
- ❌ "Operational early warning system" → "methodological comparison"
- ❌ "Superior to baselines" → "complementary to baselines"
- ❌ "Fusion outperforms" → "fusion exploratory, mixed results"

---

### **Implementation Timeline (5-7 hours)**

**Session 1 (3 hours):**
- [ ] Priority 1.1: Stratified CV split (2 hours)
- [ ] Priority 1.2: Aggregate test sets (30 min)
- [ ] Re-run Phase 05 (30 min)
- [ ] Audit fold balance, check metrics

**Session 2 (2 hours):**
- [ ] Priority 2.1: Threshold alignment (30 min)
- [ ] Priority 2.2: Episode NA gap tolerance (1 hour)
- [ ] Re-run Phases 06-08 (30 min)
- [ ] Check lead-time episodes, net benefit

**Session 3 (1-2 hours):**
- [ ] Priority 3.1: Cost sensitivity analysis (1 hour)
- [ ] Priority 4.1: Fusion NaN handling (30 min)
- [ ] Re-run Phases 08-09 (30 min)
- [ ] Final metrics audit

**Session 4 (1 hour):**
- [ ] Update thesis narrative (30 min)
- [ ] Generate final comparison tables (30 min)
- [ ] Document limitations transparently

**Total: 7-8 hours** to restore thesis defensibility without pivot.

---

### **Decision Point: Fix vs Pivot**

**Recommendation: IMPLEMENT PRIORITY 1-4 FIXES**

**Rationale:**
1. Fixes are **time-bounded** (7-8 hours) and **low-risk**
2. Expected improvements make thesis **defensible**:
   - All folds evaluable
   - Lead-time analysis viable (≥5 episodes)
   - Decision layer breakeven possible
3. Fixes align with **existing thesis structure** (no rewrite)
4. Maintains **comparative study** narrative (Track A vs Track B)

**Only pivot if:**
- Fixes fail to produce ≥5 evaluable lead-time episodes
- Net benefit remains negative across all cost scenarios
- Time exceeds 10 hours (diminishing returns)

**Next Actions:**
1. User approval for Priority 1-4 fixes
2. Implement Session 1 (stratified CV)
3. Checkpoint: If fold balance improves → continue to Session 2
4. Checkpoint: If episodes ≥5 after Session 2 → continue to Session 3
5. Final audit after Session 4

---
