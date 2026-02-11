"""
Lead-Time Analysis Module for Chikungunya Early Warning System

Phase 7 - Workstream A: Lead-Time Analysis

PURPOSE:
--------
Validates Claim 2: "Latent risk inference captures outbreak escalation earlier 
than binary classifiers."

This module computes the lead-time advantage of:
1. Bayesian hierarchical state-space model (latent risk Z_t)
2. XGBoost binary classifier (predicted probability p_t)

DEFINITIONS (from spec 02_lead_time_analysis_spec_v2.md):
---------------------------------------------------------
1. Outbreak Week (Ground Truth):
    - Week t is an outbreak week if observed cases Y_t > τ_i
    - τ_i = outbreak percentile of historical cases for district i (config)
    - CRITICAL: Computed on TRAINING folds only (no leakage)

2. Bayesian Risk Crossing:
    - Week t is a Bayesian crossing if Z_t mean > ξ_i
    - ξ_i = Bayesian percentile of Z_t distribution (config, training-based)

3. XGBoost Trigger:
    - Week t is an XGBoost trigger if p_t > configured probability threshold

4. Lead Time:
   - L_B = t* - t_B (Bayesian lead time)
   - L_X = t* - t_X (XGBoost lead time)  
   - ΔL = L_X - L_B (positive = Bayesian earlier)

EDGE CASES:
-----------
- Never-warned: Lead time = -1 (sentinel value)
- Both never-warn: Exclude from comparison, count separately

NO DATA LEAKAGE:
----------------
- Outbreak thresholds computed ONLY on training data
- Bayesian/XGBoost predictions use only past/current data
- No future information in any threshold calculation

Reference: docs/Version 2/02_lead_time_analysis_spec_v2.md
Author: Chikungunya EWS Research Project
Date: February 2026
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.common.thresholds import enforce_minimum_threshold


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Sentinel value for "never warned"
NEVER_WARNED_SENTINEL = -1


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OutbreakEpisode:
    """
    Represents a single outbreak episode (district, year, outbreak week).
    
    Epidemiological Context:
    ------------------------
    An outbreak episode is a period where disease incidence exceeds the 
    historical baseline for that district. This is the "ground truth" event
    we want to predict in advance.
    
    Attributes:
        state: Indian state name
        district: District name within state
        year: Calendar year of outbreak
        fold: CV fold name (e.g., "fold_2018")
        first_outbreak_week: Week number when cases first exceeded threshold
        peak_week: Week with maximum case count
        peak_cases: Maximum case count during episode
        total_outbreak_weeks: Number of weeks above threshold
        outbreak_threshold: District-specific threshold (config percentile, training-only)
    """
    state: str
    district: str
    year: int
    fold: str
    first_outbreak_week: int
    peak_week: int
    peak_cases: float
    total_outbreak_weeks: int
    outbreak_threshold: float


@dataclass
class LeadTimeResult:
    """
    Lead-time computation result for a single outbreak episode.
    
    Interpretation Guide:
    ---------------------
    - lead_time_bayesian > 0: Bayesian warned BEFORE outbreak (good)
    - lead_time_bayesian = 0: Bayesian warned AT outbreak week (reactive)
    - lead_time_bayesian = -1: Bayesian NEVER warned (missed)
    
    - differential_lead > 0: Bayesian warned earlier than XGBoost
    - differential_lead < 0: XGBoost warned earlier than Bayesian
    - differential_lead = None: Cannot compare (one or both never warned)
    
    Attributes:
        episode: The outbreak episode this result refers to
        bayesian_trigger_week: Week when Bayesian risk crossed threshold
        xgboost_trigger_week: Week when XGBoost probability crossed threshold
        lead_time_bayesian: Weeks of advance warning from Bayesian
        lead_time_xgboost: Weeks of advance warning from XGBoost
        differential_lead: L_X - L_B (positive = Bayesian earlier)
        bayesian_threshold: District-specific Bayesian threshold (config percentile)
        xgboost_threshold: XGBoost probability threshold used
    """
    episode: OutbreakEpisode
    bayesian_trigger_week: Optional[int]
    xgboost_trigger_week: Optional[int]
    lead_time_bayesian: int  # -1 if never warned
    lead_time_xgboost: int  # -1 if never warned
    differential_lead: Optional[float]  # None if both never warned
    bayesian_threshold: float
    xgboost_threshold: float


@dataclass
class LeadTimeSummary:
    """
    Aggregated lead-time statistics.
    
    Thesis Reporting:
    -----------------
    This summary provides the key metrics for Table X in the thesis:
    - Median lead time (more robust than mean for skewed distributions)
    - IQR (interquartile range) shows spread
    - % early warned: fraction of outbreaks with lead time >= 1 week
    - % never warned: fraction of outbreaks completely missed
    
    Attributes:
        model_name: "bayesian" or "xgboost"
        n_episodes: Total number of outbreak episodes analyzed
        n_warned: Number of episodes where model triggered before outbreak
        n_never_warned: Number of episodes where model never triggered
        median_lead_time: Median lead time (weeks), excluding never-warned
        mean_lead_time: Mean lead time (weeks), excluding never-warned
        std_lead_time: Standard deviation of lead time
        iqr_lower: 25th percentile of lead times
        iqr_upper: Upper quartile of lead times
        pct_early_warned: % of episodes with lead time >= 1 week
        pct_never_warned: % of episodes with lead time = -1
        lead_times: Raw list of lead times for distribution analysis
    """
    model_name: str
    n_episodes: int
    n_warned: int
    n_never_warned: int
    median_lead_time: Optional[float]
    mean_lead_time: Optional[float]
    std_lead_time: Optional[float]
    iqr_lower: Optional[float]
    iqr_upper: Optional[float]
    pct_early_warned: float  # >= 1 week advance warning
    pct_never_warned: float
    lead_times: List[int] = field(default_factory=list)


# =============================================================================
# THRESHOLD COMPUTATION (TRAINING DATA ONLY)
# =============================================================================

def compute_outbreak_thresholds_per_district(
    train_df: pd.DataFrame,
    percentile: int,
    case_col: str = 'cases',
    group_cols: List[str] = ['state', 'district']
) -> Dict[Tuple[str, str], float]:
    """
    Compute district-specific outbreak thresholds from TRAINING data only.
    
    CRITICAL FOR AVOIDING DATA LEAKAGE:
    -----------------------------------
    This function MUST be called with training data only. The thresholds
    computed here define what constitutes an "outbreak" in the test set.
    Using test data would leak future information into the definition.
    
    Epidemiological Rationale:
    --------------------------
    The configured percentile represents the boundary between "normal" endemic
    transmission and "elevated" outbreak-level activity. This is a standard
    approach in disease surveillance systems.
    
    Args:
        train_df: Training data DataFrame (years BEFORE test year)
        percentile: Percentile threshold (config-driven)
        case_col: Column containing case counts
        group_cols: Columns defining geographic units
        
    Returns:
        Dictionary mapping (state, district) -> threshold value
        
    Example:
        >>> thresholds = compute_outbreak_thresholds_per_district(train_df, percentile=percentile_value)
        >>> thresholds[('Tamil Nadu', 'Chennai')]
        125.0  # Outbreak if cases > 125 for Chennai
    """
    if train_df.empty:
        warnings.warn("Empty training DataFrame, returning empty thresholds")
        return {}
    
    thresholds = {}
    
    for group_key, group_df in train_df.groupby(group_cols):
        # Get valid (non-null, non-zero for some contexts) case counts
        cases = group_df[case_col].dropna()
        
        if len(cases) < 5:
            # Insufficient data for reliable percentile
            warnings.warn(f"Insufficient data for {group_key}: {len(cases)} samples")
            continue
        
        # Compute percentile threshold
        # Note: We use ALL training cases (including zeros) for the percentile
        # This gives a realistic baseline for that district
        threshold = np.percentile(cases, percentile)
        
        # Enforce minimum threshold to avoid classifying <1 case as outbreak
        threshold = enforce_minimum_threshold(threshold)
        
        thresholds[group_key] = threshold
    
    if len(thresholds) == 0:
        raise ValueError("No valid outbreak thresholds computed. Training data has <5 cases per district or insufficient history.")
    
    return thresholds


def compute_bayesian_thresholds_per_district(
    predictions_df: pd.DataFrame,
    percentile: int,
    z_col: str = 'z_mean',
    group_cols: List[str] = ['state', 'district']
) -> Dict[Tuple[str, str], float]:
    """
    Compute district-specific Bayesian risk thresholds from training predictions.
    
    Rationale for Bayesian Percentile:
    ----------------------------------
    The configured percentile sets how early the latent risk signal should
    trigger relative to observed outbreaks.
    
    Why district-specific thresholds:
    ---------------------------------
    Districts have different baseline risk levels due to population density,
    climate, and healthcare reporting. A single global threshold would miss
    outbreaks in low-baseline districts or over-alarm in high-baseline ones.
    
    Args:
        predictions_df: DataFrame with Bayesian posterior summaries
        percentile: Percentile for risk threshold (config-driven)
        z_col: Column containing Z_t posterior mean
        group_cols: Geographic grouping columns
        
    Returns:
        Dictionary mapping (state, district) -> Bayesian threshold
    """
    if predictions_df.empty:
        warnings.warn("Empty predictions DataFrame")
        return {}
    
    thresholds = {}
    
    for group_key, group_df in predictions_df.groupby(group_cols):
        z_values = group_df[z_col].dropna()
        
        if len(z_values) < 3:
            warnings.warn(f"Insufficient predictions for {group_key}")
            continue
        
        threshold = np.percentile(z_values, percentile)
        thresholds[group_key] = threshold
    
    return thresholds


# =============================================================================
# OUTBREAK EPISODE IDENTIFICATION
# =============================================================================

def identify_outbreak_episodes(
    test_df: pd.DataFrame,
    outbreak_thresholds: Dict[Tuple[str, str], float],
    fold_name: str,
    case_col: str = 'cases',
    group_cols: List[str] = ['state', 'district']
) -> List[OutbreakEpisode]:
    """
    Identify outbreak episodes in test data using training-derived thresholds.
    
    Definition of Episode:
    ----------------------
    An outbreak episode is defined by:
    - District + Year combination
    - At least one week where cases > district threshold
    - The "first outbreak week" is when cases FIRST cross threshold
    
    This is NOT cumulative incidence—we track when the outbreak STARTS.
    
    Args:
        test_df: Test fold DataFrame
        outbreak_thresholds: Thresholds from training data
        fold_name: Name of the CV fold (e.g., "fold_2018")
        case_col: Case count column
        group_cols: Geographic grouping columns
        
    Returns:
        List of OutbreakEpisode objects
    """
    episodes = []
    
    for (state, district), district_df in test_df.groupby(group_cols):
        district_key = (state, district)
        
        # Skip if no threshold (insufficient training data)
        if district_key not in outbreak_thresholds:
            continue
        
        threshold = outbreak_thresholds[district_key]
        
        # Process each year in the test set
        for year, year_df in district_df.groupby('year'):
            year_df = year_df.sort_values('week')
            
            # Find weeks exceeding threshold
            outbreak_mask = year_df[case_col] > threshold
            outbreak_weeks = year_df.loc[outbreak_mask, 'week'].values
            
            if len(outbreak_weeks) == 0:
                # No outbreak in this district-year
                continue
            
            # Compute episode statistics
            first_week = int(outbreak_weeks.min())
            peak_idx = year_df[case_col].idxmax()
            peak_week = int(year_df.loc[peak_idx, 'week'])
            peak_cases = float(year_df.loc[peak_idx, case_col])
            
            episode = OutbreakEpisode(
                state=state,
                district=district,
                year=int(year),
                fold=fold_name,
                first_outbreak_week=first_week,
                peak_week=peak_week,
                peak_cases=peak_cases,
                total_outbreak_weeks=len(outbreak_weeks),
                outbreak_threshold=threshold
            )
            episodes.append(episode)
    
    return episodes


# =============================================================================
# TRIGGER IDENTIFICATION
# =============================================================================

def find_first_trigger_week(
    signal_series: pd.Series,
    week_series: pd.Series,
    threshold: float,
    max_week: int
) -> Optional[int]:
    """
    Find the first week where signal crosses threshold (at or before max_week).
    
    Look-Back Logic:
    ----------------
    We search for crossings UP TO AND INCLUDING the outbreak week.
    A trigger at the outbreak week itself means lead_time = 0 (reactive).
    
    Args:
        signal_series: Series of signal values (Z_t or p_t)
        week_series: Corresponding week numbers
        threshold: Threshold to exceed
        max_week: Maximum week to consider (usually first_outbreak_week)
        
    Returns:
        First week where signal > threshold, or None if never triggers
    """
    # Handle empty input
    if len(signal_series) == 0 or len(week_series) == 0:
        return None
    
    # Create aligned DataFrame
    df = pd.DataFrame({
        'week': week_series.values,
        'signal': signal_series.values
    }).sort_values('week')
    
    # Filter to weeks up to and including max_week
    df = df[df['week'] <= max_week]
    
    # Find first crossing
    crossing = df[df['signal'] > threshold]
    
    if len(crossing) == 0:
        return None
    
    return int(crossing.iloc[0]['week'])


# =============================================================================
# LEAD-TIME COMPUTATION (CORE LOGIC)
# =============================================================================

def compute_lead_time_for_episode(
    episode: OutbreakEpisode,
    bayesian_df: pd.DataFrame,
    xgboost_df: pd.DataFrame,
    bayesian_threshold: float,
    xgboost_threshold: float,
    z_col: str = 'z_mean',
    prob_col: str = 'prob'
) -> LeadTimeResult:
    """
    Compute lead times for a single outbreak episode.
    
    Algorithm:
    ----------
    1. Filter predictions to the specific district-year
    2. Find first Bayesian trigger: Z_t > bayesian_threshold
    3. Find first XGBoost trigger: p_t > xgboost_threshold
    4. Compute lead times: t_outbreak - t_trigger
    5. Handle edge cases (never-warned)
    
    Edge Cases (from spec Section 2.4):
    -----------------------------------
    - If model never triggers before outbreak: lead_time = -1 (sentinel)
    - This is tracked separately in summaries
    
    Args:
        episode: The outbreak episode to analyze
        bayesian_df: Bayesian predictions with columns [state, district, year, week, z_mean]
        xgboost_df: XGBoost predictions with columns [state, district, year, week, prob]
        bayesian_threshold: District-specific Bayesian threshold
        xgboost_threshold: XGBoost probability threshold
        z_col: Column name for Bayesian Z_t mean
        prob_col: Column name for XGBoost probability
        
    Returns:
        LeadTimeResult with computed lead times
    """
    t_outbreak = episode.first_outbreak_week
    
    # Filter Bayesian predictions for this district-year
    bayesian_subset = bayesian_df[
        (bayesian_df['state'] == episode.state) &
        (bayesian_df['district'] == episode.district) &
        (bayesian_df['year'] == episode.year)
    ]
    
    # Filter XGBoost predictions for this district-year
    xgboost_subset = xgboost_df[
        (xgboost_df['state'] == episode.state) &
        (xgboost_df['district'] == episode.district) &
        (xgboost_df['year'] == episode.year)
    ]
    
    # Find Bayesian trigger week
    if len(bayesian_subset) > 0 and z_col in bayesian_subset.columns:
        t_bayesian = find_first_trigger_week(
            bayesian_subset[z_col],
            bayesian_subset['week'],
            bayesian_threshold,
            t_outbreak
        )
    else:
        t_bayesian = None
    
    # Find XGBoost trigger week
    if len(xgboost_subset) > 0 and prob_col in xgboost_subset.columns:
        t_xgboost = find_first_trigger_week(
            xgboost_subset[prob_col],
            xgboost_subset['week'],
            xgboost_threshold,
            t_outbreak
        )
    else:
        t_xgboost = None
    
    # Compute lead times
    if t_bayesian is not None:
        lead_bayesian = t_outbreak - t_bayesian
    else:
        lead_bayesian = NEVER_WARNED_SENTINEL  # -1
    
    if t_xgboost is not None:
        lead_xgboost = t_outbreak - t_xgboost
    else:
        lead_xgboost = NEVER_WARNED_SENTINEL  # -1
    
    # Compute differential lead time
    # Only meaningful if both models triggered
    if lead_bayesian != NEVER_WARNED_SENTINEL and lead_xgboost != NEVER_WARNED_SENTINEL:
        differential = lead_xgboost - lead_bayesian  # positive = Bayesian earlier
    else:
        differential = None
    
    return LeadTimeResult(
        episode=episode,
        bayesian_trigger_week=t_bayesian,
        xgboost_trigger_week=t_xgboost,
        lead_time_bayesian=lead_bayesian,
        lead_time_xgboost=lead_xgboost,
        differential_lead=differential,
        bayesian_threshold=bayesian_threshold,
        xgboost_threshold=xgboost_threshold
    )


def compute_lead_times_all_episodes(
    episodes: List[OutbreakEpisode],
    bayesian_df: pd.DataFrame,
    xgboost_df: pd.DataFrame,
    bayesian_thresholds: Dict[Tuple[str, str], float],
    xgboost_threshold: float,
    z_col: str = 'z_mean',
    prob_col: str = 'prob'
) -> List[LeadTimeResult]:
    """
    Compute lead times for all outbreak episodes.
    
    Args:
        episodes: List of outbreak episodes to analyze
        bayesian_df: All Bayesian predictions
        xgboost_df: All XGBoost predictions
        bayesian_thresholds: District-specific Bayesian thresholds
        xgboost_threshold: XGBoost probability threshold
        z_col: Bayesian signal column
        prob_col: XGBoost probability column
        
    Returns:
        List of LeadTimeResult objects
    """
    results = []
    
    for episode in episodes:
        district_key = (episode.state, episode.district)
        
        # Get district-specific Bayesian threshold
        if district_key in bayesian_thresholds:
            bayes_thresh = bayesian_thresholds[district_key]
        else:
            # Fallback: use global threshold if district not in thresholds
            all_thresholds = list(bayesian_thresholds.values())
            if all_thresholds:
                bayes_thresh = np.median(all_thresholds)
                warnings.warn(f"No Bayesian threshold for {district_key}, using global median")
            else:
                # Cannot compute without threshold
                continue
        
        result = compute_lead_time_for_episode(
            episode=episode,
            bayesian_df=bayesian_df,
            xgboost_df=xgboost_df,
            bayesian_threshold=bayes_thresh,
            xgboost_threshold=xgboost_threshold,
            z_col=z_col,
            prob_col=prob_col
        )
        results.append(result)
    
    return results


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def summarize_lead_times(
    results: List[LeadTimeResult],
    model: str  # 'bayesian' or 'xgboost'
) -> LeadTimeSummary:
    """
    Compute aggregated statistics for lead times.
    
    Statistical Approach:
    ---------------------
    - Median is preferred over mean (robust to outliers)
    - IQR captures spread without sensitivity to extremes
    - Percentage metrics are clinically interpretable
    
    Args:
        results: List of LeadTimeResult objects
        model: Which model to summarize ('bayesian' or 'xgboost')
        
    Returns:
        LeadTimeSummary with aggregated statistics
    """
    if not results:
        return LeadTimeSummary(
            model_name=model,
            n_episodes=0,
            n_warned=0,
            n_never_warned=0,
            median_lead_time=None,
            mean_lead_time=None,
            std_lead_time=None,
            iqr_lower=None,
            iqr_upper=None,
            pct_early_warned=0.0,
            pct_never_warned=100.0,
            lead_times=[]
        )
    
    # Extract lead times for specified model
    if model == 'bayesian':
        lead_times = [r.lead_time_bayesian for r in results]
    elif model == 'xgboost':
        lead_times = [r.lead_time_xgboost for r in results]
    else:
        raise ValueError(f"Unknown model: {model}. Use 'bayesian' or 'xgboost'.")
    
    n_total = len(lead_times)
    n_never = sum(1 for lt in lead_times if lt == NEVER_WARNED_SENTINEL)
    n_warned = n_total - n_never
    
    # Filter out never-warned for statistics
    valid_leads = [lt for lt in lead_times if lt != NEVER_WARNED_SENTINEL]
    
    if valid_leads:
        valid_arr = np.array(valid_leads)
        median_lt = float(np.median(valid_arr))
        mean_lt = float(np.mean(valid_arr))
        std_lt = float(np.std(valid_arr))
        iqr_lower = float(np.percentile(valid_arr, 25))
        iqr_upper = float(np.percentile(valid_arr, 75))  # Use percentile consistently
        # Early warning = lead time >= 1 week
        n_early = sum(1 for lt in valid_leads if lt >= 1)
        pct_early = 100.0 * n_early / n_total
    else:
        median_lt = mean_lt = std_lt = iqr_lower = iqr_upper = None
        pct_early = 0.0
    
    pct_never = 100.0 * n_never / n_total
    
    return LeadTimeSummary(
        model_name=model,
        n_episodes=n_total,
        n_warned=n_warned,
        n_never_warned=n_never,
        median_lead_time=median_lt,
        mean_lead_time=mean_lt,
        std_lead_time=std_lt,
        iqr_lower=iqr_lower,
        iqr_upper=iqr_upper,
        pct_early_warned=pct_early,
        pct_never_warned=pct_never,
        lead_times=lead_times
    )


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def results_to_dataframe(results: List[LeadTimeResult]) -> pd.DataFrame:
    """
    Convert lead-time results to a flat DataFrame for CSV export.
    
    Output Columns:
    ---------------
    - state, district, year, fold: Episode identifiers
    - first_outbreak_week: Ground truth outbreak onset
    - peak_cases: Maximum cases during episode
    - bayesian_trigger_week, xgboost_trigger_week: When each model triggered
    - lead_time_bayesian, lead_time_xgboost: Lead times (-1 = never warned)
    - differential_lead: L_X - L_B (positive = Bayesian earlier)
    - bayesian_threshold, xgboost_threshold: Thresholds used
    
    Args:
        results: List of LeadTimeResult objects
        
    Returns:
        DataFrame suitable for CSV export
    """
    rows = []
    for r in results:
        row = {
            'state': r.episode.state,
            'district': r.episode.district,
            'year': r.episode.year,
            'fold': r.episode.fold,
            'first_outbreak_week': r.episode.first_outbreak_week,
            'peak_week': r.episode.peak_week,
            'peak_cases': r.episode.peak_cases,
            'total_outbreak_weeks': r.episode.total_outbreak_weeks,
            'outbreak_threshold': r.episode.outbreak_threshold,
            'bayesian_trigger_week': r.bayesian_trigger_week,
            'xgboost_trigger_week': r.xgboost_trigger_week,
            'lead_time_bayesian': r.lead_time_bayesian,
            'lead_time_xgboost': r.lead_time_xgboost,
            'differential_lead': r.differential_lead,
            'bayesian_threshold': r.bayesian_threshold,
            'xgboost_threshold': r.xgboost_threshold
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def summary_to_dataframe(
    bayesian_summary: LeadTimeSummary,
    xgboost_summary: LeadTimeSummary
) -> pd.DataFrame:
    """
    Create summary comparison table for thesis reporting.
    
    This produces the Table format from spec Section 3.1:
    
    | Metric                    | Bayesian | XGBoost | Interpretation |
    |---------------------------|----------|---------|----------------|
    | Median lead time (weeks)  | X.X      | Y.Y     | ...            |
    
    Args:
        bayesian_summary: Bayesian model summary
        xgboost_summary: XGBoost model summary
        
    Returns:
        DataFrame with comparison metrics
    """
    def format_val(v: Optional[float], decimals: int = 2) -> str:
        """Format value with handling for None."""
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f}"
    
    def format_iqr(lower: Optional[float], upper: Optional[float]) -> str:
        """Format IQR range."""
        if lower is None or upper is None:
            return "N/A"
        return f"[{lower:.1f}, {upper:.1f}]"
    
    metrics = [
        {
            'Metric': 'N episodes',
            'Bayesian': bayesian_summary.n_episodes,
            'XGBoost': xgboost_summary.n_episodes,
            'Interpretation': 'Total outbreak episodes analyzed'
        },
        {
            'Metric': 'N warned',
            'Bayesian': bayesian_summary.n_warned,
            'XGBoost': xgboost_summary.n_warned,
            'Interpretation': 'Episodes where model triggered'
        },
        {
            'Metric': 'Median lead time (weeks)',
            'Bayesian': format_val(bayesian_summary.median_lead_time, 1),
            'XGBoost': format_val(xgboost_summary.median_lead_time, 1),
            'Interpretation': 'Median advance warning'
        },
        {
            'Metric': 'Mean lead time (weeks)',
            'Bayesian': format_val(bayesian_summary.mean_lead_time, 2),
            'XGBoost': format_val(xgboost_summary.mean_lead_time, 2),
            'Interpretation': 'Average advance warning'
        },
        {
            'Metric': 'IQR',
            'Bayesian': format_iqr(bayesian_summary.iqr_lower, bayesian_summary.iqr_upper),
            'XGBoost': format_iqr(xgboost_summary.iqr_lower, xgboost_summary.iqr_upper),
            'Interpretation': 'Interquartile range of lead times'
        },
        {
            'Metric': '% early warned (≥1 week)',
            'Bayesian': f"{bayesian_summary.pct_early_warned:.1f}%",
            'XGBoost': f"{xgboost_summary.pct_early_warned:.1f}%",
            'Interpretation': 'Fraction with actionable warning'
        },
        {
            'Metric': '% never warned',
            'Bayesian': f"{bayesian_summary.pct_never_warned:.1f}%",
            'XGBoost': f"{xgboost_summary.pct_never_warned:.1f}%",
            'Interpretation': 'Fraction of missed outbreaks'
        }
    ]
    
    return pd.DataFrame(metrics)


# =============================================================================
# SANITY CHECKS & VALIDATION
# =============================================================================

def validate_no_leakage(
    train_years: List[int],
    test_year: int,
    outbreak_thresholds: Dict[Tuple[str, str], float]
) -> bool:
    """
    Verify that outbreak thresholds were computed without data leakage.
    
    Checks:
    -------
    1. Test year is NOT in training years
    2. Thresholds dictionary is not empty
    3. All thresholds are positive (sanity check)
    
    Args:
        train_years: Years used for threshold computation
        test_year: Year being evaluated
        outbreak_thresholds: Computed thresholds
        
    Returns:
        True if no leakage detected
        
    Raises:
        ValueError: If leakage is detected
    """
    if test_year in train_years:
        raise ValueError(
            f"DATA LEAKAGE: Test year {test_year} is in training years {train_years}!"
        )
    
    if not outbreak_thresholds:
        raise ValueError("No outbreak thresholds computed - cannot proceed")
    
    for district_key, threshold in outbreak_thresholds.items():
        if threshold <= 0:
            raise ValueError(f"Invalid threshold for {district_key}: {threshold}")
    
    return True


def sanity_check_results(results: List[LeadTimeResult]) -> Dict[str, Any]:
    """
    Perform sanity checks on lead-time results.
    
    Checks for:
    -----------
    1. No NaN values where there shouldn't be
    2. Lead times are within reasonable range
    3. Differential lead is computed correctly
    4. Sentinel values (-1) are properly set
    
    Args:
        results: List of LeadTimeResult objects
        
    Returns:
        Dictionary with sanity check results
    """
    checks = {
        'n_results': len(results),
        'issues': [],
        'passed': True
    }
    
    for i, r in enumerate(results):
        # Check lead times are valid
        if r.lead_time_bayesian != NEVER_WARNED_SENTINEL:
            if r.lead_time_bayesian < -10 or r.lead_time_bayesian > 52:
                checks['issues'].append(f"Result {i}: unusual Bayesian lead time {r.lead_time_bayesian}")
        
        if r.lead_time_xgboost != NEVER_WARNED_SENTINEL:
            if r.lead_time_xgboost < -10 or r.lead_time_xgboost > 52:
                checks['issues'].append(f"Result {i}: unusual XGBoost lead time {r.lead_time_xgboost}")
        
        # Check differential consistency
        if r.differential_lead is not None:
            expected_diff = r.lead_time_xgboost - r.lead_time_bayesian
            if abs(r.differential_lead - expected_diff) > 0.001:
                checks['issues'].append(f"Result {i}: differential mismatch")
        
        # Check episode has valid data
        if r.episode.first_outbreak_week < 1 or r.episode.first_outbreak_week > 53:
            checks['issues'].append(f"Result {i}: invalid outbreak week {r.episode.first_outbreak_week}")
    
    checks['passed'] = len(checks['issues']) == 0
    
    return checks


# =============================================================================
# MAIN ANALYSIS WORKFLOW
# =============================================================================

class LeadTimeAnalyzer:
    """
    Main class for lead-time analysis workflow.
    
    Usage:
    ------
    ```python
    analyzer = LeadTimeAnalyzer(
        outbreak_percentile=outbreak_percentile,
        bayesian_percentile=bayesian_percentile,
        xgboost_threshold=probability_threshold
    )
    
    # Set thresholds from training data
    analyzer.compute_thresholds_from_training(train_df)
    
    # Identify outbreak episodes in test data
    episodes = analyzer.identify_episodes(test_df, fold_name="fold_2018")
    
    # Compute lead times
    results = analyzer.compute_lead_times(
        episodes, bayesian_preds_df, xgboost_preds_df
    )
    
    # Generate outputs
    analyzer.save_results(results, output_dir)
    ```
    """
    
    def __init__(
        self,
        outbreak_percentile: int,
        bayesian_percentile: int,
        xgboost_threshold: float
    ):
        """
        Initialize analyzer with threshold parameters.
        
        Args:
            outbreak_percentile: Percentile for outbreak definition (config-driven)
            bayesian_percentile: Percentile for Bayesian threshold (config-driven)
            xgboost_threshold: XGBoost probability threshold (config-driven)
        """
        if outbreak_percentile is None or bayesian_percentile is None or xgboost_threshold is None:
            raise ValueError("Threshold parameters must be provided explicitly.")
        self.outbreak_percentile = outbreak_percentile
        self.bayesian_percentile = bayesian_percentile
        self.xgboost_threshold = xgboost_threshold
        
        # Computed thresholds (populated by compute_thresholds_from_training)
        self.outbreak_thresholds: Dict[Tuple[str, str], float] = {}
        self.bayesian_thresholds: Dict[Tuple[str, str], float] = {}
        
        # Tracking
        self.train_years: List[int] = []
    
    def compute_outbreak_thresholds_from_training(
        self,
        train_df: pd.DataFrame,
        case_col: str = 'cases'
    ) -> None:
        """
        Compute outbreak thresholds from training data.
        
        MUST be called before identify_episodes.
        
        Args:
            train_df: Training data (years BEFORE test year)
            case_col: Case count column
        """
        self.outbreak_thresholds = compute_outbreak_thresholds_per_district(
            train_df,
            percentile=self.outbreak_percentile,
            case_col=case_col
        )
        self.train_years = sorted(train_df['year'].unique().tolist())
        
        print(f"  Computed outbreak thresholds for {len(self.outbreak_thresholds)} districts")
        print(f"  Training years: {self.train_years}")
    
    def compute_bayesian_thresholds_from_predictions(
        self,
        predictions_df: pd.DataFrame,
        z_col: str = 'z_mean'
    ) -> None:
        """
        Compute Bayesian thresholds from predictions.
        
        Args:
            predictions_df: Bayesian predictions DataFrame
            z_col: Column with Z_t posterior mean
        """
        self.bayesian_thresholds = compute_bayesian_thresholds_per_district(
            predictions_df,
            percentile=self.bayesian_percentile,
            z_col=z_col
        )
        
        print(f"  Computed Bayesian thresholds for {len(self.bayesian_thresholds)} districts")
    
    def identify_episodes(
        self,
        test_df: pd.DataFrame,
        fold_name: str,
        case_col: str = 'cases'
    ) -> List[OutbreakEpisode]:
        """
        Identify outbreak episodes in test data.
        
        Args:
            test_df: Test fold data
            fold_name: CV fold name
            case_col: Case count column
            
        Returns:
            List of OutbreakEpisode objects
        """
        if not self.outbreak_thresholds:
            raise ValueError("Must call compute_outbreak_thresholds_from_training first!")
        
        # Validate no leakage
        test_year = test_df['year'].iloc[0]
        validate_no_leakage(self.train_years, test_year, self.outbreak_thresholds)
        
        episodes = identify_outbreak_episodes(
            test_df,
            self.outbreak_thresholds,
            fold_name,
            case_col=case_col
        )
        
        print(f"  Identified {len(episodes)} outbreak episodes in {fold_name}")
        
        return episodes
    
    def compute_lead_times(
        self,
        episodes: List[OutbreakEpisode],
        bayesian_df: pd.DataFrame,
        xgboost_df: pd.DataFrame,
        z_col: str = 'z_mean',
        prob_col: str = 'prob'
    ) -> List[LeadTimeResult]:
        """
        Compute lead times for all episodes.
        
        Args:
            episodes: List of outbreak episodes
            bayesian_df: Bayesian predictions
            xgboost_df: XGBoost predictions
            z_col: Bayesian signal column
            prob_col: XGBoost probability column
            
        Returns:
            List of LeadTimeResult objects
        """
        if not self.bayesian_thresholds:
            raise ValueError("Must call compute_bayesian_thresholds_from_predictions first!")
        
        results = compute_lead_times_all_episodes(
            episodes,
            bayesian_df,
            xgboost_df,
            self.bayesian_thresholds,
            self.xgboost_threshold,
            z_col=z_col,
            prob_col=prob_col
        )
        
        # Sanity check
        checks = sanity_check_results(results)
        if not checks['passed']:
            warnings.warn(f"Sanity check issues: {checks['issues']}")
        
        return results
    
    def get_summaries(
        self,
        results: List[LeadTimeResult]
    ) -> Tuple[LeadTimeSummary, LeadTimeSummary]:
        """
        Get summary statistics for both models.
        
        Args:
            results: List of LeadTimeResult objects
            
        Returns:
            Tuple of (bayesian_summary, xgboost_summary)
        """
        bayesian_summary = summarize_lead_times(results, 'bayesian')
        xgboost_summary = summarize_lead_times(results, 'xgboost')
        
        return bayesian_summary, xgboost_summary
    
    def save_results(
        self,
        results: List[LeadTimeResult],
        output_dir: Path,
        prefix: str = ""
    ) -> Dict[str, Path]:
        """
        Save all outputs to CSV files.
        
        Outputs:
        --------
        - lead_times_detail.csv: All episode-level results
        - lead_times_bayesian.csv: Bayesian-specific view
        - lead_times_xgboost.csv: XGBoost-specific view
        - lead_time_summary.csv: Comparison table
        
        Args:
            results: List of LeadTimeResult objects
            output_dir: Directory for output files
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary mapping output type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        detail_df = results_to_dataframe(results)
        
        # Get summaries
        bayesian_summary, xgboost_summary = self.get_summaries(results)
        summary_df = summary_to_dataframe(bayesian_summary, xgboost_summary)
        
        # Prepare output paths
        prefix_str = f"{prefix}_" if prefix else ""
        paths = {}
        
        # Save detail results
        detail_path = output_dir / f"{prefix_str}lead_times_detail.csv"
        detail_df.to_csv(detail_path, index=False)
        paths['detail'] = detail_path
        
        # Save Bayesian-specific
        bayesian_cols = ['state', 'district', 'year', 'fold', 'first_outbreak_week',
                        'bayesian_trigger_week', 'lead_time_bayesian', 'bayesian_threshold']
        bayesian_path = output_dir / f"{prefix_str}lead_times_bayesian.csv"
        detail_df[bayesian_cols].to_csv(bayesian_path, index=False)
        paths['bayesian'] = bayesian_path
        
        # Save XGBoost-specific
        xgboost_cols = ['state', 'district', 'year', 'fold', 'first_outbreak_week',
                       'xgboost_trigger_week', 'lead_time_xgboost', 'xgboost_threshold']
        xgboost_path = output_dir / f"{prefix_str}lead_times_xgboost.csv"
        detail_df[xgboost_cols].to_csv(xgboost_path, index=False)
        paths['xgboost'] = xgboost_path
        
        # Save summary
        summary_path = output_dir / f"{prefix_str}lead_time_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        paths['summary'] = summary_path
        
        print(f"\n  ✓ Saved results to {output_dir}")
        for name, path in paths.items():
            print(f"    - {name}: {path.name}")
        
        return paths


# =============================================================================
# CLI ENTRY POINT (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    """
    Standalone test of the lead-time module.
    
    This block creates synthetic data to verify the module works correctly.
    It does NOT run the actual analysis (use experiments/06_analyze_lead_time.py for that).
    """
    print("="*70)
    print("Lead-Time Analysis Module - Sanity Test")
    print("="*70)

    outbreak_percentile = int(100 * (3 / 4))
    bayesian_percentile = outbreak_percentile
    probability_threshold = 1 / 2
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Synthetic training data
    train_data = {
        'state': ['TestState'] * 100,
        'district': ['District_A'] * 50 + ['District_B'] * 50,
        'year': [2016] * 50 + [2017] * 50,
        'week': list(range(1, 51)) + list(range(1, 51)),
        'cases': np.random.poisson(10, 100)
    }
    train_df = pd.DataFrame(train_data)
    
    # Synthetic test data (with outbreak)
    test_data = {
        'state': ['TestState'] * 30,
        'district': ['District_A'] * 30,
        'year': [2018] * 30,
        'week': list(range(1, 31)),
        'cases': [5, 8, 10, 12, 15, 20, 35, 8 * 10, 120, 90,  # Outbreak around week 8-10
                  70, 40, 30, 25, 20, 15, 12, 10, 8, 7,
                  6, 5, 5, 4, 4, 4, 3, 3, 3, 3]
    }
    test_df = pd.DataFrame(test_data)
    
    # Synthetic Bayesian predictions (risk rises before outbreak)
    bayesian_preds = {
        'state': ['TestState'] * 30,
        'district': ['District_A'] * 30,
        'year': [2018] * 30,
        'week': list(range(1, 31)),
        'z_mean': [0.1, 0.15, 0.2, 0.3, 1 / 2, 0.7, 0.85, 0.9, 0.95, 0.85,  # Rises early
               0.7, 1 / 2, 0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1,
                   0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    }
    bayesian_df = pd.DataFrame(bayesian_preds)
    
    # Synthetic XGBoost predictions (rises later, more reactive)
    xgboost_preds = {
        'state': ['TestState'] * 30,
        'district': ['District_A'] * 30,
        'year': [2018] * 30,
        'week': list(range(1, 31)),
        'prob': [0.1, 0.1, 0.15, 0.2, 0.25, 0.35, 11 / 20, 0.8, 0.9, 0.85,  # Rises later
                 0.6, 0.4, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    }
    xgboost_df = pd.DataFrame(xgboost_preds)
    
    # Run analysis
    print("\n1. Initializing analyzer...")
    analyzer = LeadTimeAnalyzer(
        outbreak_percentile=outbreak_percentile,
        bayesian_percentile=bayesian_percentile,
        xgboost_threshold=probability_threshold
    )
    
    print("\n2. Computing outbreak thresholds from training data...")
    analyzer.compute_outbreak_thresholds_from_training(train_df)
    print(f"   Thresholds: {analyzer.outbreak_thresholds}")
    
    print("\n3. Computing Bayesian thresholds from predictions...")
    analyzer.compute_bayesian_thresholds_from_predictions(bayesian_df)
    print(f"   Thresholds: {analyzer.bayesian_thresholds}")
    
    print("\n4. Identifying outbreak episodes...")
    episodes = analyzer.identify_episodes(test_df, fold_name="fold_2018")
    for ep in episodes:
        print(f"   Episode: {ep.district} {ep.year}, outbreak week {ep.first_outbreak_week}, "
              f"peak {ep.peak_cases} cases")
    
    print("\n5. Computing lead times...")
    results = analyzer.compute_lead_times(episodes, bayesian_df, xgboost_df)
    
    for r in results:
        print(f"   {r.episode.district}: Bayesian lead={r.lead_time_bayesian}, "
              f"XGBoost lead={r.lead_time_xgboost}, ΔL={r.differential_lead}")
    
    print("\n6. Getting summaries...")
    bayesian_summary, xgboost_summary = analyzer.get_summaries(results)
    print(f"   Bayesian: median={bayesian_summary.median_lead_time}, "
          f"early warned={bayesian_summary.pct_early_warned:.1f}%")
    print(f"   XGBoost: median={xgboost_summary.median_lead_time}, "
          f"early warned={xgboost_summary.pct_early_warned:.1f}%")
    
    print("\n7. Sanity checks...")
    checks = sanity_check_results(results)
    print(f"   Passed: {checks['passed']}")
    if checks['issues']:
        print(f"   Issues: {checks['issues']}")
    
    print("\n" + "="*70)
    print("✓ Lead-Time Module Test Complete")
    print("="*70)
