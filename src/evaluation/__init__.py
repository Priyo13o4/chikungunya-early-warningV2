"""Evaluation module - metrics, cross-validation, and lead-time analysis."""

from src.evaluation.cv import (
    create_rolling_origin_splits,
    CVFold,
    cv_split_generator,
    prepare_train_test,
    get_valid_samples
)

from src.evaluation.metrics import (
    compute_auc,
    compute_classification_metrics,
    compute_brier_score
)

from src.evaluation.lead_time import (
    LeadTimeAnalyzer,
    LeadTimeResult,
    LeadTimeSummary,
    OutbreakEpisode,
    compute_outbreak_thresholds_per_district,
    compute_bayesian_thresholds_per_district,
    identify_outbreak_episodes,
    compute_lead_times_all_episodes,
    summarize_lead_times,
    results_to_dataframe,
    summary_to_dataframe,
    sanity_check_results,
    NEVER_WARNED_SENTINEL
)

__all__ = [
    # CV module
    'create_rolling_origin_splits',
    'CVFold',
    'cv_split_generator',
    'prepare_train_test',
    'get_valid_samples',
    # Metrics module
    'compute_auc',
    'compute_classification_metrics',
    'compute_brier_score',
    # Lead-time module
    'LeadTimeAnalyzer',
    'LeadTimeResult',
    'LeadTimeSummary',
    'OutbreakEpisode',
    'compute_outbreak_thresholds_per_district',
    'compute_bayesian_thresholds_per_district',
    'identify_outbreak_episodes',
    'compute_lead_times_all_episodes',
    'summarize_lead_times',
    'results_to_dataframe',
    'summary_to_dataframe',
    'sanity_check_results',
    'NEVER_WARNED_SENTINEL'
]
