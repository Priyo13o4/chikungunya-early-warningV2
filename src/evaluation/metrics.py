"""
Evaluation Metrics for Chikungunya EWS

Implements metrics specific to early warning systems:
- AUC-ROC
- Lead time (weeks in advance)
- False alarm rate
- Sensitivity / Specificity
- Brier score (calibration)
- F1 score

Reference: 05_experiments.md Section 5.5
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    brier_score_loss,
    confusion_matrix
)


def compute_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute AUC-ROC score.
    
    Args:
        y_true: Binary true labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        AUC score (random baseline to perfect)
    """
    try:
        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred_proba)
    except:
        return np.nan


def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute classification metrics at a given threshold.
    
    Args:
        y_true: Binary true labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold (config-driven)
        
    Returns:
        Dictionary with metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics = {
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'false_alarm_rate': fp / (fp + tp) if (fp + tp) > 0 else 0.0,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }
    
    return metrics


def compute_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute Brier score (calibration metric).
    
    Lower is better. 0.0 = perfect, 0.25 = random for balanced classes.
    
    Args:
        y_true: Binary true labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Brier score
    """
    try:
        return brier_score_loss(y_true, y_pred_proba)
    except:
        return np.nan


def compute_lead_time(
    df: pd.DataFrame,
    pred_col: str = 'pred_proba',
    true_col: str = 'label_outbreak',
    threshold: float = None,
    group_cols: List[str] = ['state', 'district']
) -> Dict[str, float]:
    """
    Compute lead time: how many weeks before actual outbreak
    did the model first predict high risk?
    
    Args:
        df: DataFrame with predictions and true labels, sorted by time
        pred_col: Column with predicted probabilities
        true_col: Column with true outbreak labels
        threshold: Prediction threshold (config-driven)
        group_cols: Grouping columns (district)
        
    Returns:
        Dictionary with lead time statistics
    """
    if threshold is None:
        raise ValueError("threshold must be provided explicitly")

    lead_times = []
    
    for _, group in df.groupby(group_cols):
        group = group.sort_values(['year', 'week'])
        
        # Find outbreak events (positive labels)
        outbreak_idx = group[group[true_col] == 1].index.tolist()
        
        for outbreak_i in outbreak_idx:
            # Look backwards to find first prediction above threshold
            pred_above = group[(group.index <= outbreak_i) & 
                              (group[pred_col] >= threshold)]
            
            if len(pred_above) > 0:
                first_pred_idx = pred_above.index[0]
                # Compute lead time (weeks between first prediction and outbreak)
                lead_weeks = group.loc[:outbreak_i].index.get_loc(outbreak_i) - \
                            group.index.get_loc(first_pred_idx)
                lead_times.append(lead_weeks)
    
    if len(lead_times) == 0:
        return {
            'lead_time_median': np.nan,
            'lead_time_mean': np.nan,
            'lead_time_min': np.nan,
            'lead_time_max': np.nan,
            'n_detected': 0
        }
    
    return {
        'lead_time_median': float(np.median(lead_times)),
        'lead_time_mean': float(np.mean(lead_times)),
        'lead_time_min': float(np.min(lead_times)),
        'lead_time_max': float(np.max(lead_times)),
        'n_detected': len(lead_times)
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Binary true labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold (config-driven)
        
    Returns:
        Dictionary with all metrics
    """
    # Ensure proper dtypes
    y_true = np.asarray(y_true, dtype=np.float64).astype(int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    
    metrics = {}
    
    # AUC
    metrics['auc'] = compute_auc(y_true, y_pred_proba)
    
    # Classification metrics
    class_metrics = compute_classification_metrics(y_true, y_pred_proba, threshold)
    metrics.update(class_metrics)
    
    # Brier score
    metrics['brier'] = compute_brier_score(y_true, y_pred_proba)
    
    # Sample counts
    metrics['n_samples'] = len(y_true)
    metrics['n_positive'] = int(np.sum(y_true == 1))
    metrics['n_negative'] = int(np.sum(y_true == 0))
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Pretty print metrics."""
    print(f"\n{title}")
    print("-" * 40)
    print(f"  AUC:          {metrics.get('auc', np.nan):.3f}")
    print(f"  Sensitivity:  {metrics.get('sensitivity', np.nan):.3f}")
    print(f"  Specificity:  {metrics.get('specificity', np.nan):.3f}")
    print(f"  F1:           {metrics.get('f1', np.nan):.3f}")
    print(f"  FAR:          {metrics.get('false_alarm_rate', np.nan):.3f}")
    print(f"  Brier:        {metrics.get('brier', np.nan):.3f}")
    print(f"  Samples:      {metrics.get('n_samples', 0)} ({metrics.get('n_positive', 0)} pos)")
