"""
Threshold computation utilities for outbreak detection.

This module provides standardized threshold enforcement to ensure
consistency across model training, evaluation, and label generation.
"""


def enforce_minimum_threshold(threshold_value: float, minimum: float = 1.0) -> float:
    """
    Enforce minimum threshold to avoid classifying <1 case as outbreak.
    
    Many outbreak detection systems use percentile-based thresholds on
    historical case counts. When the data includes many zeros or low counts,
    percentiles (e.g., 75th) can fall below 1.0, leading to the nonsensical
    situation where 0.5 cases would trigger an outbreak alert.
    
    This function ensures thresholds are at least 1.0 case, providing a
    sensible lower bound that aligns with the definition of an outbreak
    requiring at least one case.
    
    Args:
        threshold_value: The computed threshold (e.g., from np.percentile)
        minimum: Minimum allowed threshold value (default: 1.0)
        
    Returns:
        The threshold value, but no less than the specified minimum
        
    Examples:
        >>> enforce_minimum_threshold(0.3)
        1.0
        >>> enforce_minimum_threshold(5.2)
        5.2
        >>> enforce_minimum_threshold(0.8, minimum=2.0)
        2.0
    """
    return max(threshold_value, minimum)
