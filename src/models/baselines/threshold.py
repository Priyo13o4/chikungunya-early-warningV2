"""
Threshold Baseline for Chikungunya EWS

Simple rule-based baseline: alert if cases > μ + k*σ
No ML involved - serves as minimum viable comparison.

Reference: 03_tdd.md Section 3.4.1
"""
import numpy as np
from typing import Dict, Optional

from ..base import BaseModel


class ThresholdBaseline(BaseModel):
    """Simple threshold-based alerting model."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name="threshold", config=config)
        
        # k: number of standard deviations above mean
        self.k = config.get('multiplier', 2.0) if config else 2.0
        self.neutral_probability = config.get('neutral_probability') if config else None
        
        # Fitted parameters
        self.mean_ = None
        self.std_ = None
        self.threshold_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ThresholdBaseline':
        """
        Fit threshold based on historical case data.
        
        Assumes first feature column is a case-related feature (e.g., lag-1 or MA).
        """
        # Use first feature column as the signal
        signal = np.asarray(X[:, 0], dtype=np.float64)
        signal = signal[~np.isnan(signal)]
        
        self.mean_ = np.mean(signal)
        self.std_ = np.std(signal)
        self.threshold_ = self.mean_ + self.k * self.std_
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pseudo-probability based on distance from threshold.
        
        Uses sigmoid transformation of z-score.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use first feature column
        signal = np.asarray(X[:, 0], dtype=np.float64)
        
        # Compute z-score relative to threshold
        z = (signal - self.threshold_) / (self.std_ + 1e-6)
        
        # Sigmoid to convert to probability
        prob = 1 / (1 + np.exp(-z))
        
        if self.neutral_probability is None:
            raise ValueError("neutral_probability must be provided via config")

        # Handle NaN
        prob = np.nan_to_num(prob, nan=self.neutral_probability)
        
        return prob
    
    def get_threshold_info(self) -> Dict[str, float]:
        """Get fitted threshold parameters."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return {
            'mean': float(self.mean_),
            'std': float(self.std_),
            'k': float(self.k),
            'threshold': float(self.threshold_)
        }
