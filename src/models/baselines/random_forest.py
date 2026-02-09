"""
Random Forest Baseline for Chikungunya EWS

Non-linear tree ensemble baseline.
Good for capturing feature interactions.

Reference: 03_tdd.md Section 3.4.1
"""
import numpy as np
from typing import Dict, Optional, List
from sklearn.ensemble import RandomForestClassifier

from ..base import BaseModel


class RandomForestBaseline(BaseModel):
    """Random Forest classifier baseline."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name="random_forest", config=config)
        
        # Default parameters
        cfg = config or {}
        self.n_estimators = cfg.get('n_estimators', 100)
        self.max_depth = cfg.get('max_depth', 15)
        self.min_samples_leaf = cfg.get('min_samples_leaf', 5)
        self.n_jobs = cfg.get('n_jobs', -1)
        self.random_state = cfg.get('random_state', 42)
        
        # Initialize sklearn model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            class_weight='balanced'
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestBaseline':
        """Fit Random Forest."""
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importances."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        importances = self.model.feature_importances_
        
        if feature_names:
            return dict(zip(feature_names, importances))
        return {'importance': importances.tolist()}
