"""
XGBoost Baseline for Chikungunya EWS

Gradient boosted trees - often best performing baseline.
Good feature importance and handling of missing values.

Reference: 03_tdd.md Section 3.4.1
"""
import numpy as np
from typing import Dict, Optional, List

from ..base import BaseModel

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostBaseline(BaseModel):
    """XGBoost classifier baseline."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name="xgboost", config=config)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        # Default parameters
        cfg = config or {}
        self.n_estimators = cfg.get('n_estimators', 100)
        self.max_depth = cfg.get('max_depth', 5)
        self.learning_rate = cfg.get('learning_rate', 0.1)
        self.subsample = cfg.get('subsample', 0.8)
        self.colsample_bytree = cfg.get('colsample_bytree', 0.8)
        self.reg_alpha = cfg.get('reg_alpha', 0.01)
        self.reg_lambda = cfg.get('reg_lambda', 1.0)
        self.random_state = cfg.get('random_state', 42)
        
        # Initialize model
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostBaseline':
        """Fit XGBoost model."""
        # XGBoost handles NaN natively, but ensure float type
        X = X.astype(np.float32)
        
        # Handle class imbalance via scale_pos_weight
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        if n_pos > 0:
            self.model.set_params(scale_pos_weight=n_neg/n_pos)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X.astype(np.float32)
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importances."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        importances = self.model.feature_importances_
        
        if feature_names:
            return dict(zip(feature_names, importances))
        return {'importance': importances.tolist()}
