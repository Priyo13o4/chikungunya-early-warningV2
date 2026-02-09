"""
Logistic Regression Baseline for Chikungunya EWS

Simple but interpretable baseline model.
L2 regularization, balanced class weights.

Reference: 03_tdd.md Section 3.4.1
"""
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..base import BaseModel


class LogisticBaseline(BaseModel):
    """L2-regularized logistic regression baseline."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name="logistic", config=config)
        
        # Default parameters
        self.C = config.get('C', 0.01) if config else 0.01
        self.solver = config.get('solver', 'lbfgs') if config else 'lbfgs'
        self.max_iter = config.get('max_iter', 1000) if config else 1000
        self.class_weight = config.get('class_weight', 'balanced') if config else 'balanced'
        
        # Initialize sklearn model
        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=42
        )
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticBaseline':
        """Fit logistic regression."""
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get feature coefficients for interpretation."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return {
            'intercept': float(self.model.intercept_[0]),
            'coefficients': self.model.coef_[0].tolist()
        }
