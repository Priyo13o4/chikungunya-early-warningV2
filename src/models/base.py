"""
Base Model Interface for Chikungunya EWS

Abstract base class that all models must implement.
Ensures consistent API across baselines and Bayesian models.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import pickle
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all EWS models."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize model.
        
        Args:
            name: Model identifier
            config: Model-specific configuration
        """
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self.feature_names = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Fit model to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probability of positive class (n_samples,)
        """
        pass
    
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold (config-driven)
            
        Returns:
            Binary predictions (n_samples,)
        """
        if threshold is None:
            raise ValueError("threshold must be provided explicitly")
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
