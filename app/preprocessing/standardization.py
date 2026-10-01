# backend/app/preprocessing/standardization.py
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
from typing import Optional
import pickle
from pathlib import Path

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)

class Standardizer:
    """Standardize feature data for machine learning"""
    
    def __init__(self, method: str = 'minmax'):
        """
        Args:
            method: 'minmax', 'standard', 'robust'
        """
        self.method = method
        self.scaler = None
        self.scale_params = {}
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown standardization method: {method}")
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """Fit scaler and transform data"""
        logger.info(f"Fitting {self.method} scaler on {X.shape[1]} features...")
        
        # Handle NaN values
        X_clean = self._handle_nan(X)
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Store parameters
        if hasattr(self.scaler, 'scale_'):
            self.scale_params['scale'] = self.scaler.scale_
        if hasattr(self.scaler, 'mean_'):
            self.scale_params['mean'] = self.scaler.mean_
        if hasattr(self.scaler, 'min_'):
            self.scale_params['min'] = self.scaler.min_
        if hasattr(self.scaler, 'data_min_'):
            self.scale_params['data_min'] = self.scaler.data_min_
        if hasattr(self.scaler, 'data_max_'):
            self.scale_params['data_max'] = self.scaler.data_max_
        
        self.scale_params['feature_names'] = feature_names
        self.scale_params['n_features'] = X.shape[1]
        
        logger.info(f"Scaling completed. Output shape: {X_scaled.shape}")
        logger.info(f"  Feature range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
        
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        X_clean = self._handle_nan(X)
        return self.scaler.transform(X_clean)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        return self.scaler.inverse_transform(X_scaled)
    
    def _handle_nan(self, X: np.ndarray) -> np.ndarray:
        """Handle NaN values by replacing with mean of each column"""
        X_clean = X.copy()
        for i in range(X.shape[1]):
            col = X[:, i]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                mean_val = np.nanmean(col)
                if np.isnan(mean_val):
                    mean_val = 0
                X_clean[nan_mask, i] = mean_val
                logger.info(f"Replaced {nan_mask.sum():,} NaN values in column {i} with {mean_val:.4f}")
        return X_clean
    
    def save_scaler(self, output_dir: str):
        """Save fitted scaler to disk"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted")
        
        output_path = Path(output_dir) / "scaler.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'params': self.scale_params,
                'method': self.method
            }, f)
        logger.info(f"Saved scaler to {output_path}")
    
    def load_scaler(self, path: str):
        """Load fitted scaler from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.scale_params = data['params']
            self.method = data['method']
        logger.info(f"Loaded scaler from {path}")