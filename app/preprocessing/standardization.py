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
    
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (z-score, default), 'minmax', 'robust'
        """
        self.method = method
        self.scaler = None
        self.scale_params = {}
        self.fill_values_ = None  # per-column medians from the fit data
        
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
        self.fill_values_ = self._column_medians(X)
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
    
    @staticmethod
    def _column_medians(X: np.ndarray) -> np.ndarray:
        med = np.nanmedian(X, axis=0)
        return np.where(np.isnan(med), 0.0, med)

    def _handle_nan(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN with the per-column MEDIAN (fit-time medians when available),
        consistent with feature_extraction."""
        X_clean = X.copy()
        fill = self.fill_values_ if self.fill_values_ is not None else self._column_medians(X)
        for i in range(X.shape[1]):
            nan_mask = np.isnan(X_clean[:, i])
            if nan_mask.any():
                X_clean[nan_mask, i] = fill[i]
                logger.info(f"Replaced {nan_mask.sum():,} NaN values in column {i} with median {fill[i]:.4f}")
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
                'fill_values': self.fill_values_,
                'method': self.method
            }, f)
        logger.info(f"Saved scaler to {output_path}")
    
    def load_scaler(self, path: str):
        """Load fitted scaler from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.scale_params = data['params']
            self.fill_values_ = data.get('fill_values')
            self.method = data['method']
        logger.info(f"Loaded scaler from {path}")