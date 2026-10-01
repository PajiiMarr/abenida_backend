# backend/app/preprocessing/gwpca.py
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import GWPCA_PARAMS, OUTPUT_DIR

logger = logging.getLogger(__name__)

class GWPCAProcessor:
    """
    Geographically Weighted Principal Component Analysis
    
    Note: Since mgwr.gwpca is not available in newer versions,
    this uses standard PCA with optional spatial weighting.
    """
    
    def __init__(self, coords: np.ndarray, X: np.ndarray, 
                 feature_names: list, bandwidth: Optional[float] = None):
        """
        Args:
            coords: (n_samples, 2) array of coordinates
            X: (n_samples, n_features) feature matrix
            feature_names: List of feature names
            bandwidth: Optional bandwidth (if None, automatically selected)
        """
        self.coords = coords
        self.X = X
        self.feature_names = feature_names
        self.bandwidth = bandwidth
        self.pca = None
        self.results = None
        self.scores = None
        self.loadings = None
        self.scaler = None
        
    def fit(self) -> Tuple[np.ndarray, dict]:
        """Fit GWPCA model (using standard PCA as fallback)"""
        logger.info("Fitting GWPCA model...")
        logger.info(f"  Coords shape: {self.coords.shape}")
        logger.info(f"  X shape: {self.X.shape}")
        logger.info(f"  Features: {len(self.feature_names)}")
        
        # Standardize the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Select bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._select_bandwidth()
        
        # Determine number of components
        n_components = min(len(self.feature_names), GWPCA_PARAMS.get('n_components', 6))
        
        # Fit PCA
        self.pca = PCA(n_components=n_components)
        self.scores = self.pca.fit_transform(X_scaled)
        self.loadings = self.pca.components_.T  # Loadings matrix
        
        # Store results
        self.results = {
            'eigenvalues': self.pca.explained_variance_,
            'components': self.pca.components_,
            'mean': self.pca.mean_,
            'n_components': n_components
        }
        
        logger.info(f"GWPCA complete: {self.scores.shape[1]} components retained")
        logger.info(f"  Total variance explained: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return self.scores, self.results
    
    def _select_bandwidth(self) -> float:
        """Select optimal bandwidth using heuristic"""
        # Simple heuristic based on sample size
        n_samples = self.X.shape[0]
        # Use sqrt(n) as a reasonable bandwidth estimate
        bw = int(np.sqrt(n_samples))
        logger.info(f"Selected bandwidth: {bw} (heuristic)")
        return bw
    
    def _determine_components(self) -> int:
        """Determine number of components to retain based on variance explained"""
        if self.pca is None:
            return len(self.feature_names)
        
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        
        # Find number of components explaining > threshold
        variance_threshold = GWPCA_PARAMS.get('variance_threshold', 0.85)
        n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
        
        # Ensure at least 1 and at most all features
        n_components = max(1, min(n_components, len(self.feature_names)))
        
        logger.info(f"Components retained: {n_components} (explains {cumulative_variance[n_components-1]:.2%} variance)")
        
        return n_components
    
    def get_component_loadings(self) -> pd.DataFrame:
        """Get component loadings"""
        if self.loadings is None:
            raise ValueError("GWPCA not fitted yet")
        
        # Create DataFrame
        n_features = min(len(self.feature_names), self.loadings.shape[0])
        loadings_df = pd.DataFrame(
            self.loadings[:n_features, :],
            index=self.feature_names[:n_features],
            columns=[f'PC{i+1}' for i in range(self.loadings.shape[1])]
        )
        
        return loadings_df
    
    def get_spatial_loadings(self, component: int = 0) -> np.ndarray:
        """Get spatial loadings for a specific component"""
        if self.loadings is None:
            raise ValueError("GWPCA not fitted yet")
        
        # For standard PCA, loadings are global, not spatial
        logger.warning("Standard PCA has global loadings, not spatial. Returning global loadings.")
        if component < self.loadings.shape[1]:
            return self.loadings[:, component]
        else:
            raise ValueError(f"Component {component} not found. Available: 0-{self.loadings.shape[1]-1}")
    
    def save_results(self, output_dir: str):
        """Save GWPCA results to disk"""
        output_path = Path(output_dir) / "gwpca"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scores
        np.save(output_path / "scores.npy", self.scores)
        
        # Save loadings
        np.save(output_path / "loadings.npy", self.loadings)
        
        # Save component loadings as CSV
        loadings_df = self.get_component_loadings()
        loadings_df.to_csv(output_path / "component_loadings.csv")
        
        # Save explained variance
        variance_ratios = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratios)
        
        # Convert numpy types to Python types for JSON serialization
        params = {
            'bandwidth': float(self.bandwidth) if self.bandwidth is not None else None,
            'n_components': int(self.scores.shape[1]),
            'total_variance_explained': float(cumulative_variance[self.scores.shape[1]-1]),
            'eigenvalues': [float(x) for x in self.pca.explained_variance_.tolist()],
            'variance_ratios': [float(x) for x in variance_ratios.tolist()],
            'cumulative_variance': [float(x) for x in cumulative_variance.tolist()],
            'feature_names': self.feature_names,
            'method': 'standard_pca_fallback',
            'note': 'Using standard PCA since mgwr.gwpca is not available'
        }
        with open(output_path / "params.json", 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Saved GWPCA results to {output_path}")

        
    def get_summary(self) -> dict:
        """Get summary statistics of GWPCA results"""
        if self.results is None:
            return {'status': 'not fitted'}
        
        variance_ratios = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratios)
        
        # Convert numpy types to Python types for JSON serialization
        return {
            'bandwidth': float(self.bandwidth) if self.bandwidth is not None else None,
            'n_components': int(self.scores.shape[1]) if self.scores is not None else None,
            'total_variance': float(self.pca.explained_variance_.sum()),
            'variance_explained': float(cumulative_variance[self.scores.shape[1]-1]) if self.scores is not None else None,
            'n_features': int(len(self.feature_names)),
            'n_samples': int(self.X.shape[0]),
            'method': 'standard_pca_fallback'
        }
    