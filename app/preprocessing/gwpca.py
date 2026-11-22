# backend/app/preprocessing/gwpca.py
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler

from pygwrx import GWPCA

from .config import GWPCA_PARAMS, OUTPUT_DIR

logger = logging.getLogger(__name__)


class GWPCAProcessor:
    """
    Geographically Weighted Principal Component Analysis using pyGWRx.
    """

    def __init__(self, coords: np.ndarray, X: np.ndarray,
                 feature_names: list, bandwidth: Optional[float] = None):
        self.coords = coords
        self.X = X
        self.feature_names = feature_names
        self.bandwidth = bandwidth
        self.gwpca = None
        self.results = None
        self.scores = None
        self.loadings = None
        self.scaler = None
        self.local_eigenvalues = None
        self.local_var_ratios = None

    def fit(self) -> Tuple[np.ndarray, dict]:
        logger.info("Fitting GWPCA model...")
        logger.info(f"  Coords shape: {self.coords.shape}")
        logger.info(f"  X shape: {self.X.shape}")
        logger.info(f"  Features: {len(self.feature_names)}")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        n_components = min(len(self.feature_names), GWPCA_PARAMS.get('n_components', 6))

        if self.bandwidth is None:
            self.bandwidth = self._select_bandwidth()
        if self.bandwidth <= n_components:
            self.bandwidth = n_components + 1
            logger.warning(f"Bandwidth increased to {self.bandwidth} to support {n_components} components.")

        logger.info(f"Initializing pyGWRx GWPCA with bandwidth (neighbors): {int(self.bandwidth)}")
        self.gwpca = GWPCA(
            n_components=n_components,
            bandwidth=int(self.bandwidth),
            adaptive=True,
            compute_scores=True,
        )

        logger.info("Running GWPCA (this may take a while)...")
        self.gwpca.fit(X_scaled, self.coords, eval_coords=self.coords)

        # ---- Loadings ----
        raw = self.gwpca.loadings_
        if raw.ndim == 3:
            # (n_samples, n_features, n_components) -> average over locations
            self.loadings = raw.mean(axis=0)   # (n_features, n_components)
            logger.info(f"Averaged local loadings {raw.shape} -> {self.loadings.shape}")
        elif raw.ndim == 2:
            self.loadings = raw if raw.shape[0] == len(self.feature_names) else raw.T
            logger.info(f"Using global loadings {raw.shape} -> {self.loadings.shape}")
        else:
            raise ValueError(f"Unexpected loadings ndim={raw.ndim}")

        # ---- Scores ----
        # NOTE: gwpca.transform() requires each coordinate to match exactly ONE
        # fitted location. Our sample points contain duplicate coordinates
        # (same pixel sampled across events), so transform() fails. Since we
        # fitted with eval_coords=self.coords, row i has its own local loadings;
        # compute scores by row index instead.
        if raw.ndim == 3:
            lw = raw[:, :, :n_components]                       # (n, f, k)
            self.scores = np.einsum('nf,nfk->nk', X_scaled, lw)  # (n, k)
        else:
            self.scores = X_scaled @ self.loadings[:, :n_components]
        logger.info(f"GWPCA scores shape: {self.scores.shape}")

        # ---- Variance ----
        self.local_var_ratios = self.gwpca.local_pv_[:, :n_components]
        mean_var_ratios = self.local_var_ratios.mean(axis=0)
        self.local_eigenvalues = self.local_var_ratios * X_scaled.shape[1]

        self.results = {
            'eigenvalues': self.local_eigenvalues.mean(axis=0),
            'components': self.loadings.T,
            'n_components': n_components,
        }

        logger.info(f"GWPCA complete: {self.scores.shape[1]} components retained")
        logger.info(f"  Mean total variance explained: {mean_var_ratios.sum():.4f}")

        return self.scores, self.results

    def _select_bandwidth(self) -> int:
        n_samples = self.X.shape[0]
        bw = int(np.sqrt(n_samples))
        logger.info(f"Selected bandwidth: {bw} (heuristic based on sqrt(n))")
        return bw

    def _determine_components(self) -> int:
        if self.local_var_ratios is None:
            return len(self.feature_names)
        mean_var_ratios = self.local_var_ratios.mean(axis=0)
        cumulative_variance = np.cumsum(mean_var_ratios)
        variance_threshold = GWPCA_PARAMS.get('variance_threshold', 0.85)
        n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
        n_components = max(1, min(n_components, len(self.feature_names)))
        logger.info(f"Components retained: {n_components} (explains {cumulative_variance[n_components-1]:.2%} variance)")
        return n_components

    def get_component_loadings(self) -> pd.DataFrame:
        if self.loadings is None:
            raise ValueError("GWPCA not fitted yet")
        n_features = min(len(self.feature_names), self.loadings.shape[0])
        loadings_df = pd.DataFrame(
            self.loadings[:n_features, :],
            index=self.feature_names[:n_features],
            columns=[f'PC{i+1}' for i in range(self.loadings.shape[1])]
        )
        return loadings_df

    def get_spatial_loadings(self, component: int = 0) -> np.ndarray:
        if self.gwpca is None:
            raise ValueError("GWPCA not fitted yet")
        raw = self.gwpca.loadings_
        if raw.ndim == 2:
            logger.warning("Global loadings only; no spatial variation.")
            return raw[component, :] if raw.shape[0] < raw.shape[1] else raw[:, component]
        return raw[:, :, component]

    def save_results(self, output_dir: str):
        output_path = Path(output_dir) / "gwpca"
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / "scores.npy", self.scores)
        np.save(output_path / "loadings.npy", self.loadings)

        raw = self.gwpca.loadings_
        if raw.ndim == 3:
            np.save(output_path / "local_loadings.npy", raw)
        else:
            logger.warning("Skipping local_loadings.npy — global loadings only.")

        loadings_df = self.get_component_loadings()
        loadings_df.to_csv(output_path / "component_loadings.csv")

        variance_ratios = self.local_var_ratios.mean(axis=0)
        cumulative_variance = np.cumsum(variance_ratios)

        params = {
            'bandwidth_neighbors': int(self.bandwidth) if self.bandwidth is not None else None,
            'n_components': int(self.scores.shape[1]),
            'total_variance_explained': float(cumulative_variance[-1]),
            'eigenvalues': [float(x) for x in self.results['eigenvalues'].tolist()],
            'variance_ratios': [float(x) for x in variance_ratios.tolist()],
            'cumulative_variance': [float(x) for x in cumulative_variance.tolist()],
            'feature_names': self.feature_names,
            'method': 'gwpca',
            'note': 'Scores computed per-location from local loadings (duplicate coords safe)',
        }
        with open(output_path / "params.json", 'w') as f:
            json.dump(params, f, indent=2)

        logger.info(f"Saved GWPCA results to {output_path}")

    def get_summary(self) -> dict:
        if self.results is None:
            return {'status': 'not fitted'}
        variance_ratios = self.local_var_ratios.mean(axis=0)
        cumulative_variance = np.cumsum(variance_ratios)
        return {
            'bandwidth_neighbors': int(self.bandwidth) if self.bandwidth is not None else None,
            'n_components': int(self.scores.shape[1]) if self.scores is not None else None,
            'total_variance': float(self.results['eigenvalues'].sum()),
            'variance_explained': float(cumulative_variance[-1]) if self.scores is not None else None,
            'n_features': int(len(self.feature_names)),
            'n_samples': int(self.X.shape[0]),
            'method': 'gwpca',
        }