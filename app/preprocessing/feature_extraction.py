# backend/app/preprocessing/feature_extraction.py
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
import json
import warnings
from typing import Dict, Tuple, Optional, List, Any
import rasterio
from rasterio import features
from shapely.geometry import Point
from scipy.ndimage import map_coordinates

from .config import (
    RAW_FEATURES_TO_EXTRACT,
    ENGINEERED_FEATURES_ENABLED,
    ENGINEERED_FEATURES
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract feature values at sample points from aligned rasters
    AND create engineered features with proper data handling
    """

    def __init__(self, aligned_rasters: Dict):
        self.aligned_rasters = aligned_rasters
        self.feature_names = []
        self.raw_feature_names = []
        self.engineered_feature_names = []

        self.raster_data = {}
        for name, data in aligned_rasters.items():
            if isinstance(data, np.ndarray):
                self.raster_data[name] = data
                logger.info(f"Loaded raster: {name} with shape {data.shape}")

    def extract_features_at_points(self, points: np.ndarray, labels: np.ndarray,
                                   create_engineered: bool = True) -> np.ndarray:
        logger.info(f"Extracting features for {len(points)} points...")

        raw_features = []
        self.raw_feature_names = []

        for name in RAW_FEATURES_TO_EXTRACT:
            if name in self.raster_data:
                raster = self.raster_data[name]
                values = self._extract_values_at_points(raster, points)
                valid_count = np.sum(~np.isnan(values))
                logger.info(f"Extracted {name}: {values.shape} (valid: {valid_count}, {valid_count/len(values)*100:.1f}%)")
                raw_features.append(values)
                self.raw_feature_names.append(name)
            else:
                logger.warning(f"Raster '{name}' not found in aligned rasters, skipping")

        if not raw_features:
            raise ValueError("No features could be extracted! Check aligned rasters.")

        X_raw = np.column_stack(raw_features)
        logger.info(f"Raw feature matrix shape: {X_raw.shape}")

        if create_engineered and ENGINEERED_FEATURES_ENABLED:
            X_engineered = self._create_engineered_features(X_raw, points)
            if X_engineered.shape[1] > 0:
                X = np.hstack([X_raw, X_engineered])
                self.feature_names = self.raw_feature_names + self.engineered_feature_names
                logger.info(f"Total features: {X.shape[1]} ({len(self.raw_feature_names)} raw + {len(self.engineered_feature_names)} engineered)")
            else:
                X = X_raw
                self.feature_names = self.raw_feature_names
                logger.info(f"Using only raw features: {X.shape[1]} (no engineered features created)")
        else:
            X = X_raw
            self.feature_names = self.raw_feature_names
            logger.info(f"Using only raw features: {X.shape[1]}")

        return X

    def _extract_values_at_points(self, raster: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Extract values from raster at given points with safe handling.
        FIX: sentinel (-9990) and Inf are now actually replaced with the median,
        not left as NaN for downstream zero-filling.
        """
        rows = np.clip(points[:, 0].astype(int), 0, raster.shape[0] - 1)
        cols = np.clip(points[:, 1].astype(int), 0, raster.shape[1] - 1)
        values = raster[rows, cols].astype(np.float64)

        # Treat sentinel values as missing
        values = np.where(values <= -9990, np.nan, values)

        # Replace NaN and Inf with median of the valid pixels
        invalid_mask = np.isnan(values) | np.isinf(values)
        if invalid_mask.any():
            valid_values = values[~invalid_mask]
            fill_value = float(np.median(valid_values)) if len(valid_values) > 0 else 0.0
            values = np.where(invalid_mask, fill_value, values)
            logger.debug(f"Filled {invalid_mask.sum()} missing values with {fill_value:.4f}")

        return values.astype(np.float32)

    def _safe_transform(self, values: np.ndarray, transform_type: str) -> np.ndarray:
        values = np.clip(values, -1e308, 1e308)
        if transform_type == 'log':
            return np.log1p(np.clip(values, 0, 1e308))
        elif transform_type == 'square':
            clipped = np.clip(np.abs(values), 0, 1e154)
            return clipped ** 2
        elif transform_type == 'reciprocal':
            return 1 / (np.clip(np.abs(values), 1e-10, 1e308) + 1e-10)
        elif transform_type == 'normalize':
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            if max_val - min_val > 0:
                return (values - min_val) / (max_val - min_val)
            return values
        else:
            return values

    def _create_engineered_features(self, X: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Create engineered features from raw features with safe handling.
        Features not listed in config.ENGINEERED_FEATURES are pruned at the end.

        NOTE on 'floodplain_index':
            formula = HAND / (distance_to_river + 1)
            HAND = height above nearest drainage. Low HAND + close to river = floodplain.
            This is a terrain-only feature (no target leakage); the name is kept for
            config compatibility.
        """
        logger.info("Creating engineered features...")

        feature_indices = {name: i for i, name in enumerate(self.raw_feature_names)}

        def safe_get(name):
            if name in feature_indices:
                values = X[:, feature_indices[name]]
                return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            return None

        dem = safe_get('dem')
        slope = safe_get('slope')
        aspect = safe_get('aspect')
        twi = safe_get('twi')
        hand = safe_get('hand')
        flow_acc = safe_get('flow_accumulation')
        dist_river = safe_get('distance_to_river')
        drainage_density = safe_get('drainage_density')
        chirps = safe_get('chirps')
        gpm = safe_get('gpm')

        engineered = []
        self.engineered_feature_names = []

        def add_feature(values, name):
            if values is not None and values.size > 0:
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                engineered.append(values)
                self.engineered_feature_names.append(name)

        # 1. HYDROLOGICAL
        if twi is not None and flow_acc is not None:
            flow_safe = np.clip(flow_acc, 0, 1e6)
            add_feature(twi * np.log1p(flow_safe), 'wetness_index')

        if hand is not None and flow_acc is not None:
            saturation = hand / (np.clip(flow_acc, 1e-10, 1e308) + 1)
            add_feature(np.clip(saturation, 0, 1e6), 'saturation_index')

        if flow_acc is not None and dist_river is not None:
            flow_safe = np.clip(flow_acc, 0, 1e6)
            dist_safe = np.clip(dist_river, 1, 1e6)
            add_feature(np.log1p(flow_safe) / dist_safe, 'flood_concentration')

        if slope is not None and drainage_density is not None:
            add_feature(slope / (drainage_density + 0.01), 'drainage_capacity')

        if twi is not None and hand is not None:
            add_feature(twi / (hand + 0.01), 'twi_hand_ratio')

        # 2. TOPOGRAPHIC
        if slope is not None and aspect is not None:
            aspect_rad = np.radians(aspect)
            northness = np.cos(aspect_rad)
            eastness = np.sin(aspect_rad)
            terrain_roughness = slope * (1 + np.abs(northness) + np.abs(eastness))
            add_feature(terrain_roughness, 'terrain_roughness')
            add_feature(northness, 'northness')
            add_feature(eastness, 'eastness')

        if hand is not None and dem is not None:
            add_feature(hand / (dem + 1), 'relative_elevation')

        if slope is not None and aspect is not None:
            aspect_rad = np.radians(aspect)
            add_feature(slope * (1 + np.sin(aspect_rad)), 'slope_aspect')

        # 3. PROXIMITY
        if dist_river is not None:
            add_feature(1 / (dist_river + 1), 'river_proximity')

        if dist_river is not None and drainage_density is not None:
            add_feature(dist_river * drainage_density, 'river_influence')

        if hand is not None and dist_river is not None:
            # floodplain_index = HAND / (dist_river + 1). See docstring.
            add_feature(hand / (dist_river + 1), 'floodplain_index')

        # 4. RAINFALL  (FIX: rainfall_ratio clipped to sane range)
        rainfall_intensity = None
        if chirps is not None and gpm is not None:
            rainfall_intensity = chirps + gpm
            add_feature(rainfall_intensity, 'rainfall_intensity')

            # +1.0 denominator + clip prevents ~100x blow-ups when gpm ~ 0
            rainfall_ratio = chirps / (gpm + 1.0)
            rainfall_ratio = np.clip(rainfall_ratio, 0.1, 10.0)
            add_feature(rainfall_ratio, 'rainfall_ratio')

        if rainfall_intensity is not None and twi is not None:
            add_feature(rainfall_intensity * twi, 'rainfall_wetness')

        if rainfall_intensity is not None and flow_acc is not None:
            flow_safe = np.clip(flow_acc, 0, 1e6)
            add_feature(rainfall_intensity * np.log1p(flow_safe), 'rainfall_flow')

        # 5. COMPOSITE INDICES
        fsi_components = []
        fsi_weights = []

        if twi is not None:
            p95 = max(np.percentile(twi, 95), 0.01)
            fsi_components.append(twi / p95); fsi_weights.append(0.25)
        if flow_acc is not None:
            log_flow = np.log1p(np.clip(flow_acc, 0, 1e6))
            p95 = max(np.percentile(log_flow, 95), 0.01)
            fsi_components.append(log_flow / p95); fsi_weights.append(0.20)
        if hand is not None:
            p95 = max(np.percentile(hand, 95), 0.01)
            fsi_components.append(hand / p95); fsi_weights.append(0.15)
        if dist_river is not None:
            inv_dist = 1 / (dist_river + 1)
            p95 = max(np.percentile(inv_dist, 95), 0.01)
            fsi_components.append(inv_dist / p95); fsi_weights.append(0.15)
        if rainfall_intensity is not None:
            p95 = max(np.percentile(rainfall_intensity, 95), 0.01)
            fsi_components.append(rainfall_intensity / p95); fsi_weights.append(0.15)
        if slope is not None:
            p95 = max(np.percentile(slope, 95), 0.01)
            fsi_components.append(slope / p95); fsi_weights.append(0.10)

        if fsi_components:
            fsi_weights = np.array(fsi_weights) / sum(fsi_weights)
            fsi = np.sum([c * w for c, w in zip(fsi_components, fsi_weights)], axis=0)
            add_feature(np.clip(fsi, 0, 1), 'flood_susceptibility_index')

        if slope is not None and chirps is not None:
            add_feature(np.clip(slope * chirps, 0, 1e6), 'flash_flood_potential')

        if slope is not None and twi is not None:
            add_feature(np.clip(twi / (slope + 0.01), 0, 1e6), 'water_logging_potential')

        # 6. INTERACTIONS
        if twi is not None and hand is not None:
            add_feature(twi * hand, 'twi_hand')
        if slope is not None and flow_acc is not None:
            add_feature(slope * np.log1p(np.clip(flow_acc, 0, 1e6)), 'slope_flow')
        if dist_river is not None and slope is not None:
            add_feature(dist_river * slope, 'dist_slope')
        if drainage_density is not None and rainfall_intensity is not None:
            add_feature(drainage_density * rainfall_intensity, 'drainage_rainfall')

        # 7. POLYNOMIALS
        for name, values in [('twi', twi), ('flow_acc', flow_acc),
                             ('hand', hand), ('slope', slope)]:
            if values is not None:
                add_feature(self._safe_transform(values, 'square'), f'{name}_squared')
                add_feature(self._safe_transform(values, 'log'), f'log_{name}')

        # 8. SPATIAL
        if points is not None and len(points) > 0:
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            dist_center = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
            add_feature(dist_center / (np.max(dist_center) + 0.01), 'dist_center_norm')

        if dem is not None and dist_river is not None:
            add_feature(dem / (dist_river + 1), 'dem_river_ratio')

        # 9. RATIOS
        if hand is not None and dem is not None:
            add_feature(hand / (dem + 1), 'hand_dem_ratio')
        if twi is not None and slope is not None:
            add_feature(twi / (slope + 0.01), 'twi_slope_ratio')
        if flow_acc is not None and dist_river is not None:
            add_feature(np.log1p(np.clip(flow_acc, 0, 1e6)) / (dist_river + 1), 'flow_dist_ratio')

        # WHITELIST FILTER
        if ENGINEERED_FEATURES:
            allowed = set(ENGINEERED_FEATURES)
            kept_idx = [i for i, n in enumerate(self.engineered_feature_names) if n in allowed]
            dropped = [n for n in self.engineered_feature_names if n not in allowed]
            engineered = [engineered[i] for i in kept_idx]
            self.engineered_feature_names = [self.engineered_feature_names[i] for i in kept_idx]
            if dropped:
                logger.info(f"  Pruned {len(dropped)} engineered feature(s) not in ENGINEERED_FEATURES whitelist")

        X_engineered = np.column_stack(engineered) if engineered else np.array([]).reshape(X.shape[0], 0)
        logger.info(f"Created {len(self.engineered_feature_names)} engineered features")
        if self.engineered_feature_names:
            logger.info(f"  Engineered features: {', '.join(self.engineered_feature_names[:10])}...")

        return X_engineered

    def extract_vectors_at_points(self, points: np.ndarray, labels: np.ndarray,
                                   vector_data: Dict) -> np.ndarray:
        return np.array([])

    def handle_missing_values(self, X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
        if strategy == 'drop':
            return X[~np.isnan(X).any(axis=1)]

        X_filled = X.copy().astype(np.float64)
        for i in range(X.shape[1]):
            col = X_filled[:, i]
            valid = col[~np.isnan(col) & ~np.isinf(col)]

            if len(valid) > 0:
                if strategy == 'mean':
                    fill_value = np.mean(valid)
                elif strategy == 'median':
                    fill_value = np.median(valid)
                elif strategy == 'zero':
                    fill_value = 0.0
                else:
                    fill_value = np.mean(valid)
            else:
                fill_value = 0.0

            mask = np.isnan(col) | np.isinf(col)
            if mask.any():
                X_filled[mask, i] = fill_value
                logger.info(f"Filled {mask.sum():,} NaN/Inf values in column {i} with {fill_value:.4f}")

        return X_filled.astype(np.float32)

    def compute_feature_statistics(self, X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        stats = []
        for i, name in enumerate(self.feature_names):
            if i >= X.shape[1]:
                continue
            values = X[:, i]
            valid = values[~np.isnan(values) & ~np.isinf(values)]
            if len(valid) > 0:
                stats.append({
                    'feature': name,
                    'is_engineered': name in self.engineered_feature_names,
                    'mean': np.mean(valid),
                    'std': np.std(valid),
                    'min': np.min(valid),
                    'max': np.max(valid),
                    'flood_mean': np.mean(valid[labels[:len(valid)] == 1]) if np.any(labels[:len(valid)] == 1) else np.nan,
                    'nonflood_mean': np.mean(valid[labels[:len(valid)] == 0]) if np.any(labels[:len(valid)] == 0) else np.nan,
                })
        return pd.DataFrame(stats)

    def save_feature_data(self, X: np.ndarray, labels: np.ndarray,
                          output_dir: Path, coords: Optional[np.ndarray] = None,
                          dataset_name: str = "features") -> pd.DataFrame:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        df = pd.DataFrame(X_clean, columns=self.feature_names)
        df['flood_label'] = labels.astype(int)

        if coords is not None:
            df['row'] = coords[:, 0]
            df['col'] = coords[:, 1]

        output_path = output_dir / f"{dataset_name}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved feature data to {output_path}")

        with open(output_dir / f"{dataset_name}_feature_names.json", 'w') as f:
            json.dump({
                'raw_features': self.raw_feature_names,
                'engineered_features': self.engineered_feature_names,
                'all_features': self.feature_names,
                'n_raw': len(self.raw_feature_names),
                'n_engineered': len(self.engineered_feature_names),
                'n_total': len(self.feature_names)
            }, f, indent=2)

        return df


def main():
    raster = np.random.rand(100, 100)
    points = np.random.randint(0, 100, (50, 2))
    labels = np.random.randint(0, 2, 50)

    aligner_rasters = {
        'dem': raster, 'slope': raster, 'aspect': raster,
        'twi': raster, 'hand': raster, 'flow_accumulation': raster,
        'distance_to_river': raster, 'drainage_density': raster,
        'chirps': raster, 'gpm': raster
    }

    extractor = FeatureExtractor(aligner_rasters)
    X = extractor.extract_features_at_points(points, labels, create_engineered=True)
    print(f"Extracted features: {X.shape}")
    print(f"Raw features: {len(extractor.raw_feature_names)}")
    print(f"Engineered features: {len(extractor.engineered_feature_names)}")
    print(f"Total features: {len(extractor.feature_names)}")

    X_filled = extractor.handle_missing_values(X)
    stats = extractor.compute_feature_statistics(X, labels)
    print(stats.head(10))


if __name__ == "__main__":
    main()