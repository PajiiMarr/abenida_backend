# app/models/feature_service.py
"""
Sample aligned raw feature rasters at a lat/lng and build a model-ready vector.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyproj
import rasterio
from rasterio.transform import rowcol

from app.preprocessing.config import (
    OUTPUT_DIR,
    RAW_FEATURES_TO_EXTRACT,
    TARGET_CRS,
)
from app.models.feature_engineering import build_feature_matrix

logger = logging.getLogger(__name__)

ALIGNED_DIR = OUTPUT_DIR / "aligned"

# Map feature name -> aligned raster filename (matches your file tree)
FEATURE_RASTER_MAP: Dict[str, str] = {
    "dem": "dem_aligned.tif",
    "slope": "slope_aligned.tif",
    "aspect": "aspect_aligned.tif",
    "twi": "twi_aligned.tif",
    "hand": "hand_aligned.tif",
    "flow_accumulation": "flow_accumulation_aligned.tif",
    "distance_to_river": "distance_to_river_aligned.tif",
    "drainage_density": "drainage_density_aligned.tif",
    "chirps": "chirps_aligned.tif",
    "gpm": "gpm_aligned.tif",
    "population_density": "population_density_aligned.tif",
}

_transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)


def _training_fill(name: str) -> float:
    """Fill value used in training for missing pixels (0.0 if unavailable)."""
    try:
        from app.models.feature_engineering import get_training_stats
        v = get_training_stats()["fill"].get(name)
        return float(v) if v is not None and np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def pixel_at(lat: float, lng: float) -> Tuple[int, int]:
    """(row, col) of a point in the aligned grid, the coordinate system the suite models were trained on."""
    x, y = _transformer.transform(lng, lat)
    with rasterio.open(ALIGNED_DIR / FEATURE_RASTER_MAP["dem"]) as src:
        row, col = rowcol(src.transform, x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise ValueError("Point outside raster extent")
    return int(row), int(col)


def sample_raw_features(lat: float, lng: float) -> Dict[str, float]:
    """Read every aligned raw raster at the given point."""
    x, y = _transformer.transform(lng, lat)
    out: Dict[str, float] = {}

    for feat in RAW_FEATURES_TO_EXTRACT:
        fname = FEATURE_RASTER_MAP.get(feat)
        if fname is None:
            logger.warning(f"No aligned raster mapped for feature '{feat}'")
            out[feat] = _training_fill(feat)
            continue

        path = ALIGNED_DIR / fname
        if not path.exists():
            logger.warning(f"Missing aligned raster for '{feat}': {path}")
            out[feat] = _training_fill(feat)
            continue

        with rasterio.open(path) as src:
            row, col = rowcol(src.transform, x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                v = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
                # -9999 is the nodata sentinel written by save_aligned_rasters
                out[feat] = v if np.isfinite(v) and v > -9990 else _training_fill(feat)
            else:
                out[feat] = _training_fill(feat)

    return out


def build_vector_at_point(lat: float, lng: float, feature_names: List[str]) -> np.ndarray:
    """Return shape (1, n_features) float32 array in feature_names order."""
    raw = sample_raw_features(lat, lng)
    vec = build_feature_matrix(raw, feature_names)  # (n_features,)
    return vec.reshape(1, -1).astype(np.float32)