# app/models/feature_service.py
"""
Sample aligned raw feature rasters at a lat/lng and build a model-ready vector.
Nodata handling mirrors Phase 2 (NaN -> training median).
"""

import logging
from typing import Dict, List

import numpy as np
import pyproj
import rasterio
from rasterio.transform import rowcol

from app.preprocessing.config import (
    OUTPUT_DIR,
    RAW_FEATURES_TO_EXTRACT,
    TARGET_CRS,
)
from app.models.feature_engineering import build_feature_matrix, get_training_stats

logger = logging.getLogger(__name__)

ALIGNED_DIR = OUTPUT_DIR / "aligned"

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
}

_transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)


def sample_raw_features(lat: float, lng: float) -> Dict[str, float]:
    """Read every aligned raw raster at the point; fill nodata like Phase 2."""
    fill = get_training_stats()["fill"]
    x, y = _transformer.transform(lng, lat)
    out: Dict[str, float] = {}
    pix = None

    for feat in RAW_FEATURES_TO_EXTRACT:
        fname = FEATURE_RASTER_MAP.get(feat)
        path = ALIGNED_DIR / fname if fname else None
        if path is None or not path.exists():
            logger.warning(f"Missing aligned raster for '{feat}': {path}")
            out[feat] = fill.get(feat, 0.0)
            continue

        with rasterio.open(path) as src:
            row, col = rowcol(src.transform, x, y)
            if pix is None:
                pix = (float(row), float(col))
            v = np.nan
            if 0 <= row < src.height and 0 <= col < src.width:
                v = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
                if src.nodata is not None and v == src.nodata:
                    v = np.nan
            out[feat] = v if np.isfinite(v) else fill.get(feat, 0.0)

    if pix is not None:
        out["_row"], out["_col"] = pix
    return out


def build_vector_at_point(lat: float, lng: float, feature_names: List[str]) -> np.ndarray:
    """Return shape (1, n_features) float32 array in feature_names order."""
    raw = sample_raw_features(lat, lng)
    vec = build_feature_matrix(raw, feature_names)
    return vec.reshape(1, -1).astype(np.float32)