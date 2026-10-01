"""
Full-raster prediction: produce the 12.5 m susceptibility GeoTIFF.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio

from app.preprocessing.config import (
    OUTPUT_DIR,
    RAW_FEATURES_TO_EXTRACT,
    TARGET_RESOLUTION,
)
from app.models.feature_engineering import build_feature_matrix
from app.models.feature_service import FEATURE_RASTER_MAP

logger = logging.getLogger(__name__)

ALIGNED_DIR = OUTPUT_DIR / "aligned"
PHASE3_DIR = Path("./outputs/phase3")


def load_raw_stack() -> Tuple[Dict[str, np.ndarray], dict, dict]:
    """
    Load every aligned raw raster.
    Returns (dict of arrays, meta dict, transform), all assumed to share a grid.
    """
    arrays: Dict[str, np.ndarray] = {}
    meta = None
    transform = None

    for feat in RAW_FEATURES_TO_EXTRACT:
        fname = FEATURE_RASTER_MAP.get(feat)
        if fname is None:
            continue
        path = ALIGNED_DIR / fname
        if not path.exists():
            logger.warning(f"Missing aligned raster: {path}")
            continue
        with rasterio.open(path) as src:
            arrays[feat] = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
                transform = src.transform
            elif (src.height, src.width) != (meta["height"], meta["width"]):
                logger.error(
                    f"Grid mismatch on {feat}: {src.height}x{src.width} vs "
                    f"{meta['height']}x{meta['width']}"
                )

    return arrays, meta, transform


def predict_full_raster(model, scaler, feature_names) -> Tuple[np.ndarray, dict]:
    """
    Run the model on every valid pixel. Returns (probability_raster, meta).
    """
    raw_arrays, meta, _ = load_raw_stack()
    if not raw_arrays:
        raise RuntimeError("No aligned rasters found — run preprocessing first.")

    h, w = next(iter(raw_arrays.values())).shape
    logger.info(f"Building feature stack {h}x{w} with {len(feature_names)} features")

    # Build (h, w, n_features)
    stack = build_feature_matrix(raw_arrays, feature_names)
    n = stack.shape[-1]

    flat = stack.reshape(-1, n).astype(np.float32)
    valid = np.all(np.isfinite(flat), axis=1)
    valid_pixels = flat[valid]

    logger.info(f"Valid pixels: {valid.sum():,} / {flat.shape[0]:,}")

    if scaler is not None:
        valid_pixels = scaler.transform(valid_pixels).astype(np.float32)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(valid_pixels)[:, 1]
    else:
        proba = model.predict(valid_pixels)
    proba = np.asarray(proba, dtype=np.float32)

    out = np.full(flat.shape[0], np.nan, dtype=np.float32)
    out[valid] = proba
    raster = out.reshape(h, w)

    meta.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw",
    )
    return raster, meta


def classify_raster(raster: np.ndarray, breaks=(0.2, 0.4, 0.6, 0.8)) -> np.ndarray:
    """Reclassify probability raster into 0..4 risk classes (0=Very Low)."""
    classes = np.digitize(raster, breaks, right=False).astype(np.int8)
    classes[~np.isfinite(raster)] = -1
    return classes


def save_geotiff(array: np.ndarray, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array, 1)
    logger.info(f"Saved {path}")