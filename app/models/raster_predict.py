"""
Full-raster prediction: produce the 12.5 m susceptibility GeoTIFF.

The model was trained only on flood-able terrain (0 <= HAND <= 15 m, slope <= 5 deg,
distance to river > 25 m, on land), so it only predicts there:
  - sea / no-data ............................ NaN
  - river channel (<= CHANNEL_BUFFER_M) ...... NaN  (never seen in training)
  - land outside flood-able terrain .......... OUTSIDE_VALUE (0.0 -> "Low")
  - flood-able terrain ....................... model probability

Suite models take (coords, X); coords are the (row, col) pixel indices of the
aligned grid, matching the (x, y) used in training.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio

from app.preprocessing.config import OUTPUT_DIR, RAW_FEATURES_TO_EXTRACT
from app.models.feature_engineering import build_feature_matrix
from app.models.feature_service import FEATURE_RASTER_MAP
from app.models.suite_registry import predict_proba

logger = logging.getLogger(__name__)

ALIGNED_DIR = OUTPUT_DIR / "aligned"

# Must match create_sample_points() in sar_processor.py
HAND_MIN, HAND_MAX = 0.0, 15.0
SLOPE_MAX = 5.0
CHANNEL_BUFFER_M = 25.0
OUTSIDE_VALUE = 0.0

CHUNK_ROWS = 256


def load_raw_stack() -> Tuple[Dict[str, np.ndarray], dict, dict]:
    """
    Load every aligned raw raster (nodata -> NaN).
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
            arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            arr[arr <= -9990] = np.nan  # safety net for stray nodata sentinels
            arrays[feat] = arr
            if meta is None:
                meta = src.meta.copy()
                transform = src.transform
            elif (src.height, src.width) != (meta["height"], meta["width"]):
                logger.error(
                    f"Grid mismatch on {feat}: {src.height}x{src.width} vs "
                    f"{meta['height']}x{meta['width']}"
                )

    return arrays, meta, transform


def flood_able_mask(arrays: Dict[str, np.ndarray]):
    """Return (land, channel, domain) boolean masks matching the training domain."""
    hand = arrays.get("hand")
    slope = arrays.get("slope")
    dist = arrays.get("distance_to_river")
    if hand is None or slope is None:
        raise RuntimeError("HAND and slope rasters are required to define the flood-able domain.")

    with np.errstate(invalid="ignore"):
        # Land comes from HAND so CHIRPS gaps can't punch blocky holes in the map
        land = np.isfinite(hand)

        in_terrain = (
            np.isfinite(hand) & (hand >= HAND_MIN) & (hand <= HAND_MAX)
            & np.isfinite(slope) & (slope <= SLOPE_MAX)
        )
        if dist is not None:
            channel = np.isfinite(dist) & (dist <= CHANNEL_BUFFER_M)
        else:
            channel = np.zeros_like(land)

    domain = land & in_terrain & ~channel
    return land, channel, domain


def _training_fill(name: str) -> Optional[float]:
    """Fill value used during training (feature_engineering.get_training_stats), if available."""
    try:
        from app.models.feature_engineering import get_training_stats
        v = get_training_stats()["fill"].get(name)
        return float(v) if v is not None and np.isfinite(v) else None
    except Exception:
        return None


def _fill_median(arrays: Dict[str, np.ndarray], domain: np.ndarray) -> None:
    """Fill NaN in all raw features within the domain."""
    if not domain.any():
        return

    for name, a in arrays.items():
        missing = domain & ~np.isfinite(a)
        if not missing.any():
            continue

        fill = _training_fill(name)
        if fill is None or not np.isfinite(fill):
            fill = float(np.nanmedian(a[domain]))
            if not np.isfinite(fill):
                fill = float(np.nanmedian(a))
            if not np.isfinite(fill):
                fill = 0.0

        a[missing] = fill
        logger.info(f"Filled {int(missing.sum()):,} NaN '{name}' pixels with {fill:.4f}")


def predict_full_raster(model, feature_names, chunk_rows: int = CHUNK_ROWS) -> Tuple[np.ndarray, dict]:
    """Run a suite model on every flood-able pixel. Returns (probability_raster, meta)."""
    raw_arrays, meta, _ = load_raw_stack()
    if not raw_arrays:
        raise RuntimeError("No aligned rasters found; run preprocessing first.")

    land, channel, domain = flood_able_mask(raw_arrays)
    _fill_median(raw_arrays, domain)

    h, w = domain.shape
    logger.info(
        f"Grid {h}x{w}: land={land.sum():,}, flood-able={domain.sum():,} "
        f"({domain.sum() / max(land.sum(), 1) * 100:.1f}% of land), channel={channel.sum():,}"
    )

    out = np.full((h, w), np.nan, dtype=np.float32)
    out[land & ~domain & ~channel] = OUTSIDE_VALUE

    cols = np.arange(w)
    for r0 in range(0, h, chunk_rows):
        r1 = min(r0 + chunk_rows, h)
        dom = domain[r0:r1]
        if not dom.any():
            continue

        chunk = {k: v[r0:r1] for k, v in raw_arrays.items()}
        stack = build_feature_matrix(chunk, feature_names)
        flat = stack.reshape(-1, stack.shape[-1]).astype(np.float32)

        sel = dom.reshape(-1)
        X = np.nan_to_num(flat[sel], nan=0.0, posinf=0.0, neginf=0.0)

        # (row, col) pixel indices, same coordinate system as training
        rr, cc = np.meshgrid(np.arange(r0, r1), cols, indexing="ij")
        coords = np.column_stack([rr.ravel(), cc.ravel()])[sel]

        block = out[r0:r1].reshape(-1)
        block[sel] = predict_proba(model, coords, X).astype(np.float32)
        out[r0:r1] = block.reshape(r1 - r0, w)

    meta.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    return out, meta


def quantile_breaks(raster: np.ndarray, qs=(0.50, 0.75, 0.90)) -> Tuple[float, ...]:
    """Class breaks from flood-able pixels only (ignores NaN and OUTSIDE_VALUE)."""
    v = raster[np.isfinite(raster) & (raster > OUTSIDE_VALUE)]
    if v.size == 0:
        return (0.40, 0.55, 0.70)
    return tuple(float(x) for x in np.quantile(v, qs))


def classify_raster(raster: np.ndarray, breaks=(0.40, 0.55, 0.70)) -> np.ndarray:
    """Reclassify probability raster into 0..3 (0=Low, 1=Medium, 2=High, 3=Very High); -1 = nodata."""
    classes = np.digitize(raster, breaks, right=False).astype(np.int8)
    classes[~np.isfinite(raster)] = -1
    return classes


def class_meta(meta: dict) -> dict:
    """Metadata for saving the classified raster."""
    m = meta.copy()
    m.update(dtype="int8", nodata=-1)
    return m


def save_geotiff(array: np.ndarray, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array, 1)
    logger.info(f"Saved {path}")