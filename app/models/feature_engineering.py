# app/models/feature_engineering.py
"""
Engineered features for inference. MUST mirror
app/preprocessing/feature_extraction.py::_create_engineered_features exactly.

The set of features the model expects is auto-detected from the Phase-2
training CSV. Anything not present in the trained dataset is dropped at
inference time, so pruning features in config.py cannot silently break
this module.
"""
import logging
from typing import Any, Dict, Set

import numpy as np
import pandas as pd

from app.preprocessing.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

_STATS: Dict[str, Any] = {}
_TRAINING_COLS: Set[str] = set()


def _load_training_cols() -> Set[str]:
    """Read the trained CSV header once and cache the column names."""
    global _TRAINING_COLS
    if _TRAINING_COLS:
        return _TRAINING_COLS
    path = OUTPUT_DIR / "full_dataset_engineered.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is needed for inference statistics")
    df = pd.read_csv(path, nrows=1)
    _TRAINING_COLS = set(df.columns)
    return _TRAINING_COLS


def get_training_stats() -> Dict[str, Any]:
    """p95s, centroid and fill medians, computed once from the unscaled CSV."""
    if _STATS:
        return _STATS

    path = OUTPUT_DIR / "full_dataset_engineered.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is needed for inference statistics")
    df = pd.read_csv(path)

    def col(n):
        if n not in df.columns:
            return np.zeros(len(df), dtype=np.float64)
        return np.nan_to_num(df[n].to_numpy(dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)

    def p95(v):
        return max(float(np.percentile(v, 95)), 0.01)

    twi = col("twi")
    hand = col("hand")
    slope = col("slope")
    flow = col("flow_accumulation")
    dist = col("distance_to_river")
    ri = col("chirps") + col("gpm")

    _STATS["p95_twi"] = p95(twi)
    _STATS["p95_log_flow"] = p95(np.log1p(np.clip(flow, 0, 1e6)))
    _STATS["p95_hand"] = p95(hand)
    _STATS["p95_inv_dist"] = p95(1.0 / (dist + 1.0))
    _STATS["p95_rain"] = p95(ri)
    _STATS["p95_slope"] = p95(slope)

    if "row" in df.columns and "col" in df.columns:
        r = df["row"].to_numpy(float)
        c = df["col"].to_numpy(float)
        _STATS["cx"], _STATS["cy"] = float(r.mean()), float(c.mean())
        _STATS["max_center_dist"] = float(
            np.sqrt((r - _STATS["cx"]) ** 2 + (c - _STATS["cy"]) ** 2).max()
        ) or 1.0
    else:
        _STATS["cx"], _STATS["cy"] = 0.0, 0.0
        _STATS["max_center_dist"] = 1.0

    _STATS["training_cols"] = set(df.columns)
    _STATS["fill"] = {
        n: float(np.median(col(n)))
        for n in df.columns
        if n not in ("flood_label", "row", "col", "sample_id")
    }

    logger.info(f"Loaded inference statistics from {path.name}")
    return _STATS


def _a(raw, key):
    v = np.asarray(raw.get(key, 0.0), dtype=np.float64)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def compute_engineered_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    st = get_training_stats()

    dem = _a(raw, "dem")
    slope = _a(raw, "slope")
    aspect = _a(raw, "aspect")
    twi = _a(raw, "twi")
    hand = _a(raw, "hand")
    flow = _a(raw, "flow_accumulation")
    dist = _a(raw, "distance_to_river")
    drain = _a(raw, "drainage_density")
    chirps = _a(raw, "chirps")
    gpm = _a(raw, "gpm")

    flow_safe = np.clip(flow, 0, 1e6)
    log_flow = np.log1p(flow_safe)
    asp = np.radians(aspect)
    north, east = np.cos(asp), np.sin(asp)
    ri = chirps + gpm

    out: Dict[str, Any] = {}

    # --- compute every candidate; filter to trained columns at the end ---
    out["wetness_index"] = twi * log_flow
    out["saturation_index"] = np.clip(hand / (np.clip(flow, 1e-10, 1e308) + 1), 0, 1e6)
    out["flood_concentration"] = log_flow / np.clip(dist, 1, 1e6)
    out["drainage_capacity"] = slope / (drain + 0.01)
    out["twi_hand_ratio"] = twi / (hand + 0.01)

    out["terrain_roughness"] = slope * (1 + np.abs(north) + np.abs(east))
    out["northness"] = north
    out["eastness"] = east
    out["relative_elevation"] = hand / (dem + 1)
    out["slope_aspect"] = slope * (1 + np.sin(asp))

    out["river_proximity"] = 1 / (dist + 1)
    out["river_influence"] = dist * drain
    out["floodplain_index"] = hand / (dist + 1)

    out["rainfall_intensity"] = ri
    out["rainfall_ratio"] = chirps / (gpm + 0.01)
    out["rainfall_wetness"] = ri * twi
    out["rainfall_flow"] = ri * log_flow

    comps = [
        (twi / st["p95_twi"], 0.25),
        (log_flow / st["p95_log_flow"], 0.20),
        (hand / st["p95_hand"], 0.15),
        ((1 / (dist + 1)) / st["p95_inv_dist"], 0.15),
        (ri / st["p95_rain"], 0.15),
        (slope / st["p95_slope"], 0.10),
    ]
    wsum = sum(w for _, w in comps)
    fsi = sum(c * (w / wsum) for c, w in comps)
    out["flood_susceptibility_index"] = np.clip(fsi, 0, 1)
    out["flash_flood_potential"] = np.clip(slope * chirps, 0, 1e6)
    out["water_logging_potential"] = np.clip(twi / (slope + 0.01), 0, 1e6)

    out["twi_hand"] = twi * hand
    out["slope_flow"] = slope * log_flow
    out["dist_slope"] = dist * slope
    out["drainage_rainfall"] = drain * ri

    for name, v in (("twi", twi), ("flow_acc", flow), ("hand", hand), ("slope", slope)):
        vc = np.clip(v, -1e308, 1e308)
        out[f"{name}_squared"] = np.clip(np.abs(vc), 0, 1e154) ** 2
        out[f"log_{name}"] = np.log1p(np.clip(vc, 0, 1e308))

    # Spatial
    row, colm = raw.get("_row"), raw.get("_col")
    if row is None or colm is None:
        if dem.ndim == 2:
            row, colm = np.indices(dem.shape)
        else:
            row, colm = 0.0, 0.0
    row = np.asarray(row, dtype=np.float64)
    colm = np.asarray(colm, dtype=np.float64)
    dc = np.sqrt((row - st["cx"]) ** 2 + (colm - st["cy"]) ** 2)
    out["dist_center_norm"] = dc / (st["max_center_dist"] + 0.01)
    out["dem_river_ratio"] = dem / (dist + 1)
    out["hand_dem_ratio"] = hand / (dem + 1)
    out["twi_slope_ratio"] = twi / (slope + 0.01)
    out["flow_dist_ratio"] = log_flow / (dist + 1)

    # --- FILTER: only keep engineered features present in the trained dataset ---
    trained = st["training_cols"]
    out = {
        k: np.nan_to_num(np.asarray(v, dtype=np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)
        for k, v in out.items()
        if k in trained
    }
    return out


def build_feature_matrix(raw: Dict[str, Any], feature_names):
    engineered = compute_engineered_features(raw)
    combined = {**raw, **engineered}
    missing = [n for n in feature_names if n not in combined]
    if missing:
        raise KeyError(f"Cannot build {len(missing)} feature(s): {missing[:5]}")
    return np.stack([np.asarray(combined[n], dtype=np.float64)
                     for n in feature_names], axis=-1)