# app/models/feature_engineering.py
"""
Engineered features for inference. MUST mirror
app/preprocessing/feature_extraction.py::_create_engineered_features exactly.
Dataset-level statistics (p95 values, sample centroid, nodata fill medians)
are taken from the Phase 2 training CSV.
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.preprocessing.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

RAW_NAMES = ["dem", "slope", "aspect", "twi", "hand", "flow_accumulation",
             "distance_to_river", "drainage_density", "chirps", "gpm"]

_STATS: Dict[str, Any] = {}


def get_training_stats() -> Dict[str, Any]:
    """p95s, centroid and fill medians, computed once from the unscaled training CSV."""
    if _STATS:
        return _STATS

    path = OUTPUT_DIR / "full_dataset_engineered.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is needed for inference statistics")
    df = pd.read_csv(path)

    def col(n):
        return np.nan_to_num(df[n].to_numpy(dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)

    def p95(v):
        return max(float(np.percentile(v, 95)), 0.01)

    twi, hand, slope = col("twi"), col("hand"), col("slope")
    flow, dist = col("flow_accumulation"), col("distance_to_river")
    ri = col("chirps") + col("gpm")

    _STATS["p95_twi"] = p95(twi)
    _STATS["p95_log_flow"] = p95(np.log1p(np.clip(flow, 0, 1e6)))
    _STATS["p95_hand"] = p95(hand)
    _STATS["p95_inv_dist"] = p95(1.0 / (dist + 1.0))
    _STATS["p95_rain"] = p95(ri)
    _STATS["p95_slope"] = p95(slope)

    r, c = df["row"].to_numpy(float), df["col"].to_numpy(float)
    _STATS["cx"], _STATS["cy"] = float(r.mean()), float(c.mean())
    _STATS["max_center_dist"] = float(np.sqrt((r - _STATS["cx"]) ** 2
                                              + (c - _STATS["cy"]) ** 2).max())

    # Value Phase 2 used to fill NaN/nodata (median of valid values; 0 if none)
    _STATS["fill"] = {n: float(np.median(col(n))) for n in RAW_NAMES}
    logger.info(f"Loaded inference statistics from {path.name}")
    return _STATS


def _a(raw, key):
    v = np.asarray(raw.get(key, 0.0), dtype=np.float64)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def compute_engineered_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    st = get_training_stats()

    dem, slope, aspect = _a(raw, "dem"), _a(raw, "slope"), _a(raw, "aspect")
    twi, hand = _a(raw, "twi"), _a(raw, "hand")
    flow = _a(raw, "flow_accumulation")
    dist = _a(raw, "distance_to_river")
    drain = _a(raw, "drainage_density")
    chirps, gpm = _a(raw, "chirps"), _a(raw, "gpm")

    flow_safe = np.clip(flow, 0, 1e6)
    log_flow = np.log1p(flow_safe)
    asp = np.radians(aspect)
    north, east = np.cos(asp), np.sin(asp)
    ri = chirps + gpm

    out: Dict[str, Any] = {}

    # 1. Hydrological
    out["wetness_index"] = twi * log_flow
    out["saturation_index"] = np.clip(
        hand / (np.clip(flow, 1e-10, 1e308) + 1), 0, 1e6)
    out["flood_concentration"] = log_flow / np.clip(dist, 1, 1e6)
    out["drainage_capacity"] = slope / (drain + 0.01)
    out["twi_hand_ratio"] = twi / (hand + 0.01)

    # 2. Topographic
    out["terrain_roughness"] = slope * (1 + np.abs(north) + np.abs(east))
    out["northness"] = north
    out["eastness"] = east
    out["relative_elevation"] = hand / (dem + 1)
    out["slope_aspect"] = slope * (1 + np.sin(asp))

    # 3. Proximity
    out["river_proximity"] = 1 / (dist + 1)
    out["river_influence"] = dist * drain
    out["floodplain_index"] = hand / (dist + 1)

    # 4. Rainfall
    out["rainfall_intensity"] = ri
    out["rainfall_ratio"] = chirps / (gpm + 0.01)
    out["rainfall_wetness"] = ri * twi
    out["rainfall_flow"] = ri * log_flow

    # 5. Composite indices (training-sample p95 normalisation)
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

    # 6. Interactions
    out["twi_hand"] = twi * hand
    out["slope_flow"] = slope * log_flow
    out["dist_slope"] = dist * slope
    out["drainage_rainfall"] = drain * ri

    # 7. Polynomial / log
    for name, v in (("twi", twi), ("flow_acc", flow), ("hand", hand), ("slope", slope)):
        vc = np.clip(v, -1e308, 1e308)
        out[f"{name}_squared"] = np.clip(np.abs(vc), 0, 1e154) ** 2
        out[f"log_{name}"] = np.log1p(np.clip(vc, 0, 1e308))

    # 8. Spatial (pixel row/col, as in training)
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

    # 9. Ratios
    out["hand_dem_ratio"] = hand / (dem + 1)
    out["twi_slope_ratio"] = twi / (slope + 0.01)
    out["flow_dist_ratio"] = log_flow / (dist + 1)

    return {k: np.nan_to_num(np.asarray(v, dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)
            for k, v in out.items()}


def build_feature_matrix(raw: Dict[str, Any], feature_names):
    engineered = compute_engineered_features(raw)
    combined = {**raw, **engineered}
    missing = [n for n in feature_names if n not in combined]
    if missing:
        raise KeyError(f"Cannot build {len(missing)} feature(s): {missing[:5]}")
    return np.stack([np.asarray(combined[n], dtype=np.float64)
                     for n in feature_names], axis=-1)