import numpy as np
from typing import Dict, Any

EPS = 1e-9


def _arr(x, ref):
    """Broadcast a scalar or array to match `ref`'s shape."""
    if np.isscalar(x):
        return np.full_like(ref, x, dtype=np.float32) if hasattr(ref, "shape") else x
    return x


def compute_engineered_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a dict of raw features (either scalars or 2D arrays),
    return a dict of engineered features with the same broadcast shape.
    """
    dem = np.asarray(raw.get("dem", 0.0), dtype=np.float32)
    slope = np.asarray(raw.get("slope", 0.0), dtype=np.float32)
    aspect = np.asarray(raw.get("aspect", 0.0), dtype=np.float32)
    twi = np.asarray(raw.get("twi", 0.0), dtype=np.float32)
    hand = np.asarray(raw.get("hand", 0.0), dtype=np.float32)
    flow_acc = np.asarray(raw.get("flow_accumulation", 0.0), dtype=np.float32)
    dist_river = np.asarray(raw.get("distance_to_river", 0.0), dtype=np.float32)
    drainage = np.asarray(raw.get("drainage_density", 0.0), dtype=np.float32)
    chirps = np.asarray(raw.get("chirps", 0.0), dtype=np.float32)
    gpm = np.asarray(raw.get("gpm", 0.0), dtype=np.float32)

    out: Dict[str, Any] = {}

    # --- Hydrological composites ---
    out["wetness_index"] = twi / (1.0 + slope)
    out["saturation_index"] = twi * flow_acc / (dist_river + 1.0)
    out["flood_concentration"] = flow_acc / (hand + 1.0)
    out["drainage_capacity"] = drainage / (slope + EPS)
    out["twi_hand_ratio"] = twi / (hand + 1.0)

    # --- Terrain ---
    out["terrain_roughness"] = slope * flow_acc
    out["northness"] = np.cos(np.radians(aspect))
    out["eastness"] = np.sin(np.radians(aspect))
    out["relative_elevation"] = dem - hand
    out["slope_aspect"] = slope * aspect

    # --- River proximity ---
    out["river_proximity"] = 1.0 / (dist_river + 1.0)
    out["river_influence"] = np.exp(-dist_river / 100.0)
    out["floodplain_index"] = 1.0 / ((hand + 1.0) * (slope + 1.0))

    # --- Rainfall ---
    out["rainfall_intensity"] = (chirps + gpm) / 2.0
    out["rainfall_ratio"] = gpm / (chirps + EPS)
    out["rainfall_wetness"] = out["rainfall_intensity"] * twi
    out["rainfall_flow"] = out["rainfall_intensity"] * flow_acc

    # --- Flood indices ---
    out["flood_susceptibility_index"] = (
        out["wetness_index"] * out["river_proximity"] * out["rainfall_intensity"]
    )
    out["flash_flood_potential"] = slope * out["rainfall_intensity"] / (twi + EPS)
    out["water_logging_potential"] = (1.0 / (slope + EPS)) * twi

    # --- Interactions ---
    out["twi_hand"] = twi * hand
    out["slope_flow"] = slope * flow_acc
    out["dist_slope"] = dist_river * slope
    out["drainage_rainfall"] = drainage * out["rainfall_intensity"]

    # --- Polynomial ---
    out["twi_squared"] = twi ** 2
    out["flow_acc_squared"] = flow_acc ** 2
    out["hand_squared"] = hand ** 2
    out["slope_squared"] = slope ** 2
    out["log_twi"] = np.log1p(np.abs(twi))
    out["log_flow_acc"] = np.log1p(np.abs(flow_acc))
    out["log_hand"] = np.log1p(np.abs(hand))
    out["log_slope"] = np.log1p(np.abs(slope))

    # --- Normalized / ratios ---
    out["dist_center_norm"] = dist_river / (dist_river + 1.0)
    out["dem_river_ratio"] = dem / (dist_river + 1.0)
    out["hand_dem_ratio"] = hand / (dem + EPS)
    out["twi_slope_ratio"] = twi / (slope + EPS)
    out["flow_dist_ratio"] = flow_acc / (dist_river + 1.0)

    return out


def build_feature_matrix(raw: Dict[str, Any], feature_names):
    """
    Given raw features and the ordered list of feature names the model expects,
    return a numpy array in that exact order.
    Handles both a single point (1D) and raster grids (2D).
    """
    engineered = compute_engineered_features(raw)
    combined = {**raw, **engineered}
    missing = [n for n in feature_names if n not in combined]
    if missing:
        import logging
        logging.getLogger(__name__).warning(
            f"Missing {len(missing)} feature(s): {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    return np.stack([combined.get(n, 0.0) for n in feature_names], axis=-1)