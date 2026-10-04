# backend/app/main.py
"""
Flood Susceptibility API Server
FastAPI backend for serving flood susceptibility maps and model predictions.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.models.loader import registry
from app.models.feature_service import build_vector_at_point
from app.models.raster_predict import (
    predict_full_raster,
    classify_raster,
    save_geotiff,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Flood Susceptibility API",
    description="Zamboanga City Flood Susceptibility Mapping System",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("./outputs/phase2")
PHASE3_DIR = Path("./outputs/phase3")


# ---------- Phase 2 data (optional; API works without it) ----------
try:
    X_train = np.load(OUTPUT_DIR / "preprocessed/X_train.npy")
    X_test = np.load(OUTPUT_DIR / "preprocessed/X_test.npy")
    y_train = np.load(OUTPUT_DIR / "preprocessed/y_train.npy")
    y_test = np.load(OUTPUT_DIR / "preprocessed/y_test.npy")
    with open(OUTPUT_DIR / "preprocessed/feature_names.json") as f:
        feature_names = json.load(f)
    logger.info(f"Phase 2 arrays: X_train {X_train.shape}")
except Exception as e:
    logger.warning(f"Phase 2 preprocessed data not found: {e}")
    X_train = X_test = y_train = y_test = None
    feature_names = []

try:
    sample_points = gpd.read_file(OUTPUT_DIR / "sample_points/sample_points.shp")
    logger.info(f"Loaded {len(sample_points)} sample points")
except Exception:
    sample_points = None


# ---------- One-shot scaler diagnostic ----------
_sc = getattr(registry, "scaler", None)
if _sc is None:
    logger.info("registry.scaler is None (no scaling applied)")
elif hasattr(_sc, "transform"):
    logger.info(f"registry.scaler is a fitted {type(_sc).__name__}")
elif isinstance(_sc, dict):
    logger.info(f"registry.scaler is a dict with keys: {sorted(_sc.keys())}")
    nested = _sc.get("scaler", None)
    if nested is None:
        logger.info("  -> no nested 'scaler' key")
    elif hasattr(nested, "transform"):
        logger.info(f"  -> nested scaler is a fitted {type(nested).__name__}")
    elif isinstance(nested, dict):
        logger.info(f"  -> nested scaler is a dict: {sorted(nested.keys())}")
    else:
        logger.info(f"  -> nested scaler is {type(nested).__name__}")
else:
    logger.warning(f"registry.scaler is an unexpected type: {type(_sc).__name__}")


# ---------- Request schemas ----------
class PredictPointRequest(BaseModel):
    lat: float
    lng: float
    model: str = "stacking"
    features: Optional[dict] = None
    explain: bool = True           # include top-5 factor attribution


class PredictBatchRequest(BaseModel):
    points: List[dict]             # each: {"lat": ..., "lng": ...}
    model: str = "stacking"
    explain: bool = False          # opt-in: O(n_features) extra calls per point


class GenerateRasterRequest(BaseModel):
    model: str = "stacking"


# ---------- Risk classification ----------
# Bins MUST match ModelTrainingPipeline.risk_bins in app/training/pipeline.py.
RISK_LEVELS = [
    {"index": 0, "label": "Low Risk",       "color": "#91bfdb", "upper": 0.25},
    {"index": 1, "label": "Medium Risk",    "color": "#fee090", "upper": 0.50},
    {"index": 2, "label": "High Risk",      "color": "#fc8d59", "upper": 0.75},
    {"index": 3, "label": "Very High Risk", "color": "#d73027", "upper": 1.00},
]


def classify_risk(p: float) -> dict:
    """Map a probability to a risk class using the same bins as training."""
    p = float(np.clip(p, 0.0, 1.0))
    for r in RISK_LEVELS:
        if p <= r["upper"]:
            return r
    return RISK_LEVELS[-1]


def _risk_payload(p: float) -> dict:
    """Uniform risk payload used by every prediction endpoint."""
    r = classify_risk(p)
    return {
        "probability": float(p),
        "risk_class": r["label"],
        "risk_index": r["index"],
        "color": r["color"],
    }


def _registry_models() -> list:
    try:
        models = registry.list_models()
        return list(models) if models is not None else []
    except Exception as e:
        logger.warning(f"registry.list_models() failed: {e}")
        return []


def _registry_feature_names() -> list:
    try:
        names = getattr(registry, "feature_names", None)
        return list(names) if names else []
    except Exception:
        return []


def _active_feature_set() -> str:
    try:
        names = list(getattr(registry, "feature_names", []) or [])
        eng = getattr(registry, "_feature_names_eng", None)
        if eng is None:
            return "unknown"
        eng = list(eng)
        return "engineered" if names == eng else "raw"
    except Exception:
        return "unknown"


def _registry_scaler():
    return getattr(registry, "scaler", None)


def _apply_scaler(x: np.ndarray) -> np.ndarray:
    """
    Apply the registry scaler to a feature array.

    Handles: fitted sklearn scaler, None, or a wrapper dict of the form
    {'method': ..., 'params': ..., 'scaler': <fitted|dict>}.
    """
    x = np.asarray(x, dtype=np.float64)
    scaler = getattr(registry, "scaler", None)

    if scaler is None:
        return x.astype(np.float32)

    if hasattr(scaler, "transform"):
        return scaler.transform(x).astype(np.float32)

    if isinstance(scaler, dict):
        mean = scaler.get("mean", scaler.get("mu", scaler.get("center")))
        scale = scaler.get("scale", scaler.get("sigma", scaler.get("std")))
        if mean is not None and scale is not None:
            mean = np.asarray(mean, dtype=np.float64)
            scale = np.asarray(scale, dtype=np.float64)
            scale = np.where(scale == 0, 1.0, scale)
            return ((x - mean) / scale).astype(np.float32)

        nested = scaler.get("scaler", None)
        if nested is not None:
            if hasattr(nested, "transform"):
                return nested.transform(x).astype(np.float32)
            if isinstance(nested, dict):
                inner_mean = nested.get("mean", nested.get("mu", nested.get("center")))
                inner_scale = nested.get("scale", nested.get("sigma", nested.get("std")))
                if inner_mean is not None and inner_scale is not None:
                    inner_mean = np.asarray(inner_mean, dtype=np.float64)
                    inner_scale = np.asarray(inner_scale, dtype=np.float64)
                    inner_scale = np.where(inner_scale == 0, 1.0, inner_scale)
                    return ((x - inner_mean) / inner_scale).astype(np.float32)

        raise RuntimeError(
            f"registry.scaler is a dict but has no recognizable "
            f"mean/scale keys and no usable nested 'scaler'. "
            f"Actual keys: {sorted(scaler.keys())}"
        )

    raise RuntimeError(
        f"Unrecognized registry.scaler type: {type(scaler).__name__}"
    )


# ---------- Spatial models (GWR / MGWR) need coordinates ----------
SPATIAL_MODELS = {"gwr", "mgwr"}
# Must match the 'GWR/MGWR reduced feature set' line in the training log.
SPATIAL_COLS = ["dem", "slope", "aspect", "twi", "hand",
                "flow_accumulation", "chirps", "gpm"]
_GRID: Dict[str, Any] = {}


def _pixel_rowcol(lat: float, lng: float):
    """lat/lng -> (row, col) on the aligned grid, matching training coords."""
    import pyproj
    import rasterio
    from rasterio.transform import rowcol
    from app.preprocessing.config import TARGET_CRS

    if not _GRID:
        with rasterio.open(OUTPUT_DIR / "aligned/dem_aligned.tif") as src:
            _GRID["transform"] = src.transform
        _GRID["tf"] = pyproj.Transformer.from_crs(
            "EPSG:4326", TARGET_CRS, always_xy=True
        )
    x, y = _GRID["tf"].transform(lng, lat)
    r, c = rowcol(_GRID["transform"], x, y)
    return float(r), float(c)


class _SpatialAdapter:
    """Gives GWR/MGWR a predict_proba(X) interface for known locations."""

    def __init__(self, model, names, points):      # points: [(lat, lng), ...]
        self.model = model
        self.idx = [names.index(f) for f in SPATIAL_COLS]
        self.coords = np.array([_pixel_rowcol(la, ln) for la, ln in points])

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if len(self.coords) == len(X):
            coords = self.coords
        elif len(self.coords) == 1:
            coords = np.repeat(self.coords, len(X), axis=0)
        else:
            raise ValueError(
                f"{len(X)} rows but {len(self.coords)} coordinates")
        p = np.clip(self.model.predict_at(coords, X[:, self.idx]), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


# ---------- Feature attribution (model-agnostic) ----------

def _predict_proba_single(model, x_scaled: np.ndarray) -> float:
    """Run a single-row prediction, returning P(class=1)."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x_scaled)[0, 1])
    raw = np.asarray(model.predict(x_scaled)).ravel()
    return float(raw[0])


def _feature_attributions(
    model,
    x_scaled: np.ndarray,
    names: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Local feature attribution via mean-ablation.

    For each feature i, we ask: how much does this single point's predicted
    flood probability change if feature i is reset to its training mean and
    everything else is held fixed?

        contribution_i = p(x) - p(x with x_i = training_mean_i)

    In standardized feature space the training mean is 0.0, so ablating a
    feature means setting its scaled value back to zero.

    Positive contribution  -> feature pushed the prediction toward flood.
    Negative contribution  -> feature pushed the prediction toward non-flood.

    The magnitude |contribution| is the strength of the feature's influence
    on *this point's* prediction.

    Returns the top_k features sorted by |contribution| descending.
    """
    x_scaled = np.asarray(x_scaled, dtype=np.float64)
    if x_scaled.ndim == 1:
        x_scaled = x_scaled.reshape(1, -1)

    p_full = _predict_proba_single(model, x_scaled)

    contributions: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        if i >= x_scaled.shape[1]:
            break
        x_ablated = x_scaled.copy()
        x_ablated[0, i] = 0.0  # training mean in scaled space
        try:
            p_ablated = _predict_proba_single(model, x_ablated)
        except Exception:
            continue
        c = p_full - p_ablated
        if not np.isfinite(c):
            continue
        contributions.append({
            "feature": name,
            "contribution": float(c),
            "direction": "increases" if c > 0 else "decreases",
        })

    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions[:top_k]


# ---------- Core endpoints ----------
@app.get("/")
async def root():
    return {
        "message": "Flood Susceptibility API",
        "version": "2.0.0",
        "endpoints": [
            "/api/status",
            "/api/models",
            "/api/metrics",
            "/api/sample_points",
            "/api/features",
            "/api/sar_summary",
            "/api/gwpca",
            "/api/predict/point",
            "/api/predict/batch",
            "/api/generate_susceptibility",
            "/api/susceptibility/at",
            "/api/download/{file_type}",
        ],
    }


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "phase2_loaded": X_train is not None,
        "n_train_samples": int(len(X_train)) if X_train is not None else 0,
        "n_test_samples": int(len(X_test)) if X_test is not None else 0,
        "phase2_feature_names": len(feature_names),
        "models_loaded": _registry_models(),
        "active_feature_count": len(_registry_feature_names()),
        "active_feature_set": _active_feature_set(),
        "sample_points_loaded": sample_points is not None,
    }


@app.get("/api/models")
async def list_models():
    return {
        "loaded_models": _registry_models(),
        "is_ready": bool(getattr(registry, "is_ready", False)),
        "feature_names": _registry_feature_names(),
        "n_features": len(_registry_feature_names()),
        "risk_mapping": getattr(registry, "risk_mapping", RISK_LEVELS),
    }


@app.get("/api/metrics")
async def get_metrics():
    metrics = getattr(registry, "metrics", None)
    if not metrics:
        raise HTTPException(404, "Phase 3 metrics not found")
    return metrics


@app.get("/api/sample_points")
async def get_sample_points(format: str = Query("geojson")):
    if sample_points is None:
        raise HTTPException(404, "Sample points not found")
    if format == "csv":
        df = pd.DataFrame(sample_points.drop(columns=["geometry"]))
        return JSONResponse(df.to_dict(orient="records"))
    return JSONResponse(json.loads(sample_points.to_json()))


@app.get("/api/features")
async def get_features():
    try:
        stats = pd.read_csv(OUTPUT_DIR / "feature_statistics.csv")
        return JSONResponse(stats.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(404, f"Feature stats not found: {e}")


@app.get("/api/sar_summary")
async def get_sar_summary():
    try:
        df = pd.read_csv(OUTPUT_DIR / "sar_summary.csv")
        return JSONResponse(df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(404, f"SAR summary not found: {e}")


@app.get("/api/gwpca")
async def get_gwpca_results():
    try:
        loadings = pd.read_csv(OUTPUT_DIR / "gwpca/component_loadings.csv")
        with open(OUTPUT_DIR / "gwpca/params.json") as f:
            params = json.load(f)
        return {"loadings": loadings.to_dict(orient="records"), "params": params}
    except Exception as e:
        raise HTTPException(404, f"GWPCA results not found: {e}")


# ---------- Prediction endpoints ----------
@app.post("/api/predict/point")
async def predict_point(req: PredictPointRequest):
    """
    Predict flood susceptibility for a single lat/lng.

    If `features` are provided, uses them directly (must be in feature_names
    order). Otherwise samples the aligned rasters.

    When `explain=true` (default), the response includes `top_factors`: the
    5 features whose mean-ablation changes the predicted probability the
    most, with direction ("increases" / "decreases").

    GWR / MGWR are location-dependent: they are wrapped in a _SpatialAdapter
    so the point's pixel row/col is passed to predict_at().
    """
    model = registry.get(req.model)
    if model is None:
        raise HTTPException(404, f"Model '{req.model}' not loaded")

    names = _registry_feature_names()

    if req.model in SPATIAL_MODELS:
        try:
            model = _SpatialAdapter(model, names, [(req.lat, req.lng)])
        except Exception as e:
            raise HTTPException(
                422, f"Could not build spatial context for '{req.model}': {e}"
            )

    if req.features is not None:
        missing = [k for k in names if k not in req.features]
        if missing:
            raise HTTPException(
                422,
                f"Provided features are missing required keys: {missing[:5]}"
                + ("..." if len(missing) > 5 else ""),
            )
        x = np.array(
            [[float(req.features[k]) for k in names]],
            dtype=np.float32,
        )
    else:
        try:
            x = build_vector_at_point(req.lat, req.lng, names)
        except Exception as e:
            raise HTTPException(422, f"Could not sample features at point: {e}")

    try:
        x_scaled = _apply_scaler(x)
    except Exception as e:
        raise HTTPException(500, f"Scaling failed: {e}")

    try:
        proba = _predict_proba_single(model, x_scaled)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    if not np.isfinite(proba):
        raise HTTPException(
            500,
            "Model returned a non-finite probability. Likely NaN/Inf in "
            "the feature vector — check the aligned rasters at this point.",
        )

    payload = {
        "lat": req.lat,
        "lng": req.lng,
        "model": req.model,
        **_risk_payload(proba),
    }

    if req.explain:
        try:
            payload["top_factors"] = _feature_attributions(
                model, x_scaled, names, top_k=5,
            )
        except Exception as e:
            logger.warning(f"Feature attribution failed: {e}")
            payload["top_factors"] = []

    return payload


@app.post("/api/predict/batch")
async def predict_batch(req: PredictBatchRequest):
    """
    Predict flood susceptibility for a list of (lat, lng) points.

    When `explain=true`, each result also includes `top_factors` (top 5
    mean-ablation attributions). This costs O(n_features) extra model
    calls per point — use it for small batches.
    """
    model = registry.get(req.model)
    if model is None:
        raise HTTPException(404, f"Model '{req.model}' not loaded")

    names = _registry_feature_names()

    if req.explain and len(req.points) > 25:
        logger.warning(
            f"Batch explain=true with {len(req.points)} points will make "
            f"~{len(names) * len(req.points)} model calls; this may be slow."
        )

    rows = []
    row_idx = []
    errors: Dict[int, str] = {}
    for i, p in enumerate(req.points):
        try:
            v = build_vector_at_point(p["lat"], p["lng"], names)
            rows.append(np.asarray(v, dtype=np.float32).ravel())
            row_idx.append(i)
        except Exception as e:
            errors[i] = str(e)

    if not rows:
        raise HTTPException(422, f"No points could be sampled: {errors}")

    X = np.vstack(rows)

    # Spatial models need one coordinate per surviving row.
    raw_model = model
    if req.model in SPATIAL_MODELS:
        try:
            model = _SpatialAdapter(
                raw_model, names,
                [(req.points[i]["lat"], req.points[i]["lng"]) for i in row_idx],
            )
        except Exception as e:
            raise HTTPException(
                422, f"Could not build spatial context for '{req.model}': {e}"
            )

    try:
        X_scaled = _apply_scaler(X)
    except Exception as e:
        raise HTTPException(500, f"Scaling failed: {e}")

    try:
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_scaled)[:, 1]
        else:
            probas = np.asarray(model.predict(X_scaled)).ravel()
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    results = [None] * len(req.points)
    for local_i, original_i in enumerate(row_idx):
        p = req.points[original_i]
        proba = float(probas[local_i])
        if not np.isfinite(proba):
            results[original_i] = {
                "lat": p["lat"],
                "lng": p["lng"],
                "error": "non-finite probability",
            }
            continue

        entry: Dict[str, Any] = {
            "lat": p["lat"],
            "lng": p["lng"],
            **_risk_payload(proba),
        }

        if req.explain:
            try:
                m_explain = (
                    _SpatialAdapter(raw_model, names, [(p["lat"], p["lng"])])
                    if req.model in SPATIAL_MODELS else model
                )
                entry["top_factors"] = _feature_attributions(
                    m_explain,
                    X_scaled[local_i:local_i + 1],
                    names,
                    top_k=5,
                )
            except Exception as e:
                logger.warning(
                    f"Feature attribution failed at point {original_i}: {e}"
                )
                entry["top_factors"] = []

        results[original_i] = entry

    for i, msg in errors.items():
        p = req.points[i]
        results[i] = {"lat": p["lat"], "lng": p["lng"], "error": msg}

    return {
        "count": len(results),
        "n_ok": len(row_idx),
        "n_failed": len(errors),
        "results": results,
    }


@app.post("/api/generate_susceptibility")
async def generate_susceptibility(req: GenerateRasterRequest):
    """Run full-raster prediction and save GeoTIFF (probability + risk classes)."""
    model = registry.get(req.model)
    if model is None:
        raise HTTPException(404, f"Model '{req.model}' not loaded")

    if req.model in SPATIAL_MODELS:
        raise HTTPException(
            501,
            f"Raster generation for '{req.model}' is not supported yet: "
            f"predict_full_raster does not pass pixel coordinates, so it "
            f"would produce a constant map. Use stacking, svm or mars.",
        )

    try:
        raster, meta = predict_full_raster(
            model, _registry_scaler(), _registry_feature_names()
        )
    except Exception as e:
        raise HTTPException(500, f"Raster prediction failed: {e}")

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)

    out_proba = PHASE3_DIR / f"susceptibility_{req.model}.tif"
    save_geotiff(raster, meta, out_proba)

    classes = classify_raster(raster)
    class_meta = meta.copy()
    class_meta.update(dtype="int8", nodata=-1)
    out_classes = PHASE3_DIR / f"risk_classes_{req.model}.tif"
    save_geotiff(classes, class_meta, out_classes)

    finite = raster[np.isfinite(raster)]
    return {
        "model": req.model,
        "probability_raster": str(out_proba),
        "risk_class_raster": str(out_classes),
        "shape": list(raster.shape),
        "valid_pixels": int(finite.size),
        "probability_min": float(finite.min()) if finite.size else None,
        "probability_max": float(finite.max()) if finite.size else None,
        "probability_mean": float(finite.mean()) if finite.size else None,
    }


@app.get("/api/susceptibility/at")
async def susceptibility_at(lat: float, lng: float, model: str = "stacking"):
    """Fast lookup from the cached susceptibility GeoTIFF."""
    import pyproj
    import rasterio
    from rasterio.transform import rowcol
    from app.preprocessing.config import TARGET_CRS

    tif = PHASE3_DIR / f"susceptibility_{model}.tif"
    if not tif.exists():
        raise HTTPException(404, "Run /api/generate_susceptibility first")

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", TARGET_CRS, always_xy=True
    )
    x, y = transformer.transform(lng, lat)

    with rasterio.open(tif) as src:
        row, col = rowcol(src.transform, x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise HTTPException(422, "Point outside raster extent")
        proba = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])

    if not np.isfinite(proba):
        raise HTTPException(422, "No data at that pixel")

    return {
        "lat": lat,
        "lng": lng,
        "model": model,
        **_risk_payload(proba),
    }


# ---------- File download ----------
@app.get("/api/download/{file_type}")
async def download_file(file_type: str):
    files = {
        "features": OUTPUT_DIR / "features.csv",
        "sample_points": OUTPUT_DIR / "sample_points/sample_points.csv",
        "scores": OUTPUT_DIR / "gwpca/scores.npy",
        "train_data": OUTPUT_DIR / "preprocessed/X_train.npy",
        "test_data": OUTPUT_DIR / "preprocessed/X_test.npy",
    }
    if file_type not in files:
        raise HTTPException(404, f"File type '{file_type}' not found")
    path = files[file_type]
    if not path.exists():
        raise HTTPException(404, f"File '{file_type}' not on disk")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_ready": bool(getattr(registry, "is_ready", False)),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)