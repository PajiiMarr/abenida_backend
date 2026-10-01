# backend/app/main.py
"""
Flood Susceptibility API Server
FastAPI backend for serving flood susceptibility maps and model predictions.
"""

from pathlib import Path
from typing import List, Optional
import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
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

# ---------- Phase 2 data (existing) ----------
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


# ---------- Request schemas ----------
class PredictPointRequest(BaseModel):
    lat: float
    lng: float
    model: str = "stacking"
    features: Optional[dict] = None


class PredictBatchRequest(BaseModel):
    points: List[dict]  # each: {"lat": ..., "lng": ...}
    model: str = "stacking"


class GenerateRasterRequest(BaseModel):
    model: str = "stacking"


# ---------- Risk classification ----------
RISK_LEVELS = [
    {"index": 0, "label": "Low Risk",       "color": "#91bfdb", "upper": 0.2},
    {"index": 1, "label": "Medium Risk",    "color": "#fee090", "upper": 0.4},
    {"index": 2, "label": "High Risk",      "color": "#fc8d59", "upper": 0.6},
    {"index": 3, "label": "Very High Risk", "color": "#d73027", "upper": 1.0},
]


def classify_risk(p: float) -> dict:
    for r in RISK_LEVELS:
        if p <= r["upper"]:
            return r
    return RISK_LEVELS[-1]


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
        "n_train_samples": len(X_train) if X_train is not None else 0,
        "n_test_samples": len(X_test) if X_test is not None else 0,
        "phase2_feature_names": len(feature_names),
        "models_loaded": registry.list_models(),
        "active_feature_count": len(registry.feature_names),
        "active_feature_set": (
            "engineered" if registry.feature_names == registry._feature_names_eng
            else "raw"
        ),
        "sample_points_loaded": sample_points is not None,
    }


@app.get("/api/models")
async def list_models():
    return {
        "loaded_models": registry.list_models(),
        "is_ready": registry.is_ready,
        "feature_names": registry.feature_names,
        "n_features": len(registry.feature_names),
        "risk_mapping": registry.risk_mapping,
    }


@app.get("/api/metrics")
async def get_metrics():
    if not registry.metrics:
        raise HTTPException(404, "Phase 3 metrics not found")
    return registry.metrics


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
    If `features` are provided, uses them directly (must be in feature_names order).
    Otherwise samples the aligned rasters.
    """
    model = registry.get(req.model)
    if model is None:
        raise HTTPException(404, f"Model '{req.model}' not loaded")

    if req.features is not None:
        x = np.array([[float(req.features[k]) for k in registry.feature_names]],
                     dtype=np.float32)
    else:
        x = build_vector_at_point(req.lat, req.lng, registry.feature_names)

    if registry.scaler is not None:
        x = registry.scaler.transform(x).astype(np.float32)

    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(x)[0, 1])
        else:
            proba = float(np.asarray(model.predict(x)).ravel()[0])
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    risk = classify_risk(proba)
    return {
        "lat": req.lat,
        "lng": req.lng,
        "model": req.model,
        "probability": proba,
        "risk_class": risk["label"],
        "risk_index": risk["index"],
        "color": risk["color"],
    }


@app.post("/api/predict/batch")
async def predict_batch(req: PredictBatchRequest):
    model = registry.get(req.model)
    if model is None:
        raise HTTPException(404, f"Model '{req.model}' not loaded")

    X = np.vstack([
        build_vector_at_point(p["lat"], p["lng"], registry.feature_names)
        for p in req.points
    ])

    if registry.scaler is not None:
        X = registry.scaler.transform(X).astype(np.float32)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1]
    else:
        probas = np.asarray(model.predict(X)).ravel()

    results = []
    for p, proba in zip(req.points, probas):
        r = classify_risk(float(proba))
        results.append({
            "lat": p["lat"],
            "lng": p["lng"],
            "probability": float(proba),
            "risk_class": r["label"],
            "risk_index": r["index"],
        })
    return {"count": len(results), "results": results}


@app.post("/api/generate_susceptibility")
async def generate_susceptibility(req: GenerateRasterRequest):
    """
    Run full-raster prediction and save GeoTIFF (probability + risk classes).
    """
    model = registry.get(req.model)
    if model is None:
        raise HTTPException(404, f"Model '{req.model}' not loaded")

    try:
        raster, meta = predict_full_raster(
            model, registry.scaler, registry.feature_names
        )
    except Exception as e:
        raise HTTPException(500, f"Raster prediction failed: {e}")

    out_proba = PHASE3_DIR / f"susceptibility_{req.model}.tif"
    save_geotiff(raster, meta, out_proba)

    classes = classify_raster(raster)
    class_meta = meta.copy()
    class_meta.update(dtype="int8", nodata=-1)
    out_classes = PHASE3_DIR / f"risk_classes_{req.model}.tif"
    save_geotiff(classes, class_meta, out_classes)

    # Summary stats
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
    from rasterio.transform import rowcol
    import rasterio
    from app.preprocessing.config import TARGET_CRS

    tif = PHASE3_DIR / f"susceptibility_{model}.tif"
    if not tif.exists():
        raise HTTPException(404, "Run /api/generate_susceptibility first")

    transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x, y = transformer.transform(lng, lat)

    with rasterio.open(tif) as src:
        row, col = rowcol(src.transform, x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise HTTPException(422, "Point outside raster extent")
        proba = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])

    risk = classify_risk(proba)
    return {"lat": lat, "lng": lng, "probability": proba, **risk}


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
    return {"status": "healthy", "models_ready": registry.is_ready}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)