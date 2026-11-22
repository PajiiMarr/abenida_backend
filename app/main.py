# backend/app/main.py
"""
Flood Susceptibility API Server
FastAPI backend serving the run_suite.py model suite (m1-m5 per event / overall).
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import logging
import os
import secrets
import subprocess
import sys
import threading

import geopandas as gpd
import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# External Integrations
from supabase import create_client, Client
from azure.storage.blob import BlobClient

from app.models.suite_registry import (
    DEFAULT_MODEL,
    MODELS,
    PHASE2_DIR,
    PHASE3_DIR,
    available_events,
    load_model,
    model_path,
    predict_proba,
    resolve_model,
    safe_event,
    suite_features,
)
from app.preprocessing.config import register_dynamic_event
from app.models.feature_engineering import build_feature_matrix
from app.models.feature_service import sample_raw_features, pixel_at
from app.models.raster_predict import (
    predict_full_raster,
    classify_raster,
    quantile_breaks,
    class_meta,
    save_geotiff,
    HAND_MIN,
    HAND_MAX,
    SLOPE_MAX,
    CHANNEL_BUFFER_M,
    OUTSIDE_VALUE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Flood Susceptibility API",
    description="Zamboanga City Flood Susceptibility Mapping System",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fsizc.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_origin_regex=r"https://fsizc-.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Risk-Breaks"],  # your raster code reads this header
)

# ---------- API key auth ----------
_FSI_API_KEY_ENV = os.environ.get("FSI_API_KEY")
if _FSI_API_KEY_ENV:
    API_KEY = _FSI_API_KEY_ENV
    logger.info("FSI_API_KEY loaded from environment.")
else:
    API_KEY = secrets.token_urlsafe(32)
    logger.warning(
        "FSI_API_KEY not set in environment; generated a random one for "
        f"this session: {API_KEY}"
    )

def require_key(x_api_key: str = Header(..., description="API key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---------- Database & Storage Setup ----------
SUPABASE_URL = os.environ.get("VITE_SUPABASE_URL", "https://xflhynxdadwlrloxiogv.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
BLOB_SAS_URL = os.environ.get("BLOB_SAS_URL", "")

if not SUPABASE_KEY or not BLOB_SAS_URL:
    logger.warning("Missing Supabase or Azure credentials in environment variables. Upload endpoint may fail.")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    supabase = None

def upload_to_azure(file_bytes: bytes, blob_name: str):
    if not BLOB_SAS_URL:
        raise ValueError("BLOB_SAS_URL is not configured.")
    base_url, sas_token = BLOB_SAS_URL.split("?", 1)
    blob_url = f"{base_url}/{blob_name}?{sas_token}"
    blob_client = BlobClient.from_blob_url(blob_url)
    blob_client.upload_blob(file_bytes, overwrite=True)
    return f"{base_url}/{blob_name}"

# ---------- Paths & Data ----------
OUTPUT_DIR = PHASE2_DIR
COMPARISON_CSV = PHASE3_DIR / "suite_model_comparison.csv"

try:
    X_train = np.load(OUTPUT_DIR / "preprocessed/X_train.npy")
    X_test = np.load(OUTPUT_DIR / "preprocessed/X_test.npy")
    y_train = np.load(OUTPUT_DIR / "preprocessed/y_train.npy")
    y_test = np.load(OUTPUT_DIR / "preprocessed/y_test.npy")
    with open(OUTPUT_DIR / "preprocessed/feature_names.json") as f:
        phase2_feature_names = json.load(f)
    logger.info(f"Phase 2 arrays: X_train {X_train.shape}")
except Exception as e:
    logger.warning(f"Phase 2 preprocessed data not found: {e}")
    X_train = X_test = y_train = y_test = None
    phase2_feature_names = []

try:
    sample_points = gpd.read_file(OUTPUT_DIR / "sample_points/sample_points.shp")
    logger.info(f"Loaded {len(sample_points)} sample points")
except Exception:
    sample_points = None

# ---------- Request schemas ----------
class PredictPointRequest(BaseModel):
    lat: float
    lng: float
    model: str = DEFAULT_MODEL
    event: str = "overall"
    features: Optional[dict] = None
    explain: bool = True

class PredictBatchRequest(BaseModel):
    points: List[dict]
    model: str = DEFAULT_MODEL
    event: str = "overall"
    explain: bool = False
    simulation_override: Optional[Dict[str, float]] = None

class GenerateRasterRequest(BaseModel):
    model: str = DEFAULT_MODEL
    event: str = "overall"

# ---------- Model / event resolution ----------
def _resolve(event: str, model: str) -> str:
    """Validate event + model, return the canonical model id."""
    try:
        m = resolve_model(model)
        path = model_path(event, m)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not path.exists():
        raise HTTPException(404, f"Model '{m}' for event '{event}' not found")
    return m

def _load(event: str, model: str):
    try:
        return load_model(event, model)
    except Exception as e:
        raise HTTPException(500, f"Could not load model '{model}' for '{event}': {e}")

# ---------- Risk classification (breaks are per event + model) ----------
DEFAULT_BREAKS = [0.40, 0.55, 0.70]

RISK_LEVELS = [
    {"index": 0, "label": "Low Risk",       "color": "#91bfdb"},
    {"index": 1, "label": "Medium Risk",    "color": "#fee090"},
    {"index": 2, "label": "High Risk",      "color": "#fc8d59"},
    {"index": 3, "label": "Very High Risk", "color": "#d73027"},
]

_BREAKS_CACHE: Dict[Path, tuple] = {}

def _breaks_path(event: str, model: str) -> Path:
    return PHASE3_DIR / event / f"risk_breaks_{model}.json"

def _load_breaks(event: str = "overall", model: str = DEFAULT_MODEL) -> List[float]:
    path = _breaks_path(event, model)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return list(DEFAULT_BREAKS)
    hit = _BREAKS_CACHE.get(path)
    if hit is None or hit[0] != mtime:
        try:
            b = [float(x) for x in json.loads(path.read_text())["breaks"]]
            if len(b) != 3:
                raise ValueError(f"expected 3 breaks, got {len(b)}")
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}; using defaults")
            b = list(DEFAULT_BREAKS)
        _BREAKS_CACHE[path] = (mtime, b)
    return list(_BREAKS_CACHE[path][1])

def classify_risk(p: float, event: str = "overall", model: str = DEFAULT_MODEL) -> dict:
    p = float(np.clip(p, 0.0, 1.0))
    idx = int(np.digitize(p, _load_breaks(event, model), right=False))
    return RISK_LEVELS[idx]

def _risk_mapping(event: str = "overall", model: str = DEFAULT_MODEL) -> List[dict]:
    b = _load_breaks(event, model)
    lowers = [0.0] + b
    uppers = b + [1.0]
    return [
        {**r, "lower": lo, "upper": hi}
        for r, lo, hi in zip(RISK_LEVELS, lowers, uppers)
    ]

def _risk_payload(p: float, event: str = "overall", model: str = DEFAULT_MODEL) -> dict:
    r = classify_risk(p, event, model)
    return {
        "probability": float(p),
        "risk_class": r["label"],
        "risk_index": r["index"],
        "color": r["color"],
    }

# ---------- Flood-able terrain gating ----------
def _domain_check(raw: Dict[str, float]) -> Tuple[Optional[str], Optional[str]]:
    d = raw.get("distance_to_river")
    hand = raw.get("hand")
    slope = raw.get("slope")

    if d is not None and np.isfinite(d) and d <= CHANNEL_BUFFER_M:
        return "channel", "River channel (not modelled)"

    if hand is None or slope is None or not np.isfinite(hand) or not np.isfinite(slope):
        return None, None

    if not (HAND_MIN <= hand <= HAND_MAX) or slope > SLOPE_MAX:
        return (
            "outside",
            f"Outside flood-able terrain (HAND {HAND_MIN:.0f}-{HAND_MAX:.0f} m and "
            f"slope <= {SLOPE_MAX:.0f} deg required)",
        )
    return None, None

def _special_payload(kind: str, note: str, event: str, model: str) -> dict:
    if kind == "channel":
        return {
            "probability": None,
            "risk_class": "River channel",
            "risk_index": -1,
            "color": "#4575b4",
            "in_domain": False,
            "note": note,
        }
    return {**_risk_payload(OUTSIDE_VALUE, event, model), "in_domain": False, "note": note}

def _vector_from_raw(raw: Dict[str, float], names: List[str]) -> np.ndarray:
    vec = build_feature_matrix(raw, names)
    return np.asarray(vec, dtype=np.float32).reshape(1, -1)

# ---------- Feature attribution ----------
_BASELINE: Optional[np.ndarray] = None

def _baseline(names: List[str]) -> np.ndarray:
    global _BASELINE
    if _BASELINE is None or len(_BASELINE) != len(names):
        try:
            df = pd.read_csv(OUTPUT_DIR / "full_dataset_engineered.csv")
            _BASELINE = df[names].mean().to_numpy(dtype=np.float64)
        except Exception as e:
            logger.warning(f"Could not compute attribution baseline ({e}); using 0.0")
            _BASELINE = np.zeros(len(names), dtype=np.float64)
    return _BASELINE

def _feature_attributions(
    model,
    coords: np.ndarray,
    x: np.ndarray,
    names: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Ablate each feature to its baseline mean; one batched model call per point."""
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    k = min(len(names), x.shape[1])
    base = _baseline(names)

    X = np.repeat(x, k + 1, axis=0)          # row 0 = original, row i+1 = feature i ablated
    for i in range(k):
        X[i + 1, i] = base[i]
    p = predict_proba(model, np.repeat(coords, k + 1, axis=0), X)

    contributions = []
    for i in range(k):
        c = float(p[0] - p[i + 1])
        if np.isfinite(c):
            contributions.append({
                "feature": names[i],
                "contribution": c,
                "direction": "increases" if c > 0 else "decreases",
            })

    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions[:top_k]

# ---------- Raster generation ----------
def _build_susceptibility(model: str, event: str) -> Tuple[np.ndarray, List[float], Path, Path]:
    """Predict the full raster for one (event, model), save probability + class GeoTIFFs and breaks."""
    mdl = _load(event, model)
    raster, meta = predict_full_raster(mdl, suite_features())

    event_dir = PHASE3_DIR / event
    out_proba = event_dir / f"susceptibility_{model}.tif"
    save_geotiff(raster, meta, out_proba)

    breaks = [float(b) for b in quantile_breaks(raster)]
    _breaks_path(event, model).write_text(
        json.dumps({"model": model, "event": event, "breaks": breaks}, indent=2)
    )

    out_classes = event_dir / f"risk_classes_{model}.tif"
    save_geotiff(classify_raster(raster, breaks), class_meta(meta), out_classes)
    return raster, breaks, out_proba, out_classes

# ---------- Retraining on new events ----------
_RETRAIN_LOCK = threading.Lock()          # one retrain at a time (GWR/SVM are heavy)
_RETRAIN_STATUS: Dict[str, str] = {}

def run_retraining_pipeline(new_flood_map_path: str, new_diff_db_path: str, new_ndfi_path: str, event_name: str):
    """
    Register the new event's SAR layers, rebuild the dataset, then run run_suite.py
    for the new event, 'overall' and every holdout_<event>. Finally regenerate the
    default-model rasters for the new event and overall.
    """
    with _RETRAIN_LOCK:
        _RETRAIN_STATUS[event_name] = "running"
        logger.info(f"Starting retraining for event: {event_name}")

        env = {**os.environ, "DYNAMIC_EVENT_NAME": event_name, "SUITE_NEW_EVENT": event_name}
        steps = [
            ["register_sar_to_dem.py", "--event", event_name,
             "--flood", new_flood_map_path,
             "--diff", new_diff_db_path,
             "--ndfi", new_ndfi_path],
            ["-m", "app.preprocessing.pipeline", "--standardize"],
            ["run_suite.py"],
        ]
        try:
            for step in steps:
                result = subprocess.run(
                    [sys.executable, *step], capture_output=True, text=True, env=env
                )
                if result.returncode != 0:
                    logger.error(f"Retraining step '{step[0]}' failed:\n{result.stderr[-3000:]}")
                    _RETRAIN_STATUS[event_name] = f"failed: {step[0]}"
                    return

            # Keep the event in future runs (config.AVAILABLE_EVENTS) only once it trained OK
            register_dynamic_event(event_name)

            for ev in (event_name, "overall"):
                _build_susceptibility(DEFAULT_MODEL, ev)

            _RETRAIN_STATUS[event_name] = "done"
            logger.info(f"Retraining completed for {event_name}")
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            _RETRAIN_STATUS[event_name] = f"failed: {e}"


@app.get("/")
async def root():
    return {
        "message": "Flood Susceptibility API",
        "version": "3.0.0",
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
            "/api/events/upload",
            "/api/events/models",
            "/api/events/retrain/{event}",
        ],
    }

@app.get("/api/status")
async def get_status():
    try:
        n_features = len(suite_features())
    except Exception:
        n_features = 0
    return {
        "status": "online",
        "phase2_loaded": X_train is not None,
        "n_train_samples": int(len(X_train)) if X_train is not None else 0,
        "n_test_samples": int(len(X_test)) if X_test is not None else 0,
        "phase2_feature_names": len(phase2_feature_names),
        "models_loaded": available_events(),
        "active_feature_count": n_features,
        "sample_points_loaded": sample_points is not None,
    }

@app.get("/api/models")
async def list_models(event: str = "overall", model: str = DEFAULT_MODEL):
    try:
        names = suite_features()
    except Exception:
        names = []
    try:
        m = resolve_model(model)
        safe_event(event)
    except ValueError as e:
        raise HTTPException(422, str(e))
    events = available_events()
    return {
        "models": MODELS,
        "default_model": DEFAULT_MODEL,
        "events": events,
        "is_ready": bool(events),
        "feature_names": names,
        "n_features": len(names),
        "risk_mapping": _risk_mapping(event, m),
        "flood_able_terrain": {
            "hand_min_m": HAND_MIN,
            "hand_max_m": HAND_MAX,
            "slope_max_deg": SLOPE_MAX,
            "channel_buffer_m": CHANNEL_BUFFER_M,
        },
    }

def _comparison() -> Optional[pd.DataFrame]:
    if not COMPARISON_CSV.exists():
        return None
    return pd.read_csv(COMPARISON_CSV)

def _records(df: pd.DataFrame) -> List[dict]:
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")

@app.get("/api/events/models")
async def list_event_models():
    events = available_events()
    if not events:
        return {"events": []}

    df = _comparison()
    out = []
    for event, models in events.items():
        metrics = []
        if df is not None and "category" in df.columns:
            metrics = _records(df[df["category"] == event])
        out.append({
            "event": event,
            "available_models": models,
            "model_labels": {m: MODELS[m] for m in models},
            "metrics": metrics,
        })

    out.sort(key=lambda x: (x["event"] != "overall", x["event"]))
    return {"events": out}

@app.get("/api/events/retrain/{event}")
async def retrain_status(event: str):
    try:
        safe_event(event)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"event": event, "status": _RETRAIN_STATUS.get(event, "unknown")}

@app.get("/api/metrics")
async def get_metrics():
    df = _comparison()
    if df is None:
        raise HTTPException(404, "Phase 3 metrics not found")
    return JSONResponse(_records(df))

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

@app.post("/api/events/upload")
async def upload_flood_event(
    background_tasks: BackgroundTasks,
    eventName: str = Form(...),
    rainfallDetails: Optional[str] = Form(None),
    affectedBarangays: Optional[str] = Form(None),
    affectedFamilies: Optional[str] = Form(None),
    affectedIndividuals: Optional[str] = Form(None),
    casualties: Optional[str] = Form(None),
    infrastructureDamages: Optional[str] = Form(None),
    totalCostDamaged: Optional[str] = Form(None),
    houseDamagesDetails: Optional[str] = Form(None),
    livestockDamages: Optional[str] = Form(None),
    agriculturalDamages: Optional[str] = Form(None),
    powerSupplyDamages: Optional[str] = Form(None),
    floodMap: UploadFile = File(...),
    diffDb: UploadFile = File(...),
    ndfi: UploadFile = File(...)
):
    if not supabase:
        raise HTTPException(500, "Supabase client not configured.")

    try:
        safe_event_name = "".join([c if c.isalnum() else "_" for c in eventName]).strip("_").lower()
        if not safe_event_name:
            raise HTTPException(422, "Event name must contain letters or digits.")

        flood_map_bytes = await floodMap.read()
        diff_db_bytes = await diffDb.read()
        ndfi_bytes = await ndfi.read()

        logger.info(f"Uploading files for event: {safe_event_name} to Azure Blob...")
        flood_url = upload_to_azure(flood_map_bytes, f"events/{safe_event_name}/flood_map.tif")
        diff_url = upload_to_azure(diff_db_bytes, f"events/{safe_event_name}/diff_db.tif")
        ndfi_url = upload_to_azure(ndfi_bytes, f"events/{safe_event_name}/ndfi.tif")

        db_payload = {
            "event_name": eventName,
            "rainfall_details": rainfallDetails,
            "affected_barangays": int(affectedBarangays) if affectedBarangays else None,
            "affected_families": int(affectedFamilies) if affectedFamilies else None,
            "affected_individuals": int(affectedIndividuals) if affectedIndividuals else None,
            "casualties": int(casualties) if casualties else None,
            "infrastructure_damages": int(infrastructureDamages) if infrastructureDamages else None,
            "total_cost_damaged": float(totalCostDamaged) if totalCostDamaged else None,
            "house_damages_details": houseDamagesDetails,
            "livestock_damages": float(livestockDamages) if livestockDamages else None,
            "agricultural_damages": float(agriculturalDamages) if agriculturalDamages else None,
            "power_supply_damages": float(powerSupplyDamages) if powerSupplyDamages else None,
            "flood_map_url": flood_url,
            "diff_db_url": diff_url,
            "ndfi_url": ndfi_url
        }

        response = supabase.table("flood_events").insert(db_payload).execute()
        if not response.data:
            raise Exception("Failed to insert data into Supabase (empty response).")

        local_dir = Path(f"./data/incoming/{safe_event_name}")
        local_dir.mkdir(parents=True, exist_ok=True)

        local_flood_map = local_dir / "flood_map.tif"
        local_diff_db = local_dir / "diff_db.tif"
        local_ndfi = local_dir / "ndfi.tif"

        local_flood_map.write_bytes(flood_map_bytes)
        local_diff_db.write_bytes(diff_db_bytes)
        local_ndfi.write_bytes(ndfi_bytes)

        _RETRAIN_STATUS[safe_event_name] = "queued"
        background_tasks.add_task(
            run_retraining_pipeline,
            str(local_flood_map),
            str(local_diff_db),
            str(local_ndfi),
            safe_event_name
        )

        return {
            "status": "success",
            "message": f"Event '{eventName}' saved successfully. Model retraining has started in the background.",
            "event_id": response.data[0].get("id"),
            "event_key": safe_event_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/point")
async def predict_point(req: PredictPointRequest):
    model_id = _resolve(req.event, req.model)
    model = _load(req.event, model_id)
    names = suite_features()

    try:
        row, col = pixel_at(req.lat, req.lng)
    except Exception as e:
        raise HTTPException(422, f"Point is outside the model grid: {e}")
    coords = np.array([[row, col]], dtype=np.float64)

    if req.features is not None:
        missing = [k for k in names if k not in req.features]
        if missing:
            raise HTTPException(
                422,
                f"Provided features are missing required keys: {missing[:5]}"
                + ("..." if len(missing) > 5 else ""),
            )
        x = np.array([[float(req.features[k]) for k in names]], dtype=np.float32)
    else:
        try:
            raw = sample_raw_features(req.lat, req.lng)
        except Exception as e:
            raise HTTPException(422, f"Could not sample features at point: {e}")

        kind, note = _domain_check(raw)
        if kind is not None:
            return {
                "lat": req.lat,
                "lng": req.lng,
                "model": model_id,
                "event": req.event,
                **_special_payload(kind, note, req.event, model_id),
            }

        try:
            x = _vector_from_raw(raw, names)
        except Exception as e:
            raise HTTPException(422, f"Could not build feature vector at point: {e}")

    try:
        proba = float(predict_proba(model, coords, x)[0])
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    if not np.isfinite(proba):
        raise HTTPException(
            500,
            "Model returned a non-finite probability. Likely NaN/Inf in "
            "the feature vector; check the aligned rasters at this point.",
        )

    payload = {
        "lat": req.lat,
        "lng": req.lng,
        "model": model_id,
        "event": req.event,
        "in_domain": True,
        **_risk_payload(proba, req.event, model_id),
    }

    if req.explain:
        try:
            payload["top_factors"] = _feature_attributions(model, coords, x, names, top_k=5)
        except Exception as e:
            logger.warning(f"Feature attribution failed: {e}")
            payload["top_factors"] = []

    return payload


@app.post("/api/predict/batch")
async def predict_batch(req: PredictBatchRequest):
    model_id = _resolve(req.event, req.model)
    model = _load(req.event, model_id)
    names = suite_features()

    rows, crd, row_idx = [], [], []
    errors: Dict[int, str] = {}
    special: Dict[int, dict] = {}
    for i, p in enumerate(req.points):
        try:
            raw = sample_raw_features(p["lat"], p["lng"])
            r, c = pixel_at(p["lat"], p["lng"])

            if req.simulation_override:
                for key, delta in req.simulation_override.items():
                    if key in raw and raw[key] is not None:
                        raw[key] += delta

            kind, note = _domain_check(raw)
            if kind is not None:
                special[i] = _special_payload(kind, note, req.event, model_id)
                continue
            rows.append(_vector_from_raw(raw, names).ravel())
            crd.append((r, c))
            row_idx.append(i)
        except Exception as e:
            errors[i] = str(e)

    if not rows and not special:
        raise HTTPException(422, f"No points could be sampled: {errors}")

    probas = np.array([])
    X = np.empty((0, len(names)))
    C = np.empty((0, 2))
    if rows:
        X = np.vstack(rows)
        C = np.asarray(crd, dtype=np.float64)
        try:
            probas = predict_proba(model, C, X)
        except Exception as e:
            raise HTTPException(500, f"Prediction failed: {e}")

    results = [None] * len(req.points)
    for local_i, original_i in enumerate(row_idx):
        p = req.points[original_i]
        proba = float(probas[local_i])
        if not np.isfinite(proba):
            results[original_i] = {"lat": p["lat"], "lng": p["lng"], "error": "non-finite probability"}
            continue

        entry: Dict[str, Any] = {
            "lat": p["lat"],
            "lng": p["lng"],
            "in_domain": True,
            **_risk_payload(proba, req.event, model_id),
        }

        if req.explain:
            try:
                entry["top_factors"] = _feature_attributions(
                    model, C[local_i:local_i + 1], X[local_i:local_i + 1], names, top_k=5,
                )
            except Exception as e:
                logger.warning(f"Feature attribution failed at point {original_i}: {e}")
                entry["top_factors"] = []

        results[original_i] = entry

    for i, payload in special.items():
        p = req.points[i]
        results[i] = {"lat": p["lat"], "lng": p["lng"], **payload}

    for i, msg in errors.items():
        p = req.points[i]
        results[i] = {"lat": p["lat"], "lng": p["lng"], "error": msg}

    return {
        "model": model_id,
        "event": req.event,
        "count": len(results),
        "n_ok": len(row_idx) + len(special),
        "n_out_of_domain": len(special),
        "n_failed": len(errors),
        "results": results,
    }


@app.post(
    "/api/generate_susceptibility",
    dependencies=[Depends(require_key)],
)
def generate_susceptibility(req: GenerateRasterRequest):
    """Run full-raster prediction and save GeoTIFFs (probability + risk classes)."""
    model_id = _resolve(req.event, req.model)

    try:
        raster, breaks, out_proba, out_classes = _build_susceptibility(model_id, req.event)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Raster prediction failed: {e}")

    finite = raster[np.isfinite(raster)]
    flood_able = raster[np.isfinite(raster) & (raster > OUTSIDE_VALUE)]
    return {
        "model": model_id,
        "event": req.event,
        "probability_raster": str(out_proba),
        "risk_class_raster": str(out_classes),
        "risk_breaks": breaks,
        "shape": list(raster.shape),
        "valid_pixels": int(finite.size),
        "flood_able_pixels": int(flood_able.size),
        "probability_min": float(flood_able.min()) if flood_able.size else None,
        "probability_max": float(flood_able.max()) if flood_able.size else None,
        "probability_mean": float(flood_able.mean()) if flood_able.size else None,
    }


@app.get("/api/susceptibility/at")
async def susceptibility_at(lat: float, lng: float, model: str = DEFAULT_MODEL, event: str = "overall"):
    """Fast lookup from the cached susceptibility GeoTIFF."""
    import pyproj
    import rasterio
    from rasterio.transform import rowcol
    from app.preprocessing.config import TARGET_CRS

    try:
        model_id = resolve_model(model)
        safe_event(event)
    except ValueError as e:
        raise HTTPException(422, str(e))

    tif = PHASE3_DIR / event / f"susceptibility_{model_id}.tif"
    if not tif.exists():
        raise HTTPException(404, "Run /api/generate_susceptibility first")

    transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x, y = transformer.transform(lng, lat)

    with rasterio.open(tif) as src:
        row, col = rowcol(src.transform, x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise HTTPException(422, "Point outside raster extent")
        proba = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])

    if not np.isfinite(proba):
        raise HTTPException(422, "No data at that pixel (sea or river channel)")

    payload = {
        "lat": lat,
        "lng": lng,
        "model": model_id,
        "event": event,
        "in_domain": proba > OUTSIDE_VALUE,
        **_risk_payload(proba, event, model_id),
    }
    if proba <= OUTSIDE_VALUE:
        payload["note"] = (
            f"Outside flood-able terrain (HAND {HAND_MIN:.0f}-{HAND_MAX:.0f} m and "
            f"slope <= {SLOPE_MAX:.0f} deg required)"
        )
    return payload


@app.get(
    "/api/download/{file_type}",
    dependencies=[Depends(require_key)],
)
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
        "models_ready": bool(available_events()),
    }

@app.get("/api/raster/{event}/{model}")
async def get_raster(event: str, model: str):
    try:
        model_id = resolve_model(model)
        safe_event(event)
    except ValueError as e:
        raise HTTPException(422, str(e))
    path = PHASE3_DIR / event / f"susceptibility_{model_id}.tif"
    if not path.exists():
        raise HTTPException(404, f"Raster for {event}/{model_id} not generated yet")
    breaks = _load_breaks(event, model_id)
    return FileResponse(
        path,
        media_type="image/tiff",
        headers={"X-Risk-Breaks": ",".join(f"{b:.6f}" for b in breaks)},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)