"""
Registry for the run_suite.py models: outputs/phase3/<category>/<model>.pkl

Every model takes (coords, X). coords are the (row, col) pixel indices of the
aligned grid, exactly as used in training (row -> x, col -> y).
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(os.environ.get("BASE_DIR") or ".")
PHASE2_DIR = Path(os.environ.get("PREPROCESS_DIR") or BASE_DIR / "outputs/phase2")
PHASE3_DIR = PHASE2_DIR.parent / "phase3"

MODELS: Dict[str, str] = {
    "m1_gwpca_svm": "GWPCA + SVM",
    "m2_gwr_mgwr": "GWR + MGWR",
    "m3_gwpca_mars": "GWPCA + MARS",
    "m4_stacking_all": "Stacking ensemble",
    "m5_gwr_mars_svm_pipeline": "GWR -> MARS -> SVM pipeline",
    "m6_mgwr": "MGWR",  
}
DEFAULT_MODEL = "m4_stacking_all"

# Short / legacy names accepted by the API
_ALIASES = {
    "m1": "m1_gwpca_svm", "svm": "m1_gwpca_svm",
    "m2": "m2_gwr_mgwr", "gwr": "m2_gwr_mgwr", "mgwr": "m2_gwr_mgwr",
    "m3": "m3_gwpca_mars", "mars": "m3_gwpca_mars",
    "m4": "m4_stacking_all", "stacking": "m4_stacking_all",
    "m5": "m5_gwr_mars_svm_pipeline", "pipeline": "m5_gwr_mars_svm_pipeline",
}

# Columns run_suite.py never uses as features
_NON_FEATURES = {"flood_label", "event", "sample_id", "row", "col", "x", "y", "rainfall_mm"}
_EVENT_RE = re.compile(r"[a-z0-9_]+")

_CACHE: Dict[Path, tuple] = {}
_CACHE_MAX = 6


def resolve_model(name: str) -> str:
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return key


def safe_event(event: str) -> str:
    if not _EVENT_RE.fullmatch(event or ""):
        raise ValueError(f"Invalid event name '{event}'")
    return event


def model_path(event: str, model: str) -> Path:
    return PHASE3_DIR / safe_event(event) / f"{resolve_model(model)}.pkl"


def load_model(event: str, model: str):
    """Load a suite model; re-loads automatically when the file is retrained."""
    path = model_path(event, model)
    if not path.exists():
        raise FileNotFoundError(f"Model '{model}' for event '{event}' not found")
    mtime = path.stat().st_mtime
    hit = _CACHE.get(path)
    if hit is None or hit[0] != mtime:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[path] = (mtime, joblib.load(path))
    return _CACHE[path][1]


def suite_features() -> List[str]:
    """Feature order used in training (saved by run_suite, else derived from the CSV)."""
    saved = PHASE3_DIR / "suite_features.json"
    if saved.exists():
        return list(json.loads(saved.read_text())["features"])
    header = pd.read_csv(PHASE2_DIR / "full_dataset_engineered.csv", nrows=0).columns
    return [c for c in header if c not in _NON_FEATURES]


def available_events() -> Dict[str, List[str]]:
    """{category: [model ids with a saved .pkl]}"""
    out: Dict[str, List[str]] = {}
    if PHASE3_DIR.exists():
        for d in sorted(PHASE3_DIR.iterdir()):
            if d.is_dir():
                found = [m for m in MODELS if (d / f"{m}.pkl").exists()]
                if found:
                    out[d.name] = found
    return out


def predict_proba(model, coords, X, batch: int = 10000) -> np.ndarray:
    """Flood probability per row; batched to keep GWR distance matrices small."""
    coords = np.asarray(coords, dtype=float)
    X = np.asarray(X, dtype=float)
    parts = [
        model.predict_proba(coords[i:i + batch], X[i:i + batch])
        for i in range(0, len(X), batch)
    ]
    return np.clip(np.concatenate(parts), 0.0, 1.0) if parts else np.empty(0)