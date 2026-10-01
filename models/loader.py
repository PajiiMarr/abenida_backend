# app/models/loader.py
"""
Loads Phase 2 scaler + feature names, and Phase 3 trained models.

Loading strategy per file:
  1. pickle.load   — for models saved with pickle.dump (this pipeline)
  2. joblib.load   — fallback for models saved with joblib.dump (older runs)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

# Custom classes must be importable before unpickling anything that references them.
import app.training.models  # noqa: F401

logger = logging.getLogger(__name__)

PHASE2_DIR = Path("./outputs/phase2")
PHASE3_DIR = Path("./outputs/phase3")

MODEL_NAMES = ["svm", "mars", "gwr", "mgwr", "stacking"]


# ---------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------
def _try_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.debug(f"pickle.load failed for {path.name}: {e}")
        return None


def _try_joblib(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"joblib.load failed for {path.name}: {e}")
        return None


def load_any_model(path: Path):
    """
    Load a model file. Tries pickle first, then joblib.
    Returns (object, loader_name) or (None, None).
    """
    if not path.exists():
        logger.warning(f"Not found: {path}")
        return None, None

    obj = _try_pickle(path)
    if obj is not None:
        logger.info(f"Loaded {path.name} via pickle")
        return obj, "pickle"

    obj = _try_joblib(path)
    if obj is not None:
        logger.info(f"Loaded {path.name} via joblib")
        return obj, "joblib"

    logger.error(f"Failed to deserialize {path} with pickle or joblib")
    return None, None


# ---------------------------------------------------------
# Registry
# ---------------------------------------------------------
class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if self._loaded:
            return

        self.models: Dict[str, Any] = {}
        self.formats: Dict[str, str] = {}
        self.metrics: Dict[str, Any] = {}
        self.risk_mapping: Dict[str, Any] = {}
        self.cv_results: Dict[str, Any] = {}
        self.scaler = None
        self.scaler_format: Optional[str] = None

        self._feature_names_raw: List[str] = []
        self._feature_names_eng: List[str] = []
        self.feature_names: List[str] = []

        self._load_all()
        self._loaded = True

    @staticmethod
    def _load_json(path: Path):
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"JSON load failed for {path}: {e}")
            return None

    def _load_all(self):
        # 1. Scaler
        self.scaler, self.scaler_format = load_any_model(PHASE2_DIR / "scaler.pkl")
        if self.scaler is not None:
            logger.info(f"Scaler: {type(self.scaler).__name__} ({self.scaler_format})")

        # 2. Feature names — try raw, then engineered
        for p in [
            PHASE2_DIR / "preprocessed/feature_names.json",
            PHASE2_DIR / "feature_names.json",
        ]:
            data = self._load_json(p)
            if data:
                self._feature_names_raw = list(data)
                logger.info(f"Raw features: {len(self._feature_names_raw)} from {p.name}")
                break

        for p in [
            PHASE2_DIR / "preprocessed/feature_names_engineered.json",
            PHASE2_DIR / "features_with_engineering_feature_names.json",
        ]:
            data = self._load_json(p)
            if data:
                self._feature_names_eng = list(data)
                logger.info(f"Engineered features: {len(self._feature_names_eng)} from {p.name}")
                break

        # 3. Pick the feature set that matches the scaler
        self.feature_names = self._select_feature_set()
        logger.info(f"Active feature set: {len(self.feature_names)} features")

        # 4. Models
        for name in MODEL_NAMES:
            obj, fmt = load_any_model(PHASE3_DIR / f"{name}_model.pkl")
            if obj is not None:
                self.models[name] = obj
                self.formats[name] = fmt
                logger.info(f"Model '{name}': {type(obj).__name__}")

        # 5. Auxiliary JSON files
        self.metrics = self._load_json(PHASE3_DIR / "metrics.json") or {}
        self.risk_mapping = self._load_json(PHASE3_DIR / "risk_class_mapping.json") or {}
        self.cv_results = self._load_json(PHASE3_DIR / "cv_results.json") or {}

    def _select_feature_set(self) -> List[str]:
        if self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
            n = int(self.scaler.n_features_in_)
            logger.info(f"Scaler expects {n} features")
            if len(self._feature_names_raw) == n:
                logger.info("Using RAW feature set")
                return self._feature_names_raw
            if len(self._feature_names_eng) == n:
                logger.info("Using ENGINEERED feature set")
                return self._feature_names_eng
            logger.warning(
                f"Scaler wants {n} features but raw={len(self._feature_names_raw)} "
                f"and eng={len(self._feature_names_eng)}. Using raw."
            )
        return self._feature_names_raw or self._feature_names_eng

    def get(self, name: str):
        return self.models.get(name)

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    @property
    def is_ready(self) -> bool:
        return len(self.models) > 0


registry = ModelRegistry()