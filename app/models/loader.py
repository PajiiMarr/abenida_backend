import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PHASE2_DIR = Path("./outputs/phase2")
PHASE3_DIR = Path("./outputs/phase3")

MODEL_NAMES = ["svm", "mars", "gwr", "mgwr", "stacking"]


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
        self.metrics: Dict[str, Any] = {}
        self.risk_mapping: Dict[str, Any] = {}
        self.cv_results: Dict[str, Any] = {}
        self.scaler = None

        # Candidate feature name lists
        self._feature_names_raw: List[str] = []
        self._feature_names_eng: List[str] = []
        self.feature_names: List[str] = []

        self._load_all()
        self._loaded = True

    # ---------- helpers ----------
    @staticmethod
    def _load_pickle(path: Path):
        if not path.exists():
            logger.warning(f"Not found: {path}")
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Pickle load failed for {path}: {e}")
            return None

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

    # ---------- main ----------
    def _load_all(self):
        # 1. Scaler
        self.scaler = self._load_pickle(PHASE2_DIR / "scaler.pkl")

        # 2. Feature names (try multiple locations seen in the file tree)
        candidates_raw = [
            PHASE2_DIR / "preprocessed/feature_names.json",
            PHASE2_DIR / "feature_names.json",
        ]
        candidates_eng = [
            PHASE2_DIR / "preprocessed/feature_names_engineered.json",
            PHASE2_DIR / "features_with_engineering_feature_names.json",
        ]

        for p in candidates_raw:
            data = self._load_json(p)
            if data:
                self._feature_names_raw = data if isinstance(data, list) else list(data)
                logger.info(f"Loaded {len(self._feature_names_raw)} raw feature names from {p}")
                break

        for p in candidates_eng:
            data = self._load_json(p)
            if data:
                self._feature_names_eng = data if isinstance(data, list) else list(data)
                logger.info(f"Loaded {len(self._feature_names_eng)} engineered feature names from {p}")
                break

        # 3. Choose the set that matches the scaler
        self.feature_names = self._select_feature_set()
        logger.info(f"Active feature set: {len(self.feature_names)} features")

        # 4. Models
        for name in MODEL_NAMES:
            m = self._load_pickle(PHASE3_DIR / f"{name}_model.pkl")
            if m is not None:
                self.models[name] = m
                logger.info(f"Loaded model: {name}")

        # 5. Metrics / mappings
        self.metrics = self._load_json(PHASE3_DIR / "metrics.json") or {}
        self.risk_mapping = self._load_json(PHASE3_DIR / "risk_class_mapping.json") or {}
        self.cv_results = self._load_json(PHASE3_DIR / "cv_results.json") or {}

    def _select_feature_set(self) -> List[str]:
        if self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
            n = int(self.scaler.n_features_in_)
            if len(self._feature_names_eng) == n:
                return self._feature_names_eng
            if len(self._feature_names_raw) == n:
                return self._feature_names_raw
            logger.warning(
                f"Scaler expects {n} features but raw={len(self._feature_names_raw)} "
                f"and engineered={len(self._feature_names_eng)}. Using engineered."
            )
        return self._feature_names_eng or self._feature_names_raw

    # ---------- accessors ----------
    def get(self, name: str):
        return self.models.get(name)

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    @property
    def is_ready(self) -> bool:
        return len(self.models) > 0


registry = ModelRegistry()