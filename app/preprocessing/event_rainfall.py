# backend/app/preprocessing/event_rainfall.py
"""
Event rainfall derived from the satellite daily series (CHIRPS and GPM IMERG).

Nothing is hard-coded: for every flood event in config.FLOOD_EVENTS the rainfall
statistics are computed from the daily CSVs whose paths are defined in
config.CLIMATIC (no absolute/personal paths).

Per product (chirps / gpm), over the event window [date - days_before, date + days_after]:
    total      accumulated rainfall (mm)
    max_daily  wettest day (mm)            <- extreme-precipitation indicator
    max_3day   wettest 3-day accumulation  <- extreme-precipitation indicator
    rain_days  days with > 1 mm

Event rainfall index (the single column used by the pipeline):
    rainfall_mm = mean of the CHIRPS and GPM `max_3day` values
                  (set config.RAINFALL_EVENT['metric'] to use another statistic).
"""
import logging
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRODUCTS = ("chirps", "gpm")


class EventRainfallExtractor:
    """Extract rainfall statistics for flood events from daily CHIRPS / GPM series."""

    def __init__(self, chirps_daily_path: Path, gpm_daily_path: Path):
        self.series = {
            "chirps": self._read(chirps_daily_path),
            "gpm": self._read(gpm_daily_path),
        }

    @staticmethod
    def _read(path) -> pd.Series:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Daily rainfall CSV not found: {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        s = df.sort_values("date").drop_duplicates("date").set_index("date")["rainfall_mm"]
        return pd.to_numeric(s, errors="coerce").astype(float)

    @staticmethod
    def _stats(s: pd.Series, start, end) -> Dict[str, float]:
        w = s.loc[start:end].dropna()
        if w.empty:
            return {"total": np.nan, "max_daily": np.nan, "max_3day": np.nan, "rain_days": np.nan}
        # reindex to consecutive days so the 3-day window really means 3 calendar days
        w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D"))
        w = w.fillna(0.0)
        return {
            "total": float(w.sum()),
            "max_daily": float(w.max()),
            "max_3day": float(w.rolling(3, min_periods=1).sum().max()),
            "rain_days": float((w > 1).sum()),
        }

    def event_stats(self, event_config: dict) -> Dict[str, float]:
        """Unprefixed statistics for one event, including the `rainfall_mm` index."""
        from .config import RAINFALL_EVENT

        date = event_config["date"]
        start = date - timedelta(days=event_config.get("days_before", 7))
        end = date + timedelta(days=event_config.get("days_after", 7))

        out = {}
        for prod in PRODUCTS:
            for k, v in self._stats(self.series[prod], start, end).items():
                out[f"{prod}_{k}"] = v

        metric = RAINFALL_EVENT["metric"]
        vals = [out[f"{p}_{metric}"] for p in PRODUCTS if np.isfinite(out[f"{p}_{metric}"])]
        out["rainfall_mm"] = float(np.mean(vals)) if vals else np.nan
        return out

    def extract_event_features(self, event_config: dict) -> dict:
        """Backwards-compatible: statistics keyed '<event>_<stat>'."""
        prefix = event_config.get("sar_key") or f"event_{event_config['date']:%Y%m%d}"
        return {f"{prefix}_{k}": v for k, v in self.event_stats(event_config).items()}


# ----------------------------------------------------------------------
# Module-level helpers (paths come from config, never hard-coded here)
# ----------------------------------------------------------------------
def _extractor() -> EventRainfallExtractor:
    from .config import CLIMATIC
    return EventRainfallExtractor(CLIMATIC["chirps_daily"], CLIMATIC["gpm_daily"])


def get_event_rainfall_table(save_dir: Optional[Path] = None) -> pd.DataFrame:
    """One row per flood event in config.FLOOD_EVENTS with all rainfall statistics."""
    from .config import FLOOD_EVENTS

    ext = _extractor()
    rows = []
    for key, cfg in FLOOD_EVENTS.items():
        rows.append({"event": key, "date": cfg["date"].date(), "name": cfg.get("name", key),
                     **ext.event_stats(cfg)})
    table = pd.DataFrame(rows)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_dir / "event_rainfall_summary.csv", index=False)
        logger.info(f"Saved event rainfall summary -> {save_dir / 'event_rainfall_summary.csv'}")
    return table


def get_event_rainfall_mm() -> Dict[str, float]:
    """{event: rainfall index in mm}, derived from the CHIRPS / GPM daily series."""
    table = get_event_rainfall_table()
    return dict(zip(table["event"], table["rainfall_mm"]))


def get_event_rainfall_features(event_config: dict) -> dict:
    return _extractor().extract_event_features(event_config)


def get_all_event_rainfall_features() -> dict:
    from .config import FLOOD_EVENTS
    ext = _extractor()
    feats = {}
    for cfg in FLOOD_EVENTS.values():
        feats.update(ext.extract_event_features(cfg))
    return feats


def get_active_event_rainfall_features() -> dict:
    from .config import FLOOD_EVENTS, ACTIVE_EVENT
    if ACTIVE_EVENT not in FLOOD_EVENTS:
        logger.warning(f"Active event '{ACTIVE_EVENT}' not found in FLOOD_EVENTS")
        return {}
    return get_event_rainfall_features(FLOOD_EVENTS[ACTIVE_EVENT])