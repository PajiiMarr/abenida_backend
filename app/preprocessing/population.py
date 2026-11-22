# backend/app/preprocessing/population.py
"""Barangay population (PSA census xlsx) -> population-density raster on the aligned grid."""
import difflib
import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features

from .config import POPULATION, TARGET_CRS

logger = logging.getLogger(__name__)


def _norm(name) -> str:
    s = str(name).lower().replace("ñ", "n")
    s = re.sub(r"[()\-.,*]", " ", s)            # keeps "(Upper Bunguiao)" text, drops punctuation
    s = re.sub(r"\b(barangay|brgy|pob|poblacion)\b", " ", s)
    s = re.sub(r"\bsta\b", "santa", s)
    s = re.sub(r"\bsto\b", "santo", s)
    return re.sub(r"\s+", " ", s).strip()


def load_census_population(path=None, sheet=None) -> dict:
    """Return {normalized barangay name: population} for the city sheet."""
    path = Path(path or POPULATION["census"])
    sheet = sheet or POPULATION["sheet"]
    if not path.exists():
        raise FileNotFoundError(f"Census workbook not found: {path}")
    raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=5)
    raw = raw[[1, 3]].dropna()
    raw.columns = ["name", "pop"]
    raw["pop"] = pd.to_numeric(raw["pop"], errors="coerce")
    raw = raw.dropna()
    raw = raw[~raw["name"].astype(str).str.upper().str.startswith("CITY OF")]  # drop city total
    pop = {}
    for n, p in zip(raw["name"], raw["pop"]):
        pop[_norm(n)] = pop.get(_norm(n), 0.0) + float(p)
    logger.info(f"Census: {len(pop)} barangays, total population {sum(pop.values()):,.0f}")
    return pop


def build_population_density(shape, transform, target_crs=TARGET_CRS,
                             barangay_path=None, name_field=None) -> np.ndarray:
    """
    Persons per km2 for every pixel, from barangay population / barangay area.
    Returns float32 (rows, cols) on the given grid; pixels outside any barangay are NaN
    (FeatureExtractor median-fills them).
    """
    barangay_path = Path(barangay_path or POPULATION["barangays"])
    name_field = name_field or POPULATION["name_field"]
    if not barangay_path.exists():
        raise FileNotFoundError(f"Barangay layer not found: {barangay_path}")

    gdf = gpd.read_file(barangay_path)
    if name_field not in gdf.columns:
        raise ValueError(f"'{name_field}' not in barangay layer. Columns: {list(gdf.columns)}")
    gdf = gdf.to_crs(target_crs) if gdf.crs is not None else gdf.set_crs(target_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    gdf["area_km2"] = gdf.geometry.area / 1e6

    census = load_census_population()
    keys = list(census)

    def match(n):
        k = _norm(n)
        if k in census:
            return k
        close = difflib.get_close_matches(k, keys, n=1, cutoff=0.85)
        return close[0] if close else None

    gdf["_key"] = gdf[name_field].map(match)
    unmatched = gdf.loc[gdf["_key"].isna(), name_field].tolist()
    frac = len(unmatched) / max(len(gdf), 1)
    logger.info(f"Barangay polygons matched to census: {len(gdf) - len(unmatched)}/{len(gdf)}")
    if unmatched:
        logger.warning(f"Unmatched polygons (no population): {unmatched}")
    if frac > POPULATION["max_unmatched_frac"]:
        raise ValueError(f"{frac:.0%} of barangay polygons unmatched; check name spelling/aliases.")
    unused = sorted(set(keys) - set(gdf["_key"].dropna()))
    if unused:
        logger.warning(f"Census barangays with no polygon: {unused}")

    # density = population / total area of all polygons sharing that barangay name
    area = gdf.groupby("_key")["area_km2"].transform("sum")
    gdf["density"] = gdf["_key"].map(census) / area
    gdf = gdf.dropna(subset=["density"])

    dens = features.rasterize(
        ((g, float(d)) for g, d in zip(gdf.geometry, gdf["density"])),
        out_shape=shape, transform=transform, fill=np.nan,
        dtype="float32", all_touched=True,
    )
    valid = np.isfinite(dens)
    logger.info(f"Population density: {valid.mean()*100:.1f}% of grid covered, "
                f"range {np.nanmin(dens):.0f}-{np.nanmax(dens):.0f} persons/km2")
    return dens