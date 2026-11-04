# backend/app/preprocessing/sar_processor.py
import re
import difflib
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import rasterio
import rasterio.crs
from rasterio import features
from shapely.geometry import Point
import logging
from typing import Tuple, Optional, Dict, List
import json
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from .config import SAR_REGISTERED

logger = logging.getLogger(__name__)

# ======================================================================
# Flooded-barangay lists (ZCDRRMO "Comparative Data on Flooding Events 2022-2024")
# Keys match FLOOD_EVENTS / SAR_REGISTERED. These lists are a *guide*, not a hard
# boundary: other barangays may also have flooded (see create_sample_points).
# ======================================================================
FLOOD_BARANGAYS: Dict[str, List[str]] = {
    # Paeng, 27 Oct 2022 -> 43 (13 west + 22 central + 8 east)
    "oct_2022": [
        "Ayala", "Cawit", "La Paz", "Labuan", "Limpapa", "Maasin", "Pamucutan", "Patalon",
        "Sinunuc", "Talisayan", "Tulungatung", "Malagutay", "Recodo",
        "Baliwasan", "Cabatangan", "Calarian", "Campo Islam", "San Roque", "San Jose Gusu",
        "Sta. Maria", "Pasonanca", "Canelar", "Camino Nuevo", "Tumaga", "Tetuan", "Rio Hondo",
        "Kasanyangan", "Guiwan", "Tugbungan", "Talon-Talon", "Mampang", "Divisoria", "Boalan",
        "Lunzuran", "Zambowood",
        "Cabaluay", "Curuan", "Lumbangan", "Manicahan", "Mercedes", "Putik", "Sangali", "Tigtabon",
    ],
    # LPA & Shearline, 11 Jan 2023 -> 53 (14 + 20 + 19)
    "jan_2023": [
        "Ayala", "Cawit", "La Paz", "Labuan", "Limpapa", "Maasin", "Pamucutan", "Patalon",
        "Sinubong", "Sinunuc", "Talisayan", "Tulungatung", "Malagutay", "Recodo",
        "Baliwasan", "Cabatangan", "Calarian", "San Roque", "San Jose Gusu", "Sta. Maria",
        "Pasonanca", "Canelar", "Camino Nuevo", "Zone III", "Tumaga", "Tetuan", "Sta. Catalina",
        "Kasanyangan", "Guiwan", "Tugbungan", "Talon-Talon", "Mampang", "Divisoria", "Lunzuran",
        "Bolong", "Bunguiao", "Cabaluay", "Culianan", "Curuan", "Guisao", "Licomo", "Mangusu",
        "Manicahan", "Mercedes", "Putik", "Salaan", "Sangali", "Tictapul", "Victoria", "Vitali",
        "Cacao", "Boalan", "Zambowood",
    ],
    # Habagat, 12 Jul 2024 -> 32 (11 + 16 + 5 from the numbered lists; the PDF summary
    # slide says 9/10/4, which does not add up to its own total of 32)
    "jul_2024": [
        "Ayala", "Cawit", "Labuan", "Maasin", "Pamucutan", "Patalon", "Sinubong", "Sinunuc",
        "Talisayan", "Malagutay", "Recodo",
        "Baliwasan", "Cabatangan", "Calarian", "San Roque", "San Jose Gusu", "Sta. Maria",
        "Pasonanca", "Canelar", "Tumaga", "Tetuan", "Kasanyangan", "Guiwan", "Tugbungan",
        "Talon-Talon", "Divisoria", "Boalan",
        "Mangusu", "Putik", "Sangali", "Tictapul", "Vitali",
    ],
}

_NAME_FIELDS = ["ADM4_EN", "adm4_en", "adm4_name", "adm4_ref_name", "NAME_3",
                "BRGY_NAME", "brgy_name", "Barangay", "BARANGAY", "Brgy",
                "NAME", "Name", "name", "bgy_name", "NAME_4"]

# normalized PDF spelling -> normalized alternatives that may appear in a boundary layer
_ALIASES = {
    "zone iii": ["zone iii pob", "zone 3", "zone iii"],
    "campo islam": ["campo islam", "campo islam pob"],
    "sta catalina": ["santa catalina"],
    "sta maria": ["santa maria"],
    "talon talon": ["talon talon", "talontalon"],
    "tictapul": ["tictapul", "tictabon"],
    "rio hondo": ["rio hondo", "riohondo"],
}


def _normalize_name(name) -> str:
    s = str(name).lower()
    s = re.sub(r"\(.*?\)", " ", s)                       # "(Pob.)"
    s = re.sub(r"\b(barangay|brgy|bgy)\b\.?", " ", s)
    s = s.replace("ñ", "n").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _detect_name_field(gdf: gpd.GeoDataFrame, expected: List[str]) -> str:
    exp = {_normalize_name(n) for n in expected}
    cands = [c for c in _NAME_FIELDS if c in gdf.columns] or \
            [c for c in gdf.columns if gdf[c].dtype == object]
    if not cands:
        raise ValueError(f"No text column for barangay names. Columns: {list(gdf.columns)}")
    return max(cands, key=lambda c: len(exp & {_normalize_name(v) for v in gdf[c].dropna()}))


def _match_barangay_ids(gdf: gpd.GeoDataFrame, names: List[str], name_field: str,
                        fuzzy_cutoff: float = 0.85) -> Tuple[List[int], List[str]]:
    """Return (matched 'bid' values, unmatched PDF names). gdf needs 'bid' and `name_field`."""
    norm = gdf[name_field].map(_normalize_name)
    by_norm = {}
    for bid, n in zip(gdf["bid"], norm):
        by_norm.setdefault(n, []).append(int(bid))
    pool = list(by_norm)
    ids, unmatched = set(), []
    for raw in names:
        n = _normalize_name(raw)
        hit = next((o for o in [n] + [_normalize_name(a) for a in _ALIASES.get(n, [])]
                    if o in by_norm), None)
        if hit is None:
            starts = [p for p in pool if p.startswith(n + " ")]
            hit = starts[0] if len(starts) == 1 else None
        if hit is None:
            close = difflib.get_close_matches(n, pool, n=1, cutoff=fuzzy_cutoff)
            hit = close[0] if close else None
        if hit is None:
            unmatched.append(raw)
        else:
            ids.update(by_norm[hit])
    return sorted(ids), unmatched


class SARProcessor:
    """
    SAR Data Processor for flood inventory generation
    """

    def __init__(self, sar_dir: Path = Path("./outputs/phase2/registered_sar")):
        self.sar_dir = sar_dir
        self.sar_data = {}
        self.flood_inventory = None
        self.sample_points = None
        self.sample_labels = None
        self.sample_events = None
        self.sample_focus = None
        self.metadata = {}
        self._brgy = None          # barangay GeoDataFrame with 'bid'
        self._bid_raster = None    # int32 raster of bid on the aligned grid (0 = none)
        self._brgy_name_field = None

    def load_preprocessed_sar(self) -> Dict:
        """Load preprocessed SAR data from the registered_sar directory"""
        logger.info("Loading preprocessed SAR data from your preprocess outputs...")

        # Dynamically get all registered events from config (includes dynamic uploads)
        events = list(SAR_REGISTERED.keys())
        sar_data = {}

        for event in events:
            event_data = {}

            # Load flood map
            flood_path = self.sar_dir / f"{event}_flood_map_registered.tif"
            if flood_path.exists():
                with rasterio.open(flood_path) as src:
                    event_data['flood_map'] = src.read(1)
                    self.metadata['crs'] = src.crs.to_string() if src.crs else None
                    self.metadata['transform'] = src.transform
                    self.metadata['bounds'] = src.bounds
                logger.info(f"Loaded flood_map for {event}: {event_data['flood_map'].shape}")
            else:
                logger.warning(f"Flood map not found: {flood_path}")
                continue

            # Load NDFI
            ndfi_path = self.sar_dir / f"{event}_ndfi_registered.tif"
            if ndfi_path.exists():
                with rasterio.open(ndfi_path) as src:
                    event_data['ndfi'] = src.read(1)
            else:
                event_data['ndfi'] = None

            # Load VH (optional; oct_2022 has diff_db instead)
            vh_path = self.sar_dir / f"{event}_vh_registered.tif"
            if vh_path.exists():
                with rasterio.open(vh_path) as src:
                    event_data['vh'] = src.read(1)
            else:
                event_data['vh'] = None

            sar_data[event] = event_data

        self.sar_data = sar_data
        return sar_data

    def generate_flood_inventory(self) -> Dict:
        """Generate flood inventory from loaded flood maps"""
        if not self.sar_data:
            logger.warning("No SAR data loaded. Call load_preprocessed_sar() first.")
            self.load_preprocessed_sar()

            if not self.sar_data:
                raise ValueError("No flood maps loaded!")

        flood_inventory = {}

        for event, event_data in self.sar_data.items():
            flood_map = event_data.get('flood_map')

            if flood_map is None:
                logger.warning(f"No flood map for {event}, skipping")
                continue

            flood_pixels = np.sum(flood_map > 0)
            total_pixels = flood_map.size

            flood_inventory[event] = {
                'flood_map': flood_map,
                'flood_pixels': flood_pixels,
                'total_pixels': total_pixels,
                'flood_percentage': (flood_pixels / total_pixels) * 100,
                'ndfi': event_data.get('ndfi'),
                'vh': event_data.get('vh')
            }

            logger.info(f"Added {event}: {flood_pixels:,} flood pixels ({flood_pixels/total_pixels*100:.2f}%)")

        if not flood_inventory:
            raise ValueError("No flood maps loaded!")

        self.flood_inventory = flood_inventory

        total_flood = sum(info['flood_pixels'] for info in flood_inventory.values())
        total_pixels = sum(info['total_pixels'] for info in flood_inventory.values())
        logger.info(f"Combined flood inventory: {total_flood:,} flood pixels ({total_flood/total_pixels*100:.2f}%)")

        return flood_inventory

    def _get_flood_boundary_pixels(self, flood_map: np.ndarray, buffer_size: int = 5) -> np.ndarray:
        """Get pixels near flood boundaries (transition zones)"""
        distance = distance_transform_edt(flood_map == 0)
        boundary_mask = (distance <= buffer_size) & (distance > 0)
        boundary_rows, boundary_cols = np.where(boundary_mask)
        return np.column_stack([boundary_rows, boundary_cols])

    def _get_stratified_samples(self, coords: np.ndarray, labels: np.ndarray,
                                 n_samples: int, n_grid: int = 10) -> np.ndarray:
        """Get samples using spatial grid stratification"""
        if len(coords) <= n_samples:
            return np.arange(len(coords))

        row_min, row_max = coords[:, 0].min(), coords[:, 0].max()
        col_min, col_max = coords[:, 1].min(), coords[:, 1].max()

        row_step = (row_max - row_min) / n_grid
        col_step = (col_max - col_min) / n_grid

        grid_cells = {}

        for i, (row, col) in enumerate(coords):
            grid_row = int((row - row_min) / row_step) if row_step > 0 else 0
            grid_col = int((col - col_min) / col_step) if col_step > 0 else 0
            grid_row = min(grid_row, n_grid - 1)
            grid_col = min(grid_col, n_grid - 1)
            key = (grid_row, grid_col)

            if key not in grid_cells:
                grid_cells[key] = {'indices': [], 'flood': [], 'non_flood': []}

            grid_cells[key]['indices'].append(i)
            if labels[i] == 1:
                grid_cells[key]['flood'].append(i)
            else:
                grid_cells[key]['non_flood'].append(i)

        selected_indices = []
        n_cells = len(grid_cells)
        samples_per_cell = max(1, n_samples // n_cells)

        for cell_key, cell_data in grid_cells.items():
            cell_indices = cell_data['indices']

            if len(cell_indices) <= samples_per_cell:
                selected_indices.extend(cell_indices)
            else:
                chosen = np.random.choice(cell_indices, samples_per_cell, replace=False)
                selected_indices.extend(chosen.tolist())

        if len(selected_indices) > n_samples:
            selected_indices = np.random.choice(selected_indices, n_samples, replace=False).tolist()

        return np.array(selected_indices)

    def _get_cluster_samples(self, coords: np.ndarray, labels: np.ndarray,
                              n_samples: int, n_clusters: int = 20) -> np.ndarray:
        """Get samples using K-means clustering for spatial coverage"""
        if len(coords) <= n_samples:
            return np.arange(len(coords))

        kmeans = KMeans(n_clusters=max(1, min(n_clusters, len(coords) // 10)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)

        selected_indices = []
        samples_per_cluster = max(1, n_samples // len(np.unique(clusters)))

        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]

            if len(cluster_indices) <= samples_per_cluster:
                selected_indices.extend(cluster_indices.tolist())
            else:
                chosen = np.random.choice(cluster_indices, samples_per_cluster, replace=False)
                selected_indices.extend(chosen.tolist())

        if len(selected_indices) > n_samples:
            selected_indices = np.random.choice(selected_indices, n_samples, replace=False).tolist()

        return np.array(selected_indices)

    # ------------------------------------------------------------------
    # Sampling helpers (shared by flood / non-flood, focus / non-focus pools)
    # ------------------------------------------------------------------
    def _sample_pool(self, coords: np.ndarray, label: int, n: int, method: str,
                     flood_map: np.ndarray, buffer_px: int) -> np.ndarray:
        """Draw up to n (row, col) pixels from `coords` with the chosen method."""
        if len(coords) == 0 or n <= 0:
            return coords[:0]
        n = min(n, len(coords))
        labels = np.full(len(coords), label, dtype=float)
        if method == 'stratified':
            return coords[self._get_stratified_samples(coords, labels, n)]
        if method == 'cluster':
            return coords[self._get_cluster_samples(coords, labels, n)]
        if method == 'boundary':
            W = flood_map.shape[1]
            b = self._get_flood_boundary_pixels(flood_map, buffer_size=buffer_px)
            in_b = np.isin(coords[:, 0] * W + coords[:, 1], b[:, 0] * W + b[:, 1])
            b_idx, o_idx = np.where(in_b)[0], np.where(~in_b)[0]
            nb = min(n // 2, len(b_idx))
            pick = list(np.random.choice(b_idx, nb, replace=False)) if nb else []
            rest = min(n - nb, len(o_idx))
            pick += list(np.random.choice(o_idx, rest, replace=False)) if rest else []
            return coords[np.array(pick, dtype=int)]
        return coords[np.random.choice(len(coords), n, replace=False)]  # random

    def _sample_with_focus(self, coords, label, n, focus_mask, focus_fraction, method,
                           flood_map, buffer_px, tag):
        """focus_fraction of n from inside focus_mask, the rest from everywhere else.
        Returns (samples, in_focus_bool)."""
        if focus_mask is None:
            s = self._sample_pool(coords, label, n, method, flood_map, buffer_px)
            return s, np.zeros(len(s), dtype=bool)
        inside = focus_mask[coords[:, 0], coords[:, 1]]
        c_in, c_out = coords[inside], coords[~inside]
        want_in = int(round(n * focus_fraction))
        take_in = min(want_in, len(c_in))
        take_out = min(n - take_in, len(c_out))
        if take_in < want_in:
            logger.warning(f"    {tag}: only {len(c_in):,} px in focus area (wanted {want_in:,}); "
                           f"filling from outside")
        s_in = self._sample_pool(c_in, label, take_in, method, flood_map, buffer_px)
        s_out = self._sample_pool(c_out, label, take_out, method, flood_map, buffer_px)
        return (np.vstack([s_in, s_out]),
                np.concatenate([np.ones(len(s_in), bool), np.zeros(len(s_out), bool)]))

    # ------------------------------------------------------------------
    # Barangay focus
    # ------------------------------------------------------------------
    def _load_barangays(self, barangay_path, name_field, shape, transform, crs):
        """Load all barangay polygons once and burn their ids onto the aligned grid."""
        if barangay_path is None or not Path(barangay_path).exists():
            logger.error(f"Barangay layer not found ({barangay_path}); sampling will NOT be focused.")
            return False
        gdf = gpd.read_file(barangay_path)
        if gdf.crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
        gdf["bid"] = np.arange(1, len(gdf) + 1)
        all_names = sorted({n for v in FLOOD_BARANGAYS.values() for n in v})
        self._brgy_name_field = name_field or _detect_name_field(gdf, all_names)
        self._bid_raster = features.rasterize(
            zip(gdf.geometry, gdf["bid"].astype(int)), out_shape=shape, transform=transform,
            fill=0, dtype="int32", all_touched=True)
        self._brgy = gdf
        logger.info(f"Loaded {len(gdf)} barangay polygons (name field '{self._brgy_name_field}')")
        return True

    def _event_focus_mask(self, event, fm, auto_include_frac, auto_include_min_px):
        """Boolean mask of the focus area for one event:
        barangays listed for that event  +  any other barangay whose own SAR flood fraction
        shows it flooded too (so unlisted flooded barangays are not treated as 'outside')."""
        names = FLOOD_BARANGAYS.get(event)
        if not names:
            logger.warning(f"No barangay list for '{event}' -> unfocused")
            return None
        gdf, bid = self._brgy, self._bid_raster
        listed, unmatched = _match_barangay_ids(gdf, names, self._brgy_name_field)
        logger.info(f"  {event}: matched {len(names) - len(unmatched)}/{len(names)} listed barangays")
        if unmatched:
            logger.warning(f"  {event}: unmatched names (check spelling in layer): {unmatched}")
        mask = np.isin(bid, listed)

        nb = int(bid.max()) + 1
        valid_cnt = np.bincount(bid[fm >= 0], minlength=nb)
        flood_cnt = np.bincount(bid[fm > 0], minlength=nb)
        frac = flood_cnt / np.maximum(valid_cnt, 1)
        if auto_include_frac is not None:
            extra = [i for i in range(1, nb) if i not in set(listed)
                     and flood_cnt[i] >= auto_include_min_px and frac[i] >= auto_include_frac]
            if extra:
                nm = gdf.set_index("bid").loc[extra, self._brgy_name_field].tolist()
                logger.info(f"  {event}: +{len(extra)} UNLISTED barangays added to focus "
                            f"(SAR flood >= {auto_include_frac:.0%} of their valid land): {nm}")
                mask |= np.isin(bid, extra)

        n_flood = int((fm > 0).sum())
        if n_flood:
            out_share = (fm[(fm > 0) & ~mask]).size / n_flood
            logger.info(f"  {event}: {out_share*100:.1f}% of SAR flood pixels lie outside the focus area")
        return mask

    # ------------------------------------------------------------------
    def create_sample_points(self, flood_inventory: Dict = None,
                             n_samples: int = 20000,
                             flood_ratio: float = 0.5,
                             sampling_method: str = 'stratified',
                             land_mask_path: Path = Path("./outputs/phase2/aligned/chirps_aligned.tif"),
                             coast_buffer_m: float = 100.0,
                             drop_shore_connected: bool = True,
                             exclude_coastal_nonflood: bool = True,
                             barangay_path: Optional[Path] = None,
                             barangay_name_field: Optional[str] = None,
                             focus_fraction: float = 0.8,
                             auto_include_frac: Optional[float] = 0.05,
                             auto_include_min_px: int = 200
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Create balanced sample points on the aligned 12.5 m grid (flood-able land only).
        Each point is tagged with the flood event it was sampled from (self.sample_events).

        Barangay focus (optional, needs `barangay_path`):
          `focus_fraction` (default 0.8) of each event's flood AND non-flood points come from the
          barangays reported flooded for that event; the remaining (1 - focus_fraction) come from
          everywhere else, so nothing outside the list is excluded. Barangays NOT on the list whose
          SAR flood fraction >= `auto_include_frac` (and >= `auto_include_min_px` flood pixels) are
          counted as focus too. Set auto_include_frac=None to use the PDF lists only, or
          barangay_path=None to disable focusing entirely.

        Storm-surge exclusion (not riverine/pluvial flooding):
          - flood pixels within `coast_buffer_m` of the sea are dropped
          - if `drop_shore_connected`, whole flood patches touching that coastal zone are dropped
          - if `exclude_coastal_nonflood`, non-flood pixels in the coastal zone are also dropped
            (otherwise the model would learn 'near shore = never flood')
        """
        from rasterio.warp import reproject, Resampling
        from rasterio.coords import BoundingBox
        from scipy.ndimage import label as cc_label

        if flood_inventory is None:
            flood_inventory = self.flood_inventory

        if flood_inventory is None:
            raise ValueError("No flood inventory! Call generate_flood_inventory() first.")

        # Land mask + target grid: aligned rasters (CHIRPS is land-only, nodata = sea)
        land_mask_path = Path(land_mask_path)
        if not land_mask_path.exists():
            raise FileNotFoundError(f"Land mask raster not found: {land_mask_path}")
        with rasterio.open(land_mask_path) as src:
            land = ~np.ma.getmaskarray(src.read(1, masked=True))
            dst_transform, dst_crs = src.transform, src.crs
        src_transform = self.metadata['transform']
        src_crs = self.metadata['crs']
        logger.info(f"Land mask: {land.sum():,} land pixels of {land.size:,} ({land.mean()*100:.1f}%)")

        # Coastal zone: land pixels within coast_buffer_m of the sea (storm-surge influence)
        res_m = abs(dst_transform.a)
        coast_dist_m = distance_transform_edt(land).astype(np.float32) * res_m
        near_shore = land & (coast_dist_m <= coast_buffer_m)
        del coast_dist_m
        logger.info(f"Coastal zone (<= {coast_buffer_m:.0f} m from sea): {near_shore.sum():,} pixels")

        # Terrain mask: keep only flood-able terrain (removes radar shadow/layover false positives)
        def _load_aligned(name):
            with rasterio.open(land_mask_path.parent / f"{name}_aligned.tif") as s:
                return s.read(1, masked=True).astype(np.float32).filled(np.nan)

        hand_r = _load_aligned("hand")
        slope_r = _load_aligned("slope")
        dist_r = _load_aligned("distance_to_river")
        terrain_ok = np.isfinite(hand_r) & (hand_r >= 0) & (hand_r <= 15) & (slope_r <= 5) & (dist_r > 25)
        del hand_r, slope_r, dist_r
        logger.info(f"Terrain mask (0 <= HAND <= 15 m, slope <= 5 deg): "
                    f"{terrain_ok.sum():,} pixels ({terrain_ok.mean()*100:.1f}%)")

        focus_on = self._load_barangays(barangay_path, barangay_name_field,
                                        land.shape, dst_transform, dst_crs) if barangay_path else False

        n_flood = int(n_samples * flood_ratio)
        n_non_flood = n_samples - n_flood

        pts, lbls, evs, foc = [], [], [], []

        for event, info in flood_inventory.items():
            raw_flood_map = info['flood_map']

            # Resample the SAR flood map onto the aligned grid so (row, col) match the feature rasters
            cover = np.ones(land.shape, dtype=bool)
            same_grid = (raw_flood_map.shape == land.shape and src_transform == dst_transform
                         and rasterio.crs.CRS.from_user_input(src_crs) == dst_crs)
            if not same_grid:
                resampled = np.zeros(land.shape, dtype=raw_flood_map.dtype)
                reproject(raw_flood_map, resampled,
                          src_transform=src_transform, src_crs=src_crs,
                          dst_transform=dst_transform, dst_crs=dst_crs,
                          resampling=Resampling.nearest)
                covered = np.zeros(land.shape, dtype=np.uint8)
                reproject(np.ones(raw_flood_map.shape, dtype=np.uint8), covered,
                          src_transform=src_transform, src_crs=src_crs,
                          dst_transform=dst_transform, dst_crs=dst_crs,
                          resampling=Resampling.nearest)
                raw_flood_map, cover = resampled, covered.astype(bool)

            # Storm-surge exclusion: coastal zone + flood patches physically connected to it
            raw_flood = (raw_flood_map > 0) & land & cover
            shore_connected = np.zeros(land.shape, dtype=bool)
            if drop_shore_connected:
                lab, n_cc = cc_label(raw_flood, structure=np.ones((3, 3), dtype=bool))
                touching = np.unique(lab[raw_flood & near_shore])
                touching = touching[touching > 0]
                if touching.size:
                    lut = np.zeros(n_cc + 1, dtype=bool)
                    lut[touching] = True
                    shore_connected = lut[lab]
                del lab
            surge_zone = near_shore | shore_connected
            removed_surge = int(np.sum(raw_flood & surge_zone))

            # Sea / uncovered / non-flood-able pixels become -1: neither flood (>0) nor non-flood (==0)
            flood_map = np.where(land & cover & terrain_ok, raw_flood_map.astype(np.int16), -1)
            if exclude_coastal_nonflood:
                flood_map[surge_zone] = -1
            else:
                flood_map[(flood_map > 0) & surge_zone] = -1

            removed_sea = int(np.sum((raw_flood_map > 0) & ~land))
            removed_terrain = int(np.sum((raw_flood_map > 0) & land & ~terrain_ok))
            logger.info(f"{event}: {int((flood_map > 0).sum())} flood, {int((flood_map == 0).sum())} non-flood pixels "
                        f"({removed_sea:,} sea, {removed_terrain:,} steep/high, "
                        f"{removed_surge:,} coastal/storm-surge 'flood' pixels removed)")

            focus_mask = (self._event_focus_mask(event, flood_map, auto_include_frac, auto_include_min_px)
                          if focus_on else None)

            for lab_val, n_want, buf, name in ((1, n_flood, 10, 'flood'), (0, n_non_flood, 15, 'non-flood')):
                r, c = np.where(flood_map == lab_val)
                coords = np.column_stack([r, c])
                if len(coords) == 0:
                    continue
                s, f = self._sample_with_focus(coords, lab_val, n_want, focus_mask, focus_fraction,
                                               sampling_method, flood_map, buf, f"{event}/{name}")
                pts.append(s)
                lbls.append(np.full(len(s), lab_val, dtype=float))
                evs.append(np.full(len(s), event, dtype=object))
                foc.append(f)
                logger.info(f"  Sampled {len(s)} {name} points using {sampling_method}"
                            + (f" ({f.mean()*100:.0f}% in focus area)" if focus_mask is not None and len(f) else ""))

        # Sample points are now in aligned-grid (row, col); georeference them accordingly
        self.metadata['transform'] = dst_transform
        self.metadata['crs'] = dst_crs.to_string()
        self.metadata['bounds'] = BoundingBox(
            *rasterio.transform.array_bounds(land.shape[0], land.shape[1], dst_transform)
        )

        if pts:
            sample_points = np.vstack(pts)
            sample_labels = np.concatenate(lbls)
            sample_events = np.concatenate(evs)
            sample_focus = np.concatenate(foc)
        else:
            sample_points = np.empty((0, 2), dtype=int)
            sample_labels = np.array([])
            sample_events = np.array([], dtype=object)
            sample_focus = np.array([], dtype=bool)

        shuffle_idx = np.random.permutation(len(sample_points))
        self.sample_points = sample_points[shuffle_idx]
        self.sample_labels = sample_labels[shuffle_idx]
        self.sample_events = sample_events[shuffle_idx]
        self.sample_focus = sample_focus[shuffle_idx]

        sl, se = self.sample_labels, self.sample_events
        logger.info(f"Created {len(sl)} sample points ({int(np.sum(sl))} flood, {len(sl) - int(np.sum(sl))} non-flood)")
        logger.info(f"  Flood ratio: {np.mean(sl)*100:.2f}%")
        logger.info(f"  Sampling method: {sampling_method}")
        if focus_on:
            logger.info(f"  In focus area: {self.sample_focus.mean()*100:.1f}%")
        for ev, cnt in pd.Series(se).value_counts().items():
            n_f = int(np.sum((se == ev) & (sl == 1)))
            logger.info(f"  Event {ev}: {cnt} points ({n_f} flood, {cnt - n_f} non-flood)")

        return self.sample_points, self.sample_labels

    def save_sample_points(self, sample_points: np.ndarray = None,
                           sample_labels: np.ndarray = None,
                           output_dir: Path = Path("./outputs/phase2"),
                           metadata: Dict = None,
                           sample_events: np.ndarray = None) -> gpd.GeoDataFrame:
        """Save sample points as a master file and per-event files."""
        if sample_points is None:
            sample_points = self.sample_points

        if sample_labels is None:
            sample_labels = self.sample_labels

        if sample_events is None:
            sample_events = getattr(self, 'sample_events', None)

        if sample_points is None or sample_labels is None:
            raise ValueError("No sample points to save!")

        output_dir = Path(output_dir)
        sample_dir = output_dir / "sample_points"
        sample_dir.mkdir(parents=True, exist_ok=True)

        transform = None
        crs = None
        if metadata:
            transform = metadata.get('transform')
            crs = metadata.get('crs')

        geometries = []
        for i, (row, col) in enumerate(sample_points):
            if transform:
                try:
                    x, y = transform * (col, row)
                    geom = Point(x, y)
                except Exception:
                    geom = Point(row, col)
            else:
                geom = Point(row, col)
            geometries.append(geom)

        data = {
            'sample_id': range(len(sample_points)),
            'row': sample_points[:, 0],
            'col': sample_points[:, 1],
            'x': [g.x for g in geometries],
            'y': [g.y for g in geometries],
            'flood_label': sample_labels.astype(int),
        }
        if sample_events is not None:
            data['event'] = np.asarray(sample_events).astype(str)
        focus = getattr(self, 'sample_focus', None)
        if focus is not None and len(focus) == len(sample_points):
            data['in_focus'] = np.asarray(focus).astype(int)
        data['geometry'] = geometries

        gdf = gpd.GeoDataFrame(data, crs=crs)

        # 1. Save Master Files (All events combined)
        master_shp = sample_dir / "sample_points.shp"
        gdf.to_file(master_shp)
        master_csv = sample_dir / "sample_points.csv"
        gdf.drop(columns='geometry').to_csv(master_csv, index=False)
        logger.info(f"Saved master sample points to {sample_dir}")

        # 2. Save Per-Event Files
        if sample_events is not None:
            for event in np.unique(sample_events):
                event_gdf = gdf[gdf['event'] == event]
                if len(event_gdf) == 0:
                    continue

                event_shp = sample_dir / f"{event}_sample_points.shp"
                event_gdf.to_file(event_shp)

                event_csv = sample_dir / f"{event}_sample_points.csv"
                event_gdf.drop(columns='geometry').to_csv(event_csv, index=False)

                logger.info(f"Saved {len(event_gdf):,} points for {event} to {sample_dir}")

        # Save metadata
        metadata_clean = {}
        if metadata:
            for key, value in metadata.items():
                if key == 'transform':
                    if value is not None:
                        metadata_clean['transform'] = str(value)
                elif key == 'bounds':
                    if value is not None:
                        try:
                            metadata_clean['bounds'] = {
                                'left': float(value.left),
                                'bottom': float(value.bottom),
                                'right': float(value.right),
                                'top': float(value.top)
                            }
                        except Exception:
                            metadata_clean['bounds'] = str(value)
                else:
                    metadata_clean[key] = value

        with open(sample_dir / "metadata.json", 'w') as f:
            json.dump({
                'n_samples': len(sample_points),
                'n_flood': int(np.sum(sample_labels)),
                'n_non_flood': int(len(sample_labels) - np.sum(sample_labels)),
                'flood_ratio': float(np.mean(sample_labels)),
                'crs': crs,
                'metadata': metadata_clean
            }, f, indent=2)

        return gdf

    def get_sar_metadata(self) -> Dict:
        """Get SAR metadata"""
        if not self.metadata:
            meta_path = Path("./outputs/phase2/sample_points/metadata.json")
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    return json.load(f)
        return self.metadata

    def get_sar_summary(self) -> pd.DataFrame:
        """Get summary of SAR data"""
        if self.flood_inventory is None:
            logger.warning("No flood inventory available")
            return pd.DataFrame()

        summary = []
        for event, info in self.flood_inventory.items():
            summary.append({
                'event': event,
                'flood_pixels': info['flood_pixels'],
                'total_pixels': info['total_pixels'],
                'flood_percentage': info['flood_percentage'],
                'ndfi_mean': np.mean(info['ndfi']) if info['ndfi'] is not None else None,
                'ndfi_std': np.std(info['ndfi']) if info['ndfi'] is not None else None,
                'ndfi_min': np.min(info['ndfi']) if info['ndfi'] is not None else None,
                'ndfi_max': np.max(info['ndfi']) if info['ndfi'] is not None else None,
            })

        return pd.DataFrame(summary)


def main():
    """Test the SAR processor with improved sampling"""
    processor = SARProcessor()

    try:
        sar_data = processor.load_preprocessed_sar()
        logger.info(f"Loaded SAR data for {len(sar_data)} events")

        flood_inventory = processor.generate_flood_inventory()
        logger.info(f"Generated flood inventory with {len(flood_inventory)} events")

        methods = ['stratified', 'cluster', 'boundary', 'random']

        for method in methods:
            logger.info(f"\n--- Testing {method} sampling ---")
            points, labels = processor.create_sample_points(
                n_samples=4000,
                sampling_method=method
            )
            logger.info(f"  {method}: {len(points)} points, {np.sum(labels)} flood, {len(labels) - np.sum(labels)} non-flood")

        gdf = processor.save_sample_points(points, labels, metadata=processor.metadata)
        logger.info("Sample points saved")

        summary = processor.get_sar_summary()
        logger.info(f"\n{summary}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()