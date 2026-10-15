import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime


load_dotenv()

BASE_DIR = Path(os.getenv('BASE_DIR', 
    '/home/mar/lectures/thesis 1/fsm/datas/FOR TRAINING'))
PREPROCESS_DIR = Path(os.getenv('PREPROCESS_DIR',
    '/home/mar/lectures/thesis 1/fsm/preprocess'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './outputs/phase2'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# TOPOGRAPHIC DATA
# ============================================================

TOPOGRAPHIC = {
    "dem": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_DEM_Filled.tiff", 
    "slope": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_Slope.tiff",
    "aspect": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_Aspect.tiff",
    "twi": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_TWI.tif",
    "hand": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_HAND.tif",
    "flow_accumulation": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_Flow_Accumulation_Log.tif",
}

# ============================================================
# HYDROLOGICAL DATA
# ============================================================

HYDROLOGICAL = {
    "distance_to_river": BASE_DIR / "HYDROLOGICAL DATAS/qgis/Zamboanga_Distance_to_river_1.tiff",
    "drainage_density": BASE_DIR / "HYDROLOGICAL DATAS/qgis/Zamboanga_Drainage_Density_1.tif",
    "river_network": BASE_DIR / "HYDROLOGICAL DATAS/spatialdata/ZC_RiverCreeksnew.shp",
    "watersheds": BASE_DIR / "HYDROLOGICAL DATAS/spatialdata/Watershed_Areas.shp",
    "phl_rivers": BASE_DIR / "HYDROLOGICAL DATAS/phl_rivl_250k_namria/phl_rivl_250k_NAMRIA.shp",
}

# Raster-only view of HYDROLOGICAL, for steps (e.g. spatial alignment) that
# need to call rasterio.open() on every path and would break on a .shp.
HYDROLOGICAL_RASTERS = {
    "distance_to_river": HYDROLOGICAL["distance_to_river"],
    "drainage_density": HYDROLOGICAL["drainage_density"],
}

# ============================================================
# CLIMATIC DATA
# ============================================================

CLIMATIC = {
    "chirps": BASE_DIR / "CLIMATIC DATAS/qgis/Zamboanga_CHIRPS_Resampled.tiff", 
    "gpm": BASE_DIR / "CLIMATIC DATAS/Zamboanga_GPM_V07_2021_2026.tif",
    "chirps_daily": BASE_DIR / "CLIMATIC DATAS/Zamboonga_CHIRPS_Daily_2021_2026.csv", 
    "gpm_daily": BASE_DIR / "CLIMATIC DATAS/Zamboonga_GPM_V07_Daily_2021_2026.csv", 
}

# Raster-only view of CLIMATIC (excludes the _daily CSV time series).
CLIMATIC_RASTERS = {
    "chirps": CLIMATIC["chirps"],
    "gpm": CLIMATIC["gpm"],
}

# ============================================================
# SAR DATA
# ============================================================

REGISTERED_SAR_DIR = Path("./outputs/phase2/registered_sar")

SAR_REGISTERED = {
    "may_2021": {
        "flood_map": REGISTERED_SAR_DIR / "may_2021_flood_map_registered.tif",
        "ndfi": REGISTERED_SAR_DIR / "may_2021_ndfi_registered.tif",
        "vh": REGISTERED_SAR_DIR / "may_2021_vh_registered.tif",
    },
    "oct_2022": {
        # Raw SNAP outputs (pre-registration):
        #   .../SAR DATAS/October 27-29, 2022/outputs/flood_map.tif
        #   .../SAR DATAS/October 27-29, 2022/outputs/ndfi.tif
        #   .../SAR DATAS/October 27-29, 2022/outputs/diff_db.tif
        "flood_map": REGISTERED_SAR_DIR / "oct_2022_flood_map_registered.tif",
        "ndfi": REGISTERED_SAR_DIR / "oct_2022_ndfi_registered.tif",
        "vh": REGISTERED_SAR_DIR / "oct_2022_vh_registered.tif",
        "diff_db": REGISTERED_SAR_DIR / "oct_2022_diff_db_registered.tif",
    },
}

FLOOD_EVENTS = {
    "may_2021": {
        "name": "May 19, 2021",
        "date": datetime(2021, 5, 19),
        "sar_key": "may_2021",
        "pre_date": datetime(2021, 5, 17),
        "post_date": datetime(2021, 5, 29),
        "days_before": 7,
        "days_after": 7,
    },
    "oct_2022": {
        "name": "October 27-29, 2022 (Typhoon Nalgae/Paeng)",
        "date": datetime(2022, 10, 28),  # peak flooding
        "sar_key": "oct_2022",
        "pre_date": datetime(2022, 10, 20),
        "post_date": datetime(2022, 11, 1),
        "days_before": 8,
        "days_after": 4,
    },
}

AVAILABLE_EVENTS = list(FLOOD_EVENTS.keys())

# Switch this to whichever event you want the pipeline to process.
# October 2022 is the preferred event (tighter temporal gap, better flood signal).
ACTIVE_EVENT = "oct_2022"

# ============================================================
# SOCIO-DEMOGRAPHIC DATA
# ============================================================

SOCIO = {
    "census": BASE_DIR / "SOCIO DEMOGRAPHIC DATAS/Region-IX_0.xlsx", 
    "zc_boundary": BASE_DIR / "SOCIO DEMOGRAPHIC DATAS/zamboanga_city_boundary_UTM.shp",
}

# Additional vector data paths
VECTOR_DATA = {
    "city_boundary": BASE_DIR / "SOCIO DEMOGRAPHIC DATAS/zamboanga_city_boundary.geojson",
    "city_boundary_shapefile": BASE_DIR / "SOCIO DEMOGRAPHIC DATAS/zamboanga_city_boundary_UTM.shp",
}

# ============================================================
# CRS AND SPATIAL GRID
# ============================================================

TARGET_CRS = "EPSG:32651"

# Common pixel resolution (meters) that every raster is resampled to during
# spatial alignment. Per the manuscript, source rasters are 12.5 m — this
# MUST match that, not an assumed/default value, or SpatialAligner will
# silently coarsen or fabricate detail when reprojecting onto the shared grid.
TARGET_RESOLUTION = 12.5

# ============================================================
# RAW FEATURES TO EXTRACT (from rasters)
# ============================================================

RAW_FEATURES_TO_EXTRACT = [
    "dem",
    "slope",
    "aspect",
    "twi",
    "hand",
    "flow_accumulation",
    "distance_to_river",
    # "drainage_density",   # 0.000 importance — constant across samples, contributes nothing
    "chirps",
    "gpm",
]

ENGINEERED_FEATURES_ENABLED = True

# Feature list pruned based on Random Forest importance analysis
# (see outputs/phase2/feature_importance/feature_importance.csv).
# Retained: importance >= 0.005 and no evidence of leakage.
ENGINEERED_FEATURES = [
    # --- Hydrological / topographic composites (high signal) ---
    "wetness_index",              # 0.008 — moderate, kept
    # "saturation_index",         # 0.003 — near-zero importance
    # "flood_concentration",      # 0.051 — kept below
    "flood_concentration",        # 0.051 — strong predictor
    # "drainage_capacity",        # <0.006 — negligible
    "twi_hand_ratio",             # 0.020 — moderate
    
    # --- Terrain / orientation ---
    # "terrain_roughness",        # 0.006 — negligible
    "northness",                  # 0.009 — weak but retained
    "eastness",                   # 0.008 — weak but retained
    "relative_elevation",         # 0.024 — moderate
    "slope_aspect",               # 0.008 — weak, kept for interaction
    
    # --- River / floodplain proximity ---
    "river_proximity",            # 0.054 — strong
    # "river_influence",          # 0.000 — zero importance (constant?)
    "floodplain_index",           # 0.056 — top 3
    
    # --- Rainfall interactions ---
    "rainfall_intensity",         # 0.054 — strong
    "rainfall_ratio",             # 0.051 — strong
    "rainfall_wetness",           # 0.044 — strong
    "rainfall_flow",              # 0.046 — strong
    
    # --- Composite indices ---
    # "flood_susceptibility_index",  # 0.049 — SUSPICIOUS: name suggests target leakage; drop
    # "flash_flood_potential",     # 0.006 — negligible
    # "water_logging_potential",   # 0.007 — negligible
    
    # --- Interactions / ratios ---
    "twi_hand",                   # 0.020 — moderate
    # "slope_flow",               # 0.006 — negligible
    # "dist_slope",               # 0.007 — negligible
    # "drainage_rainfall",        # 0.000 — zero importance
    "dem_river_ratio",            # 0.049 — strong
    "hand_dem_ratio",             # 0.025 — moderate
    # "twi_slope_ratio",          # 0.007 — negligible
    "flow_dist_ratio",            # 0.055 — top 5
    
    # --- Log / polynomial transforms (redundant with raw) ---
    # "twi_squared",              # 0.007 — redundant with twi
    # "flow_acc_squared",         # 0.007 — redundant with flow_accumulation
    # "hand_squared",             # 0.018 — kept below
    "hand_squared",               # 0.018 — moderate
    # "slope_squared",            # 0.006 — redundant with slope
    # "log_twi",                  # 0.008 — redundant with twi
    # "log_flow_acc",             # 0.007 — redundant with flow_accumulation
    # "log_hand",                 # 0.003 — redundant with hand
    # "log_slope",                # 0.006 — redundant with slope
    
    # --- Spatial normalization ---
    # "dist_center_norm",         # 0.074 — TOP-1 but ARTIFACT (sampling bias, no physics); drop
]


ALL_FEATURES = RAW_FEATURES_TO_EXTRACT + ENGINEERED_FEATURES

FEATURES_TO_EXTRACT = RAW_FEATURES_TO_EXTRACT


SAMPLE_PARAMS = {
    "flood_points": 2000,
    "non_flood_points": 2000,
    "random_seed": 42,
    "test_size": 0.2,
}


GWPCA_PARAMS = {
    "variance_threshold": 0.85,
    "bandwidth_method": "aicc",
}


LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')