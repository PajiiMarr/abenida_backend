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

TOPOGRAPHIC = {
    "dem": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_DEM_Filled.tiff", 
    "slope": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_Slope.tiff",
    "aspect": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_Aspect.tiff",
    "twi": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_TWI.tif",
    "hand": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_HAND.tif",
    "flow_accumulation": BASE_DIR / "TOPOGRAPHIC DATAS/qgis/Zamboanga_Flow_Accumulation_Log.tif",
}

HYDROLOGICAL = {
    "distance_to_river": BASE_DIR / "HYDROLOGICAL DATAS/qgis/Zamboanga_Distance_to_river_1.tiff",
    "drainage_density": BASE_DIR / "HYDROLOGICAL DATAS/qgis/Zamboanga_Drainage_Density_1.tif",
    "river_network": BASE_DIR / "HYDROLOGICAL DATAS/spatialdata/ZC_RiverCreeksnew.shp",
    "watersheds": BASE_DIR / "HYDROLOGICAL DATAS/spatialdata/Watershed_Areas.shp",
    "phl_rivers": BASE_DIR / "HYDROLOGICAL DATAS/phl_rivl_250k_namria/phl_rivl_250k_NAMRIA.shp",
}


HYDROLOGICAL_RASTERS = {
    "distance_to_river": HYDROLOGICAL["distance_to_river"],
    "drainage_density": HYDROLOGICAL["drainage_density"],
}

CLIMATIC = {
    "chirps": BASE_DIR / "CLIMATIC DATAS/qgis/Zamboanga_CHIRPS_Resampled.tiff", 
    "gpm": BASE_DIR / "CLIMATIC DATAS/Zamboanga_GPM_V07_2021_2026.tif",
    "chirps_daily": BASE_DIR / "CLIMATIC DATAS/Zamboonga_CHIRPS_Daily_2021_2026.csv", 
    "gpm_daily": BASE_DIR / "CLIMATIC DATAS/Zamboonga_GPM_V07_Daily_2021_2026.csv", 
}

CLIMATIC_RASTERS = {
    "chirps": CLIMATIC["chirps"],
    "gpm": CLIMATIC["gpm"],
}

REGISTERED_SAR_DIR = Path("./outputs/phase2/registered_sar")

SAR_REGISTERED = {
    "may_2021": {
        "flood_map": REGISTERED_SAR_DIR / "may_2021_flood_map_registered.tif",
        "ndfi": REGISTERED_SAR_DIR / "may_2021_ndfi_registered.tif",
        "vh": REGISTERED_SAR_DIR / "may_2021_vh_registered.tif",
    },
    "oct_2022": {
        "flood_map": REGISTERED_SAR_DIR / "oct_2022_flood_map_registered.tif",
        "ndfi": REGISTERED_SAR_DIR / "oct_2022_ndfi_registered.tif",
        "vh": REGISTERED_SAR_DIR / "oct_2022_vh_registered.tif",
        "diff_db": REGISTERED_SAR_DIR / "oct_2022_diff_db_registered.tif",
    },
    "jan_2023": {
        "flood_map": REGISTERED_SAR_DIR / "jan_2023_flood_map_registered.tif",
        "ndfi":      REGISTERED_SAR_DIR / "jan_2023_ndfi_registered.tif",
        "diff_db":   REGISTERED_SAR_DIR / "jan_2023_diff_db_registered.tif",
    },
    "jan31_2023": {
        "flood_map": REGISTERED_SAR_DIR / "jan31_2023_flood_map_registered.tif",
        "ndfi":      REGISTERED_SAR_DIR / "jan31_2023_ndfi_registered.tif",
        "diff_db":   REGISTERED_SAR_DIR / "jan31_2023_diff_db_registered.tif",
    },
    "aug_2023": {
        "flood_map": REGISTERED_SAR_DIR / "aug_2023_flood_map_registered.tif",
        "ndfi":      REGISTERED_SAR_DIR / "aug_2023_ndfi_registered.tif",
        "diff_db":   REGISTERED_SAR_DIR / "aug_2023_diff_db_registered.tif",
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
    "jan_2023": {
        "name": "January 11, 2023 (Shear Line)",
        "date": datetime(2023, 1, 11),
        "sar_key": "jan_2023",
        "pre_date": datetime(2022, 12, 31),
        "post_date": datetime(2023, 1, 12),
        "days_before": 11,
        "days_after": 1,
    },
    "jan31_2023": {
        "name": "January 31, 2023 (Localized Thunderstorms)",
        "date": datetime(2023, 1, 31),
        "sar_key": "jan31_2023",
        "pre_date": datetime(2023, 1, 24),
        "post_date": datetime(2023, 2, 5),
        "days_before": 7,
        "days_after": 5,
    },
    "aug_2023": {
        "name": "August 12, 2023 (Southwest Monsoon)",
        "date": datetime(2023, 8, 12),
        "sar_key": "aug_2023",
        "pre_date": datetime(2023, 8, 4),
        "post_date": datetime(2023, 8, 16),
        "days_before": 8,
        "days_after": 4,
    },
}

AVAILABLE_EVENTS = list(FLOOD_EVENTS.keys())

ACTIVE_EVENT = "oct_2022"


SOCIO = {
    "census": BASE_DIR / "SOCIO DEMOGRAPHIC DATAS/Region-IX_0.xlsx", 
    "zc_boundary": BASE_DIR / "SOCIO DEMOGRAPHIC DATAS/zamboanga_city_boundary_UTM.shp",
}

VECTOR_DATA = {
    "city_boundary": BASE_DIR / "SOCIO DEMOGRAPHIC DATAS/zamboanga_city_boundary.geojson",
    "city_boundary_shapefile": BASE_DIR / "SOCIO DEMOGRAPHIC DATAS/zamboanga_city_boundary_UTM.shp",
}

TARGET_CRS = "EPSG:32651"

TARGET_RESOLUTION = 12.5

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

ENGINEERED_FEATURES = [
    "wetness_index",              # 0.008 — moderate, kept
    # "saturation_index",         # 0.003 — near-zero importance
    # "flood_concentration",      # 0.051 — kept below
    "flood_concentration",        # 0.051 — strong predictor
    # "drainage_capacity",        # <0.006 — negligible
    "twi_hand_ratio",             # 0.020 — moderate
    # "terrain_roughness",        # 0.006 — negligible
    "northness",                  # 0.009 — weak but retained
    "eastness",                   # 0.008 — weak but retained
    "relative_elevation",         # 0.024 — moderate
    "slope_aspect",               # 0.008 — weak, kept for interaction
    "river_proximity",            # 0.054 — strong
    # "river_influence",          # 0.000 — zero importance (constant?)
    "floodplain_index",           # 0.056 — top 3
    "rainfall_intensity",         # 0.054 — strong
    "rainfall_ratio",             # 0.051 — strong
    "rainfall_wetness",           # 0.044 — strong
    "rainfall_flow",              # 0.046 — strong
    # "flood_susceptibility_index",  # 0.049 — SUSPICIOUS: name suggests target leakage; drop
    # "flash_flood_potential",     # 0.006 — negligible
    # "water_logging_potential",   # 0.007 — negligible
    "twi_hand",                   # 0.020 — moderate
    # "slope_flow",               # 0.006 — negligible
    # "dist_slope",               # 0.007 — negligible
    # "drainage_rainfall",        # 0.000 — zero importance
    "dem_river_ratio",            # 0.049 — strong
    "hand_dem_ratio",             # 0.025 — moderate
    # "twi_slope_ratio",          # 0.007 — negligible
    "flow_dist_ratio",            # 0.055 — top 5
    # "twi_squared",              # 0.007 — redundant with twi
    # "flow_acc_squared",         # 0.007 — redundant with flow_accumulation
    # "hand_squared",             # 0.018 — kept below
    "hand_squared",               # 0.018 — moderate
    # "slope_squared",            # 0.006 — redundant with slope
    # "log_twi",                  # 0.008 — redundant with twi
    # "log_flow_acc",             # 0.007 — redundant with flow_accumulation
    # "log_hand",                 # 0.003 — redundant with hand
    # "log_slope",                # 0.006 — redundant with slope
    # "dist_center_norm",         # 0.074 — TOP-1 but ARTIFACT (sampling bias, no physics); drop
]


ALL_FEATURES = RAW_FEATURES_TO_EXTRACT + ENGINEERED_FEATURES

FEATURES_TO_EXTRACT = RAW_FEATURES_TO_EXTRACT


SAMPLE_PARAMS = {
    "flood_points": 10000,
    "non_flood_points": 10000,
    "random_seed": 42,
    "test_size": 0.2,
}


GWPCA_PARAMS = {
    "variance_threshold": 0.85,
    "bandwidth_method": "aicc",
}


LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')