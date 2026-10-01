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
    "distance_to_river": BASE_DIR / "HYDROLOGICAL DATAS/qgis/Zamboanga_Distance_to_river.tiff",
    "drainage_density": BASE_DIR / "HYDROLOGICAL DATAS/qgis/Zamboanga_Drainage_Density.tiff", 
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
}

FLOOD_EVENTS = {
    "may_2021": {
        "name": "May 19, 2021",
        "date": datetime(2021, 5, 19),
        "sar_key": "may_2021",
        "days_before": 7,
        "days_after": 7,
    },
}

AVAILABLE_EVENTS = list(FLOOD_EVENTS.keys())
ACTIVE_EVENT = "may_2021" 

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
    "drainage_density",
    "chirps",
    "gpm",
]


ENGINEERED_FEATURES_ENABLED = True

ENGINEERED_FEATURES = [
    "wetness_index",
    "saturation_index",
    "flood_concentration",
    "drainage_capacity",
    "twi_hand_ratio",
    
    "terrain_roughness",
    "northness",
    "eastness",
    "relative_elevation",
    "slope_aspect",
    
    "river_proximity",
    "river_influence",
    "floodplain_index",
    
    "rainfall_intensity",
    "rainfall_ratio",
    "rainfall_wetness",
    "rainfall_flow",
    
    "flood_susceptibility_index",
    "flash_flood_potential",
    "water_logging_potential",
    
    "twi_hand",
    "slope_flow",
    "dist_slope",
    "drainage_rainfall",
    
    "twi_squared",
    "flow_acc_squared",
    "hand_squared",
    "slope_squared",
    "log_twi",
    "log_flow_acc",
    "log_hand",
    "log_slope",
    
    "dist_center_norm",
    "dem_river_ratio",
    
    "hand_dem_ratio",
    "twi_slope_ratio",
    "flow_dist_ratio",
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
