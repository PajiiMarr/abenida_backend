#!/usr/bin/env python
"""
Register SAR flood maps to DEM coordinate system
This reprojects SAR data from pixel coordinates to UTM (EPSG:32651)
"""
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_origin
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/Users/mar/lectures/thesis 1/fsm/datas/FOR TRAINING")
OUTPUT_DIR = Path("./outputs/phase2/registered_sar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DEM path (reference)
DEM_PATH = BASE_DIR / "TOPOGRAPHIC DATAS/Zamboanga_DEM.tif"

# SAR paths
SAR_EVENTS = {
    'may_2021': {
        'flood_map': BASE_DIR / "SAR DATAS/May 19, 2021/outputs/flood_map_20210517_20210529.tif",
        'ndfi': BASE_DIR / "SAR DATAS/May 19, 2021/outputs/ndfi_20210517_20210529.tif",
        'vh': BASE_DIR / "SAR DATAS/May 19, 2021/outputs/vh_sigma0_db_co_event_20210529.tif",
    },
}


def create_geotransform_for_sar(sar_shape, bounds, crs):
    """
    Create a geotransform for SAR data based on DEM bounds
    """
    # Use DEM bounds as reference
    # We'll assume SAR covers a similar area to DEM
    # Calculate transform that maps SAR pixel coordinates to UTM
    
    # Use DEM bounds for reference
    with rasterio.open(DEM_PATH) as src:
        dem_bounds = src.bounds
        dem_transform = src.transform
        dem_crs = src.crs
    
    # Calculate scale to fit SAR into DEM bounds
    # We'll assume SAR covers the same geographic area as DEM
    # So we scale SAR pixels to match DEM extent
    
    sar_height, sar_width = sar_shape
    dem_width = 1961  # From DEM
    dem_height = 2416  # From DEM
    
    # Calculate pixel size to fit SAR into DEM extent
    x_res = (dem_bounds.right - dem_bounds.left) / sar_width
    y_res = (dem_bounds.top - dem_bounds.bottom) / sar_height
    
    # Create transform
    transform = from_origin(
        dem_bounds.left,  # x min
        dem_bounds.top,   # y max
        x_res, y_res
    )
    
    return transform, dem_crs


def register_sar_to_dem(sar_path, output_path, dem_reference_path):
    """
    Reproject SAR data to DEM's coordinate system
    """
    logger.info(f"Registering: {sar_path}")
    
    # Get DEM reference info
    with rasterio.open(dem_reference_path) as dem_src:
        dem_bounds = dem_src.bounds
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs
        dem_shape = (dem_src.height, dem_src.width)
    
    # Load SAR data
    with rasterio.open(sar_path) as sar_src:
        sar_data = sar_src.read(1)
        sar_shape = sar_src.shape
        sar_nodata = sar_src.nodata
        
        logger.info(f"  SAR shape: {sar_shape}")
        logger.info(f"  SAR data type: {sar_data.dtype}")
    
    # Determine if SAR needs to be reprojected or just resampled
    # Since SAR has no CRS, we need to assume it covers the same area as DEM
    
    # OPTION A: Simply resample SAR to DEM resolution
    # This assumes SAR covers the same geographic area as DEM
    
    # Calculate resampling factors
    sar_height, sar_width = sar_shape
    dem_height, dem_width = dem_shape
    
    logger.info(f"  Resampling from {sar_width}x{sar_height} to {dem_width}x{dem_height}")
    
    # Create destination array
    dst_data = np.zeros((dem_height, dem_width), dtype=np.float32)
    
    # Use nearest neighbor for binary flood map
    from scipy.ndimage import zoom
    zoom_factors = (dem_height / sar_height, dem_width / sar_width)
    resampled = zoom(sar_data.astype(np.float32), zoom_factors, order=0)
    
    # Ensure binary output
    dst_data = (resampled > 0.5).astype(np.uint8)
    
    logger.info(f"  Registered shape: {dst_data.shape}")
    logger.info(f"  Flood pixels: {dst_data.sum():,}")
    
    # Save registered raster
    meta = {
        'driver': 'GTiff',
        'height': dem_height,
        'width': dem_width,
        'count': 1,
        'dtype': 'uint8',
        'crs': dem_crs,
        'transform': dem_transform,
        'compress': 'lzw',
        'nodata': 0
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(dst_data, 1)
    
    logger.info(f"  Saved to: {output_path}")
    
    return dst_data


def main():
    """Register all SAR events to DEM"""
    logger.info("=" * 70)
    logger.info("REGISTERING SAR TO DEM COORDINATE SYSTEM")
    logger.info("=" * 70)
    
    # Register each SAR event
    for event_name, event_paths in SAR_EVENTS.items():
        logger.info(f"\n--- Processing {event_name} ---")
        
        for key, sar_path in event_paths.items():
            if not sar_path.exists():
                logger.warning(f"File not found: {sar_path}")
                continue
            
            output_path = OUTPUT_DIR / f"{event_name}_{key}_registered.tif"
            
            try:
                register_sar_to_dem(sar_path, output_path, DEM_PATH)
            except Exception as e:
                logger.error(f"Failed to register {key}: {e}")
                import traceback
                traceback.print_exc()
    
    logger.info("\n" + "=" * 70)
    logger.info("REGISTRATION COMPLETE")
    logger.info(f"Registered files saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()