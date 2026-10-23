#!/usr/bin/env python
"""
Register the October 2022 SAR layers to the DEM grid (georeferenced reprojection).

Every input must already carry a CRS and a real geotransform. Files without
them are rejected instead of guessed.
"""
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/Users/mar/lectures/thesis 1/fsm/datas/FOR TRAINING")
OUTPUT_DIR = Path("./outputs/phase2/registered_sar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEM_PATH = BASE_DIR / "TOPOGRAPHIC DATAS/Zamboanga_DEM.tif"

SAR_EVENTS = {
    'jun_2024': {
        'flood_map': BASE_DIR / "SAR DATAS/June 26, 2024/outputs/flood_map.tif",
        'ndfi':      BASE_DIR / "SAR DATAS/June 26, 2024/outputs/ndfi.tif",
        'diff_db':   BASE_DIR / "SAR DATAS/June 26, 2024/outputs/diff_db.tif",
    },
}


def register_sar_to_dem(sar_path, output_path, dem_reference_path):
    logger.info(f"Registering: {sar_path}")

    with rasterio.open(dem_reference_path) as dem_src:
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs
        dem_shape = (dem_src.height, dem_src.width)
        dem_bounds = dem_src.bounds

    with rasterio.open(sar_path) as sar_src:
        sar_data = sar_src.read(1)
        sar_transform = sar_src.transform
        sar_crs = sar_src.crs
        sar_dtype = sar_src.dtypes[0]
        sar_nodata = sar_src.nodata

    logger.info(f"  SAR shape: {sar_data.shape}")
    logger.info(f"  SAR dtype: {sar_dtype}")
    logger.info(f"  SAR CRS:   {sar_crs}")

    if sar_crs is None or sar_transform == rasterio.Affine.identity():
        raise ValueError(
            f"{sar_path} has no CRS or no geotransform. Re-export it with "
            f"georeferencing before registering."
        )

    # Make sure the SAR footprint actually overlaps the DEM before warping.
    sar_bounds = array_bounds(sar_data.shape[0], sar_data.shape[1], sar_transform)
    sar_in_dem = transform_bounds(sar_crs, dem_crs, *sar_bounds)
    overlaps = not (
        sar_in_dem[2] <= dem_bounds.left or sar_in_dem[0] >= dem_bounds.right
        or sar_in_dem[3] <= dem_bounds.bottom or sar_in_dem[1] >= dem_bounds.top
    )
    if not overlaps:
        raise ValueError(
            f"{sar_path} does not overlap the DEM.\n"
            f"  SAR bounds (in DEM CRS): {sar_in_dem}\n"
            f"  DEM bounds:              {tuple(dem_bounds)}"
        )

    is_flood = 'flood_map' in Path(sar_path).name.lower()

    if is_flood:
        # Binary mask: nearest neighbour, background = 0.
        dst_data = np.zeros(dem_shape, dtype=np.uint8)
        reproject(
            source=(sar_data > 0).astype(np.uint8),
            destination=dst_data,
            src_transform=sar_transform,
            src_crs=sar_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.nearest,
        )
        out_dtype = 'uint8'
        nodata = None
        logger.info(f"  Flood pixels: {int(dst_data.sum()):,}")
    else:
        # Continuous layer: bilinear, outside-footprint = NaN (0 is a valid
        # NDFI / dB value, so it must not be used as background).
        dst_data = np.full(dem_shape, np.nan, dtype=np.float32)
        reproject(
            source=sar_data.astype(np.float32),
            destination=dst_data,
            src_transform=sar_transform,
            src_crs=sar_crs,
            src_nodata=sar_nodata if sar_nodata is not None else np.nan,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        out_dtype = 'float32'
        nodata = np.nan
        valid = int(np.isfinite(dst_data).sum())
        logger.info(f"  Valid pixels: {valid:,} of {dst_data.size:,}")

    logger.info(f"  Registered shape: {dst_data.shape}")

    meta = {
        'driver': 'GTiff',
        'height': dem_shape[0],
        'width': dem_shape[1],
        'count': 1,
        'dtype': out_dtype,
        'crs': dem_crs,
        'transform': dem_transform,
        'compress': 'lzw',
        'nodata': nodata,
    }

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(dst_data, 1)

    logger.info(f"  Saved to: {output_path}")
    return dst_data


def main():
    logger.info("=" * 70)
    logger.info("REGISTERING SAR TO DEM COORDINATE SYSTEM")
    logger.info("=" * 70)

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