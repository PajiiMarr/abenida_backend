# backend/app/preprocessing/spatial_alignment.py
"""
Spatial alignment via true georeferenced reprojection.

The previous implementation resized numpy arrays with scipy.ndimage.zoom
purely by pixel COUNT, with no knowledge of each source raster's real
transform/CRS/resolution. It then stamped the result with a fabricated
transform when writing to disk. The written GeoTIFFs therefore had
metadata that didn't describe their actual pixel content, which is why
the "*_aligned.tif" layers don't line up with the city boundary (or
anything else) in QGIS.

This version reprojects every source raster directly from its own file
(so it always has a real src_transform/src_crs) onto one shared target
grid (defined by the boundary bounds + a fixed resolution), using
rasterio.warp.reproject. That's the same approach already used correctly
by `register_sar_to_dem` in the original file — it's just now applied
everywhere.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from .config import TARGET_CRS, TARGET_RESOLUTION  # noqa: F401 (TARGET_RESOLUTION used below)

logger = logging.getLogger(__name__)

# Raster "kinds" that are categorical / binary and must not be
# interpolated with bilinear resampling.
CATEGORICAL_RASTERS = {"flood_map", "flood_label"}

RasterSource = Union[str, Path]


class SpatialAligner:
    """Reproject and clip all rasters onto one common CRS, extent, and resolution."""

    def __init__(self, target_crs: str = TARGET_CRS, resolution: float = TARGET_RESOLUTION):
        self.target_crs = target_crs
        self.resolution = resolution
        self.target_bounds: Optional[Tuple[float, float, float, float]] = None
        self.target_shape: Optional[Tuple[int, int]] = None
        self.target_transform = None
        self.aligned_rasters: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Target grid setup
    # ------------------------------------------------------------------
    def _set_target_grid(self, bounds: Tuple[float, float, float, float]):
        """
        Define the shared pixel grid every raster will be resampled onto.

        bounds must already be in self.target_crs.
        """
        minx, miny, maxx, maxy = bounds
        width = int(np.ceil((maxx - minx) / self.resolution))
        height = int(np.ceil((maxy - miny) / self.resolution))

        # Snap maxx/maxy outward so width/height * resolution >= requested extent
        maxx_snapped = minx + width * self.resolution
        maxy_snapped = miny + height * self.resolution

        self.target_bounds = (minx, miny, maxx_snapped, maxy_snapped)
        self.target_shape = (height, width)
        self.target_transform = from_origin(minx, maxy_snapped, self.resolution, self.resolution)

        logger.info(
            f"Target grid set: shape={self.target_shape}, "
            f"bounds={self.target_bounds}, resolution={self.resolution}, "
            f"crs={self.target_crs}"
        )

    def set_target_grid_from_boundary(self, boundary_gdf: gpd.GeoDataFrame, buffer: float = 0):
        """Public helper: derive the target grid from a boundary polygon."""
        boundary_gdf = self.align_vectors(boundary_gdf)
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
        if buffer:
            minx, miny, maxx, maxy = minx - buffer, miny - buffer, maxx + buffer, maxy + buffer
        self._set_target_grid((minx, miny, maxx, maxy))

    # ------------------------------------------------------------------
    # Core reprojection
    # ------------------------------------------------------------------
    def _reproject_one(self, src_path: RasterSource, name: str) -> np.ndarray:
        """Reproject a single raster FILE onto the shared target grid."""
        if self.target_transform is None:
            raise RuntimeError("Target grid not set. Call set_target_grid_from_boundary(...) first.")

        resampling = Resampling.nearest if name in CATEGORICAL_RASTERS else Resampling.bilinear

        dst = np.full(self.target_shape, np.nan, dtype=np.float32)

        with rasterio.open(src_path) as src:
            src_band = src.read(1).astype(np.float32)
            reproject(
                source=src_band,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=self.target_transform,
                dst_crs=self.target_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
        return dst

    def align_raster_files(self, raster_paths: Dict[str, RasterSource]) -> Dict[str, np.ndarray]:
        """
        Reproject every raster in `raster_paths` onto the current target grid.

        raster_paths: name -> path to the ORIGINAL geotiff on disk. We need
        the actual file (not a pre-loaded bare array) because we need its
        real transform/CRS to reproject FROM.

        Call set_target_grid_from_boundary(...) (or _set_target_grid)
        before this.
        """
        aligned: Dict[str, np.ndarray] = {}
        for name, path in raster_paths.items():
            if path is None:
                continue
            path = Path(path)
            if not path.exists():
                logger.warning(f"Skipping {name}: file not found at {path}")
                continue
            try:
                aligned[name] = self._reproject_one(path, name)
                logger.info(f"Aligned {name}: shape={aligned[name].shape}")
            except Exception as e:
                logger.error(f"Failed to align {name} ({path}): {e}", exc_info=True)

        self.aligned_rasters = aligned
        return aligned

    def align_rasters_to_boundary(
        self,
        raster_paths: Dict[str, RasterSource],
        boundary_gdf: gpd.GeoDataFrame,
        buffer: float = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper: derive the target grid from `boundary_gdf`,
        then reproject every raster in `raster_paths` onto it.

        NOTE the change in contract from the old version: `raster_paths`
        must be name -> file path, not name -> numpy array. If all you
        have are arrays, you've already lost the georeferencing you need
        to align them correctly — load from the original files instead
        (e.g. straight from config.TOPOGRAPHIC / HYDROLOGICAL / CLIMATIC).
        """
        logger.info("Aligning rasters to city boundary...")
        self.set_target_grid_from_boundary(boundary_gdf, buffer=buffer)
        return self.align_raster_files(raster_paths)

    # ------------------------------------------------------------------
    # Vectors
    # ------------------------------------------------------------------
    def align_vectors(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Reproject vector data to target CRS."""
        if gdf.crs is None:
            logger.warning("Vector has no CRS, skipping reprojection")
            return gdf
        if str(gdf.crs) != str(self.target_crs):
            gdf = gdf.to_crs(self.target_crs)
        return gdf

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def save_aligned_rasters(self, aligned_rasters: Dict[str, np.ndarray], output_dir: Path):
        """Save aligned rasters to disk, using the ACTUAL shared target transform/CRS."""
        if self.target_transform is None:
            raise RuntimeError("No target grid set — nothing consistent to write.")

        output_path = Path(output_dir) / "aligned"
        output_path.mkdir(parents=True, exist_ok=True)

        for name, raster in aligned_rasters.items():
            if raster is None or not isinstance(raster, np.ndarray):
                continue

            out_path = output_path / f"{name}_aligned.tif"
            meta = {
                "driver": "GTiff",
                "height": raster.shape[0],
                "width": raster.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": self.target_crs,
                "transform": self.target_transform,
                "compress": "lzw",
                "nodata": -9999.0,
            }
            try:
                data_to_write = np.where(np.isnan(raster), -9999.0, raster).astype("float32")
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(data_to_write, 1)
                logger.info(f"Saved aligned raster: {out_path}")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # SAR registration (unchanged logic, now reuses the shared grid helpers)
    # ------------------------------------------------------------------
    def register_sar_to_dem(
        self, sar_flood_map: np.ndarray, sar_meta: Dict[str, Any], dem_meta: Dict[str, Any]
    ) -> np.ndarray:
        """Register a SAR-derived array to the DEM's coordinate system."""
        dst_data = np.zeros((dem_meta["height"], dem_meta["width"]), dtype=np.float32)

        reproject(
            source=sar_flood_map,
            destination=dst_data,
            src_transform=sar_meta["transform"],
            src_crs=sar_meta["crs"],
            dst_transform=dem_meta["transform"],
            dst_crs=dem_meta["crs"],
            resampling=Resampling.nearest,  # binary/categorical data
        )
        return dst_data