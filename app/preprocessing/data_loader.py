# backend/app/preprocessing/data_loader.py
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio import features
from shapely.geometry import box
import logging
import json
from typing import Dict, Optional, Tuple, Any

from .config import (
    TOPOGRAPHIC, HYDROLOGICAL, CLIMATIC, SOCIO, VECTOR_DATA,
    SAR_REGISTERED, TARGET_CRS, OUTPUT_DIR
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data Loader for Phase 2 preprocessing
    Loads all required data sources and provides metadata
    """
    
    def __init__(self):
        self.data = {}
        self.bounds = None
        self.crs = None
        
    def load_all(self) -> Dict:
        """Load all data sources"""
        logger.info("Loading all data sources...")
        
        # Load topographic data
        self.data['topographic'] = self._load_topographic()
        
        # Load hydrological data
        self.data['hydrological'] = self._load_hydrological()
        
        # Load climatic data
        self.data['climatic'] = self._load_climatic()
        
        # Load vector data (boundaries, etc.)
        self.data['vector'] = self._load_vector_data()
        
        # Load SAR data
        self.data['sar'] = self._load_sar_data()
        
        # Load socio-demographic data
        self.data['socio'] = self._load_socio_data()
        
        # Determine common bounds and CRS
        self._determine_common_bounds()
        
        logger.info(f"Data bounds: {self.bounds}")
        logger.info(f"Data CRS: {self.crs}")
        
        return self.data
    
    def _load_topographic(self) -> Dict:
        """Load topographic rasters"""
        rasters = {}
        for name, path in TOPOGRAPHIC.items():
            if path.exists():
                try:
                    with rasterio.open(path) as src:
                        rasters[name] = src.read(1)
                        if self.crs is None:
                            self.crs = src.crs.to_string() if src.crs else None
                    logger.info(f"Loaded {name}: {rasters[name].shape}")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
            else:
                logger.warning(f"File not found: {path}")
        return rasters
    
    def _load_hydrological(self) -> Dict:
        """Load hydrological rasters and vector data"""
        data = {}
        
        # Load rasters
        raster_keys = ["distance_to_river", "drainage_density"]
        for name in raster_keys:
            path = HYDROLOGICAL.get(name)
            if path and path.exists():
                try:
                    with rasterio.open(path) as src:
                        data[name] = src.read(1)
                    logger.info(f"Loaded {name}: {data[name].shape}")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
            else:
                logger.warning(f"File not found: {path}")
        
        # Load vector data
        vector_keys = ["river_network", "watersheds", "phl_rivers"]
        for name in vector_keys:
            path = HYDROLOGICAL.get(name)
            if path and path.exists():
                try:
                    data[name] = gpd.read_file(path)
                    if data[name].crs and data[name].crs != TARGET_CRS:
                        data[name] = data[name].to_crs(TARGET_CRS)
                    logger.info(f"Loaded {name}: {len(data[name])} features")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
            else:
                logger.warning(f"File not found: {path}")
        
        return data
    
    def _load_climatic(self) -> Dict:
        """Load climatic rasters and time series data"""
        data = {}
        
        # Load rasters
        raster_keys = ["chirps", "gpm"]
        for name in raster_keys:
            path = CLIMATIC.get(name)
            if path and path.exists():
                try:
                    with rasterio.open(path) as src:
                        data[name] = src.read(1)
                    logger.info(f"Loaded {name}: {data[name].shape}")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
            else:
                logger.warning(f"File not found: {path}")
        
        # Load daily time series
        ts_keys = ["chirps_daily", "gpm_daily"]
        for name in ts_keys:
            path = CLIMATIC.get(name)
            if path and path.exists():
                try:
                    data[name] = pd.read_csv(path)
                    logger.info(f"Loaded {name}: {len(data[name])} records")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
            else:
                logger.warning(f"File not found: {path}")
        
        return data
    
    def _load_vector_data(self) -> Dict:
        """Load vector data (city boundary, etc.)"""
        data = {}
        
        # Load city boundary (prefer shapefile over geojson)
        boundary_shp_path = VECTOR_DATA.get("city_boundary")
        boundary_geojson_path = VECTOR_DATA.get("city_boundary_geojson")
        
        # Try shapefile first (since it's already in UTM)
        if boundary_shp_path and boundary_shp_path.exists():
            try:
                gdf = gpd.read_file(boundary_shp_path)
                logger.info(f"Loaded city boundary from shapefile: {len(gdf)} features")
                
                # Ensure CRS is correct
                if gdf.crs is None:
                    logger.warning("Boundary has no CRS, assuming target CRS")
                    gdf = gdf.set_crs(TARGET_CRS)
                elif gdf.crs != TARGET_CRS:
                    logger.info(f"Reprojecting boundary from {gdf.crs} to {TARGET_CRS}")
                    gdf = gdf.to_crs(TARGET_CRS)
                
                data['city_boundary'] = gdf
                
                # Store bounds
                bounds = gdf.total_bounds
                logger.info(f"City boundary bounds: {bounds}")
                
                # Store as separate properties for easy access
                data['city_bounds'] = bounds
                data['city_bounds_dict'] = {
                    'minx': bounds[0],
                    'miny': bounds[1],
                    'maxx': bounds[2],
                    'maxy': bounds[3]
                }
                data['city_area'] = gdf.geometry.area.sum()
                logger.info(f"City area: {data['city_area'] / 1e6:.2f} km²")
                
            except Exception as e:
                logger.error(f"Error loading city boundary shapefile: {e}")
                
        # Fallback to GeoJSON if shapefile fails
        if 'city_boundary' not in data and boundary_geojson_path and boundary_geojson_path.exists():
            try:
                gdf = gpd.read_file(boundary_geojson_path)
                logger.info(f"Loaded city boundary from GeoJSON: {len(gdf)} features")
                
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")  # GeoJSON typically uses WGS84
                
                if gdf.crs != TARGET_CRS:
                    gdf = gdf.to_crs(TARGET_CRS)
                
                data['city_boundary'] = gdf
                data['city_bounds'] = gdf.total_bounds
                
            except Exception as e:
                logger.error(f"Error loading city boundary GeoJSON: {e}")
        
        # If no boundary loaded, create one from DEM extent as fallback
        if 'city_boundary' not in data:
            logger.warning("No city boundary found! Using DEM extent as fallback.")
            dem = self.data.get('topographic', {}).get('dem')
            if dem is not None:
                # Create a bounding box from DEM extent
                # This will use the DEM bounds
                from shapely.geometry import box
                bounds = (0, 0, dem.shape[1], dem.shape[0])  # Use pixel coordinates
                # Actually, we should use the DEM's geographic bounds
                # This is a placeholder - you'd need to get the actual bounds
                # from the DEM's metadata
                
        return data
    
    def _load_sar_data(self) -> Dict:
        """Load preprocessed SAR data metadata"""
        data = {}
        
        for event, event_data in SAR_REGISTERED.items():
            data[event] = {}
            for key, path in event_data.items():
                if path.exists():
                    try:
                        with rasterio.open(path) as src:
                            data[event][key] = {
                                'path': str(path),
                                'shape': src.shape,
                                'crs': src.crs.to_string() if src.crs else None,
                                'bounds': src.bounds
                            }
                        logger.info(f"Loaded SAR metadata: {event} - {key}")
                    except Exception as e:
                        logger.warning(f"Error reading SAR {event} {key}: {e}")
                        data[event][key] = {'path': str(path), 'exists': False}
                else:
                    logger.warning(f"SAR file not found: {path}")
                    data[event][key] = {'path': str(path), 'exists': False}
        
        return data
    
    def _load_socio_data(self) -> Dict:
        """Load socio-demographic data"""
        data = {}
        
        census_path = SOCIO.get("census")
        if census_path and census_path.exists():
            try:
                data['census'] = pd.read_excel(census_path)
                logger.info(f"Loaded census data: {len(data['census'])} records")
            except Exception as e:
                logger.error(f"Error loading census data: {e}")
        else:
            logger.warning(f"Census file not found: {census_path}")
        
        return data
    
    def _determine_common_bounds(self):
        """Determine common bounds from all data sources"""
        # Try to use city boundary first
        vector_data = self.data.get('vector', {})
        if 'city_bounds' in vector_data:
            bounds = vector_data['city_bounds']
            self.bounds = {
                'left': bounds[0],
                'bottom': bounds[1],
                'right': bounds[2],
                'top': bounds[3]
            }
            logger.info(f"Using city boundary bounds: {self.bounds}")
            return
        
        # Fallback to DEM bounds
        dem = self.data.get('topographic', {}).get('dem')
        if dem is not None:
            # Get DEM bounds from raster metadata
            # This requires the DEM file to be opened
            dem_path = TOPOGRAPHIC.get('dem')
            if dem_path and dem_path.exists():
                try:
                    with rasterio.open(dem_path) as src:
                        bounds = src.bounds
                        self.bounds = {
                            'left': bounds.left,
                            'bottom': bounds.bottom,
                            'right': bounds.right,
                            'top': bounds.top
                        }
                        self.crs = src.crs.to_string() if src.crs else None
                        logger.info(f"Using DEM bounds: {self.bounds}")
                        return
                except Exception as e:
                    logger.error(f"Error reading DEM bounds: {e}")
        
        # Create default bounds if nothing else works
        self.bounds = {
            'left': 368070.28125,
            'bottom': 753615.1875,
            'right': 449945.28125,
            'top': 832340.1875
        }
        self.crs = TARGET_CRS
        logger.warning(f"Using default bounds: {self.bounds}")
    
    def get_city_boundary(self) -> Optional[gpd.GeoDataFrame]:
        """Get the city boundary GeoDataFrame"""
        return self.data.get('vector', {}).get('city_boundary')
    
    def get_city_bounds(self) -> Optional[Dict]:
        """Get the city bounds"""
        return self.data.get('vector', {}).get('city_bounds_dict')
    
    def get_dem(self) -> Optional[np.ndarray]:
        """Get DEM array"""
        return self.data.get('topographic', {}).get('dem')


def main():
    """Test data loader"""
    loader = DataLoader()
    data = loader.load_all()
    
    print("\n=== Data Loaded ===")
    print(f"Topographic: {list(data['topographic'].keys())}")
    print(f"Hydrological: {list(data['hydrological'].keys())}")
    print(f"Climatic: {list(data['climatic'].keys())}")
    print(f"Vector: {list(data['vector'].keys())}")
    print(f"SAR: {list(data['sar'].keys())}")
    print(f"Socio: {list(data['socio'].keys())}")
    
    # Check if city boundary was loaded
    boundary = loader.get_city_boundary()
    if boundary is not None:
        print(f"\nCity boundary loaded: {len(boundary)} features")
        print(f"CRS: {boundary.crs}")
        print(f"Bounds: {boundary.total_bounds}")
    else:
        print("\nWARNING: City boundary NOT loaded!")


if __name__ == "__main__":
    main()