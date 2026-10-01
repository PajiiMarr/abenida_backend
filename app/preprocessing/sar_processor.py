# backend/app/preprocessing/sar_processor.py
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio import features
from shapely.geometry import Point
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List
import json
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class SARProcessor:
    """
    SAR Data Processor for flood inventory generation
    
    Improved sampling strategies:
    1. Spatial grid stratification
    2. Proximity-based sampling (flood boundaries)
    3. Topographic stratification
    4. Balanced sampling with spatial coverage
    """
    
    def __init__(self, sar_dir: Path = Path("./outputs/phase2/registered_sar")):
        self.sar_dir = sar_dir
        self.sar_data = {}
        self.flood_inventory = None
        self.sample_points = None
        self.sample_labels = None
        self.metadata = {}
        
    def load_preprocessed_sar(self) -> Dict:
        """Load preprocessed SAR data from the registered_sar directory"""
        logger.info("Loading preprocessed SAR data from your preprocess outputs...")
        
        events = ['may_2021']
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
                logger.info(f"Loaded ndfi for {event}: {event_data['ndfi'].shape}")
            else:
                event_data['ndfi'] = None
            
            # Load VH
            vh_path = self.sar_dir / f"{event}_vh_registered.tif"
            if vh_path.exists():
                with rasterio.open(vh_path) as src:
                    event_data['vh'] = src.read(1)
                logger.info(f"Loaded vh for {event}: {event_data['vh'].shape}")
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
        """
        Get pixels near flood boundaries (transition zones)
        
        Args:
            flood_map: Binary flood map (1=flood, 0=non-flood)
            buffer_size: Number of pixels around boundary to include
        
        Returns:
            Array of (row, col) coordinates near flood boundaries
        """
        # Calculate distance to flood boundary
        distance = distance_transform_edt(flood_map == 0)
        
        # Get boundary pixels (within buffer_size of flood edge)
        boundary_mask = (distance <= buffer_size) & (distance > 0)
        boundary_rows, boundary_cols = np.where(boundary_mask)
        
        return np.column_stack([boundary_rows, boundary_cols])
    
    def _get_stratified_samples(self, coords: np.ndarray, labels: np.ndarray, 
                                 n_samples: int, n_grid: int = 10) -> np.ndarray:
        """
        Get samples using spatial grid stratification
        
        Args:
            coords: Array of (row, col) coordinates
            labels: Array of labels
            n_samples: Number of samples to select
            n_grid: Number of grid cells per dimension (n_grid x n_grid)
        
        Returns:
            Indices of selected samples
        """
        if len(coords) <= n_samples:
            return np.arange(len(coords))
        
        # Get bounds
        row_min, row_max = coords[:, 0].min(), coords[:, 0].max()
        col_min, col_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Calculate grid cell size
        row_step = (row_max - row_min) / n_grid
        col_step = (col_max - col_min) / n_grid
        
        # Initialize grid
        grid_cells = {}
        
        # Assign each point to a grid cell
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
        
        # Sample from each grid cell
        selected_indices = []
        n_cells = len(grid_cells)
        samples_per_cell = max(1, n_samples // n_cells)
        
        for cell_key, cell_data in grid_cells.items():
            cell_indices = cell_data['indices']
            
            if len(cell_indices) <= samples_per_cell:
                selected_indices.extend(cell_indices)
            else:
                # Random sample from this cell
                chosen = np.random.choice(cell_indices, samples_per_cell, replace=False)
                selected_indices.extend(chosen.tolist())
        
        # If we have too many, randomly subsample
        if len(selected_indices) > n_samples:
            selected_indices = np.random.choice(selected_indices, n_samples, replace=False).tolist()
        
        return np.array(selected_indices)
    
    def _get_cluster_samples(self, coords: np.ndarray, labels: np.ndarray,
                              n_samples: int, n_clusters: int = 20) -> np.ndarray:
        """
        Get samples using K-means clustering for spatial coverage
        
        Args:
            coords: Array of (row, col) coordinates
            labels: Array of labels
            n_samples: Number of samples to select
            n_clusters: Number of clusters
        
        Returns:
            Indices of selected samples
        """
        if len(coords) <= n_samples:
            return np.arange(len(coords))
        
        # Cluster coordinates
        kmeans = KMeans(n_clusters=min(n_clusters, len(coords) // 10), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)
        
        # Sample from each cluster
        selected_indices = []
        samples_per_cluster = max(1, n_samples // len(np.unique(clusters)))
        
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) <= samples_per_cluster:
                selected_indices.extend(cluster_indices.tolist())
            else:
                # Sample from this cluster
                chosen = np.random.choice(cluster_indices, samples_per_cluster, replace=False)
                selected_indices.extend(chosen.tolist())
        
        # If we have too many, randomly subsample
        if len(selected_indices) > n_samples:
            selected_indices = np.random.choice(selected_indices, n_samples, replace=False).tolist()
        
        return np.array(selected_indices)
    
    def create_sample_points(self, flood_inventory: Dict = None, 
                             n_samples: int = 4000,
                             flood_ratio: float = 0.5,
                             sampling_method: str = 'stratified') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create balanced sample points using improved spatial sampling
        
        Args:
            flood_inventory: Dictionary from generate_flood_inventory()
            n_samples: Total number of samples to generate
            flood_ratio: Proportion of flood samples (0.5 = balanced)
            sampling_method: 'stratified', 'cluster', 'boundary', or 'random'
            
        Returns:
            sample_points: Array of (row, col) coordinates
            sample_labels: Array of labels (1=flood, 0=non-flood)
        """
        if flood_inventory is None:
            flood_inventory = self.flood_inventory
        
        if flood_inventory is None:
            raise ValueError("No flood inventory! Call generate_flood_inventory() first.")
        
        n_flood = int(n_samples * flood_ratio)
        n_non_flood = n_samples - n_flood
        
        all_flood_points = []
        all_non_flood_points = []
        
        for event, info in flood_inventory.items():
            flood_map = info['flood_map']
            
            # Get flood pixel coordinates
            flood_rows, flood_cols = np.where(flood_map > 0)
            flood_coords = np.column_stack([flood_rows, flood_cols])
            flood_labels = np.ones(len(flood_coords))
            
            # Get non-flood pixel coordinates
            non_flood_rows, non_flood_cols = np.where(flood_map == 0)
            non_flood_coords = np.column_stack([non_flood_rows, non_flood_cols])
            non_flood_labels = np.zeros(len(non_flood_coords))
            
            logger.info(f"{event}: {len(flood_coords)} flood, {len(non_flood_coords)} non-flood pixels")
            
            # Sample flood points using improved method
            if len(flood_coords) > 0:
                if sampling_method == 'stratified':
                    flood_idx = self._get_stratified_samples(flood_coords, flood_labels, n_flood)
                    flood_samples = flood_coords[flood_idx]
                elif sampling_method == 'cluster':
                    flood_idx = self._get_cluster_samples(flood_coords, flood_labels, n_flood)
                    flood_samples = flood_coords[flood_idx]
                elif sampling_method == 'boundary':
                    boundary_coords = self._get_flood_boundary_pixels(flood_map, buffer_size=10)
                    if len(boundary_coords) > 0:
                        # Prioritize boundary pixels
                        n_boundary = min(n_flood // 2, len(boundary_coords))
                        boundary_idx = np.random.choice(len(boundary_coords), n_boundary, replace=False)
                        boundary_samples = boundary_coords[boundary_idx]
                        
                        # Fill remaining from random flood pixels
                        remaining = n_flood - n_boundary
                        flood_coords_filtered = flood_coords[~np.isin(flood_coords, boundary_samples).all(axis=1)]
                        if len(flood_coords_filtered) > remaining:
                            random_idx = np.random.choice(len(flood_coords_filtered), remaining, replace=False)
                            random_samples = flood_coords_filtered[random_idx]
                            flood_samples = np.vstack([boundary_samples, random_samples])
                        else:
                            flood_samples = boundary_samples
                    else:
                        flood_idx = np.random.choice(len(flood_coords), min(n_flood, len(flood_coords)), replace=False)
                        flood_samples = flood_coords[flood_idx]
                else:  # random
                    flood_idx = np.random.choice(len(flood_coords), min(n_flood, len(flood_coords)), replace=False)
                    flood_samples = flood_coords[flood_idx]
                
                all_flood_points.append(flood_samples)
                logger.info(f"  Sampled {len(flood_samples)} flood points using {sampling_method}")
            
            # Sample non-flood points using improved method
            if len(non_flood_coords) > 0:
                if sampling_method == 'stratified':
                    non_flood_idx = self._get_stratified_samples(non_flood_coords, non_flood_labels, n_non_flood)
                    non_flood_samples = non_flood_coords[non_flood_idx]
                elif sampling_method == 'cluster':
                    non_flood_idx = self._get_cluster_samples(non_flood_coords, non_flood_labels, n_non_flood)
                    non_flood_samples = non_flood_coords[non_flood_idx]
                elif sampling_method == 'boundary':
                    # Sample non-flood near boundaries
                    boundary_coords = self._get_flood_boundary_pixels(flood_map, buffer_size=15)
                    if len(boundary_coords) > 0:
                        n_boundary = min(n_non_flood // 2, len(boundary_coords))
                        boundary_idx = np.random.choice(len(boundary_coords), n_boundary, replace=False)
                        boundary_samples = boundary_coords[boundary_idx]
                        
                        # Fill remaining from random non-flood
                        remaining = n_non_flood - n_boundary
                        non_flood_filtered = non_flood_coords[~np.isin(non_flood_coords, boundary_samples).all(axis=1)]
                        if len(non_flood_filtered) > remaining:
                            random_idx = np.random.choice(len(non_flood_filtered), remaining, replace=False)
                            random_samples = non_flood_filtered[random_idx]
                            non_flood_samples = np.vstack([boundary_samples, random_samples])
                        else:
                            non_flood_samples = boundary_samples
                    else:
                        non_flood_idx = np.random.choice(len(non_flood_coords), min(n_non_flood, len(non_flood_coords)), replace=False)
                        non_flood_samples = non_flood_coords[non_flood_idx]
                else:  # random
                    non_flood_idx = np.random.choice(len(non_flood_coords), min(n_non_flood, len(non_flood_coords)), replace=False)
                    non_flood_samples = non_flood_coords[non_flood_idx]
                
                all_non_flood_points.append(non_flood_samples)
                logger.info(f"  Sampled {len(non_flood_samples)} non-flood points using {sampling_method}")
        
        # Combine all points
        if all_flood_points:
            flood_points = np.vstack(all_flood_points)
        else:
            flood_points = np.array([]).reshape(0, 2)
        
        if all_non_flood_points:
            non_flood_points = np.vstack(all_non_flood_points)
        else:
            non_flood_points = np.array([]).reshape(0, 2)
        
        # Adjust if we don't have enough points
        if len(flood_points) < n_flood:
            logger.warning(f"Only {len(flood_points)} flood points available, adjusting")
            n_flood_actual = len(flood_points)
            n_non_flood_actual = n_non_flood + (n_flood - n_flood_actual)
        else:
            n_flood_actual = n_flood
        
        if len(non_flood_points) < n_non_flood:
            logger.warning(f"Only {len(non_flood_points)} non-flood points available, adjusting")
            n_non_flood_actual = len(non_flood_points)
        else:
            n_non_flood_actual = n_non_flood
        
        # Sample final sets
        if len(flood_points) > n_flood_actual:
            indices = np.random.choice(len(flood_points), n_flood_actual, replace=False)
            final_flood = flood_points[indices]
        else:
            final_flood = flood_points
        
        if len(non_flood_points) > n_non_flood_actual:
            indices = np.random.choice(len(non_flood_points), n_non_flood_actual, replace=False)
            final_non_flood = non_flood_points[indices]
        else:
            final_non_flood = non_flood_points
        
        # Combine and shuffle
        sample_points = np.vstack([final_flood, final_non_flood])
        sample_labels = np.concatenate([
            np.ones(len(final_flood)),
            np.zeros(len(final_non_flood))
        ])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(sample_points))
        sample_points = sample_points[shuffle_idx]
        sample_labels = sample_labels[shuffle_idx]
        
        self.sample_points = sample_points
        self.sample_labels = sample_labels
        
        logger.info(f"Created {len(sample_points)} sample points ({int(np.sum(sample_labels))} flood, {len(sample_labels) - int(np.sum(sample_labels))} non-flood)")
        logger.info(f"  Flood ratio: {np.mean(sample_labels)*100:.2f}%")
        logger.info(f"  Sampling method: {sampling_method}")
        
        return sample_points, sample_labels
    
    def train_test_split(self, sample_points: np.ndarray = None, 
                         sample_labels: np.ndarray = None,
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple:
        """Split samples into train and test sets"""
        if sample_points is None:
            sample_points = self.sample_points
        
        if sample_labels is None:
            sample_labels = self.sample_labels
        
        if sample_points is None or sample_labels is None:
            raise ValueError("No sample points! Call create_sample_points() first.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            sample_points,
            sample_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=sample_labels
        )
        
        logger.info(f"Train set: {len(X_train)} samples ({int(np.sum(y_train))} flood, {len(y_train) - int(np.sum(y_train))} non-flood)")
        logger.info(f"Test set: {len(X_test)} samples ({int(np.sum(y_test))} flood, {len(y_test) - int(np.sum(y_test))} non-flood)")
        
        return X_train, X_test, y_train, y_test
    
    def save_sample_points(self, sample_points: np.ndarray = None,
                           sample_labels: np.ndarray = None,
                           output_dir: Path = Path("./outputs/phase2"),
                           metadata: Dict = None) -> gpd.GeoDataFrame:
        """Save sample points as shapefile and CSV"""
        if sample_points is None:
            sample_points = self.sample_points
        
        if sample_labels is None:
            sample_labels = self.sample_labels
        
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
                except:
                    geom = Point(row, col)
            else:
                geom = Point(row, col)
            geometries.append(geom)
        
        gdf = gpd.GeoDataFrame({
            'sample_id': range(len(sample_points)),
            'row': sample_points[:, 0],
            'col': sample_points[:, 1],
            'flood_label': sample_labels.astype(int),
            'geometry': geometries
        }, crs=crs)
        
        shp_path = sample_dir / "sample_points.shp"
        gdf.to_file(shp_path)
        logger.info(f"Saved sample points to {sample_dir}")
        
        csv_path = sample_dir / "sample_points.csv"
        gdf.drop(columns='geometry').to_csv(csv_path, index=False)
        
        # Create a clean copy of metadata without non-serializable objects
        metadata_clean = {}
        if metadata:
            for key, value in metadata.items():
                if key == 'transform':
                    # Convert transform to string representation
                    if value is not None:
                        metadata_clean['transform'] = str(value)
                elif key == 'bounds':
                    # Convert bounds to string or tuple
                    if value is not None:
                        try:
                            metadata_clean['bounds'] = {
                                'left': float(value.left),
                                'bottom': float(value.bottom),
                                'right': float(value.right),
                                'top': float(value.top)
                            }
                        except:
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
        # Load SAR data
        sar_data = processor.load_preprocessed_sar()
        logger.info(f"Loaded SAR data for {len(sar_data)} events")
        
        # Generate flood inventory
        flood_inventory = processor.generate_flood_inventory()
        logger.info(f"Generated flood inventory with {len(flood_inventory)} events")
        
        # Test different sampling methods
        methods = ['stratified', 'cluster', 'boundary', 'random']
        
        for method in methods:
            logger.info(f"\n--- Testing {method} sampling ---")
            points, labels = processor.create_sample_points(
                n_samples=4000,
                sampling_method=method
            )
            logger.info(f"  {method}: {len(points)} points, {np.sum(labels)} flood, {len(labels) - np.sum(labels)} non-flood")
        
        # Save final sample points
        gdf = processor.save_sample_points(points, labels)
        logger.info("Sample points saved")
        
        # Get summary
        summary = processor.get_sar_summary()
        logger.info(f"\n{summary}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()