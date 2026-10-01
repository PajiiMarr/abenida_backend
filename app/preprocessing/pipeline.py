import logging
import numpy as np
from pathlib import Path
import argparse
import json
import sys
import rasterio
import pandas as pd

from .config import OUTPUT_DIR, TARGET_CRS, SAMPLE_PARAMS, GWPCA_PARAMS, FEATURES_TO_EXTRACT, SAR_REGISTERED as SAR
from .data_loader import DataLoader
from .spatial_alignment import SpatialAligner
from .sar_processor import SARProcessor
from .feature_extraction import FeatureExtractor
from .standardization import Standardizer
from .gwpca import GWPCAProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Complete Phase 2: Preprocessing and Feature Engineering
    
    Stages:
    1. Data Loading - Load all raw data sources
    2. Spatial Alignment - Reproject and clip to common extent
    3. SAR Processing - Load preprocessed SAR and generate sample points
    4. Feature Extraction - Extract feature values at sample points
    5. Save FULL DATASET - Output as CSV (NO SPLIT YET)
    6. OPTIONAL: Standardization and GWPCA (only for model-ready data)
    """
    
    def __init__(self):
        self.data = None
        self.aligned_rasters = None
        self.sample_points = None
        self.sample_labels = None
        self.X_all = None          # All features
        self.y_all = None          # All labels
        self.coords_all = None     # All coordinates
        self.feature_names = []
        self.scaler = None
        self.gwpca = None
        self.X_pca = None
        
    def run(self, use_gwpca: bool = False, use_standardization: bool = False, 
            save_split: bool = False, test_size: float = 0.2):
        """
        Run the complete preprocessing pipeline
        
        Args:
            use_gwpca: Apply GWPCA (default: False - save original features)
            use_standardization: Apply standardization (default: False)
            save_split: Save train/test split as NPY (default: False - save full dataset as CSV)
            test_size: Test set proportion (only used if save_split=True)
        """
        
        logger.info("=" * 60)
        logger.info("STARTING PHASE 2: PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info(f"Save split: {save_split}")
        logger.info(f"Use GWPCA: {use_gwpca}")
        logger.info(f"Use standardization: {use_standardization}")
        
        # Stage 1: Data Loading
        self._stage1_load_data()
        
        # Stage 2: Spatial Alignment
        self._stage2_align_data()
        
        # Stage 3: SAR Processing (NO SPLIT - just generate all points)
        self._stage3_process_sar()
        
        # Stage 4: Feature Extraction (EXTRACT FROM ALL POINTS)
        self._stage4_extract_features()
        
        # Stage 5: Save FULL DATASET as CSV
        self._stage5_save_full_dataset()
        
        # Stage 6: OPTIONAL - Standardization (for model-ready data)
        if use_standardization:
            self._stage6_standardize()
        
        # Stage 7: OPTIONAL - GWPCA (for model-ready data)
        if use_gwpca:
            self._stage7_gwpca()
        
        # Stage 8: OPTIONAL - Save train/test split (if requested)
        if save_split:
            self._stage8_save_split(test_size)
        
        logger.info("=" * 60)
        logger.info("PHASE 2 COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return self
    
    def _stage1_load_data(self):
        """Stage 1: Load all data sources"""
        logger.info("\n--- STAGE 1: Loading Data ---")
        loader = DataLoader()
        self.data = loader.load_all()
        logger.info("Data loading complete")
        
    def _stage2_align_data(self):
        """Stage 2: Spatial alignment of all rasters"""
        logger.info("\n--- STAGE 2: Spatial Alignment ---")
        from .config import TOPOGRAPHIC, HYDROLOGICAL_RASTERS, CLIMATIC_RASTERS, TARGET_RESOLUTION

        # IMPORTANT: use the config FILE PATHS, not self.data (DataLoader already
        # converted those to bare numpy arrays with no transform/CRS attached —
        # you can't reproject an array that's lost its own georeferencing).
        raster_paths = {}
        raster_paths.update(TOPOGRAPHIC)
        raster_paths.update(HYDROLOGICAL_RASTERS)
        raster_paths.update(CLIMATIC_RASTERS)

        boundary_gdf = self.data.get('vector', {}).get('city_boundary')

        aligner = SpatialAligner(TARGET_CRS, resolution=TARGET_RESOLUTION)

        if boundary_gdf is not None and not boundary_gdf.empty:
            logger.info("Aligning rasters to city boundary...")
            self.aligned_rasters = aligner.align_rasters_to_boundary(
                raster_paths, boundary_gdf, buffer=100
            )
            logger.info(f"Aligned to city boundary: {len(self.aligned_rasters)} rasters")
        else:
            raise ValueError("No city boundary found — cannot define a target grid.")

        aligner.save_aligned_rasters(self.aligned_rasters, OUTPUT_DIR)
        self.aligner = aligner  # keep target_transform/shape around for later stages

        logger.info("Spatial alignment complete")

        
    
    def _stage3_process_sar(self):
        """Stage 3: Load preprocessed SAR and generate ALL sample points"""
        logger.info("\n--- STAGE 3: SAR Processing (Generating sample points) ---")
        
        processor = SARProcessor()
        
        sar_data = processor.load_preprocessed_sar()
        
        # Generate flood inventory from your flood maps
        flood_inventory = processor.generate_flood_inventory()
        
        meta = processor.get_sar_metadata()
        logger.info(f"Using SAR metadata: CRS={meta.get('crs')}")
        
        # ===== IMPROVED: Use stratified sampling =====
        # Options: 'stratified', 'cluster', 'boundary', 'random'
        self.sample_points, self.sample_labels = processor.create_sample_points(
            n_samples=4000,
            flood_ratio=0.5,
            sampling_method='stratified'  # Best for spatial coverage
        )
        
        # Store coordinates for all points
        self.coords_all = self.sample_points.copy()
        
        logger.info(f"Generated {len(self.sample_points)} total sample points")
        logger.info(f"  Flood points: {np.sum(self.sample_labels)}")
        logger.info(f"  Non-flood points: {len(self.sample_labels) - np.sum(self.sample_labels)}")
        
        # Save sample points
        gdf = processor.save_sample_points(self.sample_points, self.sample_labels, OUTPUT_DIR, meta)
        
        # Get SAR summary
        sar_summary = processor.get_sar_summary()
        sar_summary.to_csv(OUTPUT_DIR / "sar_summary.csv", index=False)
        
        logger.info("SAR processing complete")

    def _stage4_extract_features(self):
        """Stage 4: Extract features AND create engineered features"""
        logger.info("\n--- STAGE 4: Feature Extraction & Engineering ---")
        
        extractor = FeatureExtractor(self.aligned_rasters)
        
        # Extract features with engineering enabled
        X_all_features = extractor.extract_features_at_points(
            self.sample_points, 
            self.sample_labels,
            create_engineered=True
        )
        X_all_features = extractor.handle_missing_values(X_all_features, 'mean')
        
        # Store all features and labels
        self.X_all = X_all_features
        self.y_all = self.sample_labels
        self.feature_names = extractor.feature_names
        
        # Store counts
        self.n_raw_features = len(extractor.raw_feature_names)
        self.n_engineered_features = len(extractor.engineered_feature_names)
        
        # Log feature breakdown
        logger.info(f"Feature extraction complete: {X_all_features.shape}")
        logger.info(f"  Raw features: {self.n_raw_features}")
        logger.info(f"  Engineered features: {self.n_engineered_features}")
        logger.info(f"  Total features: {len(self.feature_names)}")
        
        # Save features with engineered features
        extractor.save_feature_data(
            X_all_features,
            self.sample_labels,
            OUTPUT_DIR,
            coords=self.sample_points,
            dataset_name="features_with_engineering"
        )
        
        # Compute statistics
        stats = extractor.compute_feature_statistics(X_all_features, self.sample_labels)
        stats.to_csv(OUTPUT_DIR / "feature_statistics_engineered.csv", index=False)

    def _stage5_save_full_dataset(self):
        """Stage 5: Save FULL DATASET with engineered features"""
        logger.info("\n--- STAGE 5: Saving Full Dataset ---")
        
        # Create DataFrame with all features
        df = pd.DataFrame(self.X_all, columns=self.feature_names)
        
        # Add coordinates and labels
        if self.coords_all is not None:
            df['row'] = self.coords_all[:, 0]
            df['col'] = self.coords_all[:, 1]
        
        df['flood_label'] = self.y_all.astype(int)
        df['sample_id'] = range(len(df))
        
        # Reorder columns
        cols = ['sample_id']
        if self.coords_all is not None:
            cols.extend(['row', 'col'])
        cols.extend(self.feature_names)
        cols.append('flood_label')
        
        df = df[cols]
        
        # Save as CSV
        output_path = OUTPUT_DIR / "full_dataset_engineered.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved full dataset to: {output_path}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Samples: {len(df)}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Raw features: {self.n_raw_features}")
        logger.info(f"  Engineered features: {self.n_engineered_features}")
        logger.info(f"  Flood samples: {df['flood_label'].sum()}")
        logger.info(f"  Non-flood samples: {len(df) - df['flood_label'].sum()}")
        
        # Also save as NPY
        npy_path = OUTPUT_DIR / "preprocessed"
        npy_path.mkdir(parents=True, exist_ok=True)
        
        np.save(npy_path / "X_all_engineered.npy", self.X_all)
        np.save(npy_path / "y_all_engineered.npy", self.y_all)
        if self.coords_all is not None:
            np.save(npy_path / "coords_all_engineered.npy", self.coords_all)
        
        with open(npy_path / "feature_names_engineered.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save summary
        summary = {
            'total_samples': len(self.X_all),
            'n_features': len(self.feature_names),
            'n_raw_features': self.n_raw_features,
            'n_engineered_features': self.n_engineered_features,
            'feature_names': self.feature_names,
            'flood_samples': int(np.sum(self.y_all)),
            'non_flood_samples': int(len(self.y_all) - np.sum(self.y_all)),
            'flood_percentage': float(np.mean(self.y_all) * 100),
        }
        
        with open(npy_path / "dataset_summary_engineered.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
    def _stage6_standardize(self):
        """Stage 6: OPTIONAL - Standardize features for model-ready data"""
        logger.info("\n--- STAGE 6: Standardization (Optional) ---")

        # Keep an unstandardized copy in memory so we don't destroy X_all
        X_unscaled = self.X_all.copy()

        self.scaler = Standardizer('minmax')
        X_scaled = self.scaler.fit_transform(X_unscaled, self.feature_names)

        self.scaler.save_scaler(OUTPUT_DIR)

        # -------- File 1: unscaled engineered dataset (re-save explicitly) --------
        df_raw = pd.DataFrame(X_unscaled, columns=self.feature_names)
        df_raw['flood_label'] = self.y_all.astype(int)
        if self.coords_all is not None:
            df_raw['row'] = self.coords_all[:, 0]
            df_raw['col'] = self.coords_all[:, 1]
        df_raw.to_csv(OUTPUT_DIR / "full_dataset_engineered.csv", index=False)
        logger.info(f"Saved unscaled engineered dataset: full_dataset_engineered.csv  {df_raw.shape}")

        # -------- File 2: standardized engineered dataset -------------------------
        df_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        df_scaled['flood_label'] = self.y_all.astype(int)
        if self.coords_all is not None:
            df_scaled['row'] = self.coords_all[:, 0]
            df_scaled['col'] = self.coords_all[:, 1]
        df_scaled.to_csv(OUTPUT_DIR / "full_dataset_engineered_standardized.csv", index=False)
        logger.info(f"Saved standardized engineered dataset: full_dataset_engineered_standardized.csv  {df_scaled.shape}")

        # Only overwrite X_all AFTER both files are written, in case later stages
        # (GWPCA / split) want to use the scaled version.
        self.X_all = X_scaled

        logger.info("Standardization complete")

        
    def _stage7_gwpca(self):
        """Stage 7: OPTIONAL - Apply GWPCA for dimensionality reduction"""
        logger.info("\n--- STAGE 7: GWPCA (Optional) ---")
        
        if self.coords_all is None:
            logger.warning("No coordinates found. GWPCA requires spatial coordinates.")
            return
        
        logger.info(f"GWPCA input: {self.X_all.shape[0]} samples, {self.X_all.shape[1]} features")
        logger.info(f"Feature names: {self.feature_names}")
        
        # Fit GWPCA on all data
        self.gwpca = GWPCAProcessor(
            self.coords_all, 
            self.X_all, 
            self.feature_names
        )
        self.X_pca, results = self.gwpca.fit()
        
        self.gwpca.save_results(OUTPUT_DIR)
        
        n_components = self.X_pca.shape[1]
        
        # Save PCA-transformed dataset
        df_pca = pd.DataFrame(self.X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        df_pca['flood_label'] = self.y_all.astype(int)
        if self.coords_all is not None:
            df_pca['row'] = self.coords_all[:, 0]
            df_pca['col'] = self.coords_all[:, 1]
        df_pca.to_csv(OUTPUT_DIR / "full_dataset_pca.csv", index=False)
        
        logger.info(f"GWPCA complete: {n_components} components")
        logger.info(f"Summary: {json.dumps(self.gwpca.get_summary(), indent=2)}")
    
    def _stage8_save_split(self, test_size=0.2):
        """Stage 8: OPTIONAL - Save train/test split (only if requested)"""
        logger.info("\n--- STAGE 8: Train/Test Split (Optional) ---")
        
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
            self.X_all,
            self.y_all,
            self.coords_all,
            test_size=test_size,
            random_state=42,
            stratify=self.y_all
        )
        
        # Save split
        output_path = OUTPUT_DIR / "preprocessed"
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / "X_train.npy", X_train)
        np.save(output_path / "X_test.npy", X_test)
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "y_test.npy", y_test)
        np.save(output_path / "coords_train.npy", coords_train)
        np.save(output_path / "coords_test.npy", coords_test)
        
        # Also save as CSV for inspection
        train_df = pd.DataFrame(X_train, columns=self.feature_names)
        train_df['flood_label'] = y_train.astype(int)
        train_df['row'] = coords_train[:, 0]
        train_df['col'] = coords_train[:, 1]
        train_df.to_csv(OUTPUT_DIR / "train_data.csv", index=False)
        
        test_df = pd.DataFrame(X_test, columns=self.feature_names)
        test_df['flood_label'] = y_test.astype(int)
        test_df['row'] = coords_test[:, 0]
        test_df['col'] = coords_test[:, 1]
        test_df.to_csv(OUTPUT_DIR / "test_data.csv", index=False)
        
        # Save split info
        split_info = {
            'total_samples': len(self.X_all),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_flood': int(np.sum(y_train)),
            'train_nonflood': int(len(y_train) - np.sum(y_train)),
            'test_flood': int(np.sum(y_test)),
            'test_nonflood': int(len(y_test) - np.sum(y_test)),
            'test_size': test_size,
        }
        with open(output_path / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Train/Test split saved to {output_path}")
        logger.info(f"Split info: {json.dumps(split_info, indent=2)}")


def main():
    """Run the preprocessing pipeline from command line"""
    parser = argparse.ArgumentParser(description='Run Phase 2 Preprocessing Pipeline')
    parser.add_argument('--gwpca', action='store_true', help='Apply GWPCA')
    parser.add_argument('--standardize', action='store_true', help='Apply standardization')
    parser.add_argument('--split', action='store_true', help='Save train/test split')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline()
    pipeline.run(
        use_gwpca=args.gwpca,
        use_standardization=args.standardize,
        save_split=args.split,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()