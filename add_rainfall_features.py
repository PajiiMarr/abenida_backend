# Save as add_rainfall_features.py
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import json

print("=" * 70)
print("ADDING RAINFALL FEATURES")
print("=" * 70)

# Load sample points
gdf = pd.read_csv('./outputs/phase2/sample_points/sample_points.csv')
points = np.column_stack([gdf['row'].values, gdf['col'].values])
labels = gdf['label'].values

# Load CHIRPS and GPM daily data
chirps_daily = pd.read_csv('/Users/mar/lectures/thesis 1/fsm/datas/FOR TRAINING/CLIMATIC DATAS/Zamboonga_CHIRPS_Daily_2021_2026.csv')
gpm_daily = pd.read_csv('/Users/mar/lectures/thesis 1/fsm/datas/FOR TRAINING/CLIMATIC DATAS/Zamboonga_GPM_V07_Daily_2021_2026.csv')

print(f"CHIRPS daily: {len(chirps_daily)} records")
print(f"GPM daily: {len(gpm_daily)} records")

# Load aligned rainfall rasters
aligned_dir = Path('./outputs/phase2/aligned')

# Extract rainfall values at sample points
with rasterio.open(aligned_dir / 'chirps_aligned.tif') as src:
    chirps_data = src.read(1)

with rasterio.open(aligned_dir / 'gpm_aligned.tif') as src:
    gpm_data = src.read(1)

# Extract values
chirps_vals = []
gpm_vals = []

for row, col in points:
    r, c = int(round(row)), int(round(col))
    if 0 <= r < chirps_data.shape[0] and 0 <= c < chirps_data.shape[1]:
        chirps_vals.append(chirps_data[r, c])
        gpm_vals.append(gpm_data[r, c])
    else:
        chirps_vals.append(np.nan)
        gpm_vals.append(np.nan)

chirps_vals = np.array(chirps_vals)
gpm_vals = np.array(gpm_vals)

print(f"\nCHIRPS values: mean={np.nanmean(chirps_vals):.2f}, std={np.nanstd(chirps_vals):.2f}")
print(f"GPM values: mean={np.nanmean(gpm_vals):.2f}, std={np.nanstd(gpm_vals):.2f}")

# Check correlations with flood
flood_mask = labels == 1
non_flood_mask = labels == 0

print(f"\nCHIRPS - Flood: mean={np.nanmean(chirps_vals[flood_mask]):.2f}")
print(f"CHIRPS - Non-flood: mean={np.nanmean(chirps_vals[non_flood_mask]):.2f}")
print(f"CHIRPS correlation: {np.corrcoef(chirps_vals, labels)[0, 1]:.4f}")

print(f"\nGPM - Flood: mean={np.nanmean(gpm_vals[flood_mask]):.2f}")
print(f"GPM - Non-flood: mean={np.nanmean(gpm_vals[non_flood_mask]):.2f}")
print(f"GPM correlation: {np.corrcoef(gpm_vals, labels)[0, 1]:.4f}")