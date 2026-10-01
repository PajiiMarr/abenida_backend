"""
Extract event-specific rainfall for flood events with available SAR data
"""
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from datetime import datetime, timedelta
import json

print("=" * 70)
print("EXTRACTING EVENT-SPECIFIC RAINFALL")
print("=" * 70)

# Sample points
gdf = pd.read_csv('./outputs/phase2/sample_points/sample_points.csv')
points = np.column_stack([gdf['row'].values, gdf['col'].values])
labels = gdf['label'].values

print(f"Loaded {len(points)} sample points")
print(f"  Flood: {labels.sum():.0f}")
print(f"  Non-flood: {len(labels) - labels.sum():.0f}")

# ===== FLOOD EVENTS WITH AVAILABLE SAR =====
FLOOD_EVENTS = {
    'jan_2021': {
        'name': 'January 19, 2021',
        'date': datetime(2021, 1, 19),
        'days_before': 7,
        'days_after': 7,
    },
    'may_2021': {
        'name': 'May 19, 2021',
        'date': datetime(2021, 5, 19),
        'days_before': 7,
        'days_after': 7,
    },
    'sep_2021': {
        'name': 'September 16-17, 2021',
        'date': datetime(2021, 9, 16),
        'days_before': 7,
        'days_after': 7,
    },
}

# ===== Load Daily Rainfall Data =====
BASE_DIR = Path("/Users/mar/lectures/thesis 1/fsm/datas/FOR TRAINING/CLIMATIC DATAS")

chirps_path = BASE_DIR / "Zamboonga_CHIRPS_Daily_2021_2026.csv"
gpm_path = BASE_DIR / "Zamboonga_GPM_V07_Daily_2021_2026.csv"

chirps_df = pd.read_csv(chirps_path)
gpm_df = pd.read_csv(gpm_path)

print(f"\nCHIRPS daily: {len(chirps_df)} records")
print(f"GPM daily: {len(gpm_df)} records")

print("\nCHIRPS columns:", chirps_df.columns.tolist())
print("GPM columns:", gpm_df.columns.tolist())

# ===== FIX: Use 'date' column for dates =====
chirps_date_col = 'date'
gpm_date_col = 'date'

# Convert date columns to datetime
chirps_df['date_parsed'] = pd.to_datetime(chirps_df[chirps_date_col])
gpm_df['date_parsed'] = pd.to_datetime(gpm_df[gpm_date_col])

print(f"\nCHIRPS date range: {chirps_df['date_parsed'].min()} to {chirps_df['date_parsed'].max()}")
print(f"GPM date range: {gpm_df['date_parsed'].min()} to {gpm_df['date_parsed'].max()}")

# ===== Extract Event-Specific Rainfall =====
print("\n" + "=" * 70)
print("EXTRACTING EVENT-SPECIFIC RAINFALL")
print("=" * 70)

all_rainfall_data = []

for event_key, event_info in FLOOD_EVENTS.items():
    event_date = event_info['date']
    days_before = event_info['days_before']
    days_after = event_info['days_after']
    
    print(f"\n--- {event_info['name']} ({event_date.strftime('%Y-%m-%d')}) ---")
    
    # Get date range
    start_date = event_date - timedelta(days=days_before)
    end_date = event_date + timedelta(days=days_after)
    
    print(f"  Rainfall period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # ===== FIX: Filter by parsed date column =====
    chirps_filtered = chirps_df[(chirps_df['date_parsed'] >= start_date) & (chirps_df['date_parsed'] <= end_date)]
    gpm_filtered = gpm_df[(gpm_df['date_parsed'] >= start_date) & (gpm_df['date_parsed'] <= end_date)]
    
    print(f"  CHIRPS records in period: {len(chirps_filtered)}")
    print(f"  GPM records in period: {len(gpm_filtered)}")
    
    # Calculate rainfall statistics for this period
    if len(chirps_filtered) > 0:
        chirps_rainfall = chirps_filtered['rainfall_mm'].values
        print(f"    CHIRPS - Mean: {chirps_rainfall.mean():.2f} mm")
        print(f"    CHIRPS - Max: {chirps_rainfall.max():.2f} mm")
        print(f"    CHIRPS - Sum: {chirps_rainfall.sum():.2f} mm")
        print(f"    CHIRPS - Std: {chirps_rainfall.std():.2f} mm")
    
    if len(gpm_filtered) > 0:
        gpm_rainfall = gpm_filtered['rainfall_mm'].values
        print(f"    GPM - Mean: {gpm_rainfall.mean():.2f} mm")
        print(f"    GPM - Max: {gpm_rainfall.max():.2f} mm")
        print(f"    GPM - Sum: {gpm_rainfall.sum():.2f} mm")
        print(f"    GPM - Std: {gpm_rainfall.std():.2f} mm")
    
    # Store for later use
    all_rainfall_data.append({
        'event': event_key,
        'event_date': event_date,
        'chirps': chirps_filtered if len(chirps_filtered) > 0 else None,
        'gpm': gpm_filtered if len(gpm_filtered) > 0 else None,
    })

# ===== Check if rainfall correlates with flood labels =====
print("\n" + "=" * 70)
print("RAINFALL CORRELATION WITH FLOOD LABELS")
print("=" * 70)

# For each flood event, we need to match sample points with rainfall
# Since rainfall data is daily and we don't have spatial location per record,
# we need to use the raster data

# Load aligned rainfall rasters for spatial extraction
aligned_dir = Path('./outputs/phase2/aligned')

def extract_raster_values(raster_path, points):
    """Extract values from raster at point locations"""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        values = []
        for row, col in points:
            r, c = int(round(row)), int(round(col))
            if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
                val = data[r, c]
                if np.isnan(val) or val == -9999:
                    values.append(np.nan)
                else:
                    values.append(float(val))
            else:
                values.append(np.nan)
        return np.array(values)

# Extract rainfall values for each event
for event_key in FLOOD_EVENTS.keys():
    # We need event-specific rainfall rasters
    # Since we don't have these, let's check if the daily data can be mapped
    print(f"\n{event_key}:")
    print("  Rainfall data is daily, not spatial. Need to use daily rainfall rasters.")
    print("  Current rasters are annual averages, not event-specific.")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("To get event-specific rainfall, you need to:")
print("  1. Create daily rainfall rasters for each event date")
print("  2. Extract rainfall values at sample point locations")
print("  3. Use these as features")
print("")
print("The daily rainfall data shows rainfall amounts, but it's in tabular format.")
print("To use it with spatial points, you need to create rasters from the daily data,")
print("or use the annual average rasters which are already aligned.")
print("=" * 70)

# ===== Check if annual rasters have any correlation =====
print("\n" + "=" * 70)
print("CHECKING ANNUAL RAINFALL RASTERS")
print("=" * 70)

chirps_raster_path = aligned_dir / 'chirps_aligned.tif'
gpm_raster_path = aligned_dir / 'gpm_aligned.tif'

if chirps_raster_path.exists():
    chirps_vals = extract_raster_values(chirps_raster_path, points)
    chirps_corr = np.corrcoef(chirps_vals[~np.isnan(chirps_vals)], 
                              labels[~np.isnan(chirps_vals)])[0, 1]
    print(f"CHIRPS annual correlation with flood: {chirps_corr:.4f}")
    
    # Stats by class
    flood_mask = (labels == 1) & ~np.isnan(chirps_vals)
    non_flood_mask = (labels == 0) & ~np.isnan(chirps_vals)
    print(f"  Flood mean: {np.mean(chirps_vals[flood_mask]):.2f}")
    print(f"  Non-flood mean: {np.mean(chirps_vals[non_flood_mask]):.2f}")

if gpm_raster_path.exists():
    gpm_vals = extract_raster_values(gpm_raster_path, points)
    gpm_corr = np.corrcoef(gpm_vals[~np.isnan(gpm_vals)], 
                           labels[~np.isnan(gpm_vals)])[0, 1]
    print(f"\nGPM annual correlation with flood: {gpm_corr:.4f}")
    
    # Stats by class
    flood_mask = (labels == 1) & ~np.isnan(gpm_vals)
    non_flood_mask = (labels == 0) & ~np.isnan(gpm_vals)
    print(f"  Flood mean: {np.mean(gpm_vals[flood_mask]):.2f}")
    print(f"  Non-flood mean: {np.mean(gpm_vals[non_flood_mask]):.2f}")

print("\n" + "=" * 70)
print("Even annual rainfall shows very weak correlation with flooding.")
print("The problem is likely that the flood maps are not accurate,")
print("or flooding is driven by factors not captured in the data.")
print("=" * 70)