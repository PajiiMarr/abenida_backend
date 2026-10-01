#!/usr/bin/env python
"""
Diagnostic script to check coordinate system compatibility between SAR and DEM data
"""
import rasterio
import numpy as np
from pathlib import Path

print("=" * 70)
print("COORDINATE SYSTEM DIAGNOSTIC")
print("=" * 70)

# ===== CORRECT PATHS =====
BASE_DIR = Path("/Users/mar/lectures/thesis 1/fsm/datas/FOR TRAINING")

# SAR paths
SAR_PATHS = {
    'sep_2021': {
        'flood_map': BASE_DIR / "SAR DATAS/September 16-17, 2021/outputs/flood_map_20210914_20210926.tif",
        'ndfi': BASE_DIR / "SAR DATAS/September 16-17, 2021/outputs/ndfi_20210914_20210926.tif",
        'vh': BASE_DIR / "SAR DATAS/September 16-17, 2021/outputs/vh_sigma0_db_co_event_20210926.tif",
    },
    'jan_2021': {
        'flood_map': BASE_DIR / "SAR DATAS/January 19, 2021/outputs/flood_map_20210117_20210129.tif",
        'ndfi': BASE_DIR / "SAR DATAS/January 19, 2021/outputs/ndfi_20210117_20210129.tif",
        'vh': BASE_DIR / "SAR DATAS/January 19, 2021/outputs/vh_sigma0_db_co_event_20210129.tif",
    },
    'may_2021': {
        'flood_map': BASE_DIR / "SAR DATAS/May 19, 2021/outputs/flood_map_20210517_20210529.tif",
        'ndfi': BASE_DIR / "SAR DATAS/May 19, 2021/outputs/ndfi_20210517_20210529.tif",
        'vh': BASE_DIR / "SAR DATAS/May 19, 2021/outputs/vh_sigma0_db_co_event_20210529.tif",
    },
}

# TOPOGRAPHIC paths
DEM_PATH = BASE_DIR / "TOPOGRAPHIC DATAS/Zamboanga_DEM.tif"
SLOPE_PATH = BASE_DIR / "TOPOGRAPHIC DATAS/Zamboanga_Slope.tif"
TWI_PATH = BASE_DIR / "TOPOGRAPHIC DATAS/Zamboanga_TWI.tif"
FLOW_ACC_PATH = BASE_DIR / "TOPOGRAPHIC DATAS/Zamboanga_FlowAccumulation.tif"

# ===== CHECK FILES EXIST =====
print("📁 Checking if files exist...")
for name, path in [
    ("DEM", DEM_PATH),
    ("SAR Sep 2021", SAR_PATHS['sep_2021']['flood_map']),
    ("SAR Jan 2021", SAR_PATHS['jan_2021']['flood_map']),
    ("SAR May 2021", SAR_PATHS['may_2021']['flood_map']),
]:
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {name}: {path}")
    if not exists:
        print(f"       File not found at: {path}")

print("\n" + "=" * 70)

# ===== LOAD AND ANALYZE SAR =====
sar_path = SAR_PATHS['sep_2021']['flood_map']
print(f"\n📁 SAR Flood Map: {sar_path}")
try:
    with rasterio.open(sar_path) as src:
        sar_shape = src.shape
        sar_transform = src.transform
        sar_crs = src.crs
        sar_bounds = src.bounds
        sar_meta = src.meta
        sar_data = src.read(1)
        
        print(f"  Shape: {sar_shape}")
        print(f"  Transform: {sar_transform}")
        print(f"  CRS: {sar_crs}")
        print(f"  Bounds: {sar_bounds}")
        print(f"  Data type: {sar_data.dtype}")
        print(f"  Unique values: {np.unique(sar_data)}")
        print(f"  Flood pixels: {(sar_data > 0).sum():,}")
        print(f"  Percentage flooded: {100 * (sar_data > 0).sum() / sar_data.size:.2f}%")
except Exception as e:
    print(f"  ❌ Failed to load: {e}")
    sar_shape = None
    sar_transform = None
    sar_crs = None
    sar_bounds = None

# ===== LOAD AND ANALYZE DEM =====
print(f"\n📁 DEM: {DEM_PATH}")
try:
    with rasterio.open(DEM_PATH) as src:
        dem_shape = src.shape
        dem_transform = src.transform
        dem_crs = src.crs
        dem_bounds = src.bounds
        dem_meta = src.meta
        dem_data = src.read(1)
        
        print(f"  Shape: {dem_shape}")
        print(f"  Transform: {dem_transform}")
        print(f"  CRS: {dem_crs}")
        print(f"  Bounds: {dem_bounds}")
        print(f"  Data type: {dem_data.dtype}")
        print(f"  Min value: {np.nanmin(dem_data)}")
        print(f"  Max value: {np.nanmax(dem_data)}")
        print(f"  Mean value: {np.nanmean(dem_data):.2f}")
        print(f"  Std value: {np.nanstd(dem_data):.2f}")
except Exception as e:
    print(f"  ❌ Failed to load: {e}")
    dem_shape = None
    dem_transform = None
    dem_crs = None
    dem_bounds = None

# ===== CHECK COMPATIBILITY =====
print("\n" + "=" * 70)
print("COMPATIBILITY CHECK")
print("=" * 70)

if sar_crs is not None and dem_crs is not None:
    print(f"✓ Same CRS? {sar_crs == dem_crs}")
    if sar_crs != dem_crs:
        print(f"  SAR CRS: {sar_crs}")
        print(f"  DEM CRS: {dem_crs}")
        print(f"  ⚠️  CRS MISMATCH! Need to reproject.")

if sar_transform is not None and dem_transform is not None:
    sar_res = abs(sar_transform[0])
    dem_res = abs(dem_transform[0])
    print(f"✓ Same resolution (pixel size)? {sar_res == dem_res}")
    if sar_res != dem_res:
        print(f"  SAR resolution: {sar_res:.4f}")
        print(f"  DEM resolution: {dem_res:.4f}")
        print(f"  ⚠️  RESOLUTION MISMATCH! Ratio: {sar_res/dem_res:.2f}x")

if sar_shape is not None and dem_shape is not None:
    print(f"✓ Same shape? {sar_shape == dem_shape}")
    if sar_shape != dem_shape:
        print(f"  SAR shape: {sar_shape}")
        print(f"  DEM shape: {dem_shape}")
        print(f"  ⚠️  SHAPE MISMATCH! Need to align.")

if sar_bounds is not None and dem_bounds is not None:
    print(f"\n📊 Extent Overlap:")
    print(f"  SAR bounds:  ({sar_bounds[0]:.2f}, {sar_bounds[1]:.2f}) to ({sar_bounds[2]:.2f}, {sar_bounds[3]:.2f})")
    print(f"  DEM bounds:  ({dem_bounds[0]:.2f}, {dem_bounds[1]:.2f}) to ({dem_bounds[2]:.2f}, {dem_bounds[3]:.2f})")
    
    # Check if DEM contains SAR extent
    contains_sar = (dem_bounds[0] <= sar_bounds[0] <= dem_bounds[2] and
                    dem_bounds[1] <= sar_bounds[1] <= dem_bounds[3] and
                    dem_bounds[0] <= sar_bounds[2] <= dem_bounds[2] and
                    dem_bounds[1] <= sar_bounds[3] <= dem_bounds[3])
    print(f"  DEM contains SAR extent? {contains_sar}")

    # Check if SAR contains DEM extent
    contains_dem = (sar_bounds[0] <= dem_bounds[0] <= sar_bounds[2] and
                    sar_bounds[1] <= dem_bounds[1] <= sar_bounds[3] and
                    sar_bounds[0] <= dem_bounds[2] <= sar_bounds[2] and
                    sar_bounds[1] <= dem_bounds[3] <= sar_bounds[3])
    print(f"  SAR contains DEM extent? {contains_dem}")

# ===== CHECK ALL SAR EVENTS =====
print("\n" + "=" * 70)
print("ALL SAR EVENTS SUMMARY")
print("=" * 70)

for event_name, event_paths in SAR_PATHS.items():
    print(f"\n📁 {event_name}")
    flood_path = event_paths['flood_map']
    if flood_path.exists():
        try:
            with rasterio.open(flood_path) as src:
                data = src.read(1)
                flood_pixels = (data > 0).sum()
                total_pixels = data.size
                pct = 100 * flood_pixels / total_pixels
                print(f"  ✅ Flood pixels: {flood_pixels:,} ({pct:.2f}%)")
                print(f"  Shape: {src.shape}")
                print(f"  CRS: {src.crs}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    else:
        print(f"  ❌ File not found")

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

issues = []
if sar_crs != dem_crs:
    issues.append("❌ CRS MISMATCH: SAR and DEM have different coordinate systems")
if sar_transform is not None and dem_transform is not None:
    if abs(sar_transform[0]) != abs(dem_transform[0]):
        issues.append("❌ RESOLUTION MISMATCH: SAR and DEM have different pixel sizes")
if sar_shape != dem_shape:
    issues.append("⚠️  SHAPE MISMATCH: SAR and DEM have different dimensions")
if sar_bounds is not None and dem_bounds is not None:
    if not (dem_bounds[0] <= sar_bounds[0] <= dem_bounds[2] and
            dem_bounds[1] <= sar_bounds[1] <= dem_bounds[3] and
            dem_bounds[0] <= sar_bounds[2] <= dem_bounds[2] and
            dem_bounds[1] <= sar_bounds[3] <= dem_bounds[3]):
        issues.append("⚠️  EXTENT MISMATCH: SAR and DEM have different spatial extents")

if issues:
    print("🔴 ISSUES FOUND:")
    for issue in issues:
        print(f"  • {issue}")
    print("\n💡 RECOMMENDATION: You need to reproject/align SAR data to DEM's")
    print("   coordinate system before extracting features.")
    print("\n   Fix: Update SpatialAligner to include SAR in alignment.")
else:
    print("✅ All coordinate systems are compatible!")
    print("   The issue likely lies elsewhere in the feature extraction process.")

print("\n" + "=" * 70)