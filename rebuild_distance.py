import shutil, numpy as np, rasterio, geopandas as gpd
from pathlib import Path
from rasterio import features
from scipy.ndimage import distance_transform_edt

RIVERS = "/Users/mar/lectures/thesis 1/fsm/datas/FOR TRAINING/River_Creeksnew_UTM.shp"   # <- edit this
A = Path("outputs/phase2/aligned")

# 1. Compare every aligned raster to the DEM grid
with rasterio.open(A / "dem_aligned.tif") as d:
    ref = (d.shape, d.transform, d.crs)
for f in sorted(A.glob("*_aligned.tif")):
    with rasterio.open(f) as r:
        a = r.read(1, masked=True)
        same = (r.shape, r.transform, r.crs) == ref
        print(f"{f.name:34s} same_grid={same}  nodata={r.nodata}  masked={a.mask.mean():.1%}")

# 2. Rebuild distance in metres
shape, tf, crs = ref
riv = gpd.read_file(RIVERS).to_crs(crs)
m = features.rasterize(riv.geometry, out_shape=shape, transform=tf, all_touched=True, dtype="uint8")
dist = (distance_transform_edt(m == 0) * abs(tf.a)).astype("float32")

out = A / "distance_to_river_aligned.tif"
shutil.copy(out, out.with_name("distance_to_river_aligned_OLD.tif"))
with rasterio.open(A / "dem_aligned.tif") as d:
    meta = d.meta.copy()
    dem_mask = d.read(1, masked=True).mask
dist[dem_mask] = np.nan
meta.update(dtype="float32", count=1, nodata=np.nan)
with rasterio.open(out, "w", **meta) as dst:
    dst.write(dist, 1)
print("distance range (m):", np.nanmin(dist), np.nanmedian(dist), np.nanmax(dist))