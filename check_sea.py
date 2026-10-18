import pandas as pd, rasterio
df = pd.read_csv("outputs/phase2/full_dataset_engineered.csv")
with rasterio.open("outputs/phase2/aligned/chirps_aligned.tif") as r: c = r.read(1)
v = c[df.row.astype(int).clip(0, c.shape[0]-1), df.col.astype(int).clip(0, c.shape[1]-1)]
print(df.assign(sea=(v == -9999)).groupby("flood_label").sea.mean())