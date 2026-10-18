import pandas as pd
df = pd.read_csv("outputs/phase2/full_dataset_engineered.csv")
m = (df.hand - 141.3549).abs() < 1e-3
print(df[m].groupby("flood_label").distance_to_river.describe())