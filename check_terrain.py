import pandas as pd
df = pd.read_csv("outputs/phase2/full_dataset_engineered.csv")
print(df.groupby("flood_label")[["hand", "slope"]].describe(percentiles=[.5, .9]).T)