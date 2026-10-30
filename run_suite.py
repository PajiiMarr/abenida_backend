import json, pandas as pd
from app.training.models import train_model_suite

df = pd.read_csv("outputs/phase2/full_dataset_engineered.csv")
features = json.load(open("outputs/phase2/feature_names.json"))
train_model_suite(df, features, coord_cols=("x", "y"),
                  label_col="flood_label", event_col="event")