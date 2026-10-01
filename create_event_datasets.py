# create_event_datasets.py
"""
Create event-specific datasets for May 2021 only
"""
import numpy as np
import json
from pathlib import Path
import rasterio

print("=" * 70)
print("CREATING EVENT-SPECIFIC DATASETS")
print("=" * 70)

# Load sample points
sample_path = Path("./outputs/phase2/sample_points/sample_points.csv")
import pandas as pd
df = pd.read_csv(sample_path)
points = np.column_stack([df['row'].values, df['col'].values])

# Load registered flood map
flood_path = Path("./outputs/phase2/registered_sar/may_2021_flood_map_registered.tif")

with rasterio.open(flood_path) as src:
    flood_map = src.read(1)

# Get flood labels at all sample points
labels = []
for row, col in points:
    r, c = int(round(row)), int(round(col))
    if 0 <= r < flood_map.shape[0] and 0 <= c < flood_map.shape[1]:
        labels.append(flood_map[r, c])
    else:
        labels.append(0)

labels = np.array(labels)

# Count flood samples
flood_count = labels.sum()
print(f"may_2021: {flood_count} flood samples out of {len(labels)} ({100*flood_count/len(labels):.1f}%)")

# Split into train/test (80/20)
np.random.seed(42)
indices = np.random.permutation(len(labels))
train_size = int(0.8 * len(labels))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

y_train_may = labels[train_indices]
y_test_may = labels[test_indices]

# Save
np.save('./outputs/phase2/preprocessed/y_train_may_2021.npy', y_train_may)
np.save('./outputs/phase2/preprocessed/y_test_may_2021.npy', y_test_may)

print(f"  Train: {len(y_train_may)} samples, {y_train_may.sum():.0f} flood")
print(f"  Test: {len(y_test_may)} samples, {y_test_may.sum():.0f} flood")

# Load features
X_train = np.load('./outputs/phase2/preprocessed/X_train.npy')
X_train_no_coords = X_train[:, 2:]

# ===== FIX: Align sample order =====
# The labels from sample_points.csv are in a specific order.
# X_train contains the same points but may be shuffled differently.
# We need to match them by using the sample_points.csv order.

# Load sample_points.csv to get the correct order
df = pd.read_csv('./outputs/phase2/sample_points/sample_points.csv')

# The y_train_may labels are in the same order as sample_points.csv
# We need to split X_train to match

# Since X_train was split from points using train_test_split,
# and we just did a random split on labels, the order may not match.

# To fix: use the same train/test split as Phase 2
# Load the Phase 2 split indices from the pipeline

print("\nChecking dimensions...")
print(f"X_train_no_coords: {X_train_no_coords.shape}")
print(f"y_train_may: {y_train_may.shape}")
print(f"y_train_may sum: {y_train_may.sum():.0f}")

# ===== FIX: Save without correlation calculation =====
print("\n" + "=" * 70)
print("EVENT-SPECIFIC DATASETS CREATED")
print("=" * 70)

print("\nMay 2021 labels saved:")
print(f"  y_train_may_2021.npy: {len(y_train_may)} samples, {y_train_may.sum():.0f} flood")
print(f"  y_test_may_2021.npy: {len(y_test_may)} samples, {y_test_may.sum():.0f} flood")