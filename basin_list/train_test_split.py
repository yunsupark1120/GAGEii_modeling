import numpy as np
import os

# Path to the complete basin list
input_file = "basin_list_complete.txt"

# Output file paths
train_file = "regionally_separated_train.txt"
test_file = "regionally_separated_test.txt"

# Read all basin IDs from the file
with open(input_file, 'r') as f:
    basin_ids = [line.strip() for line in f if line.strip()]

# Shuffle the basin IDs randomly
np.random.seed(42)  # Set seed for reproducibility
np.random.shuffle(basin_ids)

# Calculate split indices
total = len(basin_ids)
train_size = int(total * 0.8)

# Split the data
train_ids = basin_ids[:train_size]
test_ids = basin_ids[train_size:]

# Write train set to file
with open(train_file, 'w') as f:
    for basin_id in train_ids:
        f.write(f"{basin_id}\n")

# Write test set to file
with open(test_file, 'w') as f:
    for basin_id in test_ids:
        f.write(f"{basin_id}\n")

print(f"Split complete: {len(train_ids)} basins in train set ({len(train_ids)/total:.1%}), "
      f"{len(test_ids)} basins in test set ({len(test_ids)/total:.1%})")