"""Model 1: Random Forest on downscaled, flattened galaxy images.
"""
import numpy as np

from dataset import CLASS_NAMES, load_labels, load_split

# --- Step 1: load the split and the labels ---
train_idx, val_idx, test_idx = load_split()
labels = load_labels()

y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

print(f"train: {len(train_idx)} images, val: {len(val_idx)}, test: {len(test_idx)}")
print(f"first 5 train image numbers: {train_idx[:5]} -> their labels: {y_train[:5]}")
print("\nshare of each class (should be almost equal in all three parts):")
for c, name in enumerate(CLASS_NAMES):
    shares = [np.mean(y == c) for y in (y_train, y_val, y_test)]
    print(f"  {c} {name:24s} " + "  ".join(f"{s:6.1%}" for s in shares))
