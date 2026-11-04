"""Model 1: Random Forest on downscaled, flattened galaxy images.
"""
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dataset import CLASS_NAMES, PROJECT_ROOT, RAW_PATH, load_images, load_labels, load_split

IMAGE_SIZE = 64
RESULTS_DIR = PROJECT_ROOT / "results/model1_rf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# --- Step 2: shrink the images to IMAGE_SIZE x IMAGE_SIZE ---
images = load_images(IMAGE_SIZE)
img_train, img_val, img_test = images[train_idx], images[val_idx], images[test_idx]
print(f"\nall images: {images.shape}, {images.dtype}, {images.nbytes / 1e6:.0f} MB")
print(f"train images: {img_train.shape}")

# Before / after: the first training image of each class, full size vs shrunk.
examples = [train_idx[y_train == c][0] for c in range(10)]
fig, axes = plt.subplots(2, 10, figsize=(20, 4.6))
with h5py.File(RAW_PATH, "r") as f:
    for col, i in enumerate(examples):
        axes[0, col].imshow(f["images"][i])
        axes[1, col].imshow(images[i])
        axes[0, col].set_title(CLASS_NAMES[labels[i]], fontsize=9)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel("256 x 256")
axes[1, 0].set_ylabel(f"{IMAGE_SIZE} x {IMAGE_SIZE}")
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"shrink_{IMAGE_SIZE}.png", dpi=90)
print(f"saved before/after picture to {RESULTS_DIR / f'shrink_{IMAGE_SIZE}.png'}")

# --- Step 3: turn the images into a table ---
X_train = img_train.reshape(len(img_train), -1).astype(np.float32) / 255
X_val = img_val.reshape(len(img_val), -1).astype(np.float32) / 255
X_test = img_test.reshape(len(img_test), -1).astype(np.float32) / 255

print(f"\ntable X_train: {X_train.shape[0]} rows (images) x {X_train.shape[1]} columns (numbers), "
      f"{X_train.nbytes / 1e6:.0f} MB")
print(f"values from {X_train.min():.2f} to {X_train.max():.2f}")
print(f"first image, first 6 columns (pixel (0,0) R G B, pixel (0,1) R G B): {X_train[0, :6].round(3)}")
centre = (32 * IMAGE_SIZE + 32) * 3
print(f"first image, centre pixel R G B (columns {centre}-{centre + 2}): {X_train[0, centre:centre + 3].round(3)}")

# --- Step 4: train the Random Forest ---
forest = RandomForestClassifier(
    n_estimators=300,         # number of trees that vote
    max_features="sqrt",      # each question picks from sqrt(12,288) ~ 110 random pixels
    min_samples_leaf=1,       # trees may keep splitting until every image sits alone
    class_weight="balanced",  # mistakes on rare classes (e.g. Cigar Shaped) count more
    n_jobs=-1,                # build trees on all CPU cores at once
    random_state=42,          # same random choices -> same forest every run
)
start = time.perf_counter()
forest.fit(X_train, y_train)  # learns only from the training part
print(f"\ntrained {len(forest.estimators_)} trees in {time.perf_counter() - start:.0f} s")

depths = [tree.get_depth() for tree in forest.estimators_]
leaves = [tree.get_n_leaves() for tree in forest.estimators_]
print(f"questions in a row per tree (depth): {min(depths)}-{max(depths)}, "
      f"final answers per tree (leaves): ~{int(np.mean(leaves))}")

# Score on images it has already seen: shows memorisation, not real skill (that is step 5).
print(f"accuracy on the TRAINING images: {forest.score(X_train, y_train):.1%}")
