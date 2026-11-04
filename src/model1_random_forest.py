"""Model 1: Random Forest on downscaled, flattened galaxy images.
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np

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
