"""Clean Galaxy10 DECaLS and split it into train / validation / test.

The raw file is never changed. We only decide which image numbers (indices)
to use, and save those lists to data/processed/split.npz.

Run from the project root:
    python src/prepare_data.py
"""
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

RAW_PATH = "data/raw/Galaxy10_DECals_NoDuplicated.h5"
OUT_PATH = "data/processed/split.npz"
SEED = 42                  # fixed seed -> the same split every time
SAME_OBJECT_ARCSEC = 30    # closer than this on the sky = the same object photographed twice

# --- Step 1: load labels and sky positions (small arrays, one number per image) ---
with h5py.File(RAW_PATH, "r") as f:
    labels = f["ans"][:]
    ra = f["ra"][:]
    dec = f["dec"][:]

    # --- Step 2: find blank images (every pixel has the same value) ---
    # All images together are ~3.5 GB, so we read them in blocks of 500.
    images = f["images"]
    blank = []
    for start in range(0, len(labels), 500):
        block = images[start:start + 500]
        is_blank = block.max(axis=(1, 2, 3)) == block.min(axis=(1, 2, 3))
        blank += (start + np.flatnonzero(is_blank)).tolist()

print(f"images in file: {len(labels)}")
print(f"blank images: {blank}")

# --- Step 3: find the same galaxy photographed twice ---
# BallTree finds all neighbours within a radius; "haversine" = distance on a sphere (the sky).
coords = np.radians(np.c_[dec, ra])
tree = BallTree(coords, metric="haversine")
neighbours = tree.query_radius(coords, r=np.radians(SAME_OBJECT_ARCSEC / 3600))

duplicates = set()
for i, near in enumerate(neighbours):
    if i in duplicates:
        continue
    for j in near:
        if j > i:              # keep the first copy (i), drop the later one (j)
            duplicates.add(int(j))
duplicates = sorted(duplicates)
print(f"duplicates (same object within {SAME_OBJECT_ARCSEC}\"): {len(duplicates)}")

removed = set(blank) | set(duplicates)
keep = np.array([i for i in range(len(labels)) if i not in removed])
print(f"images kept: {len(keep)}")

# --- Step 4: split 70% train / 15% validation / 15% test ---
# stratify=labels keeps the class proportions the same in every part.
train, rest = train_test_split(keep, test_size=0.30, stratify=labels[keep], random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=SEED)

print("\nimages per class:")
print(f"{'class':>5} {'train':>6} {'val':>5} {'test':>5}")
for c in range(10):
    print(f"{c:>5} {np.sum(labels[train] == c):>6} {np.sum(labels[val] == c):>5} {np.sum(labels[test] == c):>5}")
print(f"{'total':>5} {len(train):>6} {len(val):>5} {len(test):>5}")

# --- Step 5: save the index lists (a few KB instead of copying 2.7 GB of images) ---
np.savez(OUT_PATH, train=np.sort(train), val=np.sort(val), test=np.sort(test),
         removed_blank=np.array(blank), removed_duplicates=np.array(duplicates))
print(f"\nsaved to {OUT_PATH}")
