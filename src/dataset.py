"""Shared data code for all three models: paths, class names, split and labels.

Every model imports from here, so all of them use exactly the same data.
"""
from pathlib import Path

import h5py
import numpy as np

# Paths are built from this file's location, so they work from any folder (scripts, notebooks).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data/raw/Galaxy10_DECals_NoDuplicated.h5"
SPLIT_PATH = PROJECT_ROOT / "data/processed/split.npz"

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar Shaped Smooth",
    "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral",
    "Edge-on without Bulge", "Edge-on with Bulge",
]


def load_labels():
    """Label (class number 0-9) of every image in the .h5 file, in file order."""
    with h5py.File(RAW_PATH, "r") as f:
        return f["ans"][:]


def load_split():
    """Image numbers of the train / validation / test parts, made by prepare_data.py."""
    split = np.load(SPLIT_PATH)
    return split["train"], split["val"], split["test"]


def load_images(size=64):
    """All images shrunk from 256x256 to size x size, as uint8 (0-255), in file order.

    The first call reads the .h5 file and saves the result to
    data/processed/images_<size>.npy; later calls just load that file.
    """
    cache_path = PROJECT_ROOT / f"data/processed/images_{size}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    factor = 256 // size
    assert size * factor == 256, "size must divide 256 (e.g. 128, 64, 32, 16)"

    with h5py.File(RAW_PATH, "r") as f:
        images = f["images"]
        small = np.empty((len(images), size, size, 3), dtype=np.uint8)
        # The file is stored compressed in pieces ("chunks") of 553 images; reading exactly
        # one chunk at a time is fastest, and needs only ~0.5 GB of memory.
        block_size = images.chunks[0]
        for start in range(0, len(images), block_size):
            block = images[start:start + block_size].astype(np.float32)
            n = len(block)
            # Shrink by averaging each factor x factor square of pixels into one pixel.
            block = block.reshape(n, size, factor, size, factor, 3).mean(axis=(2, 4))
            small[start:start + n] = np.round(block).astype(np.uint8)

    np.save(cache_path, small)
    return small
