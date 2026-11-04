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
