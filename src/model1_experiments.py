"""Model 1, step 6: compare a few variants of the Random Forest on the validation set.
"""
import csv
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from dataset import PROJECT_ROOT, load_images, load_labels, load_split

RESULTS_DIR = PROJECT_ROOT / "results/model1_rf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    {"name": "baseline",        "size": 64, "pca": None, "min_samples_leaf": 1},
    {"name": "tiny images",     "size": 16, "pca": None, "min_samples_leaf": 1},
    {"name": "smaller images",  "size": 32, "pca": None, "min_samples_leaf": 1},
    {"name": "bigger images",   "size": 128, "pca": None, "min_samples_leaf": 1},
    {"name": "PCA-50",          "size": 64, "pca": 50,   "min_samples_leaf": 1},
    {"name": "PCA-256",         "size": 64, "pca": 256,  "min_samples_leaf": 1},
    {"name": "less memorising", "size": 64, "pca": None, "min_samples_leaf": 5},
]

train_idx, val_idx, _ = load_split()   # "_": the test part is not loaded at all
labels = load_labels()
y_train, y_val = labels[train_idx], labels[val_idx]


def to_table(images, idx):
    """Pick images by number, flatten each into one row, scale to 0-1 (same as model 1, step 3)."""
    return images[idx].reshape(len(idx), -1).astype(np.float32) / 255


results = []
for variant in VARIANTS:
    start = time.perf_counter()

    images = load_images(variant["size"])
    X_train, X_val = to_table(images, train_idx), to_table(images, val_idx)

    if variant["pca"]:
        # PCA learns its components from the training images only, then applies them to validation.
        pca = PCA(n_components=variant["pca"], svd_solver="randomized", random_state=42)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)

    # Same settings as model 1, except min_samples_leaf, which some variants change.
    forest = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=variant["min_samples_leaf"],
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_val)

    results.append({
        "variant": variant["name"],
        "image_size": variant["size"],
        "pca": variant["pca"] or "none",
        "min_samples_leaf": variant["min_samples_leaf"],
        "columns": X_train.shape[1],
        "val_accuracy": round(accuracy_score(y_val, y_pred), 4),
        "val_macro_f1": round(f1_score(y_val, y_pred, average="macro"), 4),
        "seconds": round(time.perf_counter() - start),
    })
    print(f"done: {variant['name']}")

print(f"\n{'variant':16s} {'size':>4s} {'pca':>5s} {'leaf':>4s} {'columns':>7s} {'accuracy':>8s} {'macro F1':>8s} {'time':>5s}")
for r in results:
    print(f"{r['variant']:16s} {r['image_size']:>4d} {str(r['pca']):>5s} {r['min_samples_leaf']:>4d} "
          f"{r['columns']:>7d} {r['val_accuracy']:>8.1%} {r['val_macro_f1']:>8.3f} {r['seconds']:>4d}s")

with open(RESULTS_DIR / "experiments.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nsaved to {RESULTS_DIR / 'experiments.csv'}")
