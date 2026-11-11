import csv

from dataset import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results/model3_convnext"
RUNS = [
    # folder, variant name, image size, early stopping patience
    ("size_128_patience4", "early stopping", 128, 4),
    ("size_128", "baseline (full schedule)", 128, None),
    ("size_256", "256x256 images", 256, None),
]

results = []
for folder, name, size, patience in RUNS:
    history_path = RESULTS_DIR / folder / "history.csv"
    if not history_path.exists():
        print(f"skipped {name}: no {history_path} yet")
        continue
    with open(history_path) as f:
        history = list(csv.DictReader(f))
    best = max(history, key=lambda row: float(row["val_macro_f1"]))   # same rule as fit(): highest macro F1
    results.append({
        "variant": name,
        "image_size": size,
        "patience": patience if patience is not None else "none",
        "epochs_trained": len(history),
        "best_epoch": int(best["epoch"]),
        "val_accuracy": float(best["val_accuracy"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "train_seconds": round(sum(float(row["train_seconds"]) for row in history)),
    })

print(f"{'variant':26s} {'size':>4s} {'patience':>8s} {'epochs':>6s} {'best':>4s} {'accuracy':>8s} "
      f"{'macro F1':>8s} {'train time':>10s}")
for r in results:
    print(f"{r['variant']:26s} {r['image_size']:>4d} {str(r['patience']):>8s} {r['epochs_trained']:>6d} "
          f"{r['best_epoch']:>4d} {r['val_accuracy']:>8.1%} {r['val_macro_f1']:>8.3f} {r['train_seconds']:>9d}s")

with open(RESULTS_DIR / "experiments.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nsaved to {RESULTS_DIR / 'experiments.csv'}")
