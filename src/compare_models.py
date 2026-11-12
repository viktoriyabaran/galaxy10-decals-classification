import csv
import json

from dataset import CLASS_NAMES, PROJECT_ROOT

MODELS = [
    ("Random Forest", PROJECT_ROOT / "results/model1_rf"),
    ("CNN", PROJECT_ROOT / "results/model2_cnn"),
    ("ConvNeXt-Tiny", PROJECT_ROOT / "results/model3_convnext/size_128"),   # the run chosen on validation
]
OUT_DIR = PROJECT_ROOT / "results/comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_timings(path):
    """part -> seconds, from a timings.csv."""
    with open(path) as f:
        return {row["part"]: float(row["seconds"]) for row in csv.DictReader(f)}


overall, per_class = [], []
for name, folder in MODELS:
    if not (folder / "metrics_test.json").exists():
        print(f"skipped {name}: no test results yet in {folder}")
        continue
    metrics = json.load(open(folder / "metrics_test.json"))
    timings = read_timings(folder / "timings.csv")
    train_seconds = next(s for part, s in timings.items() if part.startswith("training (fit"))
    overall.append({
        "model": name,
        "test_accuracy": round(metrics["accuracy"], 4),
        "test_macro_f1": round(metrics["macro avg"]["f1-score"], 4),
        "train_seconds": round(train_seconds, 1),
        "test_predict_seconds": round(timings["test (predict)"], 3),
    })
    for class_name in CLASS_NAMES:
        per_class.append({
            "class": class_name,
            "model": name,
            "f1": round(metrics[class_name]["f1-score"], 3),
            "recall": round(metrics[class_name]["recall"], 3),
        })

for rows, file_name in [(overall, "overall.csv"), (per_class, "per_class.csv")]:
    with open(OUT_DIR / file_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {OUT_DIR / file_name} ({len(rows)} rows)")

for row in overall:
    print(row)
