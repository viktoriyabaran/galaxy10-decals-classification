"""Shared evaluation for all three models: metrics, confusion matrix, saved results.

Every model calls evaluate() the same way, so their scores are directly comparable.
"""
import json

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)

from dataset import CLASS_NAMES


def evaluate(y_true, y_pred, part, out_dir):
    """Compare true labels with predictions; print and save the results.

    part: which data was used, e.g. "val" or "test" (used in file names).
    out_dir: folder for the results, e.g. results/model1_rf.
    """
    # Accuracy: share of all images that were classified correctly.
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, recall and F1 for every class, plus their averages.
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    print(f"\n=== {part}: accuracy {accuracy:.1%} ===")
    print(report)

    # Confusion matrix: row = true class, column = predicted class.
    # normalize="true" shows each row as percentages, so rare and common classes are comparable.
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, normalize="true",
        values_format=".0%", cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"{part}: accuracy {accuracy:.1%}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{part}.png", dpi=100)
    plt.close(fig)

    with open(out_dir / f"metrics_{part}.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"saved confusion_{part}.png and metrics_{part}.json to {out_dir}")

    return accuracy
