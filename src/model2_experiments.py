import csv

import torch
from torch import nn

from dataset import CLASS_NAMES, PROJECT_ROOT
from small_cnn import SmallCNN
from training import class_weights, fit, get_device, make_loaders

BATCH_SIZE = 64
SEED = 42
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 10

RESULTS_DIR = PROJECT_ROOT / "results/model2_cnn"
VARIANTS = [
    {"name": "no augmentation", "key": "no_augment", "size": 64,  "augment": False},
    {"name": "128x128 images",  "key": "size_128",   "size": 128, "augment": True},
]

device = get_device()


def summary(name, size, augment, history_path):
    with open(history_path) as f:
        history = list(csv.DictReader(f))
    best = max(history, key=lambda row: float(row["val_macro_f1"]))
    return {
        "variant": name,
        "image_size": size,
        "augment": augment,
        "epochs_trained": len(history),
        "best_epoch": int(best["epoch"]),
        "val_accuracy": float(best["val_accuracy"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "train_seconds": round(sum(float(row["train_seconds"]) for row in history)),
    }


results = [summary("baseline", 64, True, RESULTS_DIR / "history.csv")]

for variant in VARIANTS:
    print(f"\n=== {variant['name']} ===")
    out_dir = RESULTS_DIR / "experiments" / variant["key"]
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)   # same starting point as the baseline
    train_loader, val_loader, _, _, _ = make_loaders(variant["size"], BATCH_SIZE, SEED)  # test part unused
    model = SmallCNN(len(CLASS_NAMES)).to(device)
    weights = class_weights(train_loader.dataset.tensors[1], len(CLASS_NAMES))
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    fit(model, train_loader, val_loader, loss_fn, optimizer, device, EPOCHS, out_dir,
        scheduler=scheduler, checkpoint_path=PROJECT_ROOT / f"models/model2_exp_{variant['key']}.pt",
        patience=PATIENCE, use_augment=variant["augment"])
    results.append(summary(variant["name"], variant["size"], variant["augment"], out_dir / "history.csv"))

print(f"\n{'variant':16s} {'size':>4s} {'augment':>7s} {'epochs':>6s} {'best':>4s} {'accuracy':>8s} "
      f"{'macro F1':>8s} {'train time':>10s}")
for r in results:
    print(f"{r['variant']:16s} {r['image_size']:>4d} {str(r['augment']):>7s} {r['epochs_trained']:>6d} "
          f"{r['best_epoch']:>4d} {r['val_accuracy']:>8.1%} {r['val_macro_f1']:>8.3f} {r['train_seconds']:>9d}s")

with open(RESULTS_DIR / "experiments.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nsaved to {RESULTS_DIR / 'experiments.csv'}")
