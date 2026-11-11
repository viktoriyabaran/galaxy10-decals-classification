import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from dataset import CLASS_NAMES, PROJECT_ROOT, load_images, load_labels, load_split
from evaluate import evaluate
from training import class_weights, fit, get_device, predict

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=128, choices=[128, 256], help="image size in pixels")
parser.add_argument("--quick", action="store_true", help="tiny run (few images, 1 epoch) to test the script")
args = parser.parse_args()

IMAGE_SIZE = args.size
BATCH_SIZE = 32
SEED = 42
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
EPOCHS = 15
PATIENCE = None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

run_name = f"size_{IMAGE_SIZE}" + ("_quick" if args.quick else "")
RESULTS_DIR = PROJECT_ROOT / "results/model3_convnext" / run_name
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = PROJECT_ROOT / f"models/model3_convnext_{run_name}.pt"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
device = get_device()
print(f"device: {device}, image size: {IMAGE_SIZE}, run: {run_name}")


# --- Step 1: data ---
class GalaxyImages(Dataset):
    def __init__(self, images, idx, labels):
        self.images, self.idx, self.labels = images, idx, labels

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n = self.idx[i]
        x = torch.from_numpy(self.images[n]).permute(2, 0, 1).float() / 255   # (H, W, 3) 0-255 -> (3, H, W) 0-1
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, int(self.labels[n])


images = load_images(IMAGE_SIZE)
labels = load_labels()
train_idx, val_idx, _ = load_split()
if args.quick:
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = rng.choice(train_idx, 320, replace=False), rng.choice(val_idx, 160, replace=False)

generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(GalaxyImages(images, train_idx, labels), batch_size=BATCH_SIZE,
                          shuffle=True, generator=generator)
val_loader = DataLoader(GalaxyImages(images, val_idx, labels), batch_size=BATCH_SIZE, shuffle=False)
print(f"batches per epoch: train {len(train_loader)}, val {len(val_loader)}")


# --- Step 2: the pretrained network, with a new last layer ---
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)   # downloads ~110 MB the first time
old_last_layer = model.classifier[2]
model.classifier[2] = nn.Linear(old_last_layer.in_features, len(CLASS_NAMES))
model = model.to(device)
print(f"last layer: {old_last_layer.in_features} -> {old_last_layer.out_features} (ImageNet) "
      f"replaced by {old_last_layer.in_features} -> {len(CLASS_NAMES)} (galaxies)")
print(f"learnable numbers (parameters): {sum(p.numel() for p in model.parameters()):,}")


# --- Step 3: loss and optimizer ---
weights = class_weights(torch.from_numpy(labels[train_idx]).long(), len(CLASS_NAMES))
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
epochs = 1 if args.quick else EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# --- Step 4: training (same loop as model 2; the best epoch is saved after every improvement) ---
history, best_epoch = fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, RESULTS_DIR,
                          scheduler=scheduler, checkpoint_path=MODEL_PATH, patience=PATIENCE)
best = history[best_epoch - 1]
print(f"\nbest epoch {best_epoch}: val accuracy {best['val_accuracy']:.1%}, macro F1 {best['val_macro_f1']:.3f}")
print(f"total training time: {sum(row['train_seconds'] for row in history) / 60:.0f} min for {len(history)} epochs")
print(f"saved the best network to {MODEL_PATH}, history.csv and learning_curves.png to {RESULTS_DIR}")


# --- Step 5: evaluate the best epoch on the validation images ---
_, y_val_pred, y_val = predict(model, val_loader, loss_fn, device)
evaluate(y_val, y_val_pred, "val", RESULTS_DIR)
