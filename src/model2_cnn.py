"""Model 2: a small convolutional neural network (CNN), trained from scratch.
"""
import torch

from dataset import CLASS_NAMES, PROJECT_ROOT
from training import get_device, make_loaders

IMAGE_SIZE = 64
BATCH_SIZE = 64     # images per training step
SEED = 42           # fixes all random choices (start values of the network, shuffling)

RESULTS_DIR = PROJECT_ROOT / "results/model2_cnn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
device = get_device()
print(f"device: {device}")

# --- Step 1: data for PyTorch ---
train_loader, val_loader, test_loader, mean, std = make_loaders(IMAGE_SIZE, BATCH_SIZE, SEED)

print(f"batches per epoch: train {len(train_loader)}, val {len(val_loader)}, test {len(test_loader)}")
print(f"training images per colour channel (R, G, B) before normalising: "
      f"mean {mean.numpy().round(3)}, std {std.numpy().round(3)}")

x, y = next(iter(train_loader))   # take the first batch
print(f"\none batch: images {tuple(x.shape)} (batch, colour, height, width), labels {tuple(y.shape)}")
print(f"pixel values after normalising: mean {x.mean():.2f}, std {x.std():.2f}, "
      f"from {x.min():.2f} to {x.max():.2f}")
print(f"labels in this batch: {y[:16].tolist()} ...")
print(f"classes in this batch: {len(y.unique())} of {len(CLASS_NAMES)}")
