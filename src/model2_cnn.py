"""Model 2: a small convolutional neural network (CNN), trained from scratch.
"""
import matplotlib.pyplot as plt
import torch

from dataset import CLASS_NAMES, PROJECT_ROOT
from training import augment, get_device, make_loaders

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

# --- Step 2: augmentation (training images only, new random orientation every time) ---
x_all, y_all = train_loader.dataset.tensors
DEMO_CLASS = 0                                    # Disturbed: the first one has a long curved tail
galaxy = x_all[(y_all == DEMO_CLASS).nonzero()[0]]  # first such galaxy in training: shape (1, 3, 64, 64)


def to_picture(img):
    """Undo the normalising so the image can be shown: (3, 64, 64) -> (64, 64, 3), values 0-1."""
    return (img * std[:, None, None] + mean[:, None, None]).permute(1, 2, 0).clamp(0, 1)


all_eight = [torch.rot90(galaxy[0], k, dims=(1, 2)) for k in range(4)]
all_eight += [img.flip(dims=(2,)) for img in all_eight]
random_eight = [augment(galaxy)[0] for _ in range(8)]

fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
for col in range(8):
    axes[0, col].imshow(to_picture(all_eight[col]))
    axes[1, col].imshow(to_picture(random_eight[col]))
    axes[0, col].set_title(f"{90 * (col % 4)} deg{', flipped' if col >= 4 else ''}", fontsize=9)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel("all 8 versions")
axes[1, 0].set_ylabel("8 random draws")
plt.suptitle(f"augmentation: one {CLASS_NAMES[DEMO_CLASS]} galaxy")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "augmentation.png", dpi=90)
print(f"\nsaved {RESULTS_DIR / 'augmentation.png'}")

# Same pixels, only moved: augmentation never changes the brightness values themselves.
print(f"pixel values sorted are identical after augmenting: "
      f"{torch.equal(galaxy.flatten().sort().values, augment(galaxy).flatten().sort().values)}")
