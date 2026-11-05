"""Model 2: a small convolutional neural network (CNN), trained from scratch.
"""
import copy

import matplotlib.pyplot as plt
import torch
from torch import nn

from dataset import CLASS_NAMES, PROJECT_ROOT
from training import augment, class_weights, fit, get_device, make_loaders

IMAGE_SIZE = 64
BATCH_SIZE = 64     # images per training step
SEED = 42           # fixes all random choices (start values of the network, shuffling)
LEARNING_RATE = 1e-3   # how big a step the optimizer takes after each batch
EPOCHS = 30            # how many times the network sees all training images

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


# --- Step 3: build the network ---
def conv_block(channels_in, channels_out):
    """Find patterns, normalise, keep the positive answers, halve the image size."""
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),  # channels_out filters of 3x3
        nn.BatchNorm2d(channels_out),   # keeps the numbers in a stable range -> faster, steadier training
        nn.ReLU(),                      # negative -> 0: "pattern not here"
        nn.MaxPool2d(2),                # keep the strongest answer in each 2x2 square: 64 -> 32 -> ...
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # Each block learns more filters, and sees a larger area of the galaxy than the one before.
        self.features = nn.Sequential(
            conv_block(3, 32),     # 64x64 -> 32x32: edges, bright spots
            conv_block(32, 64),    # 32x32 -> 16x16: blobs, curves
            conv_block(64, 128),   # 16x16 -> 8x8:   larger shapes
            conv_block(128, 256),  # 8x8   -> 4x4:   whole structures (arms, bars, disks)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # average each of the 256 feature maps to one number
            nn.Flatten(),              # (batch, 256, 1, 1) -> (batch, 256)
            nn.Dropout(0.3),           # during training, randomly switch off 30% of them (less memorising)
            nn.Linear(256, n_classes), # 256 numbers -> 10 scores, one per class
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN(len(CLASS_NAMES)).to(device)

# eval mode: look at the network without changing it (training mode would update BatchNorm's statistics)
model.eval()
print("\nimage shape after each part of the network:")
out = x[:1].to(device)   # one image from the first batch
print(f"  {'input':28s} {tuple(out.shape)}")
with torch.no_grad():
    for i, layer in enumerate(list(model.features) + list(model.classifier)):
        out = layer(out)
        name = f"block {i + 1}" if i < len(model.features) else type(layer).__name__
        print(f"  {name:28s} {tuple(out.shape)}")

n_params = sum(p.numel() for p in model.parameters())
print(f"\nlearnable numbers (parameters): {n_params:,}")
for name, part in [("conv blocks", model.features), ("final layer", model.classifier)]:
    print(f"  {name}: {sum(p.numel() for p in part.parameters()):,}")

# The untrained network: random filters -> its 10 scores mean nothing yet.
with torch.no_grad():
    probabilities = torch.softmax(model(x[:1].to(device)), dim=1)[0].cpu()
print(f"\nuntrained guess for one image (probability per class): {probabilities.numpy().round(2)}")
print(f"true class: {y[0].item()} ({CLASS_NAMES[y[0]]})")

# --- Step 4: the loss and the optimizer ---
y_train_all = train_loader.dataset.tensors[1]
weights = class_weights(y_train_all, len(CLASS_NAMES))
print("\nclass weights (rare classes count more):")
for name, w in zip(CLASS_NAMES, weights):
    print(f"  {name:24s} {w:.2f}")

# Cross-entropy: -log(probability given to the true class), multiplied by that class's weight.
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
# Adam adjusts every one of the network's numbers a little after each batch.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# The loss for the untrained guess from step 3, by hand (without the class weight):
p_true = probabilities[y[0]].item()
print(f"\nuntrained: probability for the true class {p_true:.3f} -> loss = -log({p_true:.3f}) = "
      f"{-torch.log(torch.tensor(p_true)).item():.3f}")
print(f"a perfect guess (probability 1.0) would give -log(1.0) = 0")

# One learning step, shown on a copy of the network (the real one stays untouched for training).
demo_model = copy.deepcopy(model).train()
demo_optimizer = torch.optim.Adam(demo_model.parameters(), lr=LEARNING_RATE)
x_batch, y_batch = x.to(device), y.to(device)
for step in range(1, 6):
    loss = loss_fn(demo_model(x_batch), y_batch)  # 1. guess for 64 images and measure how wrong
    demo_optimizer.zero_grad()                     # 2. forget the directions from the previous step
    loss.backward()                                # 3. direction to change each number (gradient)
    demo_optimizer.step()                          # 4. change every number a little in that direction
    print(f"learning step {step} on the same batch: loss {loss.item():.3f}")

# --- Step 5: the training loop ---
history = fit(model, train_loader, val_loader, loss_fn, optimizer, device, EPOCHS, RESULTS_DIR)
print(f"\ntotal training time: {sum(row['train_seconds'] for row in history):.0f} s")
print(f"saved history.csv and learning_curves.png to {RESULTS_DIR}")
