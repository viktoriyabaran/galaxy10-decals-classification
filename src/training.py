"""Shared PyTorch code for the neural network models (2 and 3).
"""
import csv
import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_images, load_labels, load_split


def get_device():
    """The Mac's GPU ("mps") if available, otherwise the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def augment(x):
    """Give every image in a batch a random orientation: rotated by 0/90/180/270 degrees, flipped or not.

    That makes 8 possible versions of each galaxy. Galaxies have no "up" in space,
    so every version shows the same galaxy type.
    """
    turns = torch.randint(0, 4, (len(x),)).to(x.device)       # how many 90-degree turns per image
    flip = (torch.rand(len(x)) < 0.5).to(x.device)            # mirror half of the images
    x = x.clone()                                              # leave the original batch unchanged
    for k in (1, 2, 3):
        x[turns == k] = torch.rot90(x[turns == k], k, dims=(2, 3))
    x[flip] = x[flip].flip(dims=(3,))
    return x


def to_tensors(images, idx, labels):
    """Pick images by number and turn them into PyTorch tensors.

    numpy stores an image as height x width x colour (64, 64, 3),
    PyTorch expects colour x height x width (3, 64, 64), so the axes are reordered.
    """
    x = torch.from_numpy(images[idx]).permute(0, 3, 1, 2).float() / 255   # 0-255 -> 0-1
    y = torch.from_numpy(labels[idx]).long()
    return x, y


def make_loaders(size=64, batch_size=64, seed=42):
    """Train / validation / test data, normalised and grouped into batches.

    Returns the three loaders and the per-channel mean and std of the training images.
    """
    images = load_images(size)
    labels = load_labels()
    train_idx, val_idx, test_idx = load_split()

    x_train, y_train = to_tensors(images, train_idx, labels)
    x_val, y_val = to_tensors(images, val_idx, labels)
    x_test, y_test = to_tensors(images, test_idx, labels)

    # Normalise each colour channel to mean 0 and std 1, using statistics of the training images only
    # (like PCA in model 1: validation and test must not influence anything the model uses).
    mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
    std = x_train.std(dim=(0, 2, 3), keepdim=True)
    x_train, x_val, x_test = (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std

    # Training batches are shuffled anew every epoch (the file is sorted by class!);
    # the generator with a fixed seed makes the shuffling the same in every run.
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size,
                              shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mean.flatten(), std.flatten()


def class_weights(y_train, n_classes=10):
    """Weight per class so that every class counts equally in the loss.

    Same formula as class_weight="balanced" in scikit-learn (model 1):
    number of images / (number of classes x images in that class). Rare classes get large weights.
    """
    counts = torch.bincount(y_train, minlength=n_classes).float()
    return len(y_train) / (n_classes * counts)


def train_one_epoch(model, loader, loss_fn, optimizer, device, use_augment=True):
    """One pass over all training batches; the network learns after every batch. Returns the mean loss."""
    model.train()                       # training mode: Dropout on, BatchNorm learns statistics
    total_loss, n_images = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_augment:
            x = augment(x)              # new random orientation for every image, every epoch
        loss = loss_fn(model(x), y)     # 1. guess and measure how wrong
        optimizer.zero_grad()           # 2. clear the previous step's directions
        loss.backward()                 # 3. how should every number change?
        optimizer.step()                # 4. change every number a little
        total_loss += loss.item() * len(y)
        n_images += len(y)
    return total_loss / n_images


@torch.no_grad()                        # only measuring: no gradients needed (faster, less memory)
def predict(model, loader, loss_fn, device):
    """Guess the class of every image in a loader, without learning. Returns mean loss, guesses, truths."""
    model.eval()                        # evaluation mode: Dropout off, BatchNorm uses learned statistics
    total_loss, guesses, truths = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        scores = model(x)
        total_loss += loss_fn(scores, y).item() * len(y)
        guesses.append(scores.argmax(dim=1).cpu())   # the class with the highest score
        truths.append(y.cpu())
    guesses, truths = torch.cat(guesses).numpy(), torch.cat(truths).numpy()
    return total_loss / len(truths), guesses, truths


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, out_dir,
        scheduler=None, checkpoint_path=None, patience=None, use_augment=True):
    history = []
    best_f1, best_epoch = -1.0, 0
    print(f"\n{'epoch':>5s} {'lr':>8s} {'train loss':>10s} {'val loss':>8s} {'val acc':>7s} {'val F1':>6s} "
          f"{'seconds':>7s}")
    for epoch in range(1, epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, use_augment)
        train_seconds = time.perf_counter() - start
        if scheduler is not None:
            scheduler.step()

        val_loss, val_guesses, val_truths = predict(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(accuracy_score(val_truths, val_guesses), 4),
            "val_macro_f1": round(f1_score(val_truths, val_guesses, average="macro"), 4),
            "train_seconds": round(train_seconds, 2),
        }
        history.append(row)
        save_history(history, out_dir / "history.csv")   # after every epoch: nothing is lost if a run stops early

        is_best = row["val_macro_f1"] > best_f1
        if is_best:
            best_f1, best_epoch = row["val_macro_f1"], epoch
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)   # all learned numbers, as a file
        print(f"{epoch:>5d} {learning_rate:>8.6f} {row['train_loss']:>10.3f} {row['val_loss']:>8.3f} "
              f"{row['val_accuracy']:>7.1%} {row['val_macro_f1']:>6.3f} {row['train_seconds']:>7.1f}"
              f"{'  <- best so far' if is_best else ''}")

        if patience is not None and epoch - best_epoch >= patience:
            print(f"early stop: no improvement for {patience} epochs")
            break

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))         # back to the best epoch

    plot_history(history, out_dir / "learning_curves.png", best_epoch)
    return history, best_epoch


def save_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, path, best_epoch=None):
    """Learning curves: loss (left) and validation scores (right) after every epoch."""
    epochs = [row["epoch"] for row in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.plot(epochs, [row["train_loss"] for row in history], marker="o", label="training")
    left.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    left.set_xlabel("epoch")
    left.set_ylabel("loss (lower is better)")
    left.legend()
    right.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="accuracy")
    right.plot(epochs, [row["val_macro_f1"] for row in history], marker="o", label="macro F1")
    right.set_xlabel("epoch")
    right.set_ylabel("validation score (higher is better)")
    if best_epoch:
        right.set_title(f"best epoch: {best_epoch} (dashed line)", fontsize=10)
    right.legend()
    for ax in (left, right):
        if best_epoch:
            ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
