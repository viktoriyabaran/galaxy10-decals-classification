"""Shared PyTorch code for the neural network models (2 and 3).
"""
import torch
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
