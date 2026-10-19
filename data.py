"""Read EuroSAT tiles, split them and normalise the bands.

EuroSAT_MS stores one 64 x 64 px GeoTIFF per tile with all 13 Sentinel-2 bands,
in one folder per land-cover class.
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

RGB = [3, 2, 1]  # band indices of B04 (red), B03 (green) and B02 (blue)


def find_tiles(root):
    """Return all .tif paths below root, their class labels and the class names.

    The class of a tile is the name of the folder it sits in.
    """
    paths = sorted(Path(root).rglob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No .tif files found below {root}")
    classes = sorted({p.parent.name for p in paths})
    labels = np.array([classes.index(p.parent.name) for p in paths])
    return paths, labels, classes


def split_indices(labels, fractions=(0.7, 0.15, 0.15), seed=42):
    """Stratified train/val/test split: every class is split with the same fractions."""
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_train = round(fractions[0] * len(idx))
        n_val = round(fractions[1] * len(idx))
        split["train"].extend(idx[:n_train])
        split["val"].extend(idx[n_train:n_train + n_val])
        split["test"].extend(idx[n_train + n_val:])
    return {name: np.sort(np.array(idx, dtype=int)) for name, idx in split.items()}


def read_tile(path, bands=None):
    """Read a tile as a float32 array of shape (bands, height, width)."""
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
    return image if bands is None else image[bands]


def band_stats(paths, bands=None, max_tiles=2000, seed=0):
    """Per-band mean and standard deviation over (a random sample of) the given tiles.

    Call it with the training tiles only, so nothing about the validation or
    test tiles leaks into the normalisation.
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), size=min(max_tiles, len(paths)), replace=False)
    total, total_sq, n_pixels = 0.0, 0.0, 0
    for i in pick:
        image = read_tile(paths[i], bands).astype(np.float64)
        total += image.sum(axis=(1, 2))
        total_sq += (image ** 2).sum(axis=(1, 2))
        n_pixels += image.shape[1] * image.shape[2]
    mean = total / n_pixels
    std = np.sqrt(np.maximum(total_sq / n_pixels - mean ** 2, 0.0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


class EuroSAT(Dataset):
    """Normalised tiles as float tensors, with optional rotation and flip augmentation."""

    def __init__(self, paths, labels, mean, std, bands=None, augment=False):
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.bands = bands
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = torch.from_numpy((read_tile(self.paths[i], self.bands) - self.mean) / self.std)
        if self.augment:
            # Images taken from above have no "up", so 90-degree turns and flips stay realistic
            image = torch.rot90(image, int(torch.randint(4, (1,))), dims=(1, 2))
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=(2,))
        return image, int(self.labels[i])
