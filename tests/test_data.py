import numpy as np
import torch

from data import RGB, EuroSAT, band_stats, find_tiles, split_indices


def test_class_comes_from_folder_name(fake_eurosat):
    paths, labels, classes = find_tiles(fake_eurosat)
    assert classes == ["Forest", "Highway", "River"]
    assert len(paths) == 30
    assert all(classes[label] == path.parent.name for path, label in zip(paths, labels))


def test_no_tile_is_in_two_splits_and_none_is_lost():
    labels = np.repeat(np.arange(10), 100)
    split = split_indices(labels)
    train, val, test = (set(split[name]) for name in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert train | val | test == set(range(len(labels)))


def test_every_class_is_split_with_the_same_fractions():
    labels = np.repeat(np.arange(10), 100)
    split = split_indices(labels, fractions=(0.7, 0.15, 0.15))
    for name, share in [("train", 0.7), ("val", 0.15), ("test", 0.15)]:
        counts = np.bincount(labels[split[name]], minlength=10)
        assert np.all(counts == round(share * 100))


def test_same_seed_gives_same_split():
    labels = np.repeat(np.arange(3), 50)
    first, again, other = split_indices(labels, seed=1), split_indices(labels, seed=1), split_indices(labels, seed=2)
    assert all(np.array_equal(first[name], again[name]) for name in first)
    assert not np.array_equal(first["test"], other["test"])


def test_tiles_come_out_as_normalised_13_band_tensors(fake_eurosat):
    paths, labels, _ = find_tiles(fake_eurosat)
    mean, std = band_stats(paths)
    data = EuroSAT(paths, labels, mean, std)
    image, label = data[0]
    assert image.shape == (13, 64, 64) and image.dtype == torch.float32
    assert label == labels[0]
    stack = torch.stack([data[i][0] for i in range(len(data))])
    assert torch.allclose(stack.mean(dim=(0, 2, 3)), torch.zeros(13), atol=1e-3)
    assert torch.allclose(stack.std(dim=(0, 2, 3)), torch.ones(13), atol=1e-2)


def test_rgb_keeps_only_three_bands(fake_eurosat):
    paths, labels, _ = find_tiles(fake_eurosat)
    mean, std = band_stats(paths, RGB)
    image, _ = EuroSAT(paths, labels, mean, std, bands=RGB)[0]
    assert image.shape == (3, 64, 64)


def test_augmentation_only_moves_pixels(fake_eurosat):
    paths, labels, _ = find_tiles(fake_eurosat)
    mean, std = band_stats(paths)
    plain, _ = EuroSAT(paths, labels, mean, std)[0]
    torch.manual_seed(0)
    augmented, _ = EuroSAT(paths, labels, mean, std, augment=True)[0]
    assert augmented.shape == plain.shape
    assert torch.equal(augmented.flatten().sort().values, plain.flatten().sort().values)
