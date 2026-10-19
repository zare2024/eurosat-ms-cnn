import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import EuroSAT, band_stats, find_tiles
from model import SmallCNN
from train import run_epoch


@pytest.mark.parametrize("bands", [3, 13])
def test_one_score_per_class_for_rgb_and_13_bands(bands):
    model = SmallCNN(in_channels=bands, num_classes=10)
    assert model(torch.randn(4, bands, 64, 64)).shape == (4, 10)


def test_model_can_memorise_one_batch():
    """If the model can't fit 16 images, something in the training setup is broken."""
    torch.manual_seed(0)
    images, labels = torch.randn(16, 13, 64, 64), torch.randint(0, 10, (16,))
    model = SmallCNN(13, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_one_epoch_reports_loss_and_accuracy(fake_eurosat):
    paths, labels, classes = find_tiles(fake_eurosat)
    mean, std = band_stats(paths)
    loader = DataLoader(EuroSAT(paths, labels, mean, std), batch_size=8)
    model = SmallCNN(13, len(classes))
    loss, accuracy = run_epoch(model, loader, nn.CrossEntropyLoss(),
                               torch.optim.Adam(model.parameters()))
    assert loss > 0 and 0 <= accuracy <= 1
