"""A small CNN whose first layer accepts any number of input bands."""
from torch import nn


def conv_block(c_in, c_out):
    """Two 3x3 convolutions with batch norm and ReLU, then 2x2 max pooling."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Three conv blocks (64 -> 32 -> 16 -> 8 px), global average pooling and a linear classifier.

    A network pretrained on ImageNet expects 3 RGB channels. Training from scratch
    lets the first layer take all 13 Sentinel-2 bands directly.
    """

    def __init__(self, in_channels=13, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
