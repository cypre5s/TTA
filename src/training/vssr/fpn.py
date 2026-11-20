"""Top-down feature pyramid at strides 4, 8, 16 and 32."""

from torch import nn
from torch.nn import functional as F


class FPN(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.smooth = nn.ModuleList(
            nn.Conv2d(width, width, 3, padding=1, groups=width) for _ in channels
        )

    def forward(self, maps):
        features = [conv(level) for conv, level in zip(self.lateral, maps, strict=True)]
        for i in range(len(features) - 2, -1, -1):
            features[i] = features[i] + F.interpolate(features[i + 1], features[i].shape[-2:], mode="nearest")
        return [conv(level) for conv, level in zip(self.smooth, features, strict=True)]
