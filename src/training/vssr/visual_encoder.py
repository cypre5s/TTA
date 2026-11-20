"""Pyramid tokens retain normalized spatial locations and scale identity."""

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .fpn import FPN


class VisualEncoder(nn.Module):
    def __init__(self, config, **backbone_options):
        super().__init__()
        self.backbone = Backbone(config, **backbone_options)
        self.fpn = FPN(self.backbone.channels, config.hidden)
        self.pools = config.visual_pools
        self.scale = nn.Embedding(4, config.hidden)
        self.xy = nn.Linear(2, config.hidden)
        self.norm = nn.LayerNorm(config.hidden)

    def forward(self, images):
        raw = self.backbone(images)
        pyramid = self.fpn(raw)
        tokens, positions = [], []
        for level, (feature, requested) in enumerate(zip(pyramid, self.pools, strict=True)):
            height = min(requested, feature.shape[-2])
            width = min(requested, feature.shape[-1])
            grid_y = (torch.arange(height, device=feature.device, dtype=torch.float32) + 0.5) / height
            grid_x = (torch.arange(width, device=feature.device, dtype=torch.float32) + 0.5) / width
            y, x = torch.meshgrid(grid_y, grid_x, indexing="ij")
            xy = torch.stack((x, y), dim=-1).reshape(-1, 2).to(feature.dtype)
            pooled = F.adaptive_avg_pool2d(feature, (height, width)).flatten(2).transpose(1, 2)
            position = self.xy(xy) + self.scale.weight[level].to(feature.dtype)
            tokens.append(pooled + position)
            positions.append(xy)
        return dict(
            tokens=self.norm(torch.cat(tokens, dim=1)), xy=torch.cat(positions, dim=0),
            backbone_shapes=tuple(tuple(x.shape) for x in raw),
            fpn_shapes=tuple(tuple(x.shape) for x in pyramid),
        )
