"""Encode normalized boxes using independently recorded validity and source."""

import torch
from torch import nn


BOX_SOURCES = {"missing": 0, "ground_truth": 1, "predicted": 2, "inferred": 3}


class GeometryEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(8, 128), nn.GELU(), nn.Linear(128, width))
        self.source = nn.Embedding(len(BOX_SOURCES), width)

    def forward(self, boxes, valid, source):
        safe = torch.where(valid[..., None], boxes, 0)
        top_left, bottom_right = safe.chunk(2, dim=-1)
        vector = torch.cat((safe, bottom_right - top_left, (top_left + bottom_right) * 0.5), dim=-1)
        projected = self.project(vector)
        return self.source(torch.where(valid, source, 0)) + projected * valid[..., None]
