"""Differentiable grid regularizers. They leave the proposal set intact."""

import torch
from torch.nn import functional as F

from .cell_proposal import box_metrics


def grid_constraints(proposal, target_boxes=None):
    boxes = proposal["cell_boxes"].float()
    exists = proposal["cell_existence_logits"].float().sigmoid()
    relations = proposal["relation_logits"].float().sigmoid()
    n = boxes.shape[1]
    pair = exists[:, :, None] * exists[:, None, :]
    pair = pair * ~torch.eye(n, dtype=torch.bool, device=boxes.device)
    a, b = boxes[:, :, None], boxes[:, None, :]
    iou, _ = box_metrics(a, b)
    delta = (a - b).abs()
    rows = torch.minimum(delta[..., 1], delta[..., 3])
    columns = torch.minimum(delta[..., 0], delta[..., 2])
    shared_edge = torch.stack(((a[..., 2] - b[..., 0]).abs(), (b[..., 2] - a[..., 0]).abs(),
                               (a[..., 3] - b[..., 1]).abs(), (b[..., 3] - a[..., 1]).abs()), -1).amin(-1)

    def weighted(values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1)

    coverage = boxes.sum() * 0
    if target_boxes is not None:
        points = (torch.arange(16, device=boxes.device, dtype=torch.float32) + 0.5) / 16
        y, x = torch.meshgrid(points, points, indexing="ij")
        xy = torch.stack((x, y), -1).reshape(-1, 2)
        mask = (xy[None, None] >= target_boxes[:, :, None, :2]) & (xy[None, None] <= target_boxes[:, :, None, 2:])
        required = mask.all(-1).any(1)
        low = (xy[None, None] - boxes[:, :, None, :2]) * 40
        high = (boxes[:, :, None, 2:] - xy[None, None]) * 40
        occupancy = (low.sigmoid() * high.sigmoid()).prod(-1) * exists[..., None]
        union = 1 - (1 - occupancy).prod(1)
        coverage = weighted(F.relu(0.9 - union), required)
    return dict(
        grid_coverage=coverage, cell_overlap=weighted(iou, pair),
        row_alignment=weighted(rows, pair * relations[..., 0]),
        column_alignment=weighted(columns, pair * relations[..., 1]),
        span_consistency=weighted(shared_edge, pair * relations[..., 2] * relations[..., 3]),
    )
