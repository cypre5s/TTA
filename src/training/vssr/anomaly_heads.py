"""Predict global and local visual–structure disagreement."""

import torch
from torch import nn


class AnomalyHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden
        self.global_head = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 2))
        self.cell = nn.Linear(d, 1)
        self.region = nn.Linear(d, 1)
        self.edge = nn.Bilinear(d, d, 1)
        self.threshold = config.abstain_threshold

    def forward(self, fused, batch):
        vision, struct, cells = fused["visual"], fused["tokens"], fused["cells"]
        token_mask = batch["token_mask"][..., None]
        summary = (struct * token_mask).sum(1) / token_mask.sum(1).clamp_min(1)
        flags = batch["parser_error"].to(vision.dtype)[:, None]
        logits = self.global_head(torch.cat((vision.mean(1), summary, flags), dim=-1))
        # Compute the same bilinear map without materializing [B,N,N,D].
        left = torch.matmul(cells, self.edge.weight[0])
        relations = torch.bmm(left, cells.transpose(1, 2)) + self.edge.bias
        confidence = logits[:, 1].float().sigmoid()
        return dict(
            anomaly_logits=logits[:, 0], cell_anomaly_logits=self.cell(cells).squeeze(-1),
            missing_region_logits=self.region(vision).squeeze(-1), relation_anomaly_logits=relations,
            confidence_logits=logits[:, 1], confidence=confidence,
            abstain=(confidence < self.threshold) | batch["structure_truncated"],
        )
