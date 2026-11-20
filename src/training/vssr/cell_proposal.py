"""Cell queries, normalized boxes and mask-aware Hungarian assignment."""

import torch
from torch import nn


def corners(raw):
    first, second = raw.sigmoid().chunk(2, dim=-1)
    return torch.cat((torch.minimum(first, second), torch.maximum(first, second)), dim=-1)


def area(box):
    return (box[..., 2:] - box[..., :2]).clamp_min(0).prod(dim=-1)


def box_metrics(first, second):
    intersection = (
        torch.minimum(first[..., 2:], second[..., 2:]) - torch.maximum(first[..., :2], second[..., :2])
    ).clamp_min(0).prod(dim=-1)
    union = area(first) + area(second) - intersection
    enclosing = (
        torch.maximum(first[..., 2:], second[..., 2:]) - torch.minimum(first[..., :2], second[..., :2])
    ).clamp_min(0).prod(dim=-1)
    iou = intersection / union.clamp_min(1e-7)
    return iou, iou - (enclosing - union) / enclosing.clamp_min(1e-7)


class CellProposal(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden
        self.queries = nn.Embedding(config.cell_queries, d)
        layer = nn.TransformerDecoderLayer(
            d, config.heads, 2 * d, config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decode = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(d))
        self.existence = nn.Linear(d, 1)
        self.box = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.types = nn.Linear(d, 4)

    def forward(self, fused):
        memory = fused["memory"]
        visual_mask = torch.ones(fused["visual"].shape[:2], dtype=torch.bool, device=memory.device)
        mask = torch.cat((visual_mask, fused["structure_mask"]), dim=1)
        queries = self.queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        cells = self.decode(queries, memory, memory_key_padding_mask=~mask)
        return dict(cell_embeddings=cells, cell_existence_logits=self.existence(cells).squeeze(-1),
                    cell_boxes=corners(self.box(cells)), cell_type_logits=self.types(cells))


@torch.no_grad()
def hungarian(existence, boxes, labels, target_boxes, box_mask, target_types=None, type_mask=None):
    """Return prediction and target indices. Only annotated coordinates affect cost."""
    from scipy.optimize import linear_sum_assignment

    cost = -existence.float().sigmoid()[:, None].expand(-1, len(target_boxes)).clone()
    if labels is not None and target_types is not None:
        type_cost = -labels.float().softmax(-1)[:, target_types]
        cost += type_cost * type_mask[None] if type_mask is not None else type_cost
    safe = torch.where(box_mask[:, None], target_boxes, 0).float()
    bbox = torch.cdist(boxes.float(), safe, p=1)
    _, giou = box_metrics(boxes.float()[:, None], safe[None])
    cost += (5 * bbox - 2 * giou) * box_mask[None]
    i, j = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(i, device=boxes.device), torch.as_tensor(j, device=boxes.device)
