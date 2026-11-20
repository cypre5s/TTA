"""Nested region queries condition child cells on their selected parent."""

import torch
from torch import nn

from .cell_proposal import corners
from .relation_decoder import RelationDecoder


class HierarchyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden
        self.queries = nn.Embedding(config.nested_queries, d)
        layer = nn.TransformerDecoderLayer(d, config.heads, 2 * d, config.dropout, batch_first=True, norm_first=True)
        self.decode = nn.TransformerDecoder(layer, 1, norm=nn.LayerNorm(d))
        self.parent_query = nn.Linear(d, d)
        self.parent_key = nn.Linear(d, d)
        self.exists = nn.Linear(d, 1)
        self.region_box = nn.Linear(d, 4)
        self.child_queries = nn.Embedding(config.child_queries, d)
        self.child_visual = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.child_norm = nn.LayerNorm(d)
        self.child_exists = nn.Linear(d, 1)
        self.child_box = nn.Linear(d, 4)
        self.child_type = nn.Linear(d, 4)
        self.child_relations = RelationDecoder(d)

    def forward(self, fused, proposal):
        return self.decode_level(fused["visual"], proposal["cell_embeddings"])

    def decode_level(self, vision, parents):
        """Return region boxes in the same local coordinate frame as the parents."""
        b, _, d = parents.shape
        nested = self.decode(self.queries.weight[None].expand(b, -1, -1), parents)
        association = torch.bmm(self.parent_query(nested), self.parent_key(parents).transpose(1, 2)) / d**0.5
        parent_context = torch.bmm(association.float().softmax(-1).to(parents.dtype), parents)
        regions = corners(self.region_box(nested + parent_context))
        base = nested + parent_context
        n = base.shape[1]
        child = base[:, :, None] + self.child_queries.weight[None, None].to(base.dtype)
        flat = child.flatten(1, 2)
        flat = self.child_norm(flat + self.child_visual(flat, vision, vision, need_weights=False)[0])
        child = flat.reshape(b, n, -1, d)
        exists = self.exists(base).squeeze(-1)
        relations = self.child_relations(child.flatten(0, 1)).reshape(b, n, child.shape[2], child.shape[2], -1)
        return dict(
            nested_logits=exists, nested_probability=exists.float().sigmoid(), nested_region=regions,
            parent_logits=association, parent_cell_id=association.argmax(-1), child_table_region=regions,
            child_embeddings=child, child_cell_logits=self.child_exists(child).squeeze(-1),
            child_boxes=corners(self.child_box(child)), child_type_logits=self.child_type(child),
            child_relation_logits=relations,
        )
