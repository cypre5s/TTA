"""Encode OTSL bytes and a typed cell graph while preserving parser failures."""

import torch
from torch import nn

from src.training.repair import VOCAB


class StructureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden
        self.token = nn.Embedding(VOCAB, d, padding_idx=0)
        self.position = nn.Embedding(config.max_tokens, d)
        self.parser_state = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=config.structure_ffn,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.structure_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False,
        )
        self.cell_text = nn.Embedding(257, d, padding_idx=0)
        self.grid = nn.Linear(5, d)
        self.cell_type = nn.Embedding(4, d)
        self.graph = nn.Linear(5 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, batch):
        ids = batch["tokens"]
        positions = torch.arange(ids.shape[1], device=ids.device)
        values = self.token(ids) + self.position(positions)
        values = values + self.parser_state(batch["parser_error"].long())[:, None]
        tokens = self.encoder(values, src_key_padding_mask=~batch["token_mask"])
        text_ids = batch["cell_text"]
        texts = self.cell_text(text_ids).sum(dim=2) / text_ids.ne(0).sum(dim=2).clamp_min(1)[..., None]
        nodes = self.grid(batch["cell_grid"]) + texts + self.cell_type(batch["cell_type"])
        neighbors = [nodes]
        for k in range(4):
            edges = batch["cell_edges"][..., k].to(nodes.dtype)
            neighbors.append(torch.bmm(edges, nodes) / edges.sum(-1, keepdim=True).clamp_min(1))
        cells = self.norm(nodes + self.graph(torch.cat(neighbors, dim=-1)))
        return tokens, cells * batch["cell_mask"][..., None]
