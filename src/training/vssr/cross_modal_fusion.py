"""Each branch attends to the other using separate learned projections."""

import torch
from torch import nn


class FusionLayer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.visual_attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.structure_attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.visual_norm = nn.LayerNorm(d)
        self.structure_norm = nn.LayerNorm(d)
        self.visual_out = nn.LayerNorm(d)
        self.structure_out = nn.LayerNorm(d)
        self.visual_ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.structure_ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual, structure, structure_mask):
        v, s = self.visual_norm(visual), self.structure_norm(structure)
        visual = visual + self.dropout(self.visual_attention(v, s, s, key_padding_mask=~structure_mask, need_weights=False)[0])
        structure = structure + self.dropout(self.structure_attention(s, v, v, need_weights=False)[0])
        visual = visual + self.dropout(self.visual_ff(self.visual_out(visual)))
        structure = structure + self.dropout(self.structure_ff(self.structure_out(structure)))
        return visual, structure * structure_mask[..., None]


class CrossModalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            FusionLayer(config.hidden, config.heads, config.dropout) for _ in range(config.fusion_layers)
        )

    def forward(self, visual, tokens, cells, geometry, token_mask, cell_mask):
        structure = torch.cat((tokens, cells + geometry), dim=1)
        mask = torch.cat((token_mask, cell_mask), dim=1)
        for layer in self.layers:
            visual, structure = layer(visual, structure, mask)
        n = tokens.shape[1]
        return dict(visual=visual, tokens=structure[:, :n], cells=structure[:, n:], structure_mask=mask,
                    memory=torch.cat((visual, structure), dim=1))
