"""Multilabel cell relations, including an explicit no-relation class."""

import torch
from torch import nn


RELATIONS = ("SAME_ROW", "SAME_COLUMN", "ADJACENT", "SPAN", "NONE")


def grid_edges(cells):
    if not cells:
        return torch.zeros(0, 0, len(RELATIONS), dtype=torch.bool)
    geometry = torch.tensor([(c.row, c.column, c.rowspan, c.colspan) for c in cells])
    row, col, rs, cs = geometry.unbind(1)
    end_row, end_col = row + rs, col + cs
    same_r = (row[:, None] < end_row[None]) & (row[None] < end_row[:, None])
    same_c = (col[:, None] < end_col[None]) & (col[None] < end_col[:, None])
    row_touch = (end_row[:, None] == row[None]) | (end_row[None] == row[:, None])
    col_touch = (end_col[:, None] == col[None]) | (end_col[None] == col[:, None])
    close = (same_r & col_touch) | (same_c & row_touch)
    span = close & ((same_r & (rs[:, None] != rs[None])) | (same_c & (cs[:, None] != cs[None])))
    edges = torch.stack((same_r, same_c, close, span, ~(same_r | same_c | close | span)), dim=-1)
    return edges & ~torch.eye(len(cells), dtype=torch.bool)[..., None]


class RelationDecoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        rank = max(8, width // 4)
        self.heads = len(RELATIONS)
        self.left = nn.Linear(width, self.heads * rank)
        self.right = nn.Linear(width, self.heads * rank)
        self.bias = nn.Parameter(torch.zeros(self.heads))
        self.rank = rank

    def forward(self, embeddings):
        shape = (*embeddings.shape[:2], self.heads, self.rank)
        left, right = self.left(embeddings).reshape(shape), self.right(embeddings).reshape(shape)
        out = torch.einsum("bikr,bjkr->bijk", left, right) / self.rank**0.5
        return (out + out.transpose(1, 2)) * 0.5 + self.bias
