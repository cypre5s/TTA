"""Configuration and stable identities for VSSR checkpoints."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import yaml


@dataclass(frozen=True)
class VSSRConfig:
    backbone: str = "repvit_m1_0.dist_450e_in1k"
    backbone_revision: str = "0e20f9f1bfa9772e63b813835bab2d6f0d168f89"
    image_size: int = 1024
    hidden: int = 256
    heads: int = 8
    structure_layers: int = 4
    structure_ffn: int = 1024
    fusion_layers: int = 2
    dropout: float = 0.1
    visual_pools: tuple[int, ...] = (12, 8, 4, 2)
    max_tokens: int = 2048
    max_cells: int = 512
    max_text_bytes: int = 64
    cell_queries: int = 128
    nested_queries: int = 8
    child_queries: int = 24
    max_nesting_depth: int = 1
    freeze_backbone: bool = True
    unfreeze_last_stage: bool = False
    proposal_threshold: float = 0.5
    abstain_threshold: float = 0.5
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(
        anomaly=1.0, cell=1.0, bbox=2.0, relation=1.0, alignment=0.2,
        hierarchy=1.0, constraint=0.2, repair=0.5,
    ))

    def __post_init__(self):
        object.__setattr__(self, "visual_pools", tuple(self.visual_pools))
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        sizes = (
            self.hidden, self.heads, self.structure_layers, self.structure_ffn,
            self.fusion_layers, self.max_tokens, self.max_cells, self.max_text_bytes,
            self.cell_queries, self.nested_queries, self.child_queries, *self.visual_pools,
        )
        if min(sizes) < 1 or self.hidden % self.heads or len(self.visual_pools) != 4:
            raise ValueError("Invalid VSSR widths, heads, sequence limits or pyramid pools")
        if self.max_nesting_depth not in (1, 2):
            raise ValueError("VSSR supports one or two nested table levels")
        if not 0 <= self.dropout < 1 or not all(
            0 < x < 1 for x in (self.proposal_threshold, self.abstain_threshold)
        ):
            raise ValueError("Invalid dropout or probability threshold")
        defaults = {"anomaly", "cell", "bbox", "relation", "alignment", "hierarchy", "constraint", "repair"}
        if set(self.loss_weights) != defaults or any(w < 0 for w in self.loss_weights.values()):
            raise ValueError("Loss weights must cover the eight named nonnegative terms")

    @classmethod
    def load(cls, path: Path):
        return cls(**yaml.safe_load(path.read_text()))

    def as_dict(self):
        return asdict(self)

    def digest(self):
        return config_digest(self.as_dict())


def config_digest(values):
    body = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()
