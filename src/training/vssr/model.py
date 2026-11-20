"""Three progressively inherited VSSR architectures."""

from torch import nn

from src.training.repair import VOCAB
from .anomaly_heads import AnomalyHeads
from .cell_proposal import CellProposal
from .cross_modal_fusion import CrossModalFusion
from .geometry_encoder import GeometryEncoder
from .hierarchy_decoder import HierarchyDecoder
from .relation_decoder import RelationDecoder
from .structure_encoder import StructureEncoder
from .visual_encoder import VisualEncoder


class VSSR1(nn.Module):
    version = "vssr-1"

    def __init__(self, config, **backbone_options):
        super().__init__()
        self.config = config
        self.visual_encoder = VisualEncoder(config, **backbone_options)
        self.structure_encoder = StructureEncoder(config)
        self.geometry_encoder = GeometryEncoder(config.hidden)
        self.fusion = CrossModalFusion(config)
        self.anomaly = AnomalyHeads(config)

    def forward(self, batch):
        feature = self.visual_encoder(batch["images"])
        tokens, cells = self.structure_encoder(batch)
        geometry = self.geometry_encoder(batch["boxes"], batch["bbox_valid"], batch["bbox_source"])
        features = self.fusion(feature["tokens"], tokens, cells, geometry, batch["token_mask"], batch["cell_mask"])
        return dict(
            self.anomaly(features, batch), features=features, parser_error=batch["parser_error"],
            visual_xy=feature["xy"], backbone_shapes=feature["backbone_shapes"], fpn_shapes=feature["fpn_shapes"],
        )


class VSSR2(VSSR1):
    version = "vssr-2"

    def __init__(self, config, **backbone_options):
        super().__init__(config, **backbone_options)
        self.proposal = CellProposal(config)
        self.relations = RelationDecoder(config.hidden)
        self.repair = nn.Linear(config.hidden, VOCAB)

    def forward(self, batch):
        out = super().forward(batch)
        out.update(self.proposal(out["features"]))
        out["relation_logits"] = self.relations(out["cell_embeddings"])
        out["repair_logits"] = self.repair(out["features"]["tokens"])
        return out


class VSSR3(VSSR2):
    version = "vssr-3"

    def __init__(self, config, **backbone_options):
        super().__init__(config, **backbone_options)
        self.hierarchy = HierarchyDecoder(config)

    def forward(self, batch):
        out = super().forward(batch)
        out.update(self.hierarchy(out["features"], out))
        return out


MODELS = {model.version: model for model in (VSSR1, VSSR2, VSSR3)}


def build_model(version, config, **backbone_options):
    if version not in MODELS:
        raise ValueError(f"Unknown model version {version}")
    return MODELS[version](config, **backbone_options)


def upgrade(previous, version):
    if (previous.version, version) not in (("vssr-1", "vssr-2"), ("vssr-2", "vssr-3")):
        raise ValueError("Upgrade VSSR one generation at a time")
    result = build_model(version, previous.config)
    differences = result.load_state_dict(previous.state_dict(), strict=False)
    prefixes = ("proposal.", "relations.", "repair.") if version == "vssr-2" else ("hierarchy.",)
    if differences.unexpected_keys or not differences.missing_keys or not all(k.startswith(prefixes) for k in differences.missing_keys):
        raise ValueError("Upgrade would replace or lose inherited parameters")
    result.visual_encoder.backbone.provenance = dict(previous.visual_encoder.backbone.provenance)
    device = next(previous.parameters()).device
    return result.to(device).train(previous.training)
