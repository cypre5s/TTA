"""Use official timm feature extractors with explicit checkpoint provenance."""

import hashlib
from pathlib import Path

from torch import nn


REPVIT = "repvit_m1_0.dist_450e_in1k"
REPVIT_REVISION = "0e20f9f1bfa9772e63b813835bab2d6f0d168f89"
REPVIT_SHA256 = "9350ed15b205ec9d4ebcde95333e8a9ad2ddda26510a358b57b81fd8a8fde055"
STRIDES = (4, 8, 16, 32)


class Backbone(nn.Module):
    def __init__(self, config, *, pretrained=False, weights: Path | None = None):
        super().__init__()
        import timm

        if config.backbone not in (REPVIT, "resnet18"):
            raise ValueError("Use RepViT-M1.0 or the explicit ResNet-18 comparison")
        self.provenance = {"timm": timm.__version__, "model": config.backbone, "pretrained": pretrained}
        overlay = None
        if pretrained:
            if config.backbone not in timm.list_models(config.backbone, pretrained=True):
                raise ValueError(f"No registered pretrained weights for {config.backbone}")
            if config.backbone == REPVIT:
                if weights is None:
                    from huggingface_hub import hf_hub_download

                    weights = Path(hf_hub_download(
                        f"timm/{REPVIT}", "model.safetensors", revision=config.backbone_revision,
                    ))
                self.provenance.update(repository=f"timm/{REPVIT}", revision=config.backbone_revision)
            if weights is not None:
                with Path(weights).open("rb") as stream:
                    self.provenance["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                if config.backbone == REPVIT and config.backbone_revision == REPVIT_REVISION:
                    if self.provenance["sha256"] != REPVIT_SHA256:
                        raise ValueError("RepViT weights do not match the pinned checkpoint")
                overlay = {"file": str(weights), "hf_hub_id": None, "url": None}
        elif weights is not None:
            raise ValueError("A weights path requires pretrained=True")
        self.encoder = timm.create_model(
            config.backbone, pretrained=pretrained, pretrained_cfg_overlay=overlay,
            features_only=True, out_indices=(0, 1, 2, 3) if config.backbone == REPVIT else (1, 2, 3, 4),
        )
        self.channels = tuple(self.encoder.feature_info.channels())
        self.reductions = tuple(self.encoder.feature_info.reduction())
        if self.reductions != STRIDES:
            raise ValueError(f"Backbone returned strides {self.reductions}, expected {STRIDES}")
        last = self.encoder.feature_info.module_name()[-1]
        modules = dict(self.encoder.named_modules())
        self.last_stage_name = last if last in modules else last.replace(".", "_")
        if self.last_stage_name not in modules:
            raise ValueError(f"Cannot locate final feature stage: {last}")
        self.freeze(config.freeze_backbone, config.unfreeze_last_stage)

    def freeze(self, whole=True, last_stage=False):
        self.frozen = whole
        self.last_stage_trainable = whole and last_stage
        self.encoder.requires_grad_(not whole)
        if self.last_stage_trainable:
            self.encoder.get_submodule(self.last_stage_name).requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
            if self.last_stage_trainable:
                self.encoder.get_submodule(self.last_stage_name).train(mode)
        return self

    def forward(self, images):
        return self.encoder(images)
