"""Store versioned, config-bound VSSR checkpoints without replacing existing files."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .config import VSSRConfig, config_digest
from .model import build_model


def save_checkpoint(path, model, optimizer=None, step=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(
        format_version=1, model_version=model.version, config=model.config.as_dict(),
        config_sha256=model.config.digest(), backbone=model.visual_encoder.backbone.provenance,
        state_dict=model.state_dict(), optimizer=optimizer.state_dict() if optimizer is not None else None,
        step=step,
    )
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(record, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink()


def load_checkpoint(path, device="cpu", expected_config=None, expected_version=None):
    record = torch.load(path, map_location="cpu", weights_only=True)
    config = VSSRConfig(**record["config"])
    if record["format_version"] != 1 or config_digest(record["config"]) != record["config_sha256"]:
        raise ValueError("Checkpoint format or configuration hash mismatch")
    if expected_config is not None and config.digest() != expected_config.digest():
        raise ValueError("Checkpoint does not match the expected config")
    if expected_version is not None and record["model_version"] != expected_version:
        raise ValueError("Checkpoint does not match the expected VSSR generation")
    model = build_model(record["model_version"], config)
    model.load_state_dict(record["state_dict"], strict=True)
    model.visual_encoder.backbone.provenance = record["backbone"]
    meta = {k: value for k, value in record.items() if k != "state_dict"}
    return model.to(device).eval(), meta
