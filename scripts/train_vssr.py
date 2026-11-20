"""Train VSSR generations on immutable, project-disjoint reviewed splits."""

import argparse
from collections import Counter
from functools import partial
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from scripts.ground_truth import immutable, jbytes
from scripts.training_reference import OCR_TIER
from src.training.vssr import VSSRConfig, VSSR1, upgrade
from src.training.vssr.checkpoint import load_checkpoint, save_checkpoint
from src.training.vssr.dataset import ReviewedPilot, collate
from src.training.vssr.trainer import batch_losses, move_batch, train_step


def loader(dataset, config, batch_size, workers, device, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=partial(collate, config=config), persistent_workers=workers > 0,
                      pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed))


def reviewed_subset(dataset):
    return Subset(dataset, [i for i, row in enumerate(dataset.rows) if row.get("tier") != OCR_TIER])


def epoch(model, batches, device, bf16, optimizer=None):
    sums, counts = Counter(), Counter()
    confusion = torch.zeros(4, device=device, dtype=torch.long)
    start = time.monotonic()
    model.train(optimizer is not None)
    for batch in batches:
        size = len(batch["table_ids"])
        if optimizer is not None:
            losses = train_step(model, batch, optimizer, bf16)
        else:
            with torch.no_grad():
                out, parts = batch_losses(model, move_batch(batch, device), bf16)
                values = torch.stack(list(parts.values())).tolist()
                if not all(map(math.isfinite, values)):
                    raise FloatingPointError("Non-finite validation loss")
                losses = dict(zip(parts, values, strict=True))
                labels = [row["anomaly"] if row.get("anomaly") is not None else -1 for row in batch["targets"]]
                truth = torch.as_tensor(labels, device=device)
                pred = out["anomaly_logits"] >= 0
                confusion += torch.stack([(pred & (truth == 1)).sum(), ((~pred) & (truth == 1)).sum(),
                                          ((~pred) & (truth == 0)).sum(), (pred & (truth == 0)).sum()])
        counts["samples"] += size
        counts["batches"] += 1
        for key, value in losses.items():
            sums[key] += value * size
    if not counts["samples"]:
        raise ValueError("Empty training or validation partition")
    metrics = dict(samples=counts["samples"], batches=counts["batches"],
                   seconds=round(time.monotonic() - start, 3),
                   losses={k: v / counts["samples"] for k, v in sums.items()})
    if optimizer is None:
        tp, fn, tn, fp = confusion.tolist()
        metrics["anomaly"] = dict(tp=tp, fn=fn, tn=tn, fp=fp,
                                  balanced_accuracy=((tp / (tp + fn) + tn / (tn + fp)) / 2
                                                     if tp + fn and tn + fp else None))
    return metrics


def train_generation(model, train, validation, output, device, *, epochs, patience, min_delta, lr, bf16):
    output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    initial = epoch(model, validation, device, bf16)
    best_loss = initial["losses"]["total"]
    best_path = output / "epoch-000.pt"
    save_checkpoint(best_path, model, optimizer, 0)
    records = [dict(epoch=0, validation=initial, checkpoint=best_path.name)]
    significant_loss, stale, step = best_loss, 0, 0
    for number in range(1, epochs + 1):
        trained = epoch(model, train, device, bf16, optimizer)
        step += trained["batches"]
        measured = epoch(model, validation, device, bf16)
        loss = measured["losses"]["total"]
        row = dict(epoch=number, train=trained, validation=measured)
        if loss < best_loss:
            best_loss = loss
            best_path = output / f"epoch-{number:03d}.pt"
            save_checkpoint(best_path, model, optimizer, step)
            row["checkpoint"] = best_path.name
        if significant_loss - loss > min_delta:
            significant_loss, stale = loss, 0
        else:
            stale += 1
        records.append(row)
        print(json.dumps(dict(version=model.version, **row)), flush=True)
        if stale >= patience:
            break
    receipt = dict(version=model.version, best_checkpoint=best_path.name, best_validation_loss=best_loss,
                   stopped_early=stale >= patience, epochs_run=len(records) - 1,
                   patience=patience, min_delta=min_delta, history=records)
    immutable(output / "training.json", jbytes(receipt))
    best, _ = load_checkpoint(best_path, device, model.config, model.version)
    return best, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Workspace created by freeze_expansion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/vssr.yaml"))
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0 or args.lr <= 0 or args.min_delta < 0:
        parser.error("Invalid epochs, patience, batch size, workers, learning rate or minimum improvement")
    config, device = VSSRConfig.load(args.config), torch.device(args.device)
    train = ReviewedPilot(args.root, config, use_reference_boxes=False)
    validation = ReviewedPilot(args.root, config, split="validation", use_reference_boxes=False)
    if not train.allow_machine or not validation.allow_machine:
        parser.error("Formal training requires the confirmed 5k freeze")
    training_groups = {row["group_id"] for row in train.rows}
    if training_groups & {row["group_id"] for row in validation.rows}:
        raise ValueError("Project leakage between training and validation")
    if {row["signature"] for row in train.rows} & {row["signature"] for row in validation.rows}:
        raise ValueError("Reference leakage between training and validation")
    torch.manual_seed(args.seed)
    training_batches = loader(train, config, args.batch_size, args.workers, device, True, args.seed)
    reviewed_validation = reviewed_subset(validation)
    validation_batches = loader(reviewed_validation, config, args.batch_size, args.workers, device)
    args.output.mkdir(parents=True, exist_ok=False)
    immutable(args.output / "inputs.json", jbytes(dict(
        config_sha256=config.digest(), data_version=train.manifest_version, train_sha256=train.train_sha256,
        validation_sha256=validation.train_sha256, batch_size=args.batch_size, workers=args.workers,
        reviewed_validation_samples=len(reviewed_validation), device=str(device), bf16=args.bf16,
        seed=args.seed, learning_rate=args.lr,
    )))
    model = VSSR1(config, pretrained=True, weights=args.backbone_weights).to(device)
    for version in ("vssr-1", "vssr-2", "vssr-3"):
        if model.version != version:
            model = upgrade(model, version)
        batches = (loader(reviewed_subset(train), config, args.batch_size, args.workers, device, True, args.seed)
                   if version == "vssr-1" else training_batches)
        model, _ = train_generation(
            model, batches, validation_batches, args.output / version, device,
            epochs=args.epochs, patience=args.patience, min_delta=args.min_delta, lr=args.lr, bf16=args.bf16,
        )


if __name__ == "__main__":
    main()
