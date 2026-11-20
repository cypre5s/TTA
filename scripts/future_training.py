"""Prepare reviewed inputs for official PaddleX or probe Xiaomi LoRA on one table."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.evaluate import read_split
from scripts.ground_truth import sha


def layout(args):
    config = json.loads(Path("configs/layout_provenance.json").read_text())
    names = list(config["category_mapping"].values())
    train_doc = set()
    for split in ["train", "val"]:
        path = args.dataset / "annotations" / f"instance_{split}.json"
        d = json.loads(path.read_text())
        cats = sorted(d["categories"], key=lambda x: x["id"])
        if [c["name"] for c in cats] != names:
            raise ValueError(
                "Expected all 25 original PP-DocLayoutV3 categories in their original order"
            )
        ids = {r["id"] for r in cats}
        images = {r["id"]: r for r in d["images"]}
        if not images or not d["annotations"]:
            raise ValueError("Empty layout partition")
        for img in images.values():
            if (
                img.get("reviewer_type") != "human"
                or not img.get("reviewer")
                or not img.get("project_group")
            ):
                raise ValueError(
                    "Layout GT requires reviewer and project group on every image"
                )
            if split == "train":
                train_doc.add(img["project_group"])
            elif img["project_group"] in train_doc:
                raise ValueError("Project leakage")
            if not (args.dataset / "images" / img["file_name"]).is_file():
                raise FileNotFoundError(img["file_name"])
        for ann in d["annotations"]:
            if (
                ann["image_id"] not in images
                or ann["category_id"] not in ids
                or not ann.get("segmentation")
                or "read_order" not in ann
            ):
                raise ValueError("Invalid annotated layout region")
    cli = args.paddlex / "main.py"
    cfg = args.paddlex / "paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml"
    if not cli.is_file() or not cfg.is_file():
        raise FileNotFoundError("PaddleX training checkout required")
    if args.train and (not args.weights or not args.weights.is_file()):
        raise ValueError("Provide approved local training-format pretrained weights")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in ["check_dataset", "train"] if args.train else ["check_dataset"]:
        cmd = [
            sys.executable,
            str(cli.resolve()),
            "-c",
            str(cfg.resolve()),
            "-o",
            f"Global.mode={mode}",
            "-o",
            f"Global.dataset_dir={args.dataset.resolve()}",
            "-o",
            f"Global.output={args.output.resolve()}",
            "-o",
            "Global.device=gpu:0",
            "-o",
            "Train.num_classes=25",
            "-o",
            "Train.batch_size=1",
        ]
        if args.weights:
            cmd += ["-o", f"Train.pretrain_weight_path={args.weights.resolve()}"]
        with (args.output / (mode + ".log")).open("w") as log:
            subprocess.run(
                cmd, cwd=args.paddlex, stdout=log, stderr=subprocess.STDOUT, check=True
            )


def xiaomi_truth(row):
    ref = Path(row["gt"]).read_bytes()
    if sha(ref) != row["gt_sha256"]:
        raise ValueError("Changed reference")
    gt = json.loads(ref)
    if (
        gt["reviewer_type"] != "human"
        or gt["decision"] != "approve"
        or gt["table_id"] != row["table_id"]
    ):
        raise ValueError("Requires a matching human-approved training reference")
    image = Path("data/tables/crops") / (gt["table_id"] + ".png")
    if sha(image.read_bytes()) != gt["image_sha256"]:
        raise ValueError("Image hash mismatch")
    return gt, image


def load_lora(model_path, rank):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).cuda()
    targets = [
        n
        for n, _ in model.named_modules()
        if n.startswith("model.language_model.")
        and n.endswith(("self_attn.q_proj", "self_attn.v_proj"))
    ]
    if not targets:
        raise ValueError("Unsupported model architecture")
    model = get_peft_model(
        model, LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=targets)
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    return processor, model, targets


def xiaomi_loss(processor, model, example, cfg):
    import torch
    from PIL import Image
    from evaluation_pipeline.cropping import fit_image

    gt, image = example
    with Image.open(image) as source:
        rgb = fit_image(source.convert("RGB"))
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cfg["prompt"]},
                {"type": "image", "image": rgb},
            ],
        }
    ]
    inp = processor.apply_chat_template(
        chat,
        tokenize=True,
        enable_thinking=False,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    target = processor.tokenizer.encode(
        gt["corrected_otsl"], add_special_tokens=False
    ) + [processor.tokenizer.eos_token_id]
    n = inp.input_ids.shape[1]
    ids = torch.tensor([target])
    inp["input_ids"] = torch.cat([inp.input_ids, ids], 1)
    inp["attention_mask"] = torch.cat([inp.attention_mask, torch.ones_like(ids)], 1)
    if "mm_token_type_ids" in inp:
        inp["mm_token_type_ids"] = torch.cat(
            [inp.mm_token_type_ids, torch.zeros_like(ids)], 1
        )
    labels = inp.input_ids.clone()
    labels[:, :n] = -100
    return model(
        **inp.to("cuda", torch.bfloat16), labels=labels.cuda(), use_cache=False
    ).loss


def xiaomi(args):
    import hashlib
    import tempfile
    import random
    import torch
    import transformers
    import yaml

    training = args.cmd == "xiaomi-train"
    if args.output.exists():
        raise ValueError("Use a fresh output path")
    if args.rank not in (8, 16, 32) or (
        training and (args.epochs < 1 or args.accumulation < 1)
    ):
        raise ValueError("Invalid training configuration")
    cfg = yaml.safe_load(Path("configs/xiaomi_local.yaml").read_text())
    sys.path.insert(0, cfg["upstream"])
    rows = read_split(args.dataset, "train")
    split_sha = sha((args.dataset / "manifest.json").read_bytes())
    with (args.model / "model.safetensors").open("rb") as stream:
        weight_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if training:
        probe = json.loads(args.probe.read_text())
        if (
            probe.get("status") != "COMPLETED"
            or probe.get("rank") != args.rank
            or probe.get("split_sha256") != split_sha
            or probe.get("weight_sha256") != weight_sha
        ):
            raise ValueError(
                "Need a successful probe on this split, checkpoint and LoRA rank"
            )
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    record = dict(
        sample=rows[0]["table_id"],
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        rank=args.rank,
        precision="bf16",
        status="RUNNING",
        split_sha256=split_sha,
        weight_sha256=weight_sha,
        seed=args.seed,
    )
    start = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        processor, model, targets = load_lora(args.model, args.rank)
        record["target_modules"] = targets
        if not training:
            loss = xiaomi_loss(processor, model, xiaomi_truth(rows[0]), cfg)
            loss.backward()
            gradients = [
                p.grad
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if (
                not gradients
                or not all(torch.isfinite(g).all() for g in gradients)
                or not any(g.abs().max().item() > 0 for g in gradients)
            ):
                raise ValueError("No finite, nonzero adapter gradient")
            record["loss"] = float(loss.detach())
        else:
            from torch.optim import AdamW

            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable, lr=args.learning_rate)
            history = []
            for epoch in range(args.epochs):
                rng.shuffle(rows)
                for first in range(0, len(rows), args.accumulation):
                    batch = rows[first : first + args.accumulation]
                    optimizer.zero_grad(set_to_none=True)
                    total = 0.0
                    for row in batch:
                        loss = xiaomi_loss(processor, model, xiaomi_truth(row), cfg)
                        if not torch.isfinite(loss):
                            raise ValueError(f"Invalid loss on {row['table_id']}")
                        (loss / len(batch)).backward()
                        total += float(loss.detach())
                    grad = torch.nn.utils.clip_grad_norm_(
                        trainable, 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    history.append(
                        dict(
                            epoch=epoch + 1,
                            samples=len(batch),
                            loss=total / len(batch),
                            gradient_norm=float(grad),
                        )
                    )
            record.update(
                optimizer_steps=len(history),
                history=history,
                training_tables=len(rows),
                epochs=args.epochs,
                accumulation=args.accumulation,
                learning_rate=args.learning_rate,
            )
        record.update(
            status="COMPLETED",
            seconds=time.perf_counter() - start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        )
        if training:
            # Publish adapter and its training provenance together.
            with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
                folder = Path(tmp) / "adapter"
                model.save_pretrained(folder)
                (folder / "training_report.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2)
                )
                if not (folder / "adapter_model.safetensors").is_file():
                    raise ValueError("Missing adapter weights")
                folder.rename(args.output)
    except Exception as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        if training:
            args.output.with_name(args.output.name + ".failed.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2)
            )
        raise
    finally:
        if not training:
            args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("layout")
    q.add_argument("--paddlex", type=Path, required=True)
    q.add_argument("--weights", type=Path)
    q.add_argument("--train", action="store_true")
    t = sub.add_parser("xiaomi-probe")
    fit = sub.add_parser("xiaomi-train")
    fit.add_argument("--probe", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=1)
    fit.add_argument("--accumulation", type=int, default=8)
    fit.add_argument("--learning-rate", type=float, default=1e-4)
    for cmd in (t, fit):
        cmd.add_argument("--model", type=Path, required=True)
        cmd.add_argument("--rank", type=int, default=8)
        cmd.add_argument("--seed", type=int, default=42)
    for cmd in (q, t, fit):
        cmd.add_argument("--dataset", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    layout(args) if args.cmd == "layout" else xiaomi(args)
