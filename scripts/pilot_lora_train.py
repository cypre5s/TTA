"""Train language-side Q/V LoRA on frozen training projects, with real validation loss."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
import yaml
from PIL import Image
from scripts.evaluate import read_split
from scripts.future_training import load_lora, xiaomi_truth
from scripts.ground_truth import immutable, jbytes, sha


def target_loss(head, hidden, labels, chunk_size=128):
    # Only assistant tokens contribute to the standard shifted causal loss.
    # Recompute each vocabulary projection during backward to bound peak memory.
    from torch.utils.checkpoint import checkpoint

    targets = labels[:, 1:]
    selected = targets.ne(-100)
    states = hidden[:, :-1][selected]
    expected = targets[selected]

    def summed(states, tokens):
        return torch.nn.functional.cross_entropy(
            head(states).float(), tokens, reduction="sum"
        )

    losses = []
    for h, t in zip(states.split(chunk_size), expected.split(chunk_size)):
        value = (
            checkpoint(summed, h, t, use_reentrant=False)
            if torch.is_grad_enabled()
            else summed(h, t)
        )
        losses.append(value)
    return torch.stack(losses).sum() / len(expected)


def batch_loss(proc, model, examples, cfg, fit):
    parts = []
    for gt, image in examples:
        with Image.open(image) as im:
            rgb = fit(im.convert("RGB"))
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": cfg["prompt"]},
                    {"type": "image", "image": rgb},
                ],
            }
        ]
        prompt = proc.apply_chat_template(
            chat,
            tokenize=True,
            enable_thinking=False,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        target = torch.tensor(
            proc.tokenizer.encode(gt["corrected_otsl"], add_special_tokens=False)
            + [proc.tokenizer.eos_token_id]
        )
        parts.append((prompt, target))
    width = max(p.input_ids.shape[1] + len(target) for p, target in parts)
    ids = torch.full((len(parts), width), proc.tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    types = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (p, target) in enumerate(parts):
        n = p.input_ids.shape[1]
        end = n + len(target)
        ids[i, :n] = p.input_ids[0]
        ids[i, n:end] = target
        mask[i, :end] = 1
        labels[i, n:end] = target
        if "mm_token_type_ids" in p:
            types[i, :n] = p.mm_token_type_ids[0]
    args = {
        "input_ids": ids.cuda(),
        "attention_mask": mask.cuda(),
        "use_cache": False,
        "pixel_values": torch.cat([p.pixel_values for p, _ in parts])
        .cuda()
        .to(torch.bfloat16),
        "image_grid_thw": torch.cat([p.image_grid_thw for p, _ in parts]).cuda(),
    }
    if "mm_token_type_ids" in parts[0][0]:
        args["mm_token_type_ids"] = types.cuda()
    base = model.get_base_model()
    hidden = base.model(**args).last_hidden_state
    return target_loss(base.lm_head, hidden, labels.cuda())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/splits/ground_truth_v1"))
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--effective", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    out = args.output
    if (out / "training.json").exists():
        return
    torch.manual_seed(42)
    rng = random.Random(42)
    cfg = yaml.safe_load(Path("configs/xiaomi_local.yaml").read_text())
    sys.path.insert(0, cfg["upstream"])
    from evaluation_pipeline.cropping import fit_image

    with (Path(cfg["checkpoint"]) / "model.safetensors").open("rb") as f:
        weight = hashlib.file_digest(f, "sha256").hexdigest()
    dataset_hash = sha((args.dataset / "manifest.json").read_bytes())
    probe = json.loads(args.probe.read_bytes())
    if probe.get("status") != "COMPLETED" or (
        probe.get("split_sha256"),
        probe.get("weight_sha256"),
        probe.get("rank"),
    ) != (dataset_hash, weight, args.rank):
        raise ValueError(
            "A finite backward pass on this model, rank and dataset is required"
        )
    proc, model, targets = load_lora(cfg["checkpoint"], args.rank)
    tr = [xiaomi_truth(r) for r in read_split(args.dataset, "train")]
    val = [xiaomi_truth(r) for r in read_split(args.dataset, "validation")]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=8e-5)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Refusing to mix training attempts")
    start = time.monotonic()
    history = []
    best = float("inf")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        # Equal-length buckets reduce padding; shuffle the bucket order every epoch.
        ordered = sorted(tr, key=lambda x: len(x[0]["corrected_otsl"]))
        chunks = [
            ordered[i : i + args.effective]
            for i in range(0, len(ordered), args.effective)
        ]
        rng.shuffle(chunks)
        train_loss = 0
        for step, chunk in enumerate(chunks):
            rng.shuffle(chunk)
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(chunk), args.micro):
                batch = chunk[i : i + args.micro]
                loss = batch_loss(proc, model, batch, cfg, fit_image)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss * len(batch) / len(chunk)).backward()
                train_loss += float(loss.detach()) * len(batch)
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 5 == 0:
                print(
                    "epoch", epoch + 1, "step", step + 1, "of", len(chunks), flush=True
                )
        total = 0.0
        model.eval()
        by_length = sorted(val, key=lambda x: len(x[0]["corrected_otsl"]))
        with torch.no_grad():
            for i in range(0, len(by_length), args.micro):
                b = by_length[i : i + args.micro]
                loss = batch_loss(proc, model, b, cfg, fit_image)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite validation loss")
                total += float(loss) * len(b)
        v = total / len(val)
        epoch_dir = out / f"epoch_{epoch + 1:02d}"
        model.save_pretrained(epoch_dir)
        w = epoch_dir / "adapter_model.safetensors"
        if not w.is_file():
            raise ValueError("Missing LoRA adapter")
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=train_loss / len(tr),
                validation_loss=v,
                weight_sha256=sha(w.read_bytes()),
                directory=str(epoch_dir),
            )
        )
        if v < best:
            best = v
        print("validation", v, flush=True)
    selected = min(history, key=lambda x: x["validation_loss"])
    (out / "best").symlink_to(Path(selected["directory"]).name)
    (out / "last").symlink_to(Path(history[-1]["directory"]).name)
    immutable(
        out / "training.json",
        jbytes(
            dict(
                history=history,
                selected_epoch=selected["epoch"],
                epochs=args.epochs,
                micro_batch=args.micro,
                effective_batch=args.effective,
                train=len(tr),
                validation=len(val),
                rank=args.rank,
                seed=42,
                lr=8e-5,
                loss_implementation="shifted target-only chunked vocabulary cross entropy",
                trainable_parameters=sum(p.numel() for p in params),
                target_modules=targets,
                vision_encoder_frozen=True,
                dataset_sha256=dataset_hash,
                base_model_sha256=weight,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                seconds=time.monotonic() - start,
            )
        ),
    )


if __name__ == "__main__":
    main()
