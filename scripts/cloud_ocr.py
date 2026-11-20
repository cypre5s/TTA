"""Original checkpoint, local Transformers image-text generation, immutable outputs."""

import argparse
from dataclasses import asdict
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import time
import subprocess

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import yaml


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(path)


def generate(model, proc, batch, cfg, fit_image):
    chats = []
    for path in batch:
        with Image.open(path) as img:
            rgb = fit_image(img.convert("RGB"))
        chats.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cfg["prompt"]},
                        {"type": "image", "image": rgb},
                    ],
                }
            ]
        )
    inputs = proc.apply_chat_template(
        chats,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to("cuda", torch.bfloat16)
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=cfg["max_new_tokens"],
            use_cache=True,
        )
    return ids[:, inputs.input_ids.shape[1] :].cpu()


def tune(model, proc, cfg, fit_image, ready_dir, result_dir):
    report = Path("logs/local/ocr_batch_comparison.json")
    if report.exists():
        return json.loads(report.read_text())["selected"]
    ready = {p.stem: p for p in ready_dir.glob("*/*.png")}
    records = [
        json.loads(p.read_text()) for p in result_dir.glob("*.json") if p.stem in ready
    ]
    if len(records) < 8:
        return cfg["batch_size"]
    records.sort(key=lambda x: (x["output_tokens"], x["table_id"]))
    paths = [
        ready[records[round((i + 0.5) * len(records) / 8) - 1]["table_id"]]
        for i in range(8)
    ]
    generate(model, proc, paths[:1], cfg, fit_image)
    timings = {4: [], 8: []}
    hashes = {}
    equal = True
    for size in [4, 8, 8, 4]:
        raw = []
        start = time.perf_counter()
        for i in range(0, 8, size):
            for row in generate(model, proc, paths[i : i + size], cfg, fit_image):
                val = proc.decode(
                    row, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                raw.append(hashlib.sha256(val.encode()).hexdigest())
        timings[size].append(time.perf_counter() - start)
        if hashes and raw != next(iter(hashes.values())):
            equal = False
        hashes[size] = raw
    ratio = sum(timings[4]) / sum(timings[8])
    selected = 8 if equal and ratio > 1.05 else 4
    save(
        report,
        dict(
            sample_ids=[p.stem for p in paths],
            timings_seconds=timings,
            raw_sha256=hashes,
            raw_identical=equal,
            speedup=ratio,
            selected=selected,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=1000)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/xiaomi_local.yaml"))
    p.add_argument("--ready", type=Path, default=Path("data/local_ocr/ready"))
    p.add_argument("--output", type=Path, default=Path("data/tables/predictions"))
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if len(list(args.output.glob("*.json"))) >= args.target:
        return
    sys.path.insert(0, cfg["upstream"])
    from evaluation_pipeline.cropping import fit_image
    from postprocess.otsl import (
        TAG_RE,
        parse_grid,
        pad_rows,
        resolve_cells,
        render_html,
    )

    print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)
    if torch.cuda.mem_get_info()[0] // 2**20 < cfg["min_free_mib"]:
        raise SystemExit("Insufficient free GPU memory")
    torch.cuda.set_per_process_memory_fraction(cfg["gpu_memory_fraction"])
    proc = AutoProcessor.from_pretrained(cfg["checkpoint"], local_files_only=True)
    proc.tokenizer.padding_side = "left"
    model = (
        AutoModelForImageTextToText.from_pretrained(
            cfg["checkpoint"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    selected_variant = cfg.get("runtime_variant", "original")
    if selected_variant != "original":
        from contextlib import ExitStack
        import atexit

        from src.xiaomi_ocr.fast_decode import variant

        stack = ExitStack()
        base_variant, _, deep = selected_variant.partition("+")
        stack.enter_context(variant(model, base_variant))
        if deep:
            kernels = (
                deep.removesuffix("_graph")
                .removesuffix("_stack")
                .removesuffix("_hybrid")
            )
            if kernels not in {"graph", "stack", "hybrid"}:
                from src.xiaomi_ocr.decode_kernels import accelerated_decode

                stack.enter_context(accelerated_decode(model, kernels))
            if deep == "graph" or deep.endswith("_graph"):
                from src.xiaomi_ocr.graph_decode import graph_linear_layers

                stack.enter_context(graph_linear_layers(model))
            elif deep == "stack" or deep.endswith("_stack"):
                from src.xiaomi_ocr.graph_decode import graph_linear_stacks

                stack.enter_context(graph_linear_stacks(model))
            elif deep == "hybrid" or deep.endswith("_hybrid"):
                from src.xiaomi_ocr.attention_graph import graph_hybrid_decode

                stack.enter_context(graph_hybrid_decode(model))
        atexit.register(stack.close)
    with (Path(cfg["checkpoint"]) / "model.safetensors").open("rb") as weights:
        model_sha = hashlib.file_digest(weights, "sha256").hexdigest()
    if (
        model_sha
        != json.loads(Path("configs/xiaomi_provenance.json").read_text())[
            "weight_sha256"
        ]
    ):
        raise ValueError("Official Xiaomi checkpoint hash differs")
    output = args.output
    output.mkdir(exist_ok=True, parents=True)
    completed = set(x.stem for x in output.glob("*.json"))
    size = (
        tune(model, proc, cfg, fit_image, args.ready, args.output)
        if args.tune
        else cfg["batch_size"]
    )
    preferred_size = size
    batches_since_oom = 0
    grow_after = 12
    while len(completed) < args.target:
        pending = sorted(
            (p for p in args.ready.glob("*/*.png") if p.stem not in completed),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
        if not pending:
            if (args.ready / "DRAIN").exists():
                break
            time.sleep(4)
            continue
        if cfg.get("bucket_by_length"):
            from src.xiaomi_ocr.scheduling import order

            pending = order(pending, size)
        batch = pending[: min(args.target - len(completed), size)]
        start = time.monotonic()
        try:
            rows = generate(model, proc, batch, cfg, fit_image)
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            size = max(1, size // 2)
            # Drop captured buffers that belong to this abandoned request.
            from src.xiaomi_ocr.graph_decode import GraphedLayer

            for module in model.modules():
                if isinstance(module, GraphedLayer):
                    module.reset()
            torch.cuda.empty_cache()
            batches_since_oom = 0
            grow_after = min(96, grow_after * 2)
            print(f"OOM, retrying with batch size {size}", flush=True)
            continue
        elapsed = time.monotonic() - start
        for path, row in zip(batch, rows):
            raw = proc.decode(
                row, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            tags, texts = parse_grid(pad_rows(raw))
            cells = resolve_cells(tags, texts)
            widths = [
                len(TAG_RE.findall(line)) for line in raw.split("<nl>") if line.strip()
            ]
            eos = proc.tokenizer.eos_token_id
            ended = eos in row.tolist()
            height = len(tags)
            width = max(map(len, tags), default=0)
            coverage = [[0] * width for _ in range(height)]
            for c in cells:
                for r in range(c.row, c.row + c.rowspan):
                    for k in range(c.column, c.column + c.colspan):
                        coverage[r][k] += 1
            errors = []
            if len(set(widths)) != 1:
                errors.append("nonrectangular_rows")
            if not cells or not raw.strip().startswith(("<ecel>", "<fcel>")):
                errors.append("invalid_start_or_empty")
            if any(x != 1 for line in coverage for x in line):
                errors.append("grid_coverage")
            if not ended:
                errors.append("truncated_or_missing_eos")
            grid = [asdict(c) for c in cells]
            for c in cells:
                c.text = escape(c.text)
            save(
                output / (path.stem + ".json"),
                dict(
                    table_id=path.stem,
                    model=cfg["model"],
                    weight_sha256=model_sha,
                    image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    raw_otsl=raw,
                    device=torch.cuda.get_device_name(),
                    runtime_variant=selected_variant,
                    html=render_html(cells, height, width),
                    cells=grid,
                    raw_row_widths=widths,
                    structure_checks=errors,
                    review_status="pending",
                    generation_seconds_batch=elapsed,
                    batch_size=len(batch),
                    batch_table_ids=[item.stem for item in batch],
                    precision="bfloat16",
                    decoding="greedy",
                    max_new_tokens=cfg["max_new_tokens"],
                    output_tokens=int((row != proc.tokenizer.pad_token_id).sum()),
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                ),
            )
            completed.add(path.stem)
        print(f"{len(completed)} tables, {elapsed:.1f} seconds", flush=True)
        batches_since_oom += 1
        if size < preferred_size and batches_since_oom >= grow_after:
            size = min(preferred_size, size * 2)
            batches_since_oom = 0


if __name__ == "__main__":
    main()
