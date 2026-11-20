"""Compare full greedy generations on fixed local crops with exact OTSL checks."""

import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import yaml
from scripts.cloud_ocr import generate, save
from src.xiaomi_ocr.fast_decode import variant

out = Path("reports/performance")
cfg = yaml.safe_load(Path("configs/xiaomi_local.yaml").read_text())
sys.path.insert(0, cfg["upstream"])
from evaluation_pipeline.cropping import fit_image


def run():
    cohort = out / "xiaomi_inputs.json"
    if cohort.exists():
        paths = [Path(p) for p in json.loads(cohort.read_text())]
    else:
        rows = sorted(
            (
                json.loads(p.read_text())
                for p in Path("data/tables/predictions").glob("*.json")
            ),
            key=lambda r: (r["output_tokens"], r["table_id"]),
        )
        paths = [
            Path("data/tables/crops")
            / (rows[int((i + 0.5) * len(rows) / 8)]["table_id"] + ".png")
            for i in range(8)
        ]
        save(cohort, [str(p) for p in paths])
    torch.cuda.set_per_process_memory_fraction(0.70)
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
    fast = []
    ref = None
    for name, size in [
        ("original", 4),
        ("original", 8),
        ("packed", 8),
        ("packed_conv", 8),
        ("original", 4),
    ]:
        with variant(model, name):
            generate(model, proc, paths[:1], {**cfg, "max_new_tokens": 4}, fit_image)
            torch.cuda.reset_peak_memory_stats()
            text = []
            ntok = 0
            per_batch = []
            start = time.perf_counter()
            for i in range(0, len(paths), size):
                t = time.perf_counter()
                seq = generate(model, proc, paths[i : i + size], cfg, fit_image)
                per_batch.append(time.perf_counter() - t)
                ntok += int((seq != proc.tokenizer.pad_token_id).sum())
                for row in seq:
                    raw = proc.decode(
                        row,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    text.append(hashlib.sha256(raw.encode()).hexdigest())
            sec = time.perf_counter() - start
        if ref is None:
            ref = text
        fast.append(
            dict(
                variant=name,
                batch_size=size,
                seconds=sec,
                seconds_per_batch=per_batch,
                output_tokens=ntok,
                tokens_per_s=ntok / sec,
                raw_sha256=text,
                exact_matches=sum(x == y for x, y in zip(text, ref)),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            )
        )
        save(
            out / "xiaomi_local_comparison.json",
            dict(
                device=torch.cuda.get_device_name(),
                inputs=[str(p) for p in paths],
                variants=fast,
            ),
        )
        print(name, size, sec, fast[-1]["exact_matches"], flush=True)


if __name__ == "__main__":
    run()
