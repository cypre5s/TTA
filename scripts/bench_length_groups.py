"""Measure actual elapsed time of FIFO and length-grouped batches on the same crops."""

import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from scripts.bench_xiaomi import cfg, fit_image, out
from scripts.cloud_ocr import generate, save
from src.xiaomi_ocr.graph_decode import graph_linear_layers
from src.xiaomi_ocr.scheduling import order


def main():
    cohort = out / "xiaomi_scheduling_inputs.json"
    if not cohort.exists():
        ready = sorted(
            Path("data/local_ocr/ready/batch").glob("*.png"),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
        if len(ready) < 32:
            raise ValueError("Need 32 inputs.")
        # A contiguous FIFO window reflects actual mixed-length arrivals.
        start = (len(ready) - 32) // 2
        save(cohort, [str(p) for p in ready[start : start + 32]])
    paths = [Path(p) for p in json.loads(cohort.read_text())]
    pending = set(paths)
    grouped = []
    while pending:
        batch = order(pending, 8)[:8]
        pending.difference_update(batch)
        grouped.append(batch)
    fifo = [paths[i : i + 8] for i in range(0, len(paths), 8)]
    model = (
        AutoModelForImageTextToText.from_pretrained(
            cfg["checkpoint"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    proc = AutoProcessor.from_pretrained(cfg["checkpoint"], local_files_only=True)
    proc.tokenizer.padding_side = "left"
    trials = []
    baseline = None
    with graph_linear_layers(model):
        generate(model, proc, paths[:8], {**cfg, "max_new_tokens": 5}, fit_image)
        for name, batches in (("fifo", fifo), ("grouped", grouped)):
            rows = {}
            steps = 0
            torch.cuda.synchronize()
            start = time.perf_counter()
            for batch in batches:
                seq = generate(model, proc, batch, cfg, fit_image)
                steps += seq.shape[1]
                for path, tokens in zip(batch, seq):
                    val = proc.decode(
                        tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    rows[path.stem] = hashlib.sha256(val.encode()).hexdigest()
            seconds = time.perf_counter() - start
            if baseline is None:
                baseline = rows
            trials.append(
                dict(
                    mode=name,
                    seconds=seconds,
                    padded_steps=steps,
                    raw_sha256=rows,
                    exact_matches=sum(v == baseline[k] for k, v in rows.items()),
                    batches=[[p.stem for p in batch] for batch in batches],
                )
            )
            save(
                out / "length_groups_walltime.json",
                dict(
                    samples=len(paths),
                    trials=trials,
                    speedup=trials[0]["seconds"] / seconds,
                ),
            )
            print(name, seconds, steps, flush=True)


if __name__ == "__main__":
    main()
