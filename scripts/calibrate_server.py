"""Select a GPU batch size on validation inputs before opening the test split."""

import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import yaml

from scripts.baseline_local import document_prompt
from scripts.cloud_ocr import generate
from scripts.ground_truth import immutable, jbytes, sha


def main():
    run = Path("experiments/baseline/original_v1/calibration.json")
    if run.is_file():
        return
    cfg = yaml.safe_load(Path("configs/xiaomi_local.yaml").read_text())
    upstream = Path(cfg["upstream"])
    sys.path.insert(0, str(upstream))
    from evaluation_pipeline.cropping import fit_image

    samples = json.loads(
        Path("data/annotation/phase3_v1/validation_calibration.json").read_bytes()
    )
    valid = {
        json.loads(s)["table_id"]
        for s in Path("data/splits/ground_truth_v1/validation.jsonl")
        .read_bytes()
        .splitlines()
    }
    if not all(
        i["table_id"] in valid
        and sha(Path(i["image"]).read_bytes()) == i["image_sha256"]
        for items in samples.values()
        for i in items
    ):
        raise ValueError("Calibration split or input hash")
    proc = AutoProcessor.from_pretrained(cfg["checkpoint"], local_files_only=True)
    proc.tokenizer.padding_side = "left"
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
    prompts = {"B0": cfg["prompt"], "B0-E2E": document_prompt(upstream)}
    selected = {}
    measure = {}
    for arm, sizes in [("B0", (16, 32)), ("B0-E2E", (8, 16))]:
        paths = [Path(i["image"]) for i in samples[arm]]
        if len(paths) != max(sizes):
            raise ValueError("Unexpected validation count")
        gen_cfg = dict(prompt=prompts[arm], max_new_tokens=96)
        generate(
            model,
            proc,
            paths[:1],
            dict(prompt=prompts[arm], max_new_tokens=8),
            fit_image,
        )
        measure[arm] = []
        reference = None
        for size in sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            try:
                outputs = []
                for pos in range(0, len(paths), size):
                    ids = generate(
                        model, proc, paths[pos : pos + size], gen_cfg, fit_image
                    )
                    outputs.extend(
                        sha(s.encode())
                        for s in proc.batch_decode(
                            ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                elapsed = time.monotonic() - start
                if reference is None:
                    reference = outputs
                measure[arm].append(
                    dict(
                        size=size,
                        seconds=elapsed,
                        peak_bytes=torch.cuda.max_memory_allocated(),
                        matching_prefixes=sum(
                            a == b for a, b in zip(reference, outputs)
                        ),
                        samples=len(outputs),
                        prefixes=outputs,
                    )
                )
            except torch.OutOfMemoryError:
                measure[arm].append(dict(size=size, error="CUDA OOM"))
                break
        viable = [
            v
            for v in measure[arm]
            if "seconds" in v
            and v["peak_bytes"] < 0.9 * torch.cuda.get_device_properties(0).total_memory
        ]
        exact = [v for v in viable if v["matching_prefixes"] == v["samples"]]
        selected[arm] = (
            min(exact or viable, key=lambda v: v["seconds"])["size"]
            if viable
            else sizes[0] // 2
        )
        print(arm, selected[arm], flush=True)
    immutable(
        run,
        jbytes(
            dict(
                gpu=torch.cuda.get_device_name(),
                validation_sha256=sha(
                    Path(
                        "data/annotation/phase3_v1/validation_calibration.json"
                    ).read_bytes()
                ),
                prefix_tokens=96,
                batch_sizes=selected,
                samples=measure,
            )
        ),
    )


if __name__ == "__main__":
    main()
