"""Generate all held-out crop outputs for the tuned-detector and/or adapter arms once."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.cloud_ocr import generate
from scripts.ground_truth import immutable, jbytes

ROOT = Path("experiments/ablation")


def load(f):
    return json.loads(f.read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", choices=["original", "lora"], required=True)
    ap.add_argument("--layout", choices=["original", "fine"], required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path("configs/xiaomi_local.yaml").read_text())
    sys.path.insert(0, cfg["upstream"])
    from evaluation_pipeline.cropping import fit_image

    test = [
        json.loads(l)
        for l in Path("data/splits/ground_truth_v1/test.jsonl")
        .read_bytes()
        .splitlines()
    ]
    if a.layout == "original":
        inputs = [
            load(Path("data/annotation/packets", r["table_id"] + ".json")) for r in test
        ]
    else:
        pages = {r["table_id"].rsplit("-", 1)[0] for r in test}
        inputs = [
            t
            for pid in sorted(pages)
            for t in load(ROOT / "fine_layout/pages" / (pid + ".json"))["tables"]
        ]
    dst = ROOT / f"{a.layout}_{a.vision}"
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
    if a.vision == "lora":
        from peft import PeftModel

        found = [
            p
            for p in Path("models/xiaomi_lora").glob("*/best")
            if (p / "adapter_model.safetensors").exists()
        ]
        if len(found) != 1:
            raise ValueError("Exactly one validation-selected adapter required")
        model = (
            PeftModel.from_pretrained(model, found[0], is_trainable=False)
            .merge_and_unload()
            .eval()
        )
        adapter = hashlib.sha256(
            (found[0] / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
    else:
        adapter = None

    def cost(x):
        with Image.open(x["image"]) as im:
            return (im.width * im.height, x["table_id"])

    batch = load(Path("experiments/baseline/original_v1/calibration.json"))[
        "batch_sizes"
    ]["B0"]
    config = dict(
        layout=a.layout,
        vision=a.vision,
        adapter=adapter,
        seed=42,
        batch=batch,
        prompt=cfg["prompt"],
        max_tokens=cfg["max_new_tokens"],
        device=torch.cuda.get_device_name(),
        test_manifest=hashlib.sha256(
            Path("data/splits/ground_truth_v1/manifest.json").read_bytes()
        ).hexdigest(),
    )
    immutable(dst / "experiment.json", jbytes(config))
    pending = [
        x
        for x in sorted(inputs, key=cost)
        if not (dst / "predictions" / (x["table_id"] + ".json")).exists()
    ]

    def work(group, retry=False):
        paths = [Path(x["image"]) for x in group]
        for p, x in zip(paths, group):
            if hashlib.sha256(p.read_bytes()).hexdigest() != x["image_sha256"]:
                raise ValueError(p)
        torch.cuda.reset_peak_memory_stats()
        st = time.monotonic()
        try:
            ids = generate(model, proc, paths, cfg, fit_image)
        except torch.cuda.OutOfMemoryError:
            if retry or len(group) == 1:
                raise
            ids = None
        if ids is None:
            torch.cuda.empty_cache()
            k = len(group) // 2
            work(group[:k], True)
            work(group[k:], True)
            return
        elapsed = time.monotonic() - st
        raw = proc.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for x, txt in zip(group, raw):
            immutable(
                dst / "predictions" / (x["table_id"] + ".json"),
                jbytes(
                    dict(
                        table_id=x["table_id"],
                        raw_otsl=txt,
                        image_sha256=x["image_sha256"],
                        seconds=elapsed / len(group),
                        batch_seconds=elapsed,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        adapter=adapter,
                    )
                ),
            )
        print(a.layout, a.vision, len(group), round(elapsed, 1), flush=True)

    for i in range(0, len(pending), batch):
        work(pending[i : i + batch])


if __name__ == "__main__":
    main()
