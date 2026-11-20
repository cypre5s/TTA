"""Replay each frozen test page on its archived original PP device and compare crop pixels."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import yaml
from paddlex import create_model
from scripts.layout_pipeline import encode, load
from scripts.ground_truth import immutable, jbytes

ROOT = Path("experiments/baseline/original_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device", choices=["cpu", "gpu:0"])
    device = ap.parse_args().device
    if not (ROOT / "evaluation_lock.json").exists():
        raise ValueError("Missing paired evaluation freeze")
    test = [
        json.loads(l)
        for l in Path("data/splits/ground_truth_v1/test.jsonl")
        .read_bytes()
        .splitlines()
    ]
    pids = sorted({r["table_id"].rsplit("-", 1)[0] for r in test})
    cfg = yaml.safe_load(Path("configs/layout.yaml").read_text())
    receipt = json.loads(Path("configs/layout_provenance.json").read_text())
    for name, expected in receipt["files"].items():
        with (Path(cfg["model_dir"]) / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(name)
    cv2.setNumThreads(1)
    model = create_model(
        cfg["model"],
        model_dir=str(Path(cfg["model_dir"]).resolve()),
        device=device,
        threshold=cfg["threshold"],
    )
    for pid in pids:
        archive = json.loads(Path("data/layout/predictions", pid + ".json").read_text())
        if archive.get("device", "cpu") != device:
            continue
        dest = ROOT / "layout_replay" / (pid + ".json")
        if dest.exists():
            continue
        page = json.loads(
            Path("data/rendered", pid.split("-")[0], pid + ".json").read_text()
        )
        bgr = load(page)
        st = time.perf_counter()
        boxes = next(iter(model.predict(bgr))).json["res"]["boxes"]
        elapsed = time.perf_counter() - st
        from PIL import Image

        _, boxes, regions, _ = encode(
            page, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), boxes
        )
        hashes = {f"{pid}-r{idx:03d}": digest for idx, digest, _ in regions}
        tables = [r["table_id"] for r in test if r["table_id"].rsplit("-", 1)[0] == pid]
        matches = {}
        for tid in tables:
            crop = json.loads(
                Path("data/annotation/packets", tid + ".json").read_bytes()
            )
            matches[tid] = hashes.get(tid) == crop["image_sha256"]
        immutable(
            dest,
            jbytes(
                dict(
                    page_id=pid,
                    model=cfg["model"],
                    device=device,
                    seconds=elapsed,
                    image_sha256=page["image_sha256"],
                    input_weight_sha256=receipt["files"]["inference.pdiparams"],
                    boxes=boxes,
                    held_out_crop_hash_equal=matches,
                )
            ),
        )
        print(device, pid, all(matches.values()), flush=True)


if __name__ == "__main__":
    main()
