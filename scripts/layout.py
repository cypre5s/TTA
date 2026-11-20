"""Run the original 25-class PaddleX detector and save traceable pilot crops."""

from collections import Counter
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
from paddlex import create_model
import yaml
import cv2
from layout_pipeline import predict_pages

cfg = yaml.safe_load(Path("configs/layout.yaml").read_text())
pred = Path("data/layout/predictions")
crops = Path("data/tables/crops")
preview = Path("data/layout/previews")
preview.mkdir(parents=True, exist_ok=True)


def atomic(path, value):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def run(watch):
    cv2.setNumThreads(1)
    target = cfg["target_tables"]
    full = cfg.get("full_pilot_pages", False)
    known = {json.loads(p.read_text())["image_sha256"] for p in crops.glob("*.json")}
    done = {p.stem for p in pred.glob("*.json")}
    seen_manifests = set()
    pages = {}

    def refresh():
        for manifest in sorted(Path("data/rendered").glob("*/manifest.json")):
            if manifest in seen_manifests:
                continue
            for page in json.loads(manifest.read_text()):
                if page["page_id"] not in done:
                    pages[page["page_id"]] = page
            seen_manifests.add(manifest)

    refresh()
    if (not full and len(known) >= target) or (
        not pages and len(seen_manifests) >= cfg["target_documents"]
    ):
        return
    provenance = json.loads(Path("configs/layout_provenance.json").read_text())
    for name, expected in provenance["files"].items():
        with (Path(cfg["model_dir"]) / name).open("rb") as weights:
            if hashlib.file_digest(weights, "sha256").hexdigest() != expected:
                raise ValueError(f"Official layout checkpoint differs at {name}")
    model = create_model(
        cfg["model"],
        model_dir=str(Path(cfg["model_dir"]).resolve()),
        device=cfg["device"],
        threshold=cfg["threshold"],
    )
    counts = Counter(p.stem.split("-")[0] for p in crops.glob("*.json"))
    while full or len(known) < target:
        if not pages:
            refresh()
            if not pages:
                if not watch or len(seen_manifests) >= cfg["target_documents"]:
                    break
                time.sleep(3)
                continue
        # Prefer underrepresented documents. Refresh the queue between short batches.
        if len(known) >= target:
            from itertools import islice

            batch = list(islice(pages.values(), 64))
        else:
            ranked = sorted(
                pages.values(),
                key=lambda p: (
                    counts[p["document_sha256"]],
                    p["page_number"],
                    p["page_id"],
                ),
            )
            grid = sorted(
                (p for p in ranked if p["grid_lines"] >= 4),
                key=lambda p: p["grid_lines"] < 16,
            )
            other = [p for p in ranked if p["grid_lines"] < 4]
            batch = grid[:14] + other[:2] if grid else other[:16]
        for page, elapsed, product in predict_pages(
            model, batch, cfg.get("pipeline_io", False)
        ):
            pid = page["page_id"]
            size, boxes, regions, jpeg = product
            tables = []
            for idx, digest, blob in regions:
                tid = f"{pid}-r{idx:03d}"
                p = crops / f"{tid}.png"
                if digest in known:
                    if p.with_suffix(".json").exists():
                        tables.append(tid)
                    continue
                tmp = p.with_suffix(".part")
                tmp.write_bytes(blob)
                tmp.replace(p)
                box = boxes[idx]
                coords = box["coordinate"]
                entry = dict(
                    table_id=tid,
                    page_id=pid,
                    document_sha256=page["document_sha256"],
                    page_number=page["page_number"],
                    image=str(p),
                    image_sha256=digest,
                    bbox_px=coords,
                    segmentation=box.get("polygon_points"),
                    confidence=box["score"],
                    read_order=idx,
                    review_status="pending",
                )
                atomic(p.with_suffix(".json"), entry)
                tables.append(tid)
                known.add(digest)
                counts[page["document_sha256"]] += 1
            jpg = preview / f"{pid}.jpg"
            tmp = jpg.with_suffix(".part")
            tmp.write_bytes(jpeg)
            tmp.replace(jpg)
            atomic(
                pred / f"{pid}.json",
                dict(
                    page_id=pid,
                    image=page["image"],
                    image_sha256=page["image_sha256"],
                    size_px=size,
                    pipeline_io=cfg.get("pipeline_io", False),
                    model=cfg["model"],
                    device=cfg["device"],
                    graph=Path(cfg["model_dir"]).name,
                    runtime_seconds=elapsed,
                    boxes=boxes,
                    tables=tables,
                    preview=str(jpg),
                ),
            )
            print(pid, len(boxes), len(known), flush=True)
            pages.pop(pid)
            done.add(pid)
            if not full and len(known) >= target:
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    run(p.parse_args().watch)
