"""Feed only unsolved crops to local inference and materialize traceable results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def work(target):
    ready = Path("data/local_ocr/ready")
    lane = ready / "batch"
    lane.mkdir(parents=True, exist_ok=True)
    dest = Path("data/tables/predictions")
    assigned = {p.stem for p in lane.glob("*.png")} | {
        p.stem for p in dest.glob("*.json")
    }
    for p in sorted(Path("data/tables/crops").glob("*.json")):
        if len(assigned) >= target:
            break
        if p.stem in assigned:
            continue
        info = json.loads(p.read_text())
        image = Path(info["image"])
        if hashlib.sha256(image.read_bytes()).hexdigest() != info["image_sha256"]:
            raise ValueError(image)
        os.link(image, lane / image.name)
        assigned.add(p.stem)
    if len(assigned) >= target:
        (ready / "DRAIN").touch()
    for p in dest.glob("*.json"):
        raw = Path("data/otsl/raw") / (p.stem + ".otsl")
        html = p.with_suffix(".html")
        if raw.exists() and html.exists():
            continue
        ans = json.loads(p.read_text())
        meta = json.loads((Path("data/tables/crops") / p.name).read_text())
        if ans["table_id"] != p.stem or ans["image_sha256"] != meta["image_sha256"]:
            raise ValueError(p)
        for path, text in [
            (raw, ans["raw_otsl"]),
            (
                html,
                '<!DOCTYPE html><meta charset="utf-8"><style>table{border-collapse:collapse}td{border:1px solid;padding:.3em}</style>'
                + ans["html"],
            ),
        ]:
            tmp = path.with_suffix(".part")
            tmp.write_text(text)
            tmp.replace(path)
    return len(list(dest.glob("*.html")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    while work(1000) < 1000 and args.watch:
        time.sleep(3)
