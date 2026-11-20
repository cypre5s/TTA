"""Time full page handling with real detector, including encoding and durable writes."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
import statistics
import yaml
import cv2
from paddlex import create_model
from layout_pipeline import predict_pages


def main():
    cv2.setNumThreads(1)
    base = Path("reports/performance")
    cfg = yaml.safe_load(Path("configs/layout.yaml").read_text())
    model = create_model(
        cfg["model"],
        model_dir=str(Path(cfg["model_dir"]).resolve()),
        device=cfg["device"],
        threshold=cfg["threshold"],
    )
    cohort = base / "layout_pipeline_inputs.json"
    if not cohort.exists():
        pool = sorted(Path("data/layout/predictions").glob("*.json"))
        paths = [str(pool[int((i + 0.5) * len(pool) / 32)]) for i in range(32)]
        cohort.write_text(json.dumps(paths, indent=2))
    pages = [json.loads(Path(p).read_text()) for p in json.loads(cohort.read_text())]
    next(iter(model.predict(pages[0]["image"])))
    trials = []
    reference = None
    for pipe in (False, True, True, False):
        rows = []
        with tempfile.TemporaryDirectory(dir=base) as d:
            dest = Path(d)
            start = time.perf_counter()
            for i, (page, _, product) in enumerate(predict_pages(model, pages, pipe)):
                size, boxes, crops, jpeg = product
                files = [
                    (f"{i}.jpg", jpeg),
                    (f"{i}.json", json.dumps(boxes, sort_keys=True).encode()),
                ]
                files.extend((f"{i}-{idx}.png", img) for idx, _, img in crops)
                for name, data in files:
                    (dest / name).write_bytes(data)
                    rows.append((name, hashlib.sha256(data).hexdigest()))
            elapsed = time.perf_counter() - start
        if reference is None:
            reference = rows
        t = dict(
            pipelined=pipe,
            seconds=elapsed,
            files=len(rows),
            identical=rows == reference,
            sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        )
        print(t, flush=True)
        trials.append(t)
    avg = lambda p: statistics.mean(t["seconds"] for t in trials if t["pipelined"] == p)
    (base / "layout_pipeline_comparison.json").write_text(
        json.dumps(
            dict(pages=len(pages), trials=trials, speedup=avg(False) / avg(True)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
