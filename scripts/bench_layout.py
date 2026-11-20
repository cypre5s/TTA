"""Time identical inputs, preserve full per-box evidence for CPU/GPU comparison."""

import argparse
import json
from pathlib import Path
import time
import statistics
import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import yaml

base = Path("reports/performance")
sample = base / "layout_inputs.json"


def inputs():
    if not sample.exists():
        ids = []
        seen = set()
        for p in sorted(Path("data/layout/predictions").glob("*.json")):
            result = json.loads(p.read_text())
            key = p.stem.split("-")[0]
            if key not in seen and result["tables"]:
                ids.append(result["image"])
                seen.add(key)
                if len(ids) == 10:
                    break
        sample.write_text(json.dumps(ids, indent=2))
    return json.loads(sample.read_text())


def bench(device, graph=False, half=False):
    import paddle
    from paddlex import create_model

    cfg = yaml.safe_load(Path("configs/layout.yaml").read_text())
    paths = inputs()
    if graph:
        cfg["model_dir"] = "models/layout_shared_constants"
    model = create_model(
        cfg["model"],
        model_dir=str(Path(cfg["model_dir"]).resolve()),
        device=device,
        threshold=cfg["threshold"],
        **(
            {"engine": "paddle_static", "engine_config": {"run_mode": "paddle_fp16"}}
            if half
            else {}
        ),
    )
    next(iter(model.predict(paths[0])))
    timings = []
    boxes = {}
    for _ in range(2):
        for p in paths:
            start = time.perf_counter()
            boxes[p] = next(iter(model.predict(p))).json["res"]["boxes"]
            if device.startswith("gpu"):
                paddle.device.cuda.synchronize()
            timings.append(time.perf_counter() - start)
    report = dict(
        device=device,
        paddle=paddle.__version__,
        seconds=timings,
        mean_seconds=statistics.mean(timings),
        boxes=boxes,
    )
    if device.startswith("gpu"):
        report["peak_allocated_bytes"] = paddle.device.cuda.max_memory_allocated()
    (
        base
        / f"layout_{device.split(':')[0]}{'_cse' if graph else '_fp16' if half else ''}.json"
    ).write_text(json.dumps(report, ensure_ascii=False))


def compare(graph=False, half=False):
    from shapely.geometry import Polygon

    a, b = [
        json.loads((base / f"layout_{device}.json").read_text())
        for device in (
            ("gpu", "gpu_cse")
            if graph
            else ("gpu", "gpu_fp16")
            if half
            else ("cpu", "gpu")
        )
    ]
    assert a["boxes"].keys() == b["boxes"].keys()
    diffs = []
    table_ious = []
    for p, cpu in a["boxes"].items():
        gpu = b["boxes"][p]
        if len(cpu) != len(gpu):
            diffs.append([p, "region count", len(cpu), len(gpu)])
        for i, (c, g) in enumerate(zip(cpu, gpu)):
            if c["label"] != g["label"] or c["order"] != g["order"]:
                diffs.append([p, i, "category/order"])
                continue
            pc, pg = [Polygon(r["polygon_points"]).buffer(0) for r in (c, g)]
            union = pc.union(pg).area
            iou = pc.intersection(pg).area / union if union else 0
            if c["label"] == "table":
                table_ious.append(iou)
            if iou < 0.98:
                diffs.append([p, i, "polygon IoU", iou])
    report = dict(
        pages=len(a["boxes"]),
        baseline="gpu_original" if graph or half else "cpu_original",
        candidate="gpu_shared_constants"
        if graph
        else "gpu_fp16"
        if half
        else "gpu_original",
        baseline_mean_s=a["mean_seconds"],
        candidate_mean_s=b["mean_seconds"],
        speedup=a["mean_seconds"] / b["mean_seconds"],
        table_iou_min=min(table_ious, default=None),
        peak_allocated_bytes=b["peak_allocated_bytes"],
        differences=diffs,
    )
    (
        base
        / (
            "layout_gpu_graph_comparison.json"
            if graph
            else "layout_fp16_comparison.json"
            if half
            else "layout_comparison.json"
        )
    ).write_text(json.dumps(report, indent=2))
    print(
        {k: v for k, v in report.items() if k != "differences"},
        len(diffs),
        "differences",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("device", choices=["cpu", "gpu:0", "compare"])
    p.add_argument("--graph", action="store_true")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()
    compare(args.graph, args.fp16) if args.device == "compare" else bench(
        args.device, args.graph, args.fp16
    )
