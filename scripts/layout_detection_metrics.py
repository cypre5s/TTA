"""Evaluate original and tuned PP on masked canvases with exhaustive single-table labels."""

import argparse
from io import BytesIO
from pathlib import Path
from statistics import mean

from PIL import Image

from scripts.ground_truth import immutable, jbytes, sha

ROOT = Path("experiments/ablation/layout_localization")
OUT = Path("reports/evaluation/pilot1k")


def precision_recall(pages, threshold, operating=None):
    ranked = sorted(
        (
            (-b["confidence"], pid, i, b)
            for pid, row in pages.items()
            for i, b in enumerate(row["boxes"])
            if operating is None or b["confidence"] >= operating
        ),
        key=lambda x: x[:3],
    )
    from scripts.ablation_score import overlap

    hit = set()
    curve = []
    for index, (_, pid, _, box) in enumerate(ranked, 1):
        if (
            pid not in hit
            and overlap(box["coordinate"], pages[pid]["truth"]) >= threshold
        ):
            hit.add(pid)
        curve.append((len(hit) / len(pages), len(hit) / index))
    ap = mean(max((p for r, p in curve if r >= k / 100), default=0) for k in range(101))
    return ap, len(hit), len(ranked)


def main():
    from scripts.ablation_score import DATA, load
    from scripts.evaluate import read_split

    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["original", "fine"])
    a = ap.parse_args()
    import numpy as np
    from paddlex import create_model

    path = (
        Path("models/layout/PP-DocLayoutV3")
        if a.model == "original"
        else Path(
            load(Path("experiments/layout_finetune/pilot_server/selected.json"))[
                "export"
            ]
        )
    )
    net = create_model(
        "PP-DocLayoutV3", model_dir=str(path.resolve()), device="gpu:0", threshold=0.001
    )
    pages = {}
    for row in read_split(DATA, "test"):
        gt_path = Path(row["gt"])
        if sha(gt_path.read_bytes()) != row["gt_sha256"]:
            raise ValueError("Ground truth changed")
        gt = load(gt_path)
        tree = gt.get("structure_tree")
        if not tree:
            continue
        tid = row["table_id"]
        packet = load(Path("data/annotation/packets", tid + ".json"))
        pid = packet["page_id"]
        pg = load(Path("data/rendered", gt["document_sha256"], pid + ".json"))
        canvas_path = ROOT / "canvases" / (tid + ".png")
        dst = ROOT / a.model / (tid + ".json")
        with Image.open(pg["image"]) as im:
            box = [round(v * pg["dpi"] / 72) for v in tree["bbox_pt"]]
            box = [
                max(0, box[0]),
                max(0, box[1]),
                min(im.width, box[2]),
                min(im.height, box[3]),
            ]
            if not canvas_path.is_file():
                canvas = Image.new("RGB", im.size, "white")
                canvas.paste(im.crop(tuple(box)), (box[0], box[1]))
                stream = BytesIO()
                canvas.save(stream, format="PNG")
                immutable(canvas_path, stream.getvalue())
        if not dst.exists():
            detections = next(iter(net.predict(str(canvas_path)))).json["res"]["boxes"]
            pred = [
                dict(
                    confidence=float(b["score"]),
                    coordinate=list(map(float, b["coordinate"])),
                )
                for b in detections
                if b["label"] == "table"
            ]
            immutable(
                dst,
                jbytes(
                    dict(
                        boxes=pred,
                        truth=box,
                        table_id=tid,
                        page_id=pid,
                        canvas_sha256=sha(canvas_path.read_bytes()),
                        gt_sha256=row["gt_sha256"],
                    )
                ),
            )
        pages[tid] = load(dst)
    aps = [
        precision_recall(pages, threshold)[0]
        for threshold in np.arange(50, 100, 5) / 100
    ]
    _, tp, n = precision_recall(pages, 0.5, 0.25)
    precision = tp / n if n else 0
    recall = tp / len(pages)
    metrics = dict(
        model=a.model,
        scope="single-table test canvases with independently reconstructed source bounds",
        reference_tables=len(pages),
        source_pages=len({v["page_id"] for v in pages.values()}),
        ap50=aps[0],
        ap50_95=mean(aps),
        iou_thresholds=[i / 100 for i in range(50, 100, 5)],
        interpolation="101-point",
        operating_score=0.25,
        precision=precision,
        recall=recall,
        f1=2 * tp / (n + len(pages)),
        true_positives=tp,
        predictions=n,
        all_class_map=None,
    )
    immutable(OUT / ("layout_" + a.model + ".json"), jbytes(metrics))
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
