"""Prepare project-disjoint table-supervised pages without treating unreviewed regions as background."""

import argparse
import json
from pathlib import Path
import random

from PIL import Image

from scripts.evaluate import read_split
from scripts.ground_truth import immutable, jbytes, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train", type=int, default=64)
    ap.add_argument("--validation", type=int, default=20)
    args = ap.parse_args()
    base = Path("data/splits/ground_truth_v1")
    categories = json.loads(Path("configs/layout_provenance.json").read_bytes())[
        "category_mapping"
    ]
    cats = [
        {"id": int(i) + 1, "name": name, "supercategory": "layout"}
        for i, name in categories.items()
    ]
    table_cat = next(c["id"] for c in cats if c["name"] == "table")
    out = args.output
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta = {}
    for split, limit, alias in [
        ("train", args.train, "train"),
        ("validation", args.validation, "val"),
    ]:
        rows = read_split(base, split)
        # Prefer independent corrected PDF bounds and one candidate per page. No test input is used.
        rng = random.Random(42)
        rng.shuffle(rows)
        rows.sort(
            key=lambda r: (
                not json.loads(Path(r["gt"]).read_bytes()).get("structure_tree")
            )
        )
        images = []
        anns = []
        stamps = []
        pages = set()
        for r in rows:
            gt_path = Path(r["gt"])
            gt = json.loads(gt_path.read_bytes())
            if (
                sha(gt_path.read_bytes()) != r["gt_sha256"]
                or gt["reviewer_type"] != "human"
                or gt["decision"] != "approve"
            ):
                raise ValueError("Unapproved crop")
            tid = gt["table_id"]
            packet = json.loads(
                Path("data/annotation/packets", tid + ".json").read_bytes()
            )
            pid = packet["page_id"]
            if pid in pages:
                continue
            pg = json.loads(
                Path("data/rendered", gt["document_sha256"], pid + ".json").read_bytes()
            )
            tree = gt.get("structure_tree")
            if not tree:
                continue
            with Image.open(pg["image"]) as rendered:
                b = [round(z * pg["dpi"] / 72) for z in tree["bbox_pt"]]
                b[0] = max(0, b[0])
                b[1] = max(0, b[1])
                b[2] = min(rendered.width, b[2])
                b[3] = min(rendered.height, b[3])
                if b[2] <= b[0] or b[3] <= b[1]:
                    raise ValueError("Empty reviewed extent")
                canvas = Image.new("RGB", rendered.size, "white")
                canvas.paste(rendered.crop(tuple(b)).convert("RGB"), (b[0], b[1]))
                # Fixed aspect-preserving downscale to reduce the official training reader's I/O.
                k = min(1, 1280 / max(canvas.size))
                image = canvas.resize(
                    (round(canvas.width * k), round(canvas.height * k)),
                    Image.Resampling.LANCZOS,
                )
                bx = [
                    b[0] * image.width / canvas.width,
                    b[1] * image.height / canvas.height,
                    b[2] * image.width / canvas.width,
                    b[3] * image.height / canvas.height,
                ]
                dest = out / "images" / (pid + ".jpg")
                from io import BytesIO

                f = BytesIO()
                image.save(f, format="JPEG", quality=93)
                immutable(dest, f.getvalue())
            idx = len(images) + 1
            images.append(
                dict(
                    id=idx,
                    file_name=dest.name,
                    width=image.width,
                    height=image.height,
                    project_group=r["group_id"],
                    reviewer_type=gt["reviewer_type"],
                    reviewer=gt["reviewer"],
                    source_table_id=tid,
                    supervision="reviewed table pasted at source position on blank page",
                    source_gt_sha256=r["gt_sha256"],
                )
            )
            x, y, xx, yy = bx
            anns.append(
                dict(
                    id=idx,
                    image_id=idx,
                    category_id=table_cat,
                    bbox=[x, y, xx - x, yy - y],
                    area=(xx - x) * (yy - y),
                    segmentation=[[x, y, xx, y, xx, yy, x, yy]],
                    iscrowd=0,
                    read_order=0,
                )
            )
            stamps.append(dict(image=dest.name, sha256=sha(f.getvalue()), source=tid))
            pages.add(pid)
            if len(images) >= limit:
                break
        if not images:
            raise ValueError("Empty supervised set")
        immutable(
            out / "annotations" / ("instance_" + alias + ".json"),
            jbytes(
                dict(
                    info={
                        "description": "Reviewed table-domain pilot; masks exclude unreviewed page regions"
                    },
                    licenses=[],
                    images=images,
                    annotations=anns,
                    categories=cats,
                )
            ),
        )
        meta[split] = stamps
    immutable(
        out / "source_manifest.json",
        jbytes(
            dict(
                dataset="ground_truth_v1",
                sha256=sha((base / "manifest.json").read_bytes()),
                sources=meta,
            )
        ),
    )
    print({p: len(v) for p, v in meta.items()})


if __name__ == "__main__":
    main()
