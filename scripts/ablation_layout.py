"""Detect tables independently on every held-out full page with the validation-selected detector."""

import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from scripts.ground_truth import immutable, jbytes

OUT = Path("experiments/ablation/fine_layout")


def main():
    import paddle
    from paddlex import create_model

    root = Path("experiments/layout_finetune/pilot_server/selected.json")
    cfg = json.loads(root.read_bytes())
    model = create_model(
        "PP-DocLayoutV3",
        model_dir=str(Path(cfg["export"]).resolve()),
        device="gpu:0",
        threshold=0.25,
    )
    rows = [
        json.loads(s)
        for s in Path("data/splits/ground_truth_v1/test.jsonl")
        .read_bytes()
        .splitlines()
    ]
    pages = sorted({r["table_id"].rsplit("-", 1)[0] for r in rows})
    for pid in pages:
        target = OUT / "pages" / (pid + ".json")
        if target.exists():
            continue
        p = Path("data/rendered", pid.split("-")[0], pid + ".json")
        pg = json.loads(p.read_bytes())
        img = Path(pg["image"])
        if hashlib.sha256(img.read_bytes()).hexdigest() != pg["image_sha256"]:
            raise ValueError(img)
        st = time.monotonic()
        boxes = next(iter(model.predict(str(img)))).json["res"]["boxes"]
        paddle.device.cuda.synchronize()
        elapsed = time.monotonic() - st
        tables = []
        with Image.open(img) as im:
            for i, b in enumerate(boxes):
                if b["label"] != "table":
                    continue
                coords = list(map(int, b["coordinate"]))
                if min(coords[2] - coords[0], coords[3] - coords[1]) < 2:
                    continue
                key = f"{pid}-r{i:03d}"
                from io import BytesIO

                buf = BytesIO()
                im.crop(tuple(coords)).save(buf, format="PNG")
                dest = OUT / "crops" / (key + ".png")
                immutable(dest, buf.getvalue())
                tables.append(
                    dict(
                        table_id=key,
                        coordinate=coords,
                        confidence=float(b["score"]),
                        image=str(dest),
                        image_sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
                        read_order=i,
                    )
                )
        immutable(
            target,
            jbytes(
                dict(
                    page_id=pid,
                    boxes=boxes,
                    tables=tables,
                    seconds=elapsed,
                    source_image_sha256=pg["image_sha256"],
                )
            ),
        )
    print("held-out pages", len(pages), flush=True)


if __name__ == "__main__":
    main()
