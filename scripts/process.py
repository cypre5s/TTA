"""Render immutable, content-addressed inputs and preserve coordinates and QC."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np
import pymupdf
import pypdfium2
import yaml

cfg = yaml.safe_load(Path("configs/pdf_processor.yaml").read_text())
db = sqlite3.connect("database/pilot.sqlite", timeout=45)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def process(key, path, count):
    root = Path("data/rendered") / key
    root.mkdir(exist_ok=True, parents=True)
    mark = root / "manifest.json"
    if mark.exists():
        return
    if sha(Path(path).read_bytes()) != key:
        raise ValueError(f"Input hash mismatch {key}")
    with pypdfium2.PdfDocument(path) as check:
        if len(check) != count:
            raise ValueError(f"page count mismatch {key}")
    with pymupdf.open(path) as doc:
        records = []
        for num, page in enumerate(doc, 1):
            pid = f"{key}-p{num:04d}"
            sidecar = root / f"{pid}.json"
            if sidecar.exists():
                item = json.loads(sidecar.read_text())
                if Path(item["image"]).exists():
                    records.append(item)
                    continue
            words = page.get_text("words")
            drawings = page.get_drawings()
            lines = sum(
                1 if item[0] == "l" else 4 if item[0] == "re" else 0
                for drawing in drawings
                for item in drawing["items"]
            )
            dpi = (
                cfg["complex_dpi"] if lines >= cfg["complex_grid_lines"] else cfg["dpi"]
            )
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image = root / f"{pid}.png"
            png = pix.tobytes("png")
            image.with_suffix(".part").write_bytes(png)
            image.with_suffix(".part").replace(image)
            gray = cv2.cvtColor(
                np.frombuffer(pix.samples_mv, dtype=np.uint8).reshape(pix.h, pix.w, 3),
                cv2.COLOR_RGB2GRAY,
            )
            preview = cv2.resize(
                gray,
                (max(1, pix.w // 4), max(1, pix.h // 4)),
                interpolation=cv2.INTER_AREA,
            )
            seg = cv2.HoughLinesP(
                cv2.Canny(preview, 60, 160),
                1,
                np.pi / 180,
                45,
                minLineLength=40,
                maxLineGap=4,
            )
            angles = []
            for x1, y1, x2, y2 in [] if seg is None else seg.reshape(-1, 4):
                a = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                if abs(a) < 15:
                    angles.append(a)
            textpath = root / f"{pid}.text.json"
            textpath.write_text(
                json.dumps(
                    dict(
                        words=words,
                        text=page.get_text(),
                        coordinate_space="unrotated PDF points, top-left",
                        rotation_matrix=list(page.rotation_matrix),
                    ),
                    ensure_ascii=False,
                )
            )
            item = dict(
                page_id=pid,
                document_sha256=key,
                page_number=num,
                dpi=dpi,
                image=str(image),
                image_sha256=sha(png),
                size_px=[pix.w, pix.h],
                page_size_pt=list(page.rect),
                rotation=page.rotation,
                skew_estimate_deg=float(np.median(angles)) if angles else None,
                contrast=float(preview.std()),
                laplacian_variance=float(cv2.Laplacian(preview, cv2.CV_32F).var()),
                grid_lines=lines,
                text_layer=str(textpath),
                kind="native_or_ocr_layer"
                if words
                else "image_only"
                if page.get_images()
                else "no_text",
                text_sha256=sha(textpath.read_bytes()),
            )
            temp = sidecar.with_suffix(".tmp")
            temp.write_text(json.dumps(item))
            temp.replace(sidecar)
            records.append(item)
        tmp = mark.with_suffix(".tmp")
        tmp.write_text(json.dumps(records))
        tmp.replace(mark)
        print(key, count, flush=True)


def main(watch):
    workers = cfg.get("workers", 3)
    cv2.setNumThreads(1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            docs = db.execute(
                "select sha256,path,pages from documents order by rowid"
            ).fetchall()
            todo = [
                r
                for r in docs
                if not (Path("data/rendered") / r[0] / "manifest.json").exists()
            ]
            for job in [pool.submit(process, *r) for r in todo]:
                job.result()
            if (
                not watch
                or len(docs)
                >= yaml.safe_load(Path("configs/crawler.yaml").read_text())["target"]
            ):
                break
            time.sleep(4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    main(p.parse_args().watch)
