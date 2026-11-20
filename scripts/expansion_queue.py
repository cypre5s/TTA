"""Feed unique original crops to local OCR and publish hash-bound review packets separately from the pilot."""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from scripts.ground_truth import immutable, jbytes, sha

EXP = Path("data/expansion_v2")
LANE = EXP / "ocr/ready/batch"
PRED = EXP / "ocr/predictions"
PACKETS = EXP / "annotation/packets"


def fill():
    LANE.mkdir(parents=True, exist_ok=True)
    manifest = EXP / "crop_index.sqlite"
    with sqlite3.connect(manifest) as db:
        db.executescript(
            "CREATE TABLE IF NOT EXISTS candidates(ordinal INTEGER PRIMARY KEY,table_id TEXT UNIQUE,image_sha256 TEXT UNIQUE,metadata TEXT);"
        )
        pilot = {p.stem for p in Path("data/annotation/packets").glob("*.json")}
        prior = {
            json.loads(p.read_bytes())["image_sha256"]
            for p in Path("data/annotation/packets").glob("*.json")
        }
        index = {r[0] for r in db.execute("SELECT table_id FROM candidates")}
        sources = [Path("data/tables/crops"), EXP / "data/tables/crops"]
        for folder in sources:
            for p in sorted(folder.glob("*.json")):
                if p.stem in index or p.stem in pilot:
                    continue
                m = json.loads(p.read_bytes())
                source = p.with_suffix(".png")
                digest = m["image_sha256"]
                if digest in prior:
                    continue
                if sha(source.read_bytes()) != digest:
                    raise ValueError(source)
                cursor = db.execute(
                    "INSERT OR IGNORE INTO candidates(table_id,image_sha256,metadata) VALUES(?,?,?)",
                    (p.stem, digest, str(p)),
                )
                if not cursor.rowcount:
                    continue
                dest = LANE / source.name
                if not dest.exists():
                    os.link(source, dest)
        return db.execute("SELECT count(*) FROM candidates").fetchone()[0]


def publish():
    PACKETS.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(EXP / "crop_index.sqlite") as db:
        for p in sorted(PRED.glob("*.json")):
            dest = PACKETS / p.name
            if dest.exists():
                continue
            n, src = db.execute(
                "select ordinal,metadata from candidates where table_id=?", (p.stem,)
            ).fetchone()
            src = Path(src)
            m = json.loads(src.read_bytes())
            pred = json.loads(p.read_bytes())
            if (
                pred["image_sha256"] != m["image_sha256"]
                or pred["table_id"] != m["table_id"]
            ):
                raise ValueError(p)
            crop = src.with_suffix(".png")
            html = EXP / "ocr/html" / (p.stem + ".html")
            raw = EXP / "ocr/otsl" / (p.stem + ".otsl")
            immutable(html, ("<meta charset=utf-8>" + pred["html"]).encode())
            immutable(raw, pred["raw_otsl"].encode())
            immutable(
                dest,
                jbytes(
                    dict(
                        m,
                        ordinal=n + 1000,
                        image=str(crop),
                        raw_otsl=str(raw),
                        html=str(html),
                        prediction=str(p),
                        prediction_sha256=sha(p.read_bytes()),
                        html_sha256=sha(html.read_bytes()),
                        review_status="pending_visual",
                        weight_sha256=pred["weight_sha256"],
                        structure_checks=pred["structure_checks"],
                    )
                ),
            )
    return len(list(PACKETS.glob("*.json")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    while True:
        print("unique available", fill(), "published", publish(), flush=True)
        if not args.watch:
            break
        time.sleep(18)
