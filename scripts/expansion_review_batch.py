"""Prioritize new-looking crops for visual review without assigning any labels."""

import argparse
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from scripts.expansion_quality import EXP, OUT, fingerprint
from scripts.ground_truth import sha
from src.otsl.grid import parse


def accepted():
    signatures = set()
    for path in OUT.glob("*.json"):
        row = json.loads(path.read_bytes())
        if row["decision"] == "usable":
            signatures.add(row["content_signature"])
    for path in Path("data/tables/ground_truth").glob("*/000001.json"):
        row = json.loads(path.read_bytes())
        if row["decision"] == "approve":
            signatures.add(fingerprint(parse(row["corrected_otsl"]), row.get("structure_tree")))
    return signatures


def pick(start, end, parity, limit, sheet_dir=None, fast_only=False):
    known = accepted()
    reviewed = {p.stem for p in (EXP / "annotation/raw").glob("*.json")}
    ranks = []
    for path in (EXP / "annotation/packets").glob("*.json"):
        if path.stem in reviewed:
            continue
        row = json.loads(path.read_bytes())
        ordinal = row["ordinal"]
        if not start <= ordinal <= end or (parity is not None and ordinal % 2 != parity):
            continue
        text = Path(row["raw_otsl"]).read_text()
        try:
            grid = parse(text)
            signature = fingerprint(grid)
            cells = len(grid.cells)
        except ValueError:
            signature, cells = None, 0
        duplicate = signature in known
        complex_table = bool(signature is None or row.get("structure_checks") or cells > 45 or len(text) > 1600)
        cost = len(text) + 16 * cells
        ranks.append((duplicate, complex_table, cost, ordinal, signature, path, row, text))

    ranks.sort(key=lambda x: x[:4])
    quick_target = limit if fast_only else max(1, limit * 2 // 3)
    chosen, seen, docs = [], set(), set()
    passes = [(False, quick_target)]
    if not fast_only:
        passes.extend(((True, limit - quick_target), (None, limit)))
    for want_complex, quota in passes:
        picked = 0
        for item in ranks:
            duplicate, complex_table, _, ordinal, signature, _, row, _ = item
            if ordinal in seen or len(chosen) >= limit or picked >= quota:
                continue
            if want_complex is not None and (duplicate or complex_table != want_complex or row["document_sha256"] in docs):
                continue
            if want_complex is not None and signature is not None and any(signature == x[4] for x in chosen):
                continue
            chosen.append(item)
            seen.add(ordinal)
            docs.add(row["document_sha256"])
            picked += 1

    batch = []
    panels = []
    for duplicate, complex_table, _, ordinal, _, path, row, otsl in chosen:
        body = Path(row["image"]).read_bytes()
        if sha(body) != row["image_sha256"] or sha(Path(row["html"]).read_bytes()) != row["html_sha256"]:
            raise ValueError(f"Changed input for ordinal {ordinal}")
        batch.append(dict(ordinal=ordinal, table_id=row["table_id"], packet=str(path),
                          image=row["image"], otsl=row["raw_otsl"], html=row["html"],
                          image_sha256=row["image_sha256"], likely_duplicate=duplicate,
                          complex_table=complex_table))
        if sheet_dir is not None:
            batch[-1]["prediction"] = otsl
            with Image.open(BytesIO(body)) as source:
                image = source.convert("RGB")
            ink = ImageOps.invert(image).getbbox()
            if ink is not None:
                image = image.crop(ink)
            image.thumbnail((1080, 720), Image.Resampling.LANCZOS)
            panels.append((ordinal, image))
    result = dict(pending=len(ranks), likely_duplicates=sum(bool(x[0]) for x in ranks), batch=batch)
    if sheet_dir is not None:
        result["sheets"] = write_sheets(sheet_dir, panels)
    return result


def write_sheets(directory, panels):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(panels), 3):
        group = panels[start:start + 3]
        path = directory / ("-".join(str(n) for n, _ in group) + ".png")
        canvas = Image.new("RGB", (max(im.width for _, im in group) + 18,
                                   sum(im.height + 32 for _, im in group)), "white")
        draw = ImageDraw.Draw(canvas)
        top = 0
        for ordinal, image in group:
            draw.rectangle((0, top, canvas.width, top + 22), fill="#dfe8f2")
            draw.text((9, top + 5), str(ordinal), fill="black")
            canvas.paste(image, (9, top + 23))
            top += image.height + 32
        canvas.save(path)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--parity", choices=("even", "odd"))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--sheet-dir", type=Path)
    parser.add_argument("--fast-only", action="store_true")
    args = parser.parse_args()
    if args.start > args.end or args.limit < 1:
        parser.error("Invalid range or batch limit")
    parity = {None: None, "even": 0, "odd": 1}[args.parity]
    print(json.dumps(pick(args.start, args.end, parity, args.limit, args.sheet_dir, args.fast_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
