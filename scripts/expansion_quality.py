"""Audit each visual response, reconstruct flagged sources and materialize only attributable usable records."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

from scripts.expansion_queue import EXP, PACKETS
from scripts.freeze_pilot_repairs import verify_tree
from scripts.ground_truth import immutable, jbytes, sha
from scripts.repair_structures import repair
from scripts.review import validator
from src.otsl.grid import parse

OUT = EXP / "quality/records"
REV = EXP / "annotation/raw"


def compact(s):
    return "".join(unicodedata.normalize("NFKC", s).split())


def fingerprint(grid, tree=None):
    # Keep the attachment point and descendants when comparing two references.
    def key(g, t):
        return [
            (
                c.row,
                c.column,
                c.rowspan,
                c.colspan,
                compact(c.text),
                [key(parse(k["otsl"]), k) for k in t["cells"][i]["children"]]
                if t
                else [],
            )
            for i, c in enumerate(g.cells)
        ]

    return sha(jbytes([grid.rows, grid.columns, key(grid, tree)]))


def glyphs(grid, tree=None):
    parts = []
    for i, c in enumerate(grid.cells):
        parts.append(c.text)
        if tree:
            for sub in tree["cells"][i]["children"]:
                parts.append(glyphs(parse(sub["otsl"]), sub))
    return "".join(parts)


def audit(packet, response):
    tid = packet.stem
    m = json.loads(packet.read_bytes())
    rev = json.loads(response.read_bytes())
    validator.validate(rev)
    if rev["table_id"] != tid or m["table_id"] != tid:
        raise ValueError("Review binding")
    old = Path(m["prediction"])
    raw = json.loads(old.read_bytes())
    image = Path(m["image"])
    if (
        sha(image.read_bytes()) != m["image_sha256"]
        or sha(old.read_bytes()) != m["prediction_sha256"]
        or sha(Path(m["html"]).read_bytes()) != m["html_sha256"]
        or Path(m["raw_otsl"]).read_text() != raw["raw_otsl"]
    ):
        raise ValueError("Input changed after packet creation")
    doc = m["document_sha256"]
    data_root = (
        Path("data")
        if (Path("data/raw/pdf") / (doc + ".pdf")).exists()
        else EXP / "data"
    )
    pg = data_root / "rendered" / doc / (m["page_id"] + ".json")
    rec = dict(
        table_id=tid,
        ordinal=m["ordinal"],
        document_sha256=doc,
        page_id=m["page_id"],
        page_number=m["page_number"],
        image=str(image),
        image_sha256=m["image_sha256"],
        source_page=str(pg),
        bbox_px=m["bbox_px"],
        review_sha256=sha(response.read_bytes()),
        packet_sha256=sha(packet.read_bytes()),
        author="machine",
        label_tier="visually_adjudicated_machine",
        human_approved=False,
        decision="hold",
        visual_quality=rev["visual_quality"],
        flags={
            k: rev[k]
            for k in [
                "cell_errors",
                "header_errors",
                "merge_errors",
                "text_errors",
                "cross_page_candidates",
                "nested_table_candidates",
            ]
            if rev[k]
        },
    )
    fix = EXP / "annotation/visual_fixes" / (tid + ".json")
    if fix.exists():
        v = json.loads(fix.read_bytes())
        if (
            v["image_sha256"] != m["image_sha256"]
            or v["review_sha256"] != rec["review_sha256"]
        ):
            raise ValueError("Stale visual correction")
        if v["decision"] == "exclude":
            return dict(rec, decision="exclude", reason=v["reason"])
        tree = v.get("structure_tree")
        out = v["corrected_otsl"]
        grid = parse(out)
        if tree and (tree["otsl"] != out or verify_tree(tree) < 0):
            raise ValueError("Visual tree mismatch")
        action = "visual_correction"
    else:
        out = raw["raw_otsl"]
        tree = None
        try:
            grid = parse(out)
        except ValueError:
            grid = None
        needs = (
            not rev["structure_valid"]
            or not grid
            or raw.get("structure_checks")
            or any(
                rev[k]
                for k in [
                    "cell_errors",
                    "header_errors",
                    "merge_errors",
                    "text_errors",
                    "nested_table_candidates",
                ]
            )
        )
        if needs:
            r = dict(m, packet=str(packet), packet_sha256=rec["packet_sha256"])
            try:
                tree = repair(r, data_root)
                nested = verify_tree(tree)
                if rev["nested_table_candidates"] and not nested:
                    raise ValueError("Nested warning without source-supported child")
                natural = glyphs(parse(tree["otsl"]), tree)
                if not compact(natural):
                    raise ValueError("No usable PDF text in frame")
                original = grid and glyphs(grid) or re.sub("<[^>]+>", "", out)

                def chars(s):
                    return Counter(c for c in compact(s) if c.isalnum())

                seen, source = chars(original), chars(natural)
                if sum((seen - source).values()) > max(10, 0.12 * sum(seen.values())):
                    raise ValueError("Source text coverage gap")
                if tree["columns"] * tree["rows"] <= 1:
                    raise ValueError("Single-cell frame needs visual disposition")
                out = tree["otsl"]
                grid = parse(out)
                action = "pdf_reconstruction"
            except ValueError as exc:
                return dict(rec, reason=str(exc))
        else:
            action = "visual_accept_original"
        if rev["visual_quality"] == "poor":
            return dict(
                rec, reason="Poor visual quality; verify legibility before inclusion"
            )
    if not compact(glyphs(grid, tree)):
        return dict(rec, reason="Empty cell content")
    return dict(
        rec,
        decision="usable",
        action=action,
        corrected_otsl=out,
        structure_tree=tree,
        rows=grid.rows,
        columns=grid.columns,
        content_signature=fingerprint(grid, tree),
    )


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(REV.glob("*.json")):
        p = PACKETS / f.name
        dest = OUT / f.name
        if not p.exists():
            raise ValueError("Missing packet")
        fix = EXP / "annotation/visual_fixes" / f.name
        if (
            dest.exists()
            and max(
                f.stat().st_mtime,
                p.stat().st_mtime,
                fix.stat().st_mtime if fix.exists() else 0,
            )
            <= dest.stat().st_mtime
        ):
            continue
        result = audit(p, f)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(jbytes(result))
        tmp.replace(dest)
    counts = Counter()
    sigs = set()
    for f in OUT.glob("*.json"):
        d = json.loads(f.read_bytes())
        counts[d["decision"]] += 1
        if d["decision"] == "usable":
            sigs.add(d["content_signature"])
    return dict(counts, unique_content=len(sigs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    while True:
        print(run(), flush=True)
        if not args.watch:
            break
        time.sleep(25)
