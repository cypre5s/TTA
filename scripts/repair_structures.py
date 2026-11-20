"""Reconstruct suspected tables from the original PDF ruling paths and positioned glyphs."""

from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import median
import argparse
import hashlib
import json

import pymupdf as pdf

from scripts.ground_truth import jbytes, sha
from src.otsl.grid import parse

OUT = Path("data/tables/structure_proposals")
QUEUE = Path("data/annotation/human_review_queue.jsonl")
TOL = 1.6


def clusters(values):
    groups = []
    for v in sorted(set(values)):
        if not groups or v - groups[-1][0] > TOL:
            groups.append([])
        groups[-1].append(v)
    return [median(g) for g in groups]


def lines(page, clip):
    hs, vs = [], []

    def add(a, b):
        x, y = a
        xx, yy = b
        if abs(y - yy) < 0.35 and abs(x - xx) > 3:
            x0, x1 = sorted((x, xx))
            if clip.y0 <= y <= clip.y1 and x0 >= clip.x0 and x1 <= clip.x1:
                hs.append((y, x0, x1))
        if abs(x - xx) < 0.35 and abs(y - yy) > 3:
            y0, y1 = sorted((y, yy))
            if clip.x0 <= x <= clip.x1 and y0 >= clip.y0 and y1 <= clip.y1:
                vs.append((x, y0, y1))

    for d in page.get_drawings():
        color = d.get("color") or d.get("fill")
        if not color or max(color) > 0.78:
            continue
        for op, *args in d["items"]:
            if op == "l" and "s" in d["type"]:
                add(args[0], args[1])
            elif op == "re":
                r = args[0]
                if r.height < 1.8:
                    add((r.x0, (r.y0 + r.y1) / 2), (r.x1, (r.y0 + r.y1) / 2))
                elif r.width < 1.8:
                    add(((r.x0 + r.x1) / 2, r.y0), ((r.x0 + r.x1) / 2, r.y1))
                elif "s" in d["type"]:
                    for a, b in [
                        (r.tl, r.tr),
                        (r.bl, r.br),
                        (r.tl, r.bl),
                        (r.tr, r.br),
                    ]:
                        add(a, b)
    return hs, vs


def touches(edges, pos, at):
    return any(
        abs(p - pos) <= TOL and lo - TOL <= at <= hi + TOL for p, lo, hi in edges
    )


def continuous(edges, pos, start, end):
    stop = start
    for a, b in sorted(
        (a, b) for p, a, b in edges if abs(p - pos) <= TOL and b >= start - TOL
    ):
        if a > stop + TOL:
            break
        stop = max(stop, b)
        if stop >= end - TOL:
            return True
    return False


def candidates(h, v):
    def join(edges):
        out = []
        for p in clusters(x[0] for x in edges):
            spans = sorted((a, b) for q, a, b in edges if abs(q - p) <= TOL)
            if not spans:
                continue
            lo, hi = spans[0]
            for a, b in spans[1:]:
                if a <= hi + TOL:
                    hi = max(hi, b)
                else:
                    out.append((p, lo, hi))
                    lo, hi = a, b
            out.append((p, lo, hi))
        return list(set(edges + out))

    h, v = join(h), join(v)
    h = [(y, a, b) for y, a, b in h if touches(v, a, y) and touches(v, b, y)]
    bounds = []
    for y, a, b in sorted(h, key=lambda e: e[2] - e[1], reverse=True):
        levels = clusters(z for z, l, r in h if abs(l - a) <= TOL and abs(r - b) <= TOL)
        if len(levels) < 2 or b - a < 14:
            continue
        runs = [[levels[0]]]
        for lower, upper in zip(levels, levels[1:]):
            if continuous(v, a, lower, upper) and continuous(v, b, lower, upper):
                runs[-1].append(upper)
            else:
                runs.append([upper])
        for run in runs:
            top, bot = run[0], run[-1]
            if bot - top < 9:
                continue
            rect = pdf.Rect(a, top, b, bot)
            if not any(
                max(abs(a - b) for a, b in zip(rect, other)) <= TOL for other in bounds
            ):
                bounds.append(rect)
    bounds.sort(key=lambda r: r.get_area(), reverse=True)
    return bounds, h, v


def inside(box, other, gap=0):
    return all(
        (
            box.x0 - gap <= other.x0,
            box.y0 - gap <= other.y0,
            box.x1 + gap >= other.x1,
            box.y1 + gap >= other.y1,
        )
    )


def permitted_child(box, child):
    return (
        inside(box, child, TOL)
        and child.x0 > box.x0 + 2 * TOL
        and child.x1 < box.x1 - 2 * TOL
    )


def glyph_lines(page, clip):
    result = []
    for block in page.get_text(
        "rawdict", clip=clip, flags=pdf.TEXTFLAGS_RAWDICT & ~pdf.TEXT_PRESERVE_IMAGES
    )["blocks"]:
        for line in block.get("lines", []):
            chars = []
            for span in line["spans"]:
                for c in span["chars"]:
                    b = pdf.Rect(c["bbox"])
                    chars.append(((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, c["c"]))
            if chars:
                result.append(chars)
    return result


def emit_otsl(cells, rows, cols):
    slots = [[""] * cols for _ in range(rows)]
    for c in cells:
        r, k = c["row"], c["column"]
        for i in range(c["rowspan"]):
            for j in range(c["colspan"]):
                if slots[r + i][k + j]:
                    raise ValueError("overlapping cells")
                if not i and not j:
                    token = "<fcel>" + c["text"] if c["text"] else "<ecel>"
                elif i and j:
                    token = "<xcel>"
                else:
                    token = "<ucel>" if i else "<lcel>"
                slots[r + i][k + j] = token
    raw = "".join("".join(row) + "<nl>" for row in slots)
    parse(raw)
    return raw


def build(box, remaining, hs, vs, glyphs):
    child_boxes = []
    for candidate in remaining:
        if not permitted_child(box, candidate):
            continue
        if not any(
            candidate.x0 + TOL < x < candidate.x1 - TOL
            and candidate.y0 - TOL <= a < b <= candidate.y1 + TOL
            for x, a, b in vs
        ):
            continue
        if any(
            candidate.y0 + TOL < y < candidate.y1 - TOL
            and a < candidate.x0 - TOL
            and b > candidate.x1 + TOL
            for y, a, b in hs
        ):
            continue
        if not any(inside(b, candidate, TOL) for b in child_boxes):
            child_boxes.append(candidate)

    def contained(edge, b, vertical):
        pos, a, z = edge
        if vertical:
            return b.x0 - TOL <= pos <= b.x1 + TOL and b.y0 - TOL <= a < z <= b.y1 + TOL
        return b.y0 - TOL <= pos <= b.y1 + TOL and b.x0 - TOL <= a < z <= b.x1 + TOL

    def edges(lines, vertical):
        return [
            e
            for e in lines
            if contained(e, box, vertical)
            and not any(contained(e, b, vertical) for b in child_boxes)
        ]

    h, v = edges(hs, False), edges(vs, True)
    # Ignore isolated underlines and decorative strokes with no cross-points.
    v = [(x, a, b) for x, a, b in v if touches(h, a, x) and touches(h, b, x)]
    h = [(y, a, b) for y, a, b in h if touches(v, a, y) and touches(v, b, y)]
    xs = clusters([box.x0, box.x1] + [p for p, _, _ in v])
    ys = clusters([box.y0, box.y1] + [p for p, _, _ in h])
    rows, cols = len(ys) - 1, len(xs) - 1
    if rows * cols > 12000:
        raise ValueError("excessive grid size")
    parent = list(range(rows * cols))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    regions = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            if r and not touches(h, ys[r], (xs[c] + xs[c + 1]) / 2):
                parent[find(k)] = find(k - cols)
            if c and not touches(v, xs[c], (ys[r] + ys[r + 1]) / 2):
                parent[find(k)] = find(k - 1)
    for k in range(rows * cols):
        regions[find(k)].append(divmod(k, cols))
    cells = []
    assigned = 0
    for indices in regions.values():
        r, c = indices[0]
        ri, ci = map(max, zip(*indices))
        if len(indices) != (ri - r + 1) * (ci - c + 1):
            raise ValueError("nonrectangular border component")
        b = pdf.Rect(xs[c], ys[r], xs[ci + 1], ys[ri + 1])
        subs = [s for s in child_boxes if inside(b, s, TOL)]
        assigned += len(subs)
        pieces = []
        for line in glyphs:
            part = "".join(
                ch
                for x, y, ch in line
                if b.x0 <= x < b.x1
                and b.y0 <= y < b.y1
                and not any(s.x0 <= x < s.x1 and s.y0 <= y < s.y1 for s in subs)
            )
            if part.strip():
                pieces.append((median(y for _, y, _ in line), part.strip()))
        children = [
            build(s, [z for z in remaining if permitted_child(s, z)], hs, vs, glyphs)
            for s in subs
        ]
        ordered = [(y, escape(val)) for y, val in pieces] + [
            (s.y0, node["html"]) for s, node in zip(subs, children)
        ]
        cells.append(
            dict(
                row=r,
                column=c,
                rowspan=ri - r + 1,
                colspan=ci - c + 1,
                bbox_pt=list(b),
                text="\n".join(val for _, val in pieces),
                children=children,
                content_html="<br>".join(
                    val for _, val in sorted(ordered, key=lambda t: t[0])
                ),
            )
        )
    if assigned != len(child_boxes):
        raise ValueError("nested grid crosses an outer cell boundary")
    html = ["<table>"]
    for r in range(rows):
        html.append("<tr>")
        for c in cells:
            if c["row"] == r:
                html.append(
                    f'<td rowspan="{c["rowspan"]}" colspan="{c["colspan"]}">{c["content_html"]}</td>'
                )
        html.append("</tr>")
    html.append("</table>")
    return dict(
        bbox_pt=list(box),
        rows=rows,
        columns=cols,
        cells=cells,
        otsl=emit_otsl(cells, rows, cols),
        html="".join(html),
    )


def repair(r, data_root=Path("data")):
    packet = Path(r["packet"])
    meta = json.loads(packet.read_text())
    source = data_root / "raw/pdf" / (r["document_sha256"] + ".pdf")
    if (
        sha(packet.read_bytes()) != r["packet_sha256"]
        or sha(Path(meta["image"]).read_bytes()) != meta["image_sha256"]
    ):
        raise ValueError("packet/crop hash mismatch")
    with source.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != r["document_sha256"]:
            raise ValueError("source hash mismatch")
    page_meta = json.loads(
        (
            data_root / "rendered" / r["document_sha256"] / (meta["page_id"] + ".json")
        ).read_text()
    )
    with pdf.open(source) as doc:
        page = doc[r["page_number"] - 1]
        if page.rotation:
            raise ValueError("rotated PDF geometry")
        crop = pdf.Rect([x * 72 / page_meta["dpi"] for x in r["bbox_px"]])
        area = (crop + (-7, -7, 7, 7)) & page.rect
        hs, vs = lines(page, area)
        # Continuation pages can have aligned side edges without a top/bottom rule.
        virtual = []
        if vs:
            left, right = min(x for x, _, _ in vs), max(x for x, _, _ in vs)
            for which in [1, 2]:
                near = [z[which] for z in vs if abs(z[0] - left) <= TOL]
                target = min(near) if which == 1 else max(near)
                if any(
                    abs(z[0] - right) <= TOL and abs(z[which] - target) <= TOL
                    for z in vs
                ) and not any(
                    abs(y - target) <= TOL
                    and abs(a - left) <= TOL
                    and abs(b - right) <= TOL
                    for y, a, b in hs
                ):
                    hs.append((target, left, right))
                    virtual.append(target)
        bounds, hs, vs = candidates(hs, vs)
        if not bounds:
            raise ValueError("no complete ruled frame")
        box = bounds[0]
        if box.get_area() / crop.get_area() < 0.73:
            raise ValueError("largest frame covers too little of the original crop")
        tree = build(box, bounds[1:], hs, vs, glyph_lines(page, area))
        tree["inferred_boundary_y_pt"] = virtual
    return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinals", type=int, nargs="+")
    parser.add_argument("--triage", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    frozen_queue = Path("data/annotation/phase3_v1/packet_index.jsonl")
    source_queue = frozen_queue if frozen_queue.exists() else QUEUE
    for line in source_queue.read_text().splitlines():
        r = json.loads(line)
        extra = r["structure_valid"] and bool(
            {"nested_table_candidates", "merge_errors", "header_errors", "cell_errors"}
            & set(r["flagged_fields"])
        )
        if r["structure_valid"] and not (args.triage and extra):
            continue
        if args.ordinals and r["ordinal"] not in args.ordinals:
            continue
        entry = {
            k: r[k]
            for k in (
                "ordinal",
                "table_id",
                "document_sha256",
                "page_number",
                "packet_sha256",
                "flagged_fields",
            )
        }
        entry.update(
            authorship="machine",
            review_status="pending_human",
            source="PDF ruling paths and text layer",
        )
        try:
            correction = Path(
                "data/annotation/visual_corrections", r["table_id"] + ".json"
            )
            if correction.exists():
                override = json.loads(correction.read_bytes())
                if override["image_sha256"] != sha(Path(r["image"]).read_bytes()):
                    raise ValueError("visual correction source changed")
                entry.update(override)
            else:
                tree = repair(r)
                entry.update(candidate=tree, state="proposed")
        except ValueError as exc:
            entry.update(state="needs_visual_repair", reason=str(exc))
        if extra:
            entry["triage_only"] = True
        counts[entry["state"]] += 1
        dest = OUT / (r["table_id"] + ".json")
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(jbytes(entry))
        tmp.replace(dest)
    print(dict(counts))


if __name__ == "__main__":
    main()
