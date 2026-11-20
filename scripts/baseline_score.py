"""Freeze the paired baseline evaluation, score all held-out tables, and retain alignment evidence."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import re
from statistics import mean

from rapidfuzz.distance import Levenshtein

from scripts.evaluate import read_split, score
from scripts.ground_truth import immutable, jbytes, sha
from src.otsl.grid import parse

ROOT = Path("experiments/baseline/original_v1")
DATASET = Path("data/splits/ground_truth_v1")
OUT = Path("reports/baseline")
ARMS = ("B0", "B0-E2E")
TAG = re.compile(r"<(?:fcel|ecel|lcel|ucel|xcel|nl)>")
POLICY = dict(
    page_alignment="top-to-bottom PP table reading order only if complete region and block counts agree",
    format="raw OTSL, strict parser, no structural padding or test-reference alignment",
    teds="IBM PubTabNet TEDS/TEDS-S on outer grids; macro by reference table",
    nesting="hierarchy-aware anchors include each ancestor cell coordinate; OTSL predicts no explicit child nodes",
    text="CER on concatenated cell text in source order including nested tables; outer soft line wraps ignored in CER",
    absent="missing, ambiguous or invalid output counts as zero structural success; exact requires all descendants",
)


def load(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    return sha(path.read_bytes())


def page_blocks(text):
    # Document-mode paragraphs use blank lines. A heading can share a block with
    # the first OTSL row. Keep an unterminated final row for strict validation.
    result = []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = para.splitlines()
        active = []
        for ln in lines:
            hit = TAG.search(ln)
            if hit:
                active.append(ln[hit.start() :] if not active else ln)
            elif active:
                if ln.lstrip().startswith(("#", "~~~", "\x60\x60\x60")) or active[
                    -1
                ].rstrip().endswith("<nl>"):
                    result.append("\n".join(active).strip())
                    active = []
                elif ln.strip():
                    active.append(ln)
        if active:
            result.append("\n".join(active).strip())
    return result


def geometry(grid, path=()):
    return {(path, c.row, c.column, c.rowspan, c.colspan) for c in grid.cells}


def source_geometry(tree, path=()):
    g = geometry(parse(tree["otsl"]), path)
    count = 0
    for c in tree["cells"]:
        for i, sub in enumerate(c["children"]):
            desc, n = source_geometry(sub, path + ((c["row"], c["column"], i),))
            g.update(desc)
            count += n + 1
    return g, count


def canonical_text(txt):
    return re.sub(r"\s+", "", txt)


def tree_text(tree):
    # Text in mixed-content cells can occur both above and below an inner table.
    from lxml import html

    return canonical_text(html.fromstring(tree["html"]).text_content())


def evaluate_one(job):
    rec, arm, conf = job
    tid = rec["table_id"]
    gt_file = Path(rec["gt"])
    gt = load(gt_file)
    if (
        file_sha(gt_file) != rec["gt_sha256"]
        or gt["decision"] != "approve"
        or gt["reviewer_type"] != "human"
    ):
        raise ValueError("Reference provenance mismatch")
    m = load(Path("data/annotation/packets", tid + ".json"))
    pid = m["page_id"]
    layout_path = Path("data/layout/predictions", pid + ".json")
    layout = load(layout_path)
    layout_ids = sorted(layout["tables"], key=lambda s: int(s.rsplit("-r", 1)[1]))
    if tid not in layout_ids:
        raise ValueError("Layout crop not found")
    page_path = Path("data/rendered", m["document_sha256"], pid + ".json")
    page = load(page_path)
    source = ROOT / arm / ((tid if arm == "B0" else pid) + ".json")
    prediction = load(source)
    if prediction["config_sha256"] != conf or prediction["image_sha256"] != (
        m["image_sha256"] if arm == "B0" else page["image_sha256"]
    ):
        raise ValueError("Prediction lineage mismatch")
    if arm == "B0":
        if prediction["layout_sha256"] != file_sha(layout_path):
            raise ValueError("Layout results changed")
        raw = prediction["raw_otsl"]
        n = 1
        aligned = True
    else:
        blocks = page_blocks(prediction["raw"])
        n = len(blocks)
        aligned = n == len(layout_ids)
        raw = blocks[layout_ids.index(tid)] if aligned else ""
    result = score(raw, gt["corrected_otsl"])
    outer = parse(gt["corrected_otsl"])
    truth = gt.get("structure_tree")
    expected = geometry(outer)
    nested = 0
    text = canonical_text("".join(c.text for c in outer.cells))
    iou = None
    if truth:
        expected, nested = source_geometry(truth)
        text = tree_text(truth)
        ref = [x * page["dpi"] / 72 for x in truth["bbox_pt"]]
        box = m["bbox_px"]
        intersect = max(0, min(ref[2], box[2]) - max(ref[0], box[0])) * max(
            0, min(ref[3], box[3]) - max(ref[1], box[1])
        )
        total = (
            (ref[2] - ref[0]) * (ref[3] - ref[1])
            + (box[2] - box[0]) * (box[3] - box[1])
            - intersect
        )
        iou = intersect / total if total else None
    try:
        pred = parse(raw)
    except ValueError:
        pred = None
    predicted = geometry(pred) if pred else set()
    ptext = canonical_text("".join(c.text for c in pred.cells)) if pred else ""
    result.update(
        table_id=tid,
        page_id=pid,
        weight_arm=arm,
        source_sha256=file_sha(source),
        gt_sha256=rec["gt_sha256"],
        raw_extraction=raw,
        raw_block_count=n,
        layout_table_count=len(layout_ids),
        aligned=aligned,
        prediction_eos=prediction["ended_with_eos"],
        recognition_seconds=prediction["seconds"],
        peak_allocated_bytes=prediction["peak_allocated_bytes"],
        layout_seconds=layout["runtime_seconds"],
        reference_iou=iou,
        reference_subtables=nested,
        character_edits=Levenshtein.distance(text, ptext),
        reference_characters=len(text),
        full_exact=bool(result["exact"] and not nested),
        matched_cells=len(expected & predicted),
        predicted_cells=len(predicted),
        true_cells=len(expected),
    )
    return result


def aggregate(samples):
    n = len(samples)
    ps = sum(s["predicted_cells"] for s in samples)
    ts = sum(s["true_cells"] for s in samples)
    tp = sum(s["matched_cells"] for s in samples)
    physical = {s["page_id"]: s for s in samples}
    arm = samples[0]["weight_arm"]
    gpu_seconds = sum(
        s["recognition_seconds"]
        for s in (samples if arm == "B0" else physical.values())
    )
    layout_seconds = (
        sum(s["layout_seconds"] for s in physical.values()) if arm == "B0" else 0
    )
    ious = [s["reference_iou"] for s in samples if s["reference_iou"] is not None]
    nested = sum(s["reference_subtables"] for s in samples)
    return dict(
        tables=n,
        pages=len(physical),
        aligned=sum(s["aligned"] for s in samples),
        reference_subtables=nested,
        nested_link_recall=0 if nested else None,
        nested_link_f1=0 if nested else None,
        **{
            k: mean(s[k] for s in samples)
            for k in [
                "teds",
                "teds_s",
                "row_accuracy",
                "column_accuracy",
                "span_exact",
                "parse_ok",
                "full_exact",
            ]
        },
        cell_precision=tp / ps if ps else 0,
        cell_recall=tp / ts,
        cell_f1=2 * tp / (ps + ts),
        cer=sum(s["character_edits"] for s in samples)
        / max(1, sum(s["reference_characters"] for s in samples)),
        gpu_batch_seconds_allocated=gpu_seconds,
        archived_layout_seconds=layout_seconds,
        inference_seconds_per_page=(gpu_seconds + layout_seconds) / len(physical),
        peak_allocated_gib=max(s["peak_allocated_bytes"] for s in samples) / 1024**3,
        source_box_subgroup=len(ious),
        source_box_mean_iou=mean(ious) if ious else None,
        sources_with_iou50=sum(v >= 0.5 for v in ious),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "evaluate"])
    args = ap.parse_args()
    splits = {name: read_split(DATASET, name) for name in ["test", "hard_test"]}
    freeze = ROOT / "evaluation_lock.json"
    lock = dict(
        dataset=file_sha(DATASET / "manifest.json"),
        phase3=file_sha(Path("data/annotation/phase3_v1/completion.json")),
        inference=file_sha(ROOT / "config.json"),
        policy=POLICY,
        code={
            str(p): file_sha(p)
            for p in map(
                Path,
                [
                    __file__,
                    "scripts/evaluate.py",
                    "src/otsl/grid.py",
                    "src/evaluation/vendor/pubtabnet_teds.py",
                ],
            )
        },
    )
    if args.action == "freeze":
        immutable(freeze, jbytes(lock))
        return
    if load(freeze) != lock:
        raise ValueError("Frozen evaluation changed")
    hard = {r["table_id"] for r in splits["hard_test"]}
    for arm in ARMS:
        with ProcessPoolExecutor(max_workers=6) as pool:
            rows = list(
                pool.map(
                    evaluate_one, [(r, arm, lock["inference"]) for r in splits["test"]]
                )
            )
        metrics = {
            k: aggregate(v)
            for k, v in [
                ("test", rows),
                ("hard_test", [r for r in rows if r["table_id"] in hard]),
            ]
        }
        immutable(OUT / (arm + "_samples.jsonl"), b"".join(jbytes(x) for x in rows))
        immutable(
            OUT / (arm + "_metrics.json"),
            jbytes(dict(arm=arm, metrics=metrics, lock_sha256=file_sha(freeze))),
        )
        print(arm, metrics["test"], flush=True)


if __name__ == "__main__":
    main()
