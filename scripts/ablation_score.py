"""Score paired detector, recognizer and repair runs against the frozen pilot."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from rapidfuzz.distance import Levenshtein

from scripts import baseline_score as baseline
from scripts.evaluate import read_split, score
from scripts.ground_truth import immutable, jbytes, sha
from src.otsl.grid import parse

ROOT = Path("experiments/ablation")
OUT = Path("reports/evaluation/pilot1k")
PLANS = {
    "B1": ("fine_original", False),
    "B2": ("original_lora", False),
    "B3": ("fine_lora", False),
    "B4": ("B0", True),
    "B5": ("fine_original", True),
    "B6": ("fine_lora", True),
}
DATA = baseline.DATASET
load = baseline.load


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def lock():
    layout = load(Path("experiments/layout_finetune/pilot_server/selected.json"))
    chosen = next(t for t in layout["trials"] if t["name"] == layout["selected"])
    adapters = sorted(
        Path("models/xiaomi_lora").glob("*/best/adapter_model.safetensors")
    )
    if len(adapters) != 1:
        raise ValueError("One validation-selected LoRA checkpoint required")
    files = [
        Path(n)
        for n in [
            "scripts/ablation_score.py",
            "scripts/ablation_layout.py",
            "scripts/layout_detection_metrics.py",
            "experiments/ablation/cohorts/content_overlap.json",
            "scripts/ablation_recognition.py",
            "src/training/constrained_repair.py",
            "scripts/baseline_score.py",
            "scripts/evaluate.py",
            "src/otsl/grid.py",
            "src/evaluation/vendor/pubtabnet_teds.py",
            "models/otsl_repair/server_v1/best.pt",
            "experiments/ablation/repair_validation.json",
        ]
    ]
    return dict(
        arms=PLANS,
        split_sha256=baseline.file_sha(DATA / "manifest.json"),
        baseline_lock=baseline.file_sha(baseline.ROOT / "evaluation_lock.json"),
        selected_layout_sha256=chosen["best_sha256"],
        selected_adapter_sha256=baseline.file_sha(adapters[0]),
        matching="fine detector: greedy descending IoU, one-to-one within page, threshold 0.50; missing table scores zero",
        reference_extent="source-PDF extent when independently reconstructed, otherwise the reviewed crop",
        fine_threshold=0.25,
        validation_only_selection=True,
        files={str(p): baseline.file_sha(p) for p in files},
    )


def jobs(arm):
    source, repair = PLANS[arm]
    fine = source.startswith("fine_")
    groups = defaultdict(list)
    for row in read_split(DATA, "test"):
        tid = row["table_id"]
        pg = load(Path("data/annotation/packets", tid + ".json"))
        ref_path = Path(row["gt"])
        if baseline.file_sha(ref_path) != row["gt_sha256"]:
            raise ValueError("Reference changed")
        gt = load(ref_path)
        if gt["reviewer_type"] != "human" or gt["decision"] != "approve":
            raise ValueError("Invalid pilot reference")
        pid = pg["page_id"]
        tree = gt.get("structure_tree")
        if tree:
            dpi = load(Path("data/rendered", pg["document_sha256"], pid + ".json"))[
                "dpi"
            ]
            box = [v * dpi / 72 for v in tree["bbox_pt"]]
        else:
            box = pg["bbox_px"]
        groups[pid].append((row, box, bool(tree), pg))
    for pid, refs in sorted(groups.items()):
        if fine:
            page = load(ROOT / "fine_layout/pages" / (pid + ".json"))
            boxes = page["tables"]
            matches = {}
            used = set()
            pairs = sorted(
                (
                    (
                        -overlap(ref[1], b["coordinate"]),
                        ref[0]["table_id"],
                        b["table_id"],
                    )
                    for ref in refs
                    for b in boxes
                )
            )
            for neg, tid, pred_id in pairs:
                if -neg < 0.5:
                    break
                if tid not in matches and pred_id not in used:
                    matches[tid] = (pred_id, -neg)
                    used.add(pred_id)
            layout_seconds = page["seconds"]
        else:
            page = load(Path("data/layout/predictions", pid + ".json"))
            boxes = page["tables"]
            layout_seconds = page["runtime_seconds"]
            matches = {
                r[0]["table_id"]: (r[0]["table_id"], overlap(r[1], r[3]["bbox_px"]))
                for r in refs
            }
        for row, _, independent, _ in refs:
            tid = row["table_id"]
            pred_id, iou = matches.get(tid, (None, 0))
            yield (
                arm,
                source,
                repair,
                pid,
                row,
                pred_id,
                iou if independent else None,
                layout_seconds,
                len(boxes),
            )


def sample(args):
    arm, source, repair, pid, rec, pred_id, iou, layout_seconds, n_boxes = args
    gt = load(Path(rec["gt"]))
    raw = ""
    pred = {}
    repair_result = {}
    if pred_id:
        directory = (
            baseline.ROOT / "B0" if source == "B0" else ROOT / source / "predictions"
        )
        p = directory / (pred_id + ".json")
        pred = load(p)
        if pred["table_id"] != pred_id:
            raise ValueError("Prediction identity mismatch")
        raw = pred["raw_otsl"]
        if repair:
            repair_result = load(ROOT / "repaired" / source / (pred_id + ".json"))
            if repair_result["raw_sha256"] != sha(raw.encode()) or repair_result[
                "source_sha256"
            ] != baseline.file_sha(p):
                raise ValueError("Repair lineage mismatch")
            raw = repair_result["repaired_otsl"]
    result = score(raw, gt["corrected_otsl"])
    outer = parse(gt["corrected_otsl"])
    tree = gt.get("structure_tree")
    gold, nested = (
        baseline.source_geometry(tree) if tree else (baseline.geometry(outer), 0)
    )
    text = (
        baseline.tree_text(tree)
        if tree
        else baseline.canonical_text("".join(c.text for c in outer.cells))
    )
    parsed = parse(raw) if result["parse_ok"] else None
    geometry = baseline.geometry(parsed) if parsed else set()
    ocr = (
        baseline.canonical_text("".join(c.text for c in parsed.cells)) if parsed else ""
    )
    return dict(
        result,
        table_id=rec["table_id"],
        page_id=pid,
        weight_arm=arm,
        prediction_table_id=pred_id,
        gt_sha256=rec["gt_sha256"],
        aligned=bool(pred_id),
        reference_iou=iou,
        reference_subtables=nested,
        character_edits=Levenshtein.distance(text, ocr),
        reference_characters=len(text),
        full_exact=bool(result["exact"] and not nested),
        matched_cells=len(gold & geometry),
        predicted_cells=len(geometry),
        true_cells=len(gold),
        layout_seconds=layout_seconds,
        layout_table_count=n_boxes,
        recognition_seconds=pred.get("seconds", 0),
        repair_seconds=repair_result.get("seconds", 0),
        repair_decision=repair_result.get("decision"),
        peak_allocated_bytes=pred.get("peak_allocated_bytes", 0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "score"])
    args = ap.parse_args()
    locked = ROOT / "evaluation_lock.json"
    config = lock()
    if args.action == "freeze":
        immutable(locked, jbytes(config))
        return
    if locked.read_bytes() != jbytes(config):
        raise ValueError("Held-out evaluation lock changed")
    hard = {r["table_id"] for r in read_split(DATA, "hard_test")}
    for arm in PLANS:
        with ProcessPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(sample, jobs(arm)))
        stats = {}
        for name, subset in [
            ("test", rows),
            ("hard_test", [r for r in rows if r["table_id"] in hard]),
        ]:
            metrics = baseline.aggregate(subset)
            # The frozen baseline's latency calculation distinguishes its own two arms.
            # Full ablations also recognise predictions without reference matches.
            pages = {r["page_id"]: r for r in subset}
            source, repair = PLANS[arm]
            ids = (
                {
                    b["table_id"]
                    for pid in pages
                    for b in load(ROOT / "fine_layout/pages" / (pid + ".json"))[
                        "tables"
                    ]
                }
                if source.startswith("fine_")
                else {r["table_id"] for r in subset}
            )
            directory = (
                baseline.ROOT / "B0"
                if source == "B0"
                else ROOT / source / "predictions"
            )
            preds = [load(directory / (k + ".json")) for k in ids]
            runtime = sum(p["seconds"] for p in preds)
            rtime = (
                sum(
                    load(ROOT / "repaired" / source / (k + ".json"))["seconds"]
                    for k in ids
                )
                if repair
                else 0
            )
            layout_time = sum(v["layout_seconds"] for v in pages.values())
            metrics.update(
                recognition_seconds=runtime,
                layout_seconds=layout_time,
                repair_seconds=rtime,
                gpu_batch_seconds_allocated=runtime,
                archived_layout_seconds=layout_time
                if not source.startswith("fine_")
                else 0,
                inference_seconds_per_page=(runtime + layout_time + rtime) / len(pages),
                peak_allocated_gib=max(
                    (p["peak_allocated_bytes"] for p in preds), default=0
                )
                / 1024**3,
                recall_on_reviewed_tables=mean(r["aligned"] for r in subset),
                candidate_tables=len(ids),
            )
            stats[name] = metrics
        immutable(OUT / (arm + "_samples.jsonl"), b"".join(jbytes(row) for row in rows))
        immutable(
            OUT / (arm + "_metrics.json"),
            jbytes(dict(arm=arm, metrics=stats, lock_sha256=baseline.file_sha(locked))),
        )
        print(arm, stats["test"]["teds_s"], flush=True)


if __name__ == "__main__":
    main()
