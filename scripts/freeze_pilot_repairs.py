"""Seal a repaired pilot against a separately supplied, attributable batch confirmation."""

from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import json

from scripts.ground_truth import ingest, split, immutable, jbytes, sha
from scripts.repair_structures import OUT, emit_otsl
from src.otsl.grid import parse

FROZEN = Path("data/annotation/phase3_v1")
DATASET = Path("data/splits/ground_truth_v1")


def read(path):
    return json.loads(path.read_bytes())


def signature(grid):
    return [(c.row, c.column, c.rowspan, c.colspan) for c in grid.cells]


def verify_tree(node):
    cells = node["cells"]
    grid = parse(node["otsl"])
    if (grid.rows, grid.columns) != (node["rows"], node["columns"]) or emit_otsl(
        cells, grid.rows, grid.columns
    ) != node["otsl"]:
        raise ValueError("Grid geometry inconsistent")
    nested = 0
    for cell in cells:
        box = cell["bbox_pt"]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Invalid cell bounds")
        for child in cell["children"]:
            b = child["bbox_pt"]
            if min(b[0] - box[0], b[1] - box[1], box[2] - b[2], box[3] - b[3]) < -2:
                raise ValueError("Child table outside parent cell")
            nested += 1 + verify_tree(child)
    return nested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation", required=True, type=Path)
    args = ap.parse_args()
    receipt = read(args.confirmation)
    if (
        not receipt["reviewer"].strip()
        or not receipt["statement"].strip()
        or datetime.fromisoformat(receipt["recorded_at"]).tzinfo is None
    ):
        raise ValueError("Missing attributable confirmation")
    confirm_sha = sha(args.confirmation.read_bytes())
    queue = FROZEN / "packet_index.jsonl"
    if not queue.exists():
        immutable(queue, Path("data/annotation/human_review_queue.jsonl").read_bytes())
    q = [json.loads(l) for l in queue.read_bytes().splitlines()]
    if (
        len(q) != 1000
        or len({r["table_id"] for r in q}) != 1000
        or {r["ordinal"] for r in q} != set(range(1, 1001))
    ):
        raise ValueError("Incomplete pilot index")
    rows = []
    ledger = []
    counts = Counter()
    for r in q:
        packet = Path(r["packet"])
        m = read(packet)
        if (
            sha(packet.read_bytes()) != r["packet_sha256"]
            or sha(Path(m["image"]).read_bytes()) != m["image_sha256"]
            or sha(Path(m["prediction"]).read_bytes()) != m["prediction_sha256"]
        ):
            raise ValueError(f"Source mismatch for {r['ordinal']}")
        raw = Path(m["raw_otsl"]).read_text()
        if raw != read(Path(m["prediction"]))["raw_otsl"]:
            raise ValueError("Raw OTSL mismatch")
        original = parse(raw)
        path = OUT / (r["table_id"] + ".json")
        item = read(path) if path.exists() else None
        if not r["structure_valid"] and not item:
            raise ValueError("Unresolved structural suspicion")
        if item and (
            item["packet_sha256"] != r["packet_sha256"]
            or item["state"] not in ("proposed", "proposed_exclusion")
        ):
            raise ValueError(f"Unresolved proposal {r['ordinal']}")
        excluded = item and item["state"] == "proposed_exclusion"
        tree = item.get("candidate") if item else None
        nested = verify_tree(tree) if tree else 0
        # Additional cautions attached to a "valid" visual review only change
        # the accepted structure when the source provides a concrete discrepancy.
        structural_delta = tree and (
            nested or signature(parse(tree["otsl"])) != signature(original)
        )
        apply = tree and (not r["structure_valid"] or structural_delta)
        action = (
            "exclude_non_table"
            if excluded
            else "source_structure"
            if apply
            else "retain_reviewed_original"
        )
        decision = dict(
            table_id=r["table_id"],
            packet_sha256=r["packet_sha256"],
            reviewer_type="human",
            reviewer=receipt["reviewer"],
            reviewed_at=receipt["recorded_at"],
            confirmation_sha256=confirm_sha,
            confirmation_scope="one user-supplied batch confirmation; recorded_at is receipt time",
            correction_author="assistant" if apply or excluded else "pilot OCR",
            decision="reject" if excluded else "approve",
            revision=1,
            action=action,
            difficulty=r["flagged_fields"],
            original_prediction_sha256=m["prediction_sha256"],
        )
        if excluded:
            decision["reason"] = item["reason"]
        else:
            decision["corrected_otsl"] = tree["otsl"] if apply else raw
            if apply:
                decision["structure_tree"] = tree
        if item:
            decision["proposal_sha256"] = sha(path.read_bytes())
        rows.append(decision)
        ledger.append(
            dict(
                ordinal=r["ordinal"],
                table_id=r["table_id"],
                original_structure_flag=not r["structure_valid"],
                original_dimensions=[original.rows, original.columns],
                final_dimensions=(
                    [tree["rows"], tree["columns"]]
                    if apply
                    else None
                    if excluded
                    else [original.rows, original.columns]
                ),
                nested_tables=nested if apply else 0,
                action=action,
                crop_sha256=m["image_sha256"],
                proposal_sha256=decision.get("proposal_sha256"),
            )
        )
        counts[action] += 1
    inputs = FROZEN / "batch_decisions.jsonl"
    immutable(inputs, b"".join(jbytes(x) for x in rows))
    immutable(FROZEN / "repair_ledger.jsonl", b"".join(jbytes(x) for x in ledger))
    ingest(inputs)
    split("ground_truth_v1", 42)
    manifest = read(DATASET / "manifest.json")
    partitions = {}
    seen = set()
    for name in ["train", "validation", "test", "hard_test"]:
        f = DATASET / (name + ".jsonl")
        if sha(f.read_bytes()) != manifest["sha256"][name]:
            raise ValueError("Partition hash")
        part = [json.loads(line) for line in f.read_bytes().splitlines()]
        for record in part:
            if sha(Path(record["gt"]).read_bytes()) != record["gt_sha256"]:
                raise ValueError("Reference hash")
        groups = {s["group_id"] for s in part}
        if name != "hard_test":
            if groups & seen:
                raise ValueError("Project leakage")
            seen |= groups
        partitions[name] = len(part)
    if (
        sum(partitions[n] for n in ["train", "validation", "test"])
        != counts["source_structure"] + counts["retain_reviewed_original"]
    ):
        raise ValueError("Sample count mismatch")
    result = dict(
        phase="3",
        status="complete",
        dataset="ground_truth_v1",
        partitions=partitions,
        pilot_count=len(q),
        original_suspects=sum(not r["structure_valid"] for r in q),
        additional_triage=sum(
            r["original_structure_flag"] == False and r["proposal_sha256"] is not None
            for r in ledger
        ),
        actions=dict(counts),
        tables_with_children=sum(bool(r["nested_tables"]) for r in ledger),
        confirm_sha256=confirm_sha,
        ledger_sha256=sha((FROZEN / "repair_ledger.jsonl").read_bytes()),
        dataset_manifest_sha256=sha((DATASET / "manifest.json").read_bytes()),
    )
    immutable(FROZEN / "completion.json", jbytes(result))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
