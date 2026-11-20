"""Freeze the reviewed 5k cohort into a self-contained local experiment directory."""

import argparse
import json
from pathlib import Path

import yaml

from scripts.evaluate import read_split
from scripts.expansion_split import TARGETS, verify_usable
from scripts.ground_truth import immutable, jbytes, sha
from scripts.training_reference import MACHINE_TIERS

VERSION = "ground_truth_5k_v1"


def load(path):
    return json.loads(path.read_bytes())


def bind(workspace, alias, source):
    alias = Path(alias)
    if alias.is_absolute() or ".." in alias.parts:
        raise ValueError(f"Unsafe staged path: {alias}")
    source = Path(source).resolve(strict=True)
    target = workspace / alias
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=source.is_dir())
    except FileExistsError:
        if not target.is_symlink() or target.resolve() != source:
            raise ValueError(f"Staged path conflict: {target}") from None


def approved_review(dataset, approval_body):
    manifest_path = dataset / "manifest.json"
    manifest_body = manifest_path.read_bytes()
    manifest = json.loads(manifest_body)
    if manifest["status"] != "prepared_for_review" or manifest["sizes"] != TARGETS:
        raise ValueError("Expected a complete 5k review manifest")
    splits = {name: read_split(dataset, name) for name in TARGETS}
    approval = json.loads(approval_body)
    machine_test = sum(row["tier"] in MACHINE_TIERS for row in splits["test"])
    if (
        approval.get("review_manifest_sha256") != sha(manifest_body)
        or approval.get("decision") != "approve"
        or approval.get("reviewer_type") != "human"
        or approval.get("machine_test_count") != machine_test
        or not approval.get("reviewer", "").strip()
        or not approval.get("confirmation", "").strip()
    ):
        raise ValueError("Missing human confirmation for this exact test cohort")
    seen_tables = set()
    seen_signatures = set()
    seen_groups = set()
    for name, rows in splits.items():
        ids = {row["table_id"] for row in rows}
        signatures = {row["signature"] for row in rows}
        groups = {row["group_id"] for row in rows}
        if (
            len(rows) != TARGETS[name]
            or len(ids) != len(rows)
            or len(signatures) != len(rows)
            or seen_tables & ids
            or seen_signatures & signatures
            or seen_groups & groups
            or any(row["tier"] not in MACHINE_TIERS and row["tier"] != "human" for row in rows)
        ):
            raise ValueError(f"Invalid or overlapping partition: {name}")
        seen_tables.update(ids)
        seen_signatures.update(signatures)
        seen_groups.update(groups)
    hard = read_split(dataset, "hard_test")
    if hard != [row for row in splits["test"] if row["hard"]]:
        raise ValueError("Hard test differs from the approved test subset")
    return manifest, splits, approval


def confirmed_reference(row, split, approval, body):
    if sha(body) != row["reference_sha256"]:
        raise ValueError("Reference changed after approval")
    if row["tier"] == "human":
        return body
    evaluation = split == "test"
    reference = dict(
        json.loads(body),
        reviewer_type="human" if evaluation else "machine",
        reviewer=approval["reviewer"] if evaluation else row["tier"],
        decision="approve" if evaluation else "usable",
        human_approved=evaluation,
        original_reference_sha256=row["reference_sha256"],
    )
    if evaluation:
        reference["batch_review_manifest_sha256"] = approval["review_manifest_sha256"]
    return jbytes(reference)


def stage_inputs(workspace, row, split, approval):
    if verify_usable(row) != row:
        raise ValueError("Review input paths changed after publication")
    packet_path = Path(row["packet"])
    packet = load(packet_path)
    ref_path = Path(row["reference"])
    ref = load(ref_path)
    tid = row["table_id"]
    pid = packet["page_id"]
    doc = row["document_sha256"]
    page_alias = Path("data/rendered", doc, pid + ".json")
    page_path = Path(ref.get("source_page", page_alias))
    page = load(page_path)
    layout_alias = Path("data/layout/predictions", pid + ".json")
    data_root = page_path.parents[2]
    layout = data_root / "layout/predictions" / layout_alias.name
    layout_meta = load(layout)
    page_image = Path(page["image"])
    if not page_image.is_file():
        page_image = page_path.parent / page_image.name
    if (
        page["page_id"] != pid
        or page["document_sha256"] != doc
        or tid not in layout_meta["tables"]
        or layout_meta["image_sha256"] != page["image_sha256"]
        or sha(page_image.read_bytes()) != page["image_sha256"]
    ):
        raise ValueError(f"Unbound original layout or source page: {tid}")
    bind(workspace, Path("data/annotation/packets", tid + ".json"), packet_path)
    bind(workspace, Path("data/tables/crops", tid + ".png"), row["image"])
    bind(workspace, row["image"], row["image"])
    bind(workspace, Path("data/otsl/raw", tid + ".otsl"), packet["raw_otsl"])
    bind(workspace, packet["prediction"], packet["prediction"])
    bind(workspace, layout_alias, layout)
    # Preserve metadata paths, including paths relative to the expansion root.
    bind(workspace, page_alias, page_path)
    bind(workspace, page["image"], page_image)

    reference = Path("data/tables/frozen_5k", tid + ".json")
    if row["tier"] == "human":
        bind(workspace, reference, ref_path)
    else:
        body = confirmed_reference(row, split, approval, ref_path.read_bytes())
        immutable(workspace / reference, body)
    digest = sha((workspace / reference).read_bytes())
    return dict(
        table_id=tid,
        group_id=row["group_id"],
        gt=str(reference),
        gt_sha256=digest,
        hard=row["hard"],
        tier=row["tier"],
        signature=row["signature"],
        page_id=pid,
        page_sha256=page["image_sha256"],
    ), dict(table_id=tid, packet=str(Path("data/annotation/packets", tid + ".json")),
            packet_sha256=row["packet_sha256"], page_image=page["image"],
            page_sha256=page["image_sha256"], **{"image": row["image"], "image_sha256": row["image_sha256"]})


def stage_runtime(workspace):
    for folder in ("scripts", "src"):
        bind(workspace, folder, folder)
    asset_keys = {
        "xiaomi_local.yaml": ("checkpoint", "upstream"),
        "layout.yaml": ("model_dir",),
    }
    for path in Path("configs").glob("*"):
        if path.name not in asset_keys:
            bind(workspace, path, path)
            continue
        cfg = yaml.safe_load(path.read_text())
        for key in asset_keys[path.name]:
            cfg[key] = str(Path(cfg[key]).resolve(strict=True))
        immutable(workspace / path, yaml.safe_dump(cfg).encode())
    layout = yaml.safe_load(Path("configs/layout.yaml").read_text())
    bind(workspace, "models/layout/PP-DocLayoutV3", layout["model_dir"])
    bind(
        workspace,
        "models/layout/PP-DocLayoutV3_pretrained.pdparams",
        ".cache/layout-train/PP-DocLayoutV3_pretrained.pdparams",
    )


def freeze(dataset, approval_path, workspace, cross_page_study=None, cross_page_reviews=None):
    approval_body = approval_path.read_bytes()
    manifest, selected, approval = approved_review(dataset, approval_body)
    stage_runtime(workspace)
    final = workspace / "data/splits/ground_truth_v1"
    receipts = workspace / "data/annotation/phase3_v1"
    cross_page = None
    if cross_page_study is not None:
        from scripts.cross_page_study import review

        study, _ = review(cross_page_reviews, cross_page_study)
        test_ids = {r["table_id"] for r in selected["test"]}
        if study["cohort_sha256"] != approval["review_manifest_sha256"] or not set(study["tables"]) <= test_ids:
            raise ValueError("Cross-page review must match the approved test cohort")
        study_body = cross_page_study.read_bytes()
        review_body = cross_page_reviews.read_bytes()
        immutable(receipts / "cross_page_study.json", study_body)
        immutable(receipts / "cross_page_decisions.json", review_body)
        cross_page = dict(study_sha256=sha(study_body), human_review_sha256=sha(review_body))
    hashes = {}
    test_packets = []
    calibration = {"B0": [], "B0-E2E": []}
    validation_pages = set()
    hard = []
    for name, rows in selected.items():
        results = []
        for row in rows:
            result, packet = stage_inputs(workspace, row, name, approval)
            results.append(result)
            if name == "test":
                test_packets.append(packet)
                if row["hard"]:
                    hard.append(result)
            elif name == "validation":
                item = dict(table_id=row["table_id"], image=packet["image"], image_sha256=packet["image_sha256"])
                calibration["B0"].append(item)
                if result["page_id"] not in validation_pages:
                    validation_pages.add(result["page_id"])
                    calibration["B0-E2E"].append(dict(item, image=packet["page_image"], image_sha256=packet["page_sha256"]))
        body = b"".join(jbytes(r) for r in results)
        immutable(final / (name + ".jsonl"), body)
        hashes[name] = sha(body)
    body = b"".join(jbytes(r) for r in hard)
    immutable(final / "hard_test.jsonl", body)
    hashes["hard_test"] = sha(body)
    immutable(receipts / "packet_index.jsonl", b"".join(jbytes(r) for r in test_packets))
    immutable(receipts / "validation_calibration.json", jbytes(calibration))
    immutable(receipts / "batch_confirmation.json", approval_body)
    freeze_body = jbytes(dict(
        version=VERSION, sizes=dict(TARGETS, hard_test=len(hard)), sha256=hashes,
        tiers=manifest["tiers"], test_reference_status="human_confirmed",
        training_reference_policy="human and machine-reviewed training/validation references tracked separately",
        approval_sha256=sha(approval_body), review_manifest_sha256=approval["review_manifest_sha256"],
        cross_page=cross_page,
    ))
    immutable(final / "manifest.json", freeze_body)
    immutable(receipts / "completion.json", jbytes(dict(
        dataset=VERSION, phase="3", status="complete", partitions=dict(TARGETS, hard_test=len(hard)),
        dataset_manifest_sha256=sha(freeze_body), confirm_sha256=sha(approval_body),
    )))
    print(VERSION, sha(freeze_body), dict(TARGETS, hard_test=len(hard)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path("data/expansion_v2/review_5k"))
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("experiments/formal_5k/workspace"))
    parser.add_argument("--cross-page-study", type=Path)
    parser.add_argument("--cross-page-reviews", type=Path)
    args = parser.parse_args()
    if bool(args.cross_page_study) != bool(args.cross_page_reviews):
        parser.error("Provide both cross-page protocol and human decisions")
    freeze(args.review, args.approval, args.workspace, args.cross_page_study, args.cross_page_reviews)


if __name__ == "__main__":
    main()
