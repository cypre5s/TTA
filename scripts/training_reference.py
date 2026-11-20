"""Validate the reference tiers allowed in training and validation."""

import json
from pathlib import Path

from scripts.ground_truth import sha

VISUAL_TIER = "visually_adjudicated_machine"
OCR_TIER = "structure_screened_ocr"
MACHINE_TIERS = {VISUAL_TIER, OCR_TIER}


def permits_machine_training(dataset, root=Path(".")):
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["version"] != "ground_truth_5k_v1":
        return False
    receipt = Path(root) / "data/annotation/phase3_v1/completion.json"
    approval = receipt.with_name("batch_confirmation.json")
    recorded = json.loads(receipt.read_bytes())
    if (
        recorded["status"] != "complete"
        or recorded["dataset_manifest_sha256"] != sha(manifest_path.read_bytes())
        or sha(approval.read_bytes()) != manifest["approval_sha256"]
    ):
        raise ValueError("Missing confirmed 5k freeze receipt")
    return True


def training_reference(row, allow_machine=False):
    body = Path(row["gt"]).read_bytes()
    if sha(body) != row["gt_sha256"]:
        raise ValueError("Training reference hash mismatch")
    gt = json.loads(body)
    human = gt["reviewer_type"] == "human" and gt["decision"] == "approve"
    machine = (
        allow_machine
        and row.get("tier") in MACHINE_TIERS
        and gt["reviewer_type"] == "machine"
        and gt["decision"] == "usable"
        and gt.get("human_approved") is False
        and gt.get("original_reference_sha256")
    )
    if gt["table_id"] != row["table_id"] or not (human or machine):
        raise ValueError("Invalid training reference provenance")
    return gt
