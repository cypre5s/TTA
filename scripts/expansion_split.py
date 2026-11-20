"""Prepare a 5k project-isolated review manifest from distinct, traceable table content."""

import argparse
import hashlib
import json
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts.evaluate import read_split
from scripts.expansion_quality import EXP, OUT, fingerprint, screen
from scripts.ground_truth import immutable, jbytes, sha
from scripts.training_reference import OCR_TIER, VISUAL_TIER
from src.otsl.grid import parse

PILOT = Path("data/splits/ground_truth_v1")
TARGETS = {"train": 3500, "validation": 750, "test": 750}
REFERENCE_PRIORITY = {"human": 0, VISUAL_TIER: 1, OCR_TIER: 2}


def group_map():
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    with sqlite3.connect(
        f"file:{EXP / 'database/pilot.sqlite'}?mode=ro", uri=True
    ) as conn:
        for digest, site, number, name in conn.execute(
            "select sha256, site, project_id, project from sources"
        ):
            for ident in [number, name]:
                if ident:
                    a, b = find(digest), find(site + "|" + "".join(ident.split()))
                    parent[max(a, b)] = min(a, b)
    return {k: find(k) for k in parent}


def inventory():
    groups = group_map()
    refs = defaultdict(list)
    assignment = defaultdict(set)
    for name in TARGETS:
        for row in read_split(PILOT, name):
            p = Path(row["gt"])
            b = p.read_bytes()
            if sha(b) != row["gt_sha256"]:
                raise ValueError("Changed pilot reference")
            gt = json.loads(b)
            group = groups[gt["document_sha256"]]
            assignment[group].add(name)
            ref = dict(
                table_id=row["table_id"],
                reference=str(p),
                reference_sha256=sha(b),
                group_id=group,
                document_sha256=gt["document_sha256"],
                tier="human",
                pilot_partition=name,
                hard=row["hard"],
            )
            refs[
                fingerprint(parse(gt["corrected_otsl"]), gt.get("structure_tree"))
            ].append(ref)
    for p in sorted(OUT.glob("*.json")):
        b = p.read_bytes()
        r = json.loads(b)
        if r["decision"] != "usable":
            continue
        group = groups[r["document_sha256"]]
        refs[r["content_signature"]].append(
            dict(
                table_id=r["table_id"],
                reference=str(p),
                reference_sha256=sha(b),
                group_id=group,
                document_sha256=r["document_sha256"],
                tier=r["label_tier"],
                hard=bool(r["flags"] or r["visual_quality"] == "poor"),
            )
        )
    return refs, assignment


def choose(refs, assignment):
    groups = defaultdict(list)
    exposure = {"train": 0, "validation": 1, "test": 2}
    for signature, options in refs.items():
        prior = [o for o in options if o["tier"] == "human"]
        if prior:
            # Keep the earliest prior exposure even when that project's groups
            # merged and the original row became ineligible for this release.
            partition = min((o["pilot_partition"] for o in prior), key=exposure.get)
            options = [o for o in prior if o["pilot_partition"] == partition]
        options = [o for o in options if len(assignment[o["group_id"]]) < 2]
        if options:
            # A signature appears exactly once in the final corpus. Prefer its human-reviewed reference.
            ref = min(
                options,
                key=lambda o: (
                    REFERENCE_PRIORITY[o["tier"]],
                    exposure.get(o.get("pilot_partition"), 3),
                    o["table_id"],
                ),
            )
            groups[ref["group_id"]].append(dict(ref, signature=signature))
    bins = defaultdict(list)
    unused = []
    for group, rows in groups.items():
        if assignment[group]:
            bins[next(iter(assignment[group]))].extend(rows)
        else:
            unused.append((group, rows))
    # Assign the largest remaining project as a unit and minimise the relative size imbalance.
    for _, rows in sorted(unused, key=lambda g: (-len(g[1]), g[0])):
        split = min(TARGETS, key=lambda s: (len(bins[s]) / TARGETS[s], s))
        bins[split].extend(rows)
    for split, size in TARGETS.items():
        # A corpus larger than 5k supplies reserves for each project-isolated partition.
        bins[split].sort(
            key=lambda r: (
                REFERENCE_PRIORITY[r["tier"]],
                hashlib.sha256(("42" + r["table_id"]).encode()).digest(),
            )
        )
        bins[split] = bins[split][:size]
    return bins


def verify_usable(row):
    from PIL import Image

    from scripts.freeze_pilot_repairs import verify_tree

    body = Path(row["reference"]).read_bytes()
    gt = json.loads(body)
    if sha(body) != row["reference_sha256"] or gt["table_id"] != row["table_id"]:
        raise ValueError("Reference changed before publication")
    pilot = row["tier"] == "human"
    packet_root = (
        Path("data/annotation/packets") if pilot else EXP / "annotation/packets"
    )
    packet = packet_root / (row["table_id"] + ".json")
    pb = packet.read_bytes()
    meta = json.loads(pb)
    if (
        sha(pb) != gt["packet_sha256"]
        or meta["document_sha256"] != row["document_sha256"]
    ):
        raise ValueError("Broken packet link")
    if pilot and (gt["reviewer_type"] != "human" or gt["decision"] != "approve"):
        raise ValueError("Invalid pilot decision")
    if not pilot:
        rev = EXP / "annotation/raw" / packet.name
        if gt["decision"] != "usable" or gt["label_tier"] != row["tier"]:
            raise ValueError("Invalid expansion review")
        if row["tier"] == OCR_TIER:
            if screen(packet) != gt:
                raise ValueError("Original OCR structure screen changed")
        elif row["tier"] != VISUAL_TIER or sha(rev.read_bytes()) != gt["review_sha256"]:
            raise ValueError("Changed visual review")
    img = Path(meta["image"])
    if (
        sha(img.read_bytes()) != gt["image_sha256"]
        or gt["image_sha256"] != meta["image_sha256"]
    ):
        raise ValueError("Changed crop")
    with Image.open(img) as im:
        if min(im.size) < 2:
            raise ValueError("Empty crop")
        im.verify()
    grid = parse(gt["corrected_otsl"])
    tree = gt.get("structure_tree")
    if tree and (tree["otsl"] != gt["corrected_otsl"] or verify_tree(tree) < 0):
        raise ValueError("Invalid hierarchy")
    if fingerprint(grid, tree) != row["signature"]:
        raise ValueError("Content changed before publication")
    return dict(
        row,
        image=str(img),
        image_sha256=gt["image_sha256"],
        packet=str(packet),
        packet_sha256=gt["packet_sha256"],
    )


def prepare(output, staged, splits, tiers):
    seen = set()
    seen_groups = set()
    contents = {}
    hard = []
    for k, rows in splits.items():
        sigs = {r["signature"] for r in rows}
        groups = {r["group_id"] for r in rows}
        if len(sigs) != len(rows) or seen & sigs or seen_groups & groups:
            raise ValueError("Project or content overlap")
        seen.update(sigs)
        seen_groups.update(groups)
        checked = [verify_usable(r) for r in rows]
        for row in checked:
            body = Path(row["reference"]).read_bytes()
            if sha(body) != row["reference_sha256"]:
                raise ValueError("Reference changed during publication")
            alias = Path("references", row["table_id"] + ".json")
            immutable(staged / alias, body)
            row["reference"] = str(output / alias)
        contents[k] = b"".join(jbytes(r) for r in checked)
        if k == "test":
            hard = [r for r in checked if r["hard"]]
    if not hard:
        raise ValueError("Empty hard-test subset")
    contents["hard_test"] = b"".join(jbytes(r) for r in hard)
    manifest = dict(
        version="expansion_5k_review_v1",
        status="prepared_for_review",
        sizes=TARGETS,
        tiers=tiers,
        sha256={k: sha(v) for k, v in contents.items()},
        project_groups=len(seen_groups),
        pilot_embargo="prior projects kept in their previous partition; merged conflicting groups excluded",
        machine_labels_require_human_evaluation_clearance=True,
    )
    for name, body in contents.items():
        immutable(staged / (name + ".jsonl"), body)
    immutable(staged / "manifest.json", jbytes(manifest))


def publish(output, splits, tiers):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".5k-review-") as tmp:
        staged = Path(tmp, "prepared")
        prepare(output, staged, splits, tiers)
        staged.rename(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish-for-review", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--output", type=Path, default=EXP / "review_5k")
    args = ap.parse_args()
    if args.watch and not args.publish_for_review:
        ap.error("--watch requires --publish-for-review")
    while True:
        if args.publish_for_review and args.output.exists():
            manifest = json.loads((args.output / "manifest.json").read_bytes())
            sizes = {name: len(read_split(args.output, name)) for name in TARGETS}
            if manifest["status"] != "prepared_for_review" or sizes != TARGETS:
                raise ValueError("Existing review directory is not a complete 5k cohort")
            read_split(args.output, "hard_test")
            print("Verified existing review cohort", args.output, flush=True)
            return
        refs, assignment = inventory()
        splits = choose(refs, assignment)
        sizes = {k: len(splits[k]) for k in TARGETS}
        tiers = {k: dict(Counter(r["tier"] for r in splits[k])) for k in TARGETS}
        print(jbytes(dict(distinct=len(refs), partitions=sizes, tiers=tiers)).decode().strip(),
              flush=True)
        if not args.publish_for_review:
            return
        if sizes == TARGETS:
            publish(args.output, splits, tiers)
            return
        if not args.watch:
            raise ValueError("Insufficient unique records to fill the project-isolated 5k cohort")
        time.sleep(30)


if __name__ == "__main__":
    main()
