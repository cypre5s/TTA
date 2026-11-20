"""Prepare a 5k project-isolated review manifest from distinct, traceable table content."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from scripts.evaluate import read_split
from scripts.expansion_quality import EXP, OUT, fingerprint
from scripts.ground_truth import immutable, jbytes, sha
from src.otsl.grid import parse

PILOT = Path("data/splits/ground_truth_v1")
TARGETS = {"train": 3500, "validation": 750, "test": 750}


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
        options = [o for o in options if len(assignment[o["group_id"]]) < 2]
        if options:
            # A signature appears exactly once in the final corpus. Prefer its human-reviewed reference.
            ref = min(
                options,
                key=lambda o: (
                    o["tier"] != "human",
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
                r["tier"] != "human",
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
        if gt["decision"] != "usable" or sha(rev.read_bytes()) != gt["review_sha256"]:
            raise ValueError("Invalid expansion review")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish-for-review", action="store_true")
    ap.add_argument("--output", type=Path, default=EXP / "review_5k")
    args = ap.parse_args()
    refs, assignment = inventory()
    splits = choose(refs, assignment)
    sizes = {k: len(splits[k]) for k in TARGETS}
    tiers = {k: dict(Counter(r["tier"] for r in splits[k])) for k in TARGETS}
    print(
        json.dumps(
            dict(distinct=len(refs), partitions=sizes, tiers=tiers), ensure_ascii=False
        ),
        flush=True,
    )
    if not args.publish_for_review:
        return
    if sizes != TARGETS:
        raise ValueError(
            "Insufficient unique records to fill the project-isolated 5k cohort"
        )
    seen = set()
    seen_groups = set()
    hashes = {}
    for k, rows in splits.items():
        sigs = {r["signature"] for r in rows}
        groups = {r["group_id"] for r in rows}
        if len(sigs) != len(rows) or seen & sigs or seen_groups & groups:
            raise ValueError("Project or content overlap")
        seen.update(sigs)
        seen_groups.update(groups)
        rows[:] = [verify_usable(r) for r in rows]
        buf = b"".join(jbytes(r) for r in rows)
        immutable(args.output / (k + ".jsonl"), buf)
        hashes[k] = sha(buf)
    buf = b"".join(jbytes(r) for r in splits["test"] if r["hard"])
    immutable(args.output / "hard_test.jsonl", buf)
    hashes["hard_test"] = sha(buf)
    immutable(
        args.output / "manifest.json",
        jbytes(
            dict(
                version="expansion_5k_review_v1",
                status="prepared_for_review",
                sizes=sizes,
                tiers=tiers,
                sha256=hashes,
                project_groups=len(seen_groups),
                pilot_embargo="prior projects kept in their previous partition; merged conflicting groups excluded",
                machine_labels_require_human_evaluation_clearance=True,
            )
        ),
    )


if __name__ == "__main__":
    main()
