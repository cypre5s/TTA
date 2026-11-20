"""Import attributable human decisions and freeze project-disjoint table splits."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from src.otsl.grid import parse

root = Path("data/tables/ground_truth")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jbytes(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n").encode()


def immutable(p, b):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_bytes() != b:
            raise ValueError(f"Version conflict: {p}")
        return
    # Exclusive publication keeps a concurrent import from replacing an audit record.
    import os, tempfile

    with tempfile.NamedTemporaryFile(dir=p.parent, delete=False) as tmp:
        tmp.write(b)
    try:
        try:
            os.link(tmp.name, p)
        except FileExistsError:
            if p.read_bytes() != b:
                raise ValueError(f"Concurrent change: {p}")
    finally:
        Path(tmp.name).unlink()


def ingest(path):
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not items:
        raise ValueError("No human decisions")
    for r in items:
        tid = r["table_id"]
        if not isinstance(tid, str) or "/" in tid or not tid:
            raise ValueError("Invalid table ID")
        packet = Path("data/annotation/packets") / (tid + ".json")
        meta = json.loads(packet.read_text())
        if (
            r.get("reviewer_type") != "human"
            or not r.get("reviewer", "").strip()
            or not r.get("reviewed_at")
        ):
            raise ValueError("Named human reviewer and timestamp required")
        from datetime import datetime

        if datetime.fromisoformat(r["reviewed_at"]).tzinfo is None:
            raise ValueError("Include timezone")
        if r["packet_sha256"] != sha(packet.read_bytes()):
            raise ValueError("Stale input")
        if r["decision"] not in ("approve", "reject"):
            raise ValueError("Unknown human decision")
        if r["decision"] == "approve":
            g = parse(r["corrected_otsl"])
            r["grid"] = g.as_dict()
        r["document_sha256"] = meta["document_sha256"]
        r["image_sha256"] = meta["image_sha256"]
        revision = r.get("revision", 1)
        if type(revision) != int or revision < 1:
            raise ValueError("Invalid correction revision")
        if revision > 1 and not (root / tid / f"{revision - 1:06d}.json").exists():
            raise ValueError("Missing previous revision")
        immutable(root / tid / f"{revision:06d}.json", jbytes(r))
    print(len(items), "human decisions imported")


def split(version, seed):
    if (
        not version
        or not all(c.isalnum() or c in "-._" for c in version)
        or version in (".", "..")
    ):
        raise ValueError("Invalid dataset version")
    docs = sqlite3.connect("database/pilot.sqlite")
    parent = {}

    def leader(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for digest, site, number, name in docs.execute(
        "select sha256,site,project_id,project from sources"
    ):
        # Bridge alternate IDs across identical files and repeated project names on the same issuer.
        for item in [number, name]:
            if item:
                a, b = leader(digest), leader(site + "|" + "".join(item.split()))
                parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        revisions = sorted(d.glob("*.json"))
        if not revisions:
            continue
        gt = json.loads(revisions[-1].read_text())
        if gt["decision"] == "approve":
            g = leader(gt["document_sha256"])
            groups[g].append(
                dict(
                    table_id=gt["table_id"],
                    group_id=g,
                    gt=str(revisions[-1]),
                    gt_sha256=sha(revisions[-1].read_bytes()),
                    hard=bool(gt.get("difficulty")),
                )
            )
    if len(groups) < 3:
        raise ValueError("Need at least three distinct reviewed project groups")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    held = max(1, round(0.15 * n))
    parts = [
        ("test", keys[:held]),
        ("validation", keys[held : 2 * held]),
        ("train", keys[2 * held :]),
    ]
    out = Path("data/splits") / version
    hashes = {}
    for name, ks in parts:
        records = [r for k in ks for r in groups[k]]
        payload = b"".join(jbytes(r) for r in records)
        immutable(out / (name + ".jsonl"), payload)
        hashes[name] = sha(payload)
        if name == "test":
            hard = b"".join(jbytes(r) for r in records if r["hard"])
            immutable(out / "hard_test.jsonl", hard)
            hashes["hard_test"] = sha(hard)
    immutable(
        out / "manifest.json",
        jbytes(
            dict(
                version=version,
                seed=seed,
                group_count=n,
                sha256=hashes,
                hard_test_policy="reviewer-tagged difficulty subset of held-out test projects",
            )
        ),
    )
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cmd = p.add_subparsers(dest="op", required=True)
    c = cmd.add_parser("import")
    c.add_argument("decisions", type=Path)
    c = cmd.add_parser("split")
    c.add_argument("--version", required=True)
    c.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.op == "import":
        ingest(args.decisions)
    else:
        split(args.version, args.seed)
