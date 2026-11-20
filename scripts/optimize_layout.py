"""Share SSA constants used exclusively as read-only shape/axis/scale operands."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import os
import yaml

read_only_positions = {
    "1.reshape": {1},
    "1.scale": {1},
    "1.slice": {1, 2},
    "1.unsqueeze": {1},
    "1.squeeze": {1},
    "1.tile": {1},
    "1.expand": {1},
    "1.split": {1, 2},
    "1.sum": {1},
    "1.concat": {1},
    "1.full_like": {1},
    "1.pad": {1, 2},
}


def optimize(source, dest):
    blob = (source / "inference.json").read_bytes()
    graph = json.loads(blob)
    regions = graph["program"]["regions"]
    if len(regions) != 1 or len(regions[0]["blocks"]) != 1:
        raise ValueError("Expected single SSA block")
    block = regions[0]["blocks"][0]
    uses = defaultdict(list)
    for op in block["ops"]:
        for pos, v in enumerate(op.get("I", [])):
            uses[v["%"]].append((op["#"], pos))
    cache = {}
    alias = {}
    kept = []
    removed = Counter()
    for op in block["ops"]:
        for inp in op.get("I", []):
            inp["%"] = alias.get(inp["%"], inp["%"])
        if op["#"] in ("1.full", "1.full_int_array"):
            out = op["O"][0]
            idx = out["%"]
            if uses[idx] and all(
                pos in read_only_positions.get(kind, ()) for kind, pos in uses[idx]
            ):
                key = json.dumps(
                    [
                        op["#"],
                        [a for a in op["A"] if a["N"] != "struct_name"],
                        out["TT"],
                        op.get("OA"),
                    ],
                    sort_keys=True,
                )
                if key in cache:
                    alias[idx] = cache[key]
                    removed[op["#"]] += 1
                    continue
                cache[key] = idx
        kept.append(op)
    before = len(block["ops"])
    block["ops"] = kept
    dest.mkdir(exist_ok=True, parents=True)
    for name in ["inference.pdiparams", "inference.yml"]:
        p = dest / name
        if not p.exists():
            os.link(source / name, p)
    (dest / "inference.json").write_text(json.dumps(graph, separators=(",", ":")))
    data = dict(
        input_graph_sha256=hashlib.sha256(blob).hexdigest(),
        output_graph_sha256=hashlib.sha256(
            (dest / "inference.json").read_bytes()
        ).hexdigest(),
        ops_before=before,
        ops_after=len(kept),
        removed=removed,
    )
    Path("reports/performance/layout_graph.json").write_text(json.dumps(data, indent=2))
    print(data)


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/layout.yaml").read_text())
    optimize(Path(cfg["model_dir"]), Path("models/layout_shared_constants"))
