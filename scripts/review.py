"""Freeze pilot review packets and validate the raw visual review responses."""

import argparse
import hashlib
import json
from pathlib import Path
from jsonschema import Draft202012Validator
import yaml

root = Path("data/annotation")
raw = root / "raw"
packets = root / "packets"
raw.mkdir(exist_ok=True, parents=True)
packets.mkdir(exist_ok=True)
errors = [
    "cell_errors",
    "merge_errors",
    "header_errors",
    "cross_page_candidates",
    "nested_table_candidates",
    "text_errors",
]
fields = {
    "table_id": {"type": "string", "minLength": 1},
    "visual_quality": {"enum": ["good", "medium", "poor"]},
    "structure_valid": {"type": "boolean"},
    "review_required": {"const": True},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "reasoning_summary": {"type": "string", "minLength": 3},
}
fields.update(
    {k: {"type": "array", "items": {"type": "string", "minLength": 1}} for k in errors}
)
schema = {
    "type": "object",
    "required": list(fields),
    "additionalProperties": False,
    "properties": fields,
}

validator = Draft202012Validator(schema)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(limit):
    assigned = set(p.stem for p in packets.glob("*.json"))
    for p in sorted(Path("data/tables/predictions").glob("*.json")):
        if len(assigned) >= limit:
            break
        if p.stem in assigned:
            continue
        html = p.with_suffix(".html")
        if not html.exists():
            continue
        pred = json.loads(p.read_text())
        meta = json.loads((Path("data/tables/crops") / p.name).read_text())
        uid = len(assigned) + 1
        packet = dict(
            **meta,
            ordinal=uid,
            prediction=str(p),
            raw_otsl=f"data/otsl/raw/{p.stem}.otsl",
            html=str(html),
            structure_checks=pred["structure_checks"],
            prediction_sha256=digest(p),
            html_sha256=digest(html),
        )
        temp = packets / (p.name + ".part")
        temp.write_text(json.dumps(packet, ensure_ascii=False))
        temp.replace(packets / p.name)
        assigned.add(p.stem)
    return len(assigned)


def verify():
    config = yaml.safe_load(Path("configs/labeler.yaml").read_text())
    assigned = {}
    for rule in config["reviewers"]:
        for ordinal in range(rule["first"], rule["last"] + 1, rule["step"]):
            if ordinal in assigned:
                raise ValueError(f"Overlapping visual assignments at {ordinal}")
            assigned[ordinal] = rule["agent"]
    if set(assigned) != set(range(1, max(config["stages"]) + 1)):
        raise ValueError("Incomplete visual review allocation")
    out = []
    for p in sorted(raw.glob("*.json")):
        d = json.loads(p.read_text())
        validator.validate(d)
        packet = packets / p.name
        meta = json.loads(packet.read_text())
        assert d["table_id"] == p.stem and d["table_id"] == meta["table_id"]
        assert digest(Path(meta["image"])) == meta["image_sha256"]
        assert digest(Path(meta["prediction"])) == meta["prediction_sha256"]
        assert digest(Path(meta["html"])) == meta["html_sha256"]
        out.append(
            dict(
                ordinal=meta["ordinal"],
                table_id=p.stem,
                model=config["model"],
                provider=config["provider"],
                agent=assigned[meta["ordinal"]],
                request_id="visual-" + digest(packet)[:24],
                provider_request_id=None,
                usage_tokens=None,
                cost=None,
                packet_sha256=digest(packet),
                response_sha256=digest(p),
                status="schema_valid",
                human_review_status="pending",
            )
        )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", type=int, choices=[10, 100, 1000])
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    if args.freeze:
        previous = {10: 0, 100: 10, 1000: 100}[args.freeze]
        if previous and not set(range(1, previous + 1)) <= {
            r["ordinal"] for r in verify()
        }:
            raise SystemExit("Prior visual pilot not verified")
        print(freeze(args.freeze))
    if args.verify:
        rows = verify()
        (root / "receipt.jsonl").write_text(
            "".join(
                json.dumps(x) + "\n" for x in sorted(rows, key=lambda r: r["ordinal"])
            )
        )
        print(len(rows), "validated")
    Path("configs/annotation.schema.json").write_text(json.dumps(schema, indent=2))
