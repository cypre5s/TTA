"""Count distinct usable structures across the human pilot and reviewed expansion."""

import json
from collections import Counter
from pathlib import Path

from scripts.expansion_quality import EXP, OUT, fingerprint
from src.otsl.grid import parse


def progress():
    pilot = set()
    pilot_approved = 0
    for path in Path("data/tables/ground_truth").glob("*/000001.json"):
        item = json.loads(path.read_bytes())
        if item["decision"] == "approve":
            pilot_approved += 1
            pilot.add(
                fingerprint(parse(item["corrected_otsl"]), item.get("structure_tree"))
            )
    counts = Counter()
    expansion = set()
    for path in OUT.glob("*.json"):
        item = json.loads(path.read_bytes())
        counts[item["decision"]] += 1
        if item["decision"] == "usable":
            expansion.add(item["content_signature"])
    return dict(
        pilot_approved=pilot_approved,
        pilot_distinct=len(pilot),
        new_ocr=sum(1 for _ in (EXP / "ocr/predictions").glob("*.json")),
        new_reviews=sum(1 for _ in (EXP / "annotation/raw").glob("*.json")),
        new_audit=counts,
        new_distinct=len(expansion),
        shared_with_pilot=len(pilot & expansion),
        total_distinct=len(pilot | expansion),
        remaining_to_5000=max(0, 5000 - len(pilot | expansion)),
    )


if __name__ == "__main__":
    print(json.dumps(progress(), ensure_ascii=False))
