"""Record content overlap between the fixed pilot partitions without changing evaluation membership."""

import json
from collections import defaultdict
from pathlib import Path

from scripts.evaluate import read_split
from scripts.expansion_quality import fingerprint
from scripts.ground_truth import immutable, jbytes, sha
from src.otsl.grid import parse


def main():
    dataset = Path("data/splits/ground_truth_v1")
    digests = defaultdict(dict)
    for name in ("train", "validation", "test"):
        for row in read_split(dataset, name):
            data = Path(row["gt"]).read_bytes()
            if sha(data) != row["gt_sha256"]:
                raise ValueError("Ground truth changed")
            gt = json.loads(data)
            digests[name][row["table_id"]] = fingerprint(
                parse(gt["corrected_otsl"]), gt.get("structure_tree")
            )
    train = set(digests["train"].values())
    val = set(digests["validation"].values())
    test = digests["test"]
    record = dict(
        dataset_sha256=sha((dataset / "manifest.json").read_bytes()),
        method="NFKC and stripped whitespace in cell texts; ordered anchors and complete child structure",
        distinct={n: len(set(r.values())) for n, r in digests.items()},
        test_same_as_train=sorted(t for t, d in test.items() if d in train),
        test_same_as_validation=sorted(t for t, d in test.items() if d in val),
        validation_same_as_train=sum(
            v in train for v in digests["validation"].values()
        ),
        test_unseen=sorted(t for t, d in test.items() if d not in train | val),
    )
    immutable(Path("experiments/ablation/cohorts/content_overlap.json"), jbytes(record))
    print(
        "held-out tables with training/validation-unseen structures and text",
        len(record["test_unseen"]),
    )


if __name__ == "__main__":
    main()
