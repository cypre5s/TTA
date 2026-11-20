"""Score complete, frozen sets of human-reviewed tables with pinned TEDS."""

import argparse
import json
from pathlib import Path
from statistics import mean
from rapidfuzz.distance import Levenshtein
from scripts.ground_truth import immutable, jbytes, sha
from src.evaluation.vendor.pubtabnet_teds import TEDS
from src.otsl.grid import parse


def signature(g):
    return {(x.row, x.column, x.rowspan, x.colspan) for x in g.cells}


def score(raw, truth):
    gold = parse(truth)
    expected = "".join(x.text for x in gold.cells)
    try:
        p = parse(raw)
    except ValueError:
        p = None
    gt = signature(gold)
    predicted = signature(p) if p else set()

    def document(g):
        return "<html><body>" + g.html() + "</body></html>"

    table = document(gold)
    text = "".join(x.text for x in p.cells) if p else ""
    return dict(
        teds=TEDS(False).evaluate(document(p), table) if p else 0,
        teds_s=TEDS(True).evaluate(document(p), table) if p else 0,
        row_accuracy=int(bool(p and p.rows == gold.rows)),
        column_accuracy=int(bool(p and p.columns == gold.columns)),
        span_exact=int(p is not None and predicted == gt),
        exact=int(bool(p and gold.as_dict() == p.as_dict())),
        matched_cells=len(gt & predicted),
        predicted_cells=len(predicted),
        true_cells=len(gt),
        char_edits=Levenshtein.distance(text, expected),
        true_chars=len(expected),
        parse_ok=bool(p),
    )


def read_split(path, name):
    manifest = json.loads((path / "manifest.json").read_text())
    b = (path / (name + ".jsonl")).read_bytes()
    if sha(b) != manifest["sha256"][name]:
        raise ValueError("Split hash mismatch")
    rows = [json.loads(line) for line in b.splitlines()]
    if not rows:
        raise ValueError("Empty evaluation partition")
    return rows


def evaluate(args):
    rows = read_split(args.dataset, args.split)
    config = json.loads(args.config.read_text())
    for key in (
        "experiment_id",
        "model_version",
        "random_seed",
        "dataset_version",
        "git_commit",
        "prediction_field",
    ):
        if key not in config:
            raise ValueError(f"Missing {key}")
    version = json.loads((args.dataset / "manifest.json").read_text())["version"]
    if config["dataset_version"] != version:
        raise ValueError("Dataset version mismatch")
    lock = dict(
        config_sha256=sha(args.config.read_bytes()),
        split_manifest_sha256=sha((args.dataset / "manifest.json").read_bytes()),
        metric_sha256=sha(Path("src/evaluation/vendor/pubtabnet_teds.py").read_bytes()),
        evaluation_sha256=sha(Path(__file__).read_bytes()),
        parser_sha256=sha(Path("src/otsl/grid.py").read_bytes()),
    )
    if args.action == "freeze":
        immutable(args.freeze, jbytes(lock))
        return
    if args.split in ("test", "hard_test") and (
        not args.freeze.exists() or json.loads(args.freeze.read_text()) != lock
    ):
        raise ValueError(
            "Final test requires matching frozen config, dataset and metric"
        )
    samples = []
    for row in rows:
        b = Path(row["gt"]).read_bytes()
        if sha(b) != row["gt_sha256"]:
            raise ValueError("GT changed")
        gt = json.loads(b)
        if gt["reviewer_type"] != "human" or gt["decision"] != "approve":
            raise ValueError("Unapproved reference")
        path = args.predictions / (row["table_id"] + ".json")
        raw = ""
        if path.exists():
            pred = json.loads(path.read_text())
            if pred["table_id"] != row["table_id"]:
                raise ValueError(path)
            raw = pred[config["prediction_field"]]
        samples.append(
            dict(
                table_id=row["table_id"],
                missing=not path.exists(),
                prediction_sha256=sha(path.read_bytes()) if path.exists() else None,
                **score(raw, gt["corrected_otsl"]),
            )
        )
    metrics = {
        k: mean(x[k] for x in samples)
        for k in (
            "teds",
            "teds_s",
            "exact",
            "row_accuracy",
            "column_accuracy",
            "span_exact",
        )
    }
    n = sum(x["matched_cells"] for x in samples)
    npred = sum(x["predicted_cells"] for x in samples)
    ngt = sum(x["true_cells"] for x in samples)
    metrics.update(
        cell_precision=n / npred if npred else 0,
        cell_recall=n / ngt,
        cell_f1=2 * n / (npred + ngt),
        cer=sum(x["char_edits"] for x in samples)
        / max(1, sum(x["true_chars"] for x in samples)),
    )
    immutable(
        args.output,
        jbytes(
            dict(
                config=config,
                frozen=lock,
                split=args.split,
                sample_count=len(rows),
                metrics=metrics,
                samples=samples,
            )
        ),
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["freeze", "evaluate"])
    for name in ["config", "dataset", "freeze", "predictions", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--split", choices=["validation", "test", "hard_test"], default="validation"
    )
    evaluate(p.parse_args())
