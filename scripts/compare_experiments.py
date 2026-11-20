"""Compare compatible evaluated runs, leaving missing experiments explicit."""

import argparse, json
from pathlib import Path
from scripts.ground_truth import immutable

BASELINES = ("B0", "B0-E2E", "B1", "B2", "B3", "B4", "B5", "B6")


def compare(paths):
    runs = {}
    reference = None
    for path in paths:
        d = json.loads(path.read_text())
        key = d["config"]["experiment_id"]
        if key not in BASELINES or key in runs:
            raise ValueError(f"Unexpected or duplicate experiment {key}")
        # configs and models may differ. The actual scored references and metric must agree.
        fingerprint = (
            d["split"],
            d["config"]["dataset_version"],
            tuple(
                d["frozen"][name]
                for name in (
                    "split_manifest_sha256",
                    "evaluation_sha256",
                    "metric_sha256",
                    "parser_sha256",
                )
            ),
            tuple(sorted(r["table_id"] for r in d["samples"])),
        )
        if reference is None:
            reference = fingerprint
        if fingerprint != reference:
            raise ValueError("Incompatible test cohort or evaluation code")
        runs[key] = d
    if "B0" not in runs:
        raise ValueError("B0 evaluation is required")
    rows = [
        "# 同一数据划分上的实验结果",
        "",
        f"数据版本 `{reference[1]}`，划分 `{reference[0]}`，共 {len(reference[-1])} 张表格。",
        "",
        "| 实验 | TEDS | TEDS-S | 结构及文字完全匹配 | TEDS-S 相对 B0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in BASELINES:
        d = runs.get(key)
        if d:
            m = d["metrics"]
            ref = runs["B0"]["metrics"]["teds_s"]
            rows.append(
                f"| {key} | {m['teds']:.4f} | {m['teds_s']:.4f} | {m['exact']:.4f} | {100 * (m['teds_s'] - ref):+.2f} 个百分点 |"
            )
        else:
            rows.append(f"| {key} | 未执行 | | | |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    immutable(a.output, compare(a.metrics).encode())
