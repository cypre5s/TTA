"""Bracket FP16 timing with original FP32 runs after releasing the production GPU."""

import json
from pathlib import Path
import statistics
import tempfile
import bench_layout


def main():
    parent = Path("reports/performance")
    trials = []
    destination = parent / "layout_precision_isolated.json"
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        bench_layout.base = Path(tmp)
        for half in (False, True, True, False):
            bench_layout.bench("gpu:0", half=half)
            run = json.loads(
                (
                    Path(tmp) / ("layout_gpu_fp16.json" if half else "layout_gpu.json")
                ).read_text()
            )
            trials.append(
                {
                    "fp16": half,
                    "seconds": run["seconds"],
                    "mean_seconds": run["mean_seconds"],
                }
            )
        bench_layout.compare(half=True)
        geometry = json.loads((Path(tmp) / "layout_fp16_comparison.json").read_text())
    means = {
        half: statistics.mean(t["mean_seconds"] for t in trials if t["fp16"] == half)
        for half in (False, True)
    }
    result = dict(
        pages=10,
        passes_per_trial=2,
        trials=trials,
        fp32_seconds=means[False],
        fp16_seconds=means[True],
        speedup=means[False] / means[True],
        geometry=geometry,
    )
    destination.write_text(json.dumps(result, indent=2))
    print(result["speedup"], len(geometry["differences"]), flush=True)


if __name__ == "__main__":
    main()
