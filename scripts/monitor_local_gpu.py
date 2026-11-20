"""Sample local production progress and NVIDIA utilization while the two workers run."""

import csv
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def count(folder):
    return sum(1 for p in folder.iterdir() if p.suffix == ".json")


def main():
    log = Path("logs/local/gpu_samples.csv")
    stages = {
        "layout": Path("data/layout/predictions"),
        "ocr": Path("data/tables/predictions"),
        "reviews": Path("data/annotation/raw"),
    }
    with log.open("a", newline="") as sink:
        writer = csv.writer(sink)
        if sink.tell() == 0:
            writer.writerow(
                [
                    "utc",
                    "epoch",
                    "gpu_percent",
                    "memory_controller_percent",
                    "used_mib",
                    "power_w",
                    *stages,
                ]
            )
        while True:
            reading = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=8,
                )
                .strip()
                .split(", ")
            )
            progress = [count(folder) for folder in stages.values()]
            now = datetime.now(timezone.utc)
            writer.writerow([now.isoformat(), now.timestamp(), *reading, *progress])
            sink.flush()
            if progress[0] >= 13247 and progress[1] >= 1000:
                break
            time.sleep(5)
    with log.open() as stream:
        rows = list(csv.DictReader(stream))
    # Adjacent intervals are weighted by their observed length; exclude restart gaps.
    durations = [
        (a, b, float(b["epoch"]) - float(a["epoch"])) for a, b in zip(rows, rows[1:])
    ]
    durations = [item for item in durations if 0 < item[2] <= 15]
    seconds = sum(dt for _, _, dt in durations)
    if not seconds:
        return
    result = {
        "from_utc": rows[0]["utc"],
        "to_utc": rows[-1]["utc"],
        "samples": len(rows),
        "sampled_seconds": seconds,
    }
    for key in ("gpu_percent", "memory_controller_percent", "used_mib", "power_w"):
        vals = [float(r[key]) for r in rows]
        result[key] = {
            "mean": sum(float(a[key]) * dt for a, _, dt in durations) / seconds,
            "median": statistics.median(vals),
            "max": max(vals),
        }
    for key in stages:
        result[key + "_per_minute"] = (
            60 * sum(int(b[key]) - int(a[key]) for a, b, _ in durations) / seconds
        )
    Path("reports/performance/pilot_gpu_utilization.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
