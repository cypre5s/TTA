"""Compare one and three CPU rendering workers on identical real documents."""

import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

import cv2
import process


def main():
    cv2.setNumThreads(1)
    base = Path.cwd()
    candidates = process.db.execute(
        "select sha256,path,pages from documents where pages between 3 and 12 order by sha256"
    ).fetchall()
    docs = [(k, str((base / p).resolve()), n) for k, p, n in candidates[:9]]
    timings = {1: [], 3: []}
    outputs = []
    with tempfile.TemporaryDirectory(dir=base / ".cache") as tmp:
        try:
            for run, workers in enumerate([1, 3, 3, 1]):
                work = Path(tmp) / str(run)
                work.mkdir()
                os.chdir(work)
                start = time.perf_counter()
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers
                ) as pool:
                    jobs = [pool.submit(process.process, *row) for row in docs]
                    for job in jobs:
                        job.result()
                timings[workers].append(time.perf_counter() - start)
                outputs.append(
                    sorted(
                        (r["page_id"], r["image_sha256"], r["text_sha256"])
                        for f in Path("data/rendered").glob("*/manifest.json")
                        for r in json.loads(f.read_text())
                    )
                )
        finally:
            os.chdir(base)
    assert outputs[0] and all(x == outputs[0] for x in outputs)
    result = dict(
        documents=[k for k, _, _ in docs],
        pages=len(outputs[0]),
        timings_seconds=timings,
        speedup=statistics.mean(timings[1]) / statistics.mean(timings[3]),
        outputs_match=True,
        conditions="Concurrent production jobs active; alternating order.",
    )
    (base / "reports/performance/pdf_workers.json").write_text(
        json.dumps(result, indent=2)
    )
    print(result)


if __name__ == "__main__":
    main()
