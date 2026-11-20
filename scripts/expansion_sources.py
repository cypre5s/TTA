"""Extend the serial public-document collection if the 5k reviewer queue is undersupplied."""

import argparse
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import yaml

from scripts.expansion_queue import EXP


def count(path, table):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-target", type=int, default=6200)
    p.add_argument("--wave", type=int, default=45)
    p.add_argument("--waves", type=int, default=4)
    args = p.parse_args()
    root = EXP.resolve()
    for _ in range(args.waves):
        candidates = count(root / "crop_index.sqlite", "candidates")
        if candidates >= args.candidate_target:
            break
        documents = count(root / "database/pilot.sqlite", "documents")
        target = documents + args.wave
        for name in ("crawler", "crawler_beijing", "crawler_culr"):
            if count(root / "crop_index.sqlite", "candidates") >= args.candidate_target:
                return
            if count(root / "database/pilot.sqlite", "documents") >= target:
                break
            cfg = yaml.safe_load((root / "configs" / (name + ".yaml")).read_text())
            cfg["target"] = target
            config = root / "configs" / (name + "_to_" + str(target) + ".yaml")
            config.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            print(
                "serial source",
                cfg["site"],
                "document target",
                target,
                "candidates",
                candidates,
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("collect.py").resolve()),
                    "--config",
                    str(config),
                ],
                cwd=root,
                check=False,
            )
        after = count(root / "database/pilot.sqlite", "documents")
        if after <= documents:
            raise RuntimeError(
                "No new documents in this pass; inspect source responses"
            )
        # Give the local render/layout consumers time to expose the new candidates.
        time.sleep(600)


if __name__ == "__main__":
    main()
