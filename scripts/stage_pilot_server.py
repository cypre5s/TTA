"""Assemble the consented 1k pilot only, with independent data and code manifests."""

from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(".cache/server-pilot/payload")
WEIGHTS = Path("../TenderDoc-AI/models/layout/PP-DocLayoutV3")


def stage(path, dest=None):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    out = ROOT / (dest or p)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not p.samefile(out) and digest(out) != digest(p):
        if not str(dest or p).startswith(("scripts/", "src/", "tests/", "configs/")):
            raise ValueError(f"Staging conflict: {out}")
        temp = out.with_name(out.name + ".new")
        os.link(p, temp)
        temp.replace(out)
    if not out.exists():
        os.link(p, out)
    return out


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    root = Path("data/annotation/phase3_v1")
    index = [
        json.loads(s) for s in (root / "packet_index.jsonl").read_bytes().splitlines()
    ]
    if len(index) != 1000 or len({r["table_id"] for r in index}) != 1000:
        raise ValueError("Unexpected scope")
    for name in [
        "packet_index.jsonl",
        "repair_ledger.jsonl",
        "batch_confirmation.json",
        "completion.json",
    ]:
        stage(root / name)
    for r in index:
        m = json.loads(Path(r["packet"]).read_bytes())
        for key in ["image", "raw_otsl", "prediction", "html"]:
            stage(m[key])
        stage(r["packet"])
        stage(r["visual_review"])
        stage(Path("data/tables/ground_truth", r["table_id"], "000001.json"))
    split = Path("data/splits/ground_truth_v1")
    for f in split.iterdir():
        stage(f)
    for r in map(json.loads, (split / "test.jsonl").read_bytes().splitlines()):
        pid = r["table_id"].rsplit("-", 1)[0]
        doc = pid.split("-")[0]
        folder = Path("data/rendered", doc)
        stage(folder / (pid + ".png"))
        stage(folder / (pid + ".json"))
        stage(Path("data/layout/predictions", pid + ".json"))
    layout = Path("data/datasets/layout/reviewed_table_pilot")
    for f in layout.rglob("*"):
        if f.is_file():
            stage(f)
    for name in ["inference.yml", "inference.json", "inference.pdiparams"]:
        stage(WEIGHTS / name, Path("models/layout/PP-DocLayoutV3", name))
    stage(
        ".cache/layout-train/PP-DocLayoutV3_pretrained.pdparams",
        "models/layout/PP-DocLayoutV3_pretrained.pdparams",
    )
    for folder in ["scripts", "src", "configs", "tests"]:
        for f in Path(folder).rglob("*"):
            if (
                f.is_file()
                and "__pycache__" not in f.parts
                and not f.name.endswith(".pyc")
            ):
                stage(f)
    for f in ["Makefile", "pyproject.toml"]:
        stage(f)
    (ROOT / "source_revision.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
    )
    for name, folder in [("DATA_SHA256SUMS", "data"), ("WEIGHTS_SHA256SUMS", "models")]:
        files = sorted(f for f in (ROOT / folder).rglob("*") if f.is_file())
        (ROOT / name).write_text(
            "".join(f"{digest(f)}  {f.relative_to(ROOT)}\n" for f in files)
        )
        print(folder, len(files), sum(f.stat().st_size for f in files), flush=True)


if __name__ == "__main__":
    main()
