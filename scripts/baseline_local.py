"""Cache original-checkpoint crop and whole-page predictions with local input receipts."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.cloud_ocr import generate
from scripts.ground_truth import immutable, jbytes, sha

ROOT = Path("experiments/baseline/original_v1")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def document_prompt(upstream):
    tree = ast.parse((upstream / "evaluation_pipeline/e2e_img2md.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "PROMPTS":
                return ast.literal_eval(node.value)["document"]
    raise ValueError("Cannot identify upstream document prompt")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    dataset = Path("data/splits/ground_truth_v1")
    if not (dataset / "manifest.json").is_file():
        raise SystemExit("Phase 3 is pending: missing frozen ground_truth_v1")
    test_data = (dataset / "test.jsonl").read_bytes()
    if (
        sha(test_data)
        != json.loads((dataset / "manifest.json").read_bytes())["sha256"]["test"]
    ):
        raise SystemExit("Test split hash mismatch")
    held_out = [json.loads(line) for line in test_data.splitlines()]
    if not held_out:
        raise SystemExit("Empty Phase 3 test split")
    for record in held_out:
        data = Path(record["gt"]).read_bytes()
        decision = json.loads(data)
        if (
            sha(data) != record["gt_sha256"]
            or decision["decision"] != "approve"
            or decision["reviewer_type"] != "human"
            or decision["table_id"] != record["table_id"]
        ):
            raise SystemExit("Phase 3 reference verification failed")
    test_ids = {record["table_id"] for record in held_out}
    cfg = yaml.safe_load(Path("configs/xiaomi_local.yaml").read_text())
    upstream = Path(cfg["upstream"])
    sys.path.insert(0, str(upstream))
    from evaluation_pipeline.cropping import fit_image
    from postprocess.document import finalize_document

    ckpt = Path(cfg["checkpoint"])
    weights = digest(ckpt / "model.safetensors")
    if (
        weights
        != json.loads(Path("configs/xiaomi_provenance.json").read_text())[
            "weight_sha256"
        ]
    ):
        raise ValueError("Checkpoint hash mismatch")
    layout = json.loads(Path("configs/layout_provenance.json").read_text())
    layout_dir = Path(
        yaml.safe_load(Path("configs/layout.yaml").read_text())["model_dir"]
    )
    if (
        not layout_dir.is_dir()
        and (Path("models/layout/PP-DocLayoutV3") / "inference.pdiparams").is_file()
    ):
        layout_dir = Path("models/layout/PP-DocLayoutV3")
    for name, expected in layout["files"].items():
        if digest(layout_dir / name) != expected:
            raise ValueError("Layout weight or config mismatch")
    prompts = {"B0": cfg["prompt"], "B0-E2E": document_prompt(upstream)}
    caps = {"B0": cfg["max_new_tokens"], "B0-E2E": 16384}
    calibration_path = ROOT / "calibration.json"
    calibration = (
        json.loads(calibration_path.read_bytes())
        if calibration_path.is_file()
        else None
    )
    batch_sizes = (
        calibration["batch_sizes"] if calibration else {arm: 4 for arm in prompts}
    )
    run = dict(
        model=weights,
        layout=layout,
        decoding="greedy",
        precision="bfloat16",
        attention="sdpa",
        seed=42,
        batch_sizes=batch_sizes,
        calibration_sha256=digest(calibration_path) if calibration else None,
        test_sha256=sha(test_data),
        phase3_receipt=sha(
            Path("data/annotation/phase3_v1/completion.json").read_bytes()
        ),
        runtime="original",
        prompts=prompts,
        max_new_tokens=caps,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        script_sha256=digest(Path(__file__)),
        generate_sha256=digest(Path("scripts/cloud_ocr.py")),
        commit=(
            Path("source_revision.txt").read_text().strip()
            if Path("source_revision.txt").exists()
            else subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        ),
    )
    run_path = ROOT / "config.json"
    immutable(run_path, jbytes(run))
    torch.manual_seed(42)
    proc = AutoProcessor.from_pretrained(ckpt, local_files_only=True)
    proc.tokenizer.padding_side = "left"
    model = (
        AutoModelForImageTextToText.from_pretrained(
            ckpt,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    queue = [
        json.loads(line)
        for line in Path("data/annotation/phase3_v1/packet_index.jsonl")
        .read_text()
        .splitlines()
    ]
    queue = [r for r in queue if r["table_id"] in test_ids]
    if {r["table_id"] for r in queue} != test_ids:
        raise ValueError("Missing held-out packets")
    todo = {arm: {} for arm in prompts}
    for r in queue:
        packet = Path(r["packet"])
        m = json.loads(packet.read_text())
        if sha(packet.read_bytes()) != r["packet_sha256"]:
            raise ValueError(packet)
        pid = m["page_id"]
        base = Path("data/rendered") / m["document_sha256"] / pid
        page = json.loads(base.with_suffix(".json").read_text())
        for arm, key, path, expected in [
            ("B0-E2E", pid, Path(page["image"]), page["image_sha256"]),
            ("B0", m["table_id"], Path(m["image"]), m["image_sha256"]),
        ]:
            dest = ROOT / arm / (key + ".json")
            if digest(path) != expected:
                raise ValueError(path)
            if dest.exists():
                old = json.loads(dest.read_bytes())
                if old["image_sha256"] != expected or old["config_sha256"] != digest(
                    run_path
                ):
                    raise ValueError(dest)
            else:
                todo[arm][key] = (key, pid, path, expected, dest)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]

    def one_batch(arm, group, retried=False):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            generated = generate(
                model,
                proc,
                [x[2] for x in group],
                dict(prompt=prompts[arm], max_new_tokens=caps[arm]),
                fit_image,
            )
        except torch.OutOfMemoryError:
            if len(group) == 1 or retried:
                raise
            generated = None
        if generated is None:
            torch.cuda.empty_cache()
            half = len(group) // 2
            one_batch(arm, group[:half], True)
            one_batch(arm, group[half:], True)
            return
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        for (key, pid, path, expected, dest), ids in zip(group, generated, strict=True):
            tokens = ids.tolist()
            end = next((i + 1 for i, t in enumerate(tokens) if t in eos), len(tokens))
            raw = proc.decode(
                tokens[:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            row = dict(
                page_id=pid,
                image_sha256=expected,
                config_sha256=digest(run_path),
                raw=raw,
                output_tokens=end,
                seconds=elapsed / len(group),
                batch_seconds=elapsed,
                batch_ids=[x[0] for x in group],
                peak_allocated_bytes=peak,
                ended_with_eos=tokens[end - 1] in eos,
            )
            if arm == "B0":
                row.update(
                    table_id=key,
                    raw_otsl=raw,
                    layout_sha256=digest(
                        Path("data/layout/predictions") / (pid + ".json")
                    ),
                )
            else:
                row["markdown"] = finalize_document(raw)
            immutable(dest, jbytes(row))
        print(arm, len(group), round(elapsed, 2), flush=True)

    from PIL import Image

    def cost(item):
        with Image.open(item[2]) as im:
            return (im.width * im.height, item[0])

    work = {a: sorted(items.values(), key=cost) for a, items in todo.items()}
    for arm, items in work.items():
        step = batch_sizes[arm]
        for pos in range(0, len(items), step):
            one_batch(arm, items[pos : pos + step])


if __name__ == "__main__":
    main()
