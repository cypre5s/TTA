"""Run official PaddleX COCO validation and bounded PP-DocLayoutV3 finetunes on the pilot."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path.cwd()
RUNTIME = HERE.parent / "runtime"
DETECT = RUNTIME / "sources/PaddleDetection"
PDX = RUNTIME / "sources/PaddleX"
ROOT = HERE / "experiments/layout_finetune/pilot_server"
DATA = HERE / "data/datasets/layout/reviewed_table_pilot"
OFFICIAL = PDX / "paddlex/repo_apis/PaddleDetection_api/configs/PP-DocLayoutV3.yaml"
WEIGHT = HERE / "models/layout/PP-DocLayoutV3_pretrained.pdparams"


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(cmd, log, cwd, env):
    start = time.monotonic()
    with log.open("w") as f:
        res = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if res.returncode:
        raise RuntimeError(str(log) + ": " + log.read_text(errors="replace")[-1300:])
    return time.monotonic() - start


def train_trial(name, batch, rate, env):
    import paddle

    d = ROOT / name
    d.mkdir(exist_ok=True)
    result = d / "result.json"
    if result.is_file():
        return json.loads(result.read_bytes())
    log = d / "train.log"
    # The official dynamic training config defaults to 24 workers. Limit it to
    # the container's allocated CPUs and use small-data warmup for this pilot.
    cfg = OFFICIAL.read_text().replace("    steps: 100", "    steps: 4")
    c = d / "PP-DocLayoutV3.yaml"
    c.write_text(cfg)
    cmd = [
        sys.executable,
        str(DETECT / "tools/train.py"),
        "-c",
        str(c),
        "--eval",
        "--enable_ce",
        "True",
        "-o",
        "use_gpu=True",
        "num_classes=25",
        "worker_num=2",
        "log_iter=4",
        "epoch=2",
        "snapshot_epoch=1",
        "target_metrics=bbox",
        f"pretrain_weights={WEIGHT}",
        f"save_dir={d}",
        f"TrainReader.batch_size={batch}",
        f"LearningRate.base_lr={rate:.8f}",
        f"TrainDataset.dataset_dir={DATA}",
        f"EvalDataset.dataset_dir={DATA}",
        f"TestDataset.dataset_dir={DATA}",
    ]
    elapsed = run(cmd, log, DETECT, env)
    best = next(d.rglob("best_model.pdparams"), None)
    last = next(d.rglob("model_final.pdparams"), None)
    if not best or not last:
        raise ValueError(f"Missing best/last checkpoint in {d}")
    state = paddle.load(str(best.with_suffix(".pdstates")))
    ap = float(state["metric"])
    res = dict(
        name=name,
        batch_size=batch,
        learning_rate=rate,
        epochs=2,
        seconds=elapsed,
        validation_metric="bbox AP on reviewed table-only canvases",
        validation_ap=ap,
        best=str(best),
        last=str(last),
        best_epoch=int(state["epoch"]),
        best_sha256=digest(best),
        last_sha256=digest(last),
        config_sha256=digest(c),
        input_sha256=digest(WEIGHT),
        dataset_sha256=digest(DATA / "source_manifest.json"),
        cuda=paddle.device.cuda.get_device_name(),
        paddle=paddle.__version__,
        command=cmd,
    )
    result.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    if not (HERE / "reports/baseline/B0-E2E_metrics.json").is_file():
        raise ValueError("Phase 4 metrics required")
    if (
        digest(WEIGHT)
        != "ae9b0a9ff769a0170bb8e756d8a173992eb99c5e8c0a8176358125640ae5250b"
    ):
        raise ValueError("Incorrect training checkpoint")
    env = dict(
        os.environ,
        PADDLE_PDX_PADDLEDETECTION_PATH=str(DETECT),
        PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True",
        FLAGS_allocator_strategy="auto_growth",
        PYTHONHASHSEED="42",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(RUNTIME / "ppcompat"), str(PDX), str(DETECT), str(HERE), env.get("PYTHONPATH", "")]
    )
    ck = ROOT / "check_dataset"
    if not (ROOT / "checked.json").exists():
        cmd = [
            sys.executable,
            str(PDX / "main.py"),
            "-c",
            str(PDX / "paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml"),
            "-o",
            "Global.mode=check_dataset",
            "-o",
            f"Global.dataset_dir={DATA}",
            "-o",
            f"Global.output={ck}",
            "-o",
            "Global.device=cpu",
        ]
        run(cmd, ROOT / "check_dataset.log", PDX, env)
        outputs = list(ck.rglob("*.json"))
        if not outputs:
            raise ValueError("Missing official checker output")
        for out in outputs:
            record = json.loads(out.read_bytes())
            if record.get("check_pass") is False:
                raise ValueError(out)
        (ROOT / "checked.json").write_text(
            json.dumps({"outputs": [str(p) for p in outputs]})
        )
    trials = []
    for name, batch, rate in [("lr1e5_b4", 4, 1e-5), ("lr3e5_b8", 8, 3e-5)]:
        try:
            res = train_trial(name, batch, rate, env)
        except RuntimeError:
            message = (ROOT / name / "train.log").read_text(errors="replace").lower()
            if not any(
                x in message
                for x in ["out of memory", "resourceexhausted", "outofmemory"]
            ):
                raise
            res = train_trial(name + "_half", max(1, batch // 2), rate, env)
        trials.append(res)
    winner = max(trials, key=lambda x: x["validation_ap"])
    dest = HERE / "models/layout_tender/pilot_server"
    dest.mkdir(parents=True, exist_ok=True)
    for attr in ["best", "last"]:
        target = dest / (attr + ".pdparams")
        if not target.exists():
            os.link(winner[attr], target)
    export = dest / "export"
    if not list(export.glob("**/inference.pdiparams")):
        cmd = [
            sys.executable,
            str(DETECT / "tools/export_model.py"),
            "--for_fd",
            "-c",
            winner["command"][3],
            "--output_dir",
            str(export),
            "-o",
            f"weights={winner['best']}",
            "num_classes=25",
            "pdx_model_name=PP-DocLayoutV3",
            "export_with_pir=True",
            "uniform_output_enabled=True",
            f"TrainDataset.dataset_dir={DATA}",
            f"EvalDataset.dataset_dir={DATA}",
            f"TestDataset.dataset_dir={DATA}",
        ]
        run(cmd, ROOT / "export.log", DETECT, env)
    files = list(export.glob("**/inference.pdiparams"))
    if len(files) != 1:
        raise ValueError("Missing exported inference model")
    (ROOT / "selected.json").write_text(
        json.dumps(
            {
                "trials": trials,
                "selected": winner["name"],
                "export": str(files[0].parent),
            },
            indent=2,
        )
    )
    print(winner["name"], winner["validation_ap"], flush=True)


if __name__ == "__main__":
    main()
