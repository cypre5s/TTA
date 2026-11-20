"""Run the upgrade's bounded smoke checks and report aggregate evidence only."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import torch

from scripts.ground_truth import immutable, jbytes, sha
from src.otsl.grid import parse
from src.training.repair import encode
from src.training.vssr import VSSRConfig, VSSR1, upgrade
from src.training.vssr.backbone import Backbone
from src.training.vssr.checkpoint import load_checkpoint, save_checkpoint
from src.training.vssr.dataset import ReviewedPilot, TableSample, cell_targets, collate
from src.training.vssr.grid_reconstruction import reconstruct
from src.training.vssr.losses import compute_losses
from src.training.vssr.otsl_serializer import serialize_hierarchy
from src.training.vssr.repair_pipeline import decode_one, nested_graphs
from src.training.vssr.trainer import move_batch


def params(module):
    return dict(total=sum(p.numel() for p in module.parameters()),
                trainable=sum(p.numel() for p in module.parameters() if p.requires_grad))


def prepare_evidence(root, output, config, count):
    from PIL import Image

    dataset = ReviewedPilot(root, config)
    ids, groups, pages = [], set(), {}
    for i, row in enumerate(dataset.rows):
        if row["group_id"] in groups:
            continue
        gt = json.loads((root / row["gt"]).read_bytes())
        if not gt.get("structure_tree"):
            continue
        sample = dataset[i]
        if not any(sample.geometry) or len(encode(sample.raw_otsl)) > config.max_tokens:
            continue
        tid = row["table_id"]
        packet = json.loads((root / "data/annotation/packets" / (tid + ".json")).read_bytes())
        with Image.open(root / packet["image"]) as im:
            if min(im.size) < 2:
                continue
        pid = packet["page_id"]
        body = (root / "data/rendered" / packet["document_sha256"] / (pid + ".json")).read_bytes()
        immutable(output / (pid + ".json"), body)
        ids.append(tid)
        pages[pid] = sha(body)
        groups.add(row["group_id"])
        if len(ids) == count:
            break
    if len(ids) != count:
        raise RuntimeError("Insufficient approved crops with independent geometry")
    immutable(output / "manifest.json", jbytes(dict(train_sha256=dataset.train_sha256, table_ids=ids, pages=pages)))
    print(f"Prepared {count} existing Pilot train selections and page geometry sidecars", flush=True)


def fixture(config):
    corrected = "<fcel>项目<fcel>规格<lcel><nl>\n<ucel><fcel>A<fcel>B<nl>\n<fcel>明细<fcel>C<fcel>D<nl>"
    raw = corrected.replace("<fcel>B", "<fcel>E")
    grid = parse(corrected)
    positions = {
        (c.row, c.column, c.rowspan, c.colspan): (
            c.column / grid.columns, c.row / grid.rows,
            (c.column + c.colspan) / grid.columns, (c.row + c.rowspan) / grid.rows,
        ) for c in grid.cells
    }
    labels = cell_targets(grid, positions)
    geometry = [dict(bbox=c["bbox"], source="ground_truth") for c in labels["cells"]]
    geometry[-1] = None
    labels["cells"][-1]["bbox"] = None
    child_grid = parse("<fcel>代号<fcel>数量<nl>\n<fcel>A<fcel>1<nl>")
    child_boxes = {(c.row, c.column, c.rowspan, c.colspan): (c.column / 2, c.row / 2, (c.column + 1) / 2, (c.row + 1) / 2)
                   for c in child_grid.cells}
    nested = [dict(cell_targets(child_grid, child_boxes), parent=0, bbox=bbox)
              for bbox in ((0.025, 0.03, 0.31, 0.28), (0.025, 0.35, 0.31, 0.61))]
    labels.update(
        anomaly=1.0, confidence=1.0, cell_anomaly=[0, 0, 0, 1, 0, 0, None],
        relation_anomaly=[(1, 3, 1)], missing_regions=[(0.55, 0.55, 0.98, 0.97)],
        alignment=[dict(cell_index=0, point=(0.15, 0.22))], nested=nested,
        repair_tokens=[b if a != b else None for a, b in zip(encode(raw), encode(corrected), strict=True)],
    )
    pixels = torch.ones(3, config.image_size, config.image_size) * 0.8
    for x0, y0, x1, y1 in positions.values():
        l, t, r, b = map(lambda x: min(config.image_size - 1, round(x * config.image_size)), (x0, y0, x1, y1))
        pixels[:, [t, b], l:r + 1] = 0.1
        pixels[:, t:b + 1, [l, r]] = 0.1
    sample = TableSample("fixture", pixels, raw, corrected, geometry=geometry, supervision=labels)
    return sample, list(positions.values())


def reconstruction_fixture(sample, parent_boxes, nested):
    n = len(parent_boxes)
    boxes = torch.tensor(parent_boxes)
    types = torch.tensor([[0., 1., 0., 0.]]).repeat(n, 1)
    fragments = [dict(text=c.text, grid_anchor=(c.row, c.column, c.rowspan, c.colspan)) for c in parse(sample.raw_otsl).cells]
    graph = reconstruct("fixture", boxes, torch.ones(n) * 9, types, fragments=fragments).graph
    assert graph is not None and any(c.rowspan > 1 for c in graph.cells) and any(c.colspan > 1 for c in graph.cells)
    assert "<fcel>E" in serialize_hierarchy(graph)["parent_table"]["otsl"]
    # The fixture validates decoding independently of untrained proposal scores.
    if nested:
        children = sample.supervision["nested"]
        regions = torch.tensor([[c["bbox"] for c in children]])
        child_boxes = torch.tensor([[c["bbox"] for c in child["cells"]] for child in children])[None]
        children_out = dict(
            nested_probability=torch.ones(1, 2), parent_cell_id=torch.zeros(1, 2, dtype=torch.long),
            child_table_region=regions, child_boxes=child_boxes, child_cell_logits=torch.full((1, 2, 4), 9.),
            child_type_logits=torch.ones(1, 2, 4, 4), child_relation_logits=torch.zeros(1, 2, 4, 4, 5),
        )
        _, uncertain, _ = nested_graphs(children_out, 0, graph, fragments, 0.5)
        assert not uncertain
    result = serialize_hierarchy(graph)
    if nested:
        assert len(result["child_tables"]) == 2 and result["nesting_depth"] == 1
        assert len(set(result["parent_cell_id"])) == 1
        assert all(parse(child["otsl"]).rows == 2 for child in result["child_tables"])
    return dict(rows=graph.rows, columns=graph.columns, parent_cells=n,
                rowspan=max(c.rowspan for c in graph.cells), colspan=max(c.colspan for c in graph.cells),
                child_tables=len(result["child_tables"]), depth=result["nesting_depth"])


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak(device):
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None


def timed_forward(model, batch, device, bf16):
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode(), amp(device, bf16):
        out = model(batch)
    sync(device)
    elapsed = time.perf_counter() - start
    for name, value in out.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(name)
    for name in ("confidence", "nested_probability"):
        if name in out and not ((out[name] >= 0) & (out[name] <= 1)).all():
            raise ValueError(name)
    return out, dict(seconds=elapsed, peak_bytes=peak(device), forward_dtype=str(out["anomaly_logits"].dtype),
                     shapes={k: list(v.shape) for k, v in out.items() if torch.is_tensor(v)},
                     backbone_shapes=out["backbone_shapes"], fpn_shapes=out["fpn_shapes"],
                     fused_shapes={k: list(v.shape) for k, v in out["features"].items()},
                     parameters=params(model), modules={k: params(v) for k, v in model.named_children()})


def amp(device, bf16):
    return torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()


def real_forward(model, real, config, device, bf16):
    records, geometry = [], 0
    for i in range(0, len(real), 2):
        batch = move_batch(collate(real[i:i + 2], config), device)
        valid = int(batch["bbox_valid"].sum())
        geometry += valid
        out, result = timed_forward(model, batch, device, bf16)
        decoded = [decode_one(out, batch, j, config) for j in range(len(batch["raw_otsl"]))]
        assert all(r["original_otsl"] == raw for r, raw in zip(decoded, batch["raw_otsl"], strict=True))
        assert valid == sum(c["bbox"] is not None for r in decoded for c in r["cell_anomalies"])
        assert all(0 <= p <= 1 for r in decoded for p in r["missing_region_probability"])
        records.append(dict(
            result, probabilities=[r["anomaly_probability"] for r in decoded],
            abstained=sum(r["abstain"] for r in decoded), valid_geometry=valid,
            cell_anomaly_locations=sum(c["bbox"] is not None for r in decoded for c in r["cell_anomalies"]),
        ))
    if not geometry:
        raise ValueError("Real smoke input contained no verified cell geometry")
    return dict(count=len(real), batch_size=2, geometry=geometry, batches=records)


def backward(model, batch, device, bf16):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sample_parameter = model.repair.weight
    before = sample_parameter.detach().clone()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    with amp(device, bf16):
        out = model(batch)
    losses = compute_losses(out, batch, model.config)
    sync(device)
    start_back = time.perf_counter()
    verified = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    for name, loss in losses.items():
        if name == "total":
            continue
        grads = torch.autograd.grad(loss, trainable, allow_unused=True, retain_graph=True)
        grads = [g for g in grads if g is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().max() > 0 for g in grads):
            raise FloatingPointError(f"Loss {name} produced no finite nonzero gradient")
        verified.append(name)
    sync(device)
    repeated = time.perf_counter() - start_back
    single = time.perf_counter()
    losses["total"].backward()
    sync(device)
    elapsed = time.perf_counter() - single
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing or not all(torch.isfinite(p.grad).all() for p in trainable):
        raise FloatingPointError(f"Invalid combined gradients; missing={missing}")
    optimizer.step()
    sync(device)
    if torch.equal(before, sample_parameter):
        raise RuntimeError("Optimizer did not update the repair head")
    return dict(losses={k: float(v.detach()) for k, v in losses.items()}, verified_losses=verified,
                trainable_parameter_tensors=len(trainable), backward_seconds=elapsed,
                individual_grad_checks_seconds=repeated, forward_and_loss_seconds=start_back - started,
                peak_bytes=peak(device), optimizer_steps=1)


def run(args, config, mode):
    device = torch.device(args.device)
    bf16 = device.type == "cuda"
    if bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA BF16 is required for the RTX 4090 smoke tests")
    torch.manual_seed(42)
    if device.type == "cpu":
        torch.set_num_threads(2)
    report = dict(device=torch.cuda.get_device_name(device) if bf16 else "CPU", torch=str(torch.__version__),
                  cuda=torch.version.cuda, precision="bf16" if bf16 else "fp32", input_size=config.image_size,
                  config_sha256=config.digest(), seed=42, tests={})
    if bf16:
        report["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True,
        ).strip()
    tests = report["tests"]
    sample, boxes = fixture(config)
    weights = dict(pretrained=not args.random_backbone, weights=args.weights)
    if mode in ("all", "backbone"):
        original = Backbone(config, **weights).to(device).eval()
        pixels = sample.image[None].to(device)
        if bf16:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode(), amp(device, bf16):
            pyramid = original(pixels)
        sync(device)
        assert [p.shape[-1] for p in pyramid] == [config.image_size // s for s in (4, 8, 16, 32)]
        tests["backbone"] = dict(shapes=[list(p.shape) for p in pyramid], peak_bytes=peak(device),
                                 seconds=time.perf_counter() - start, parameters=params(original), provenance=original.provenance)
        del pyramid, original, pixels
        if mode == "backbone":
            return report
    model = VSSR1(config, **weights).to(device).eval()
    invalid = TableSample("parser-failure", sample.image, "<lcel><nl>")
    synthetic = move_batch(collate([sample, invalid], config), device)
    fallback, tests["vssr-1"] = timed_forward(model, synthetic, device, bf16)
    assert fallback["parser_error"].tolist() == [False, True] and not synthetic["bbox_valid"][1].any()
    del fallback
    real = []
    if args.pilot_root:
        dataset = ReviewedPilot(args.pilot_root, config, args.evidence)
        real = [dataset[i] for i in range(min(args.real_count, len(dataset)))]
        if len(real) != args.real_count:
            raise ValueError("Insufficient approved Pilot train examples")
        tests["real_train_forward"] = real_forward(model, real, config, device, bf16)
    if mode == "vssr-1":
        return report
    model = upgrade(model, "vssr-2").eval()
    out, tests["vssr-2"] = timed_forward(model, synthetic, device, bf16)
    assert out["relation_logits"].shape == (2, config.cell_queries, config.cell_queries, 5)
    tests["vssr-2"]["controlled_reconstruction"] = reconstruction_fixture(sample, boxes, False)
    del out
    if mode == "vssr-2":
        return report
    model = upgrade(model, "vssr-3").eval()
    out, tests["vssr-3"] = timed_forward(model, synthetic, device, bf16)
    assert out["parent_logits"].shape == (2, config.nested_queries, config.cell_queries)
    tests["vssr-3"]["controlled_hierarchy"] = reconstruction_fixture(sample, boxes, True)
    del out
    if real:
        tests["real_train_forward_vssr3"] = real_forward(model, real, config, device, bf16)
    supervised = move_batch(collate([sample], config), device)
    tests["backward"] = backward(model, supervised, device, bf16)
    model.eval()
    checkpoint = args.output / "vssr3_smoke.pt"
    save_checkpoint(checkpoint, model, step=1)
    restored, metadata = load_checkpoint(checkpoint, device, config, "vssr-3")
    with torch.inference_mode(), amp(device, bf16):
        before, after = model(supervised), restored(supervised)
    keys = ("anomaly_logits", "cell_boxes", "relation_logits", "nested_region", "parent_logits", "child_boxes")
    errors = {k: float((before[k].float() - after[k].float()).abs().max()) for k in keys}
    if any(error > 1e-5 for error in errors.values()):
        raise RuntimeError("Reloaded checkpoint changed output")
    tests["checkpoint"] = dict(step=metadata["step"], version=metadata["model_version"],
                               config_sha256=metadata["config_sha256"], max_absolute_error=errors)
    return report


def main(mode="all"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/vssr_smoke.yaml"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--real-count", type=int, default=8)
    parser.add_argument("--prepare-evidence", action="store_true")
    parser.add_argument("--random-backbone", action="store_true")
    parser.add_argument("--backbone", choices=("resnet18", "repvit_m1_0.dist_450e_in1k"))
    args = parser.parse_args()
    if not 8 <= args.real_count <= 16:
        parser.error("Pilot smoke scope must be 8–16 samples")
    if args.prepare_evidence and not args.pilot_root:
        parser.error("Evidence preparation needs --pilot-root")
    config = VSSRConfig.load(args.config)
    if args.backbone:
        config = replace(config, backbone=args.backbone)
    if args.prepare_evidence:
        prepare_evidence(args.pilot_root, args.output, config, args.real_count)
        return
    if (args.output / "smoke.json").exists():
        parser.error("Use a fresh output path")
    report = run(args, config, mode)
    immutable(args.output / "smoke.json", jbytes(report))
    print(json.dumps(report, ensure_ascii=False), flush=True)
