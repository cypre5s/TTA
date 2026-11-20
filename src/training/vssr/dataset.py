"""Read reviewed table crops without converting absent annotations into negatives."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from scripts.ground_truth import sha
from scripts.training_reference import OCR_TIER, permits_machine_training, training_reference
from src.otsl.grid import parse
from src.training.repair import encode
from .geometry_encoder import BOX_SOURCES
from .relation_decoder import grid_edges


@dataclass
class TableSample:
    table_id: str
    image: torch.Tensor
    raw_otsl: str
    corrected_otsl: str | None = None
    geometry: list[dict | None] = field(default_factory=list)
    ocr_fragments: list[dict] = field(default_factory=list)
    supervision: dict = field(default_factory=dict)


def image_tensor(path, size, expected=None):
    from io import BytesIO

    body = path.read_bytes()
    if expected is not None and sha(body) != expected:
        raise ValueError(f"Crop hash mismatch: {path}")
    with Image.open(BytesIO(body)) as original:
        rgb = original.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    pixels = pixels.permute(2, 0, 1).float() / 255
    mean = pixels.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = pixels.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (pixels - mean) / std


def checked_box(value):
    if value is None:
        return None
    if (
        len(value) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in value)
        or value[0] >= value[2] or value[1] >= value[3]
    ):
        raise ValueError("Expected normalized, positive-area xyxy coordinates")
    return tuple(value)


def cell_targets(grid, geometry=None):
    cells = [dict(bbox=(geometry or {}).get((c.row, c.column, c.rowspan, c.colspan)),
                  type=int(bool(c.text.strip()))) for c in grid.cells]
    return dict(cells=cells, relations=grid_edges(grid.cells))


class VSSRDataset(Dataset):
    """JSONL records use normalized xyxy, OTSL, optional labels and image SHA256."""

    def __init__(self, records, config, root=Path(".")):
        self.records = ([json.loads(line) for line in records.read_bytes().splitlines()]
                        if isinstance(records, Path) else list(records))
        self.root = Path(root)
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        raw, correct = row["raw_otsl"], row.get("corrected_otsl")
        labels = dict(row.get("supervision", {}))
        if correct is not None:
            labels.setdefault("anomaly", float(raw != correct))
            if "cells" not in labels:
                labels.update(cell_targets(parse(correct)))
            raw_tokens, target_tokens = encode(raw), encode(correct)
            if "repair_tokens" not in labels and len(raw_tokens) == len(target_tokens):
                labels["repair_tokens"] = [b if a != b else None for a, b in zip(raw_tokens, target_tokens, strict=True)]
        return TableSample(
            row["table_id"], image_tensor(self.root / row["image"], self.config.image_size, row.get("image_sha256")),
            raw, correct, row.get("geometry", []), row.get("ocr_fragments", []), labels,
        )


def checked_json(path, expected):
    body = path.read_bytes()
    if sha(body) != expected:
        raise ValueError(f"Changed input: {path}")
    return json.loads(body)


def root_projection(page, packet):
    if not page or page.get("rotation", 0) != 0:
        return lambda _: None
    x0, y0, x1, y1 = packet["bbox_px"]
    scale = page["dpi"] / 72
    frame = tuple(v / scale for v in (x0, y0, x1, y1))
    return lambda box: normalize_box(box, frame)


def normalize_box(box, frame):
    l, t, r, b = frame
    coords = ((box[0] - l) / (r - l), (box[1] - t) / (b - t),
              (box[2] - l) / (r - l), (box[3] - t) / (b - t))
    bounded = tuple(max(0.0, min(1.0, v)) for v in coords)
    return bounded if bounded[0] < bounded[2] and bounded[1] < bounded[3] else None


def reviewed_cell_targets(grid, tree, project):
    if tree is None:
        return cell_targets(grid)
    anchors = {(c.row, c.column, c.rowspan, c.colspan): i for i, c in enumerate(grid.cells)}
    boxes, nested = {}, []
    for cell in tree["cells"]:
        anchor = tuple(cell[k] for k in ("row", "column", "rowspan", "colspan"))
        boxes[anchor] = project(cell["bbox_pt"])
        for child in cell["children"]:
            frame = child["bbox_pt"]
            targets = reviewed_cell_targets(parse(child["otsl"]), child, lambda box: normalize_box(box, frame))
            nested.append(dict(targets, parent=anchors[anchor], bbox=project(frame)))
    return dict(cell_targets(grid, boxes), nested=nested)


class ReviewedPilot(Dataset):
    """Read frozen training/validation splits and the historical Pilot smoke cohort."""

    def __init__(self, root, config, evidence=None, *, split="train", use_reference_boxes=True):
        if split not in ("train", "validation") or (evidence is not None and split != "train"):
            raise ValueError("Expected a frozen training or validation partition")
        self.root = Path(root)
        self.config = config
        self.use_reference_boxes = use_reference_boxes
        base = self.root / "data/splits/ground_truth_v1"
        manifest = json.loads((base / "manifest.json").read_bytes())
        body = (base / (split + ".jsonl")).read_bytes()
        if sha(body) != manifest["sha256"][split]:
            raise ValueError("Reviewed partition differs from its freeze hash")
        self.allow_machine = permits_machine_training(base, self.root)
        self.manifest_version = manifest["version"]
        self.train_sha256 = sha(body)
        self.rows = [json.loads(line) for line in body.splitlines()]
        self.xiaomi_sha256 = json.loads((self.root / "configs/xiaomi_provenance.json").read_bytes())["weight_sha256"]
        self.evidence = Path(evidence) if evidence is not None else None
        self.pages = {}
        if self.evidence is not None:
            selection = json.loads((self.evidence / "manifest.json").read_bytes())
            if selection["train_sha256"] != self.train_sha256:
                raise ValueError("Smoke selection was made against another Pilot train split")
            available = {r["table_id"]: r for r in self.rows}
            self.rows = [available[tid] for tid in selection["table_ids"]]
            if len(self.rows) != len(set(selection["table_ids"])) or not 8 <= len(self.rows) <= 16:
                raise ValueError("Expected 8–16 distinct approved samples")
            self.pages = selection["pages"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        tid = row["table_id"]
        gt = training_reference(dict(row, gt=self.root / row["gt"]), self.allow_machine)
        if self.evidence is not None and gt["reviewer_type"] != "human":
            raise ValueError("Smoke selection requires human-approved labels")
        packet = checked_json(self.root / "data/annotation/packets" / (tid + ".json"), gt["packet_sha256"])
        if packet["table_id"] != tid or packet["image_sha256"] != gt["image_sha256"]:
            raise ValueError("Pilot packet does not match the reviewed crop")
        pred = checked_json(self.root / packet["prediction"], packet["prediction_sha256"])
        if pred["weight_sha256"] != self.xiaomi_sha256 or pred.get("precision", "bfloat16") != "bfloat16":
            raise ValueError("Smoke input must use the original Xiaomi BF16 checkpoint")
        raw = pred["raw_otsl"]
        truth = parse(gt["corrected_otsl"])
        page_path = self.root / "data/rendered" / packet["document_sha256"] / (packet["page_id"] + ".json")
        if self.evidence is not None and packet["page_id"] in self.pages:
            pid = packet["page_id"]
            page = checked_json(self.evidence / (pid + ".json"), self.pages[pid])
        else:
            page = json.loads(page_path.read_bytes()) if page_path.is_file() else None
        if page and (page["page_id"] != packet["page_id"] or page["document_sha256"] != packet["document_sha256"]):
            raise ValueError("Page geometry provenance mismatch")
        tree = gt.get("structure_tree")
        labels = reviewed_cell_targets(truth, tree, root_projection(page, packet))
        geometry = []
        if self.use_reference_boxes and tree is not None:
            try:
                raw_grid = parse(raw)
            except ValueError:
                raw_grid = None
            # Input boxes are projected only for OCR and reference cells that agree.
            matching = {(c.row, c.column, c.rowspan, c.colspan): c.text for c in truth.cells}
            all_boxes = {(c.row, c.column, c.rowspan, c.colspan): p["bbox"]
                         for c, p in zip(truth.cells, labels["cells"], strict=True)}
            for c in raw_grid.cells if raw_grid else ():
                key = c.row, c.column, c.rowspan, c.colspan
                bbox = all_boxes.get(key) if matching.get(key) == c.text else None
                geometry.append(dict(bbox=bbox, source="ground_truth") if bbox is not None else None)
        if row.get("tier") != OCR_TIER:
            labels["anomaly"] = float(raw != gt["corrected_otsl"])
        if not self.use_reference_boxes:
            raw_tokens, target_tokens = encode(raw), encode(gt["corrected_otsl"])
            if len(raw_tokens) == len(target_tokens):
                labels["repair_tokens"] = [b if a != b else None for a, b in zip(raw_tokens, target_tokens, strict=True)]
        return TableSample(tid, image_tensor(self.root / packet["image"], self.config.image_size, gt["image_sha256"]),
                           raw, gt["corrected_otsl"], geometry, supervision=labels)


def collate(samples, config):
    if not samples:
        raise ValueError("Empty table batch")
    grids, errors, encoded = [], [], []
    for sample in samples:
        try:
            grids.append(parse(sample.raw_otsl))
            errors.append(None)
        except ValueError as exc:
            grids.append(None)
            errors.append(str(exc))
        encoded.append(encode(sample.raw_otsl))
    b = len(samples)
    length = min(config.max_tokens, max(len(t) for t in encoded))
    slots = min(config.max_cells, max([1, *(len(g.cells) for g in grids if g is not None)]))
    truncated = [len(t) > length or bool(g and len(g.cells) > slots) for t, g in zip(encoded, grids, strict=True)]
    tokens = torch.zeros(b, length, dtype=torch.long)
    grid = torch.zeros(b, slots, 5)
    text = torch.zeros(b, slots, config.max_text_bytes, dtype=torch.long)
    types = torch.zeros(b, slots, dtype=torch.long)
    cell_mask = torch.zeros(b, slots, dtype=torch.bool)
    edges = torch.zeros(b, slots, slots, 4, dtype=torch.bool)
    coordinates = torch.zeros(b, slots, 4)
    valid = torch.zeros(b, slots, dtype=torch.bool)
    source = torch.zeros(b, slots, dtype=torch.long)
    ocr = []
    for i, (sample, sample_grid, ids) in enumerate(zip(samples, grids, encoded, strict=True)):
        ids = ids[:length]
        tokens[i, :len(ids)] = torch.tensor(ids)
        fragments = [dict(f, bbox=checked_box(f.get("bbox"))) for f in sample.ocr_fragments]
        if sample_grid is not None:
            nodes = sample_grid.cells[:slots]
            edges[i, :len(nodes), :len(nodes)] = grid_edges(nodes)[..., :4]
            for j, node in enumerate(nodes):
                cell_mask[i, j] = True
                grid[i, j] = torch.tensor((node.row / sample_grid.rows, node.column / sample_grid.columns,
                                           node.rowspan / sample_grid.rows, node.colspan / sample_grid.columns,
                                           float(not node.text.strip())))
                types[i, j] = int(bool(node.text.strip()))
                ids = [v + 1 for v in node.text.encode()[:config.max_text_bytes]]
                if ids:
                    text[i, j, :len(ids)] = torch.tensor(ids)
                pos = sample.geometry[j] if j < len(sample.geometry) else None
                bbox = checked_box(pos.get("bbox")) if pos else None
                if bbox is not None:
                    tag = BOX_SOURCES[pos["source"]]
                    if tag == BOX_SOURCES["missing"]:
                        raise ValueError("A present box needs a nonmissing provenance")
                    valid[i, j], source[i, j], coordinates[i, j] = True, tag, torch.tensor(bbox)
                if not sample.ocr_fragments and node.text:
                    fragments.append(dict(text=node.text, bbox=bbox,
                                          grid_anchor=(node.row, node.column, node.rowspan, node.colspan)))
        ocr.append(fragments)
    return dict(
        images=torch.stack([s.image for s in samples]), tokens=tokens, token_mask=tokens.ne(0),
        cell_grid=grid, cell_text=text, cell_type=types, cell_mask=cell_mask, cell_edges=edges,
        boxes=coordinates, bbox_valid=valid, bbox_source=source, parser_error=torch.tensor([e is not None for e in errors]),
        structure_truncated=torch.tensor(truncated), parser_error_details=errors,
        table_ids=[s.table_id for s in samples], raw_otsl=[s.raw_otsl for s in samples], fragments=ocr,
        targets=[s.supervision for s in samples],
    )
