"""Inference adapter for original Xiaomi OTSL, cell coordinates and text fragments."""

import torch

from .dataset import collate
from .geometry_encoder import BOX_SOURCES
from .grid_reconstruction import reconstruct
from .otsl_serializer import serialize_hierarchy
from .trainer import move_batch


def nested_graphs(out, i, graph, fragments, threshold):
    parents = {c.query_id: c for c in graph.cells if c.query_id is not None}
    requests, uncertain, consumed = [], [], set()
    order = out["nested_probability"][i].argsort(descending=True).tolist()
    for j in order:
        if out["nested_probability"][i, j] < threshold:
            continue
        parent = parents.get(int(out["parent_cell_id"][i, j]))
        region = out["child_table_region"][i, j].float()
        bbox = region.tolist()
        if parent is None or not (
            parent.bbox[0] <= bbox[0] < bbox[2] <= parent.bbox[2]
            and parent.bbox[1] <= bbox[1] < bbox[3] <= parent.bbox[3]
        ):
            uncertain.append(dict(nested_query=j, bbox=bbox, reason="region is not contained by its selected cell"))
            continue
        offset = region[:2].repeat(2)
        scale = (region[2:] - region[:2]).repeat(2)
        boxes = out["child_boxes"][i, j].float() * scale + offset
        candidates = [
            (k, f) for k, f in enumerate(fragments)
            if k not in consumed and f.get("bbox") is not None
            and bbox[0] <= f["bbox"][0] < f["bbox"][2] <= bbox[2]
            and bbox[1] <= f["bbox"][1] < f["bbox"][3] <= bbox[3]
        ]
        result = reconstruct(
            f"{graph.table_id}:child{j}", boxes, out["child_cell_logits"][i, j],
            out["child_type_logits"][i, j], out["child_relation_logits"][i, j],
            [f for _, f in candidates], threshold,
        )
        if result.graph is None:
            uncertain.append(dict(nested_query=j, bbox=bbox, reason=result.reason, cell_proposals=result.uncertain_cells))
            continue
        result.graph.parent_cell_id = parent.cell_id
        result.graph.nesting_depth = 1
        result.graph.bbox = tuple(bbox)
        for cell in result.graph.cells:
            cell.ocr_fragments = [candidates[k][0] for k in cell.ocr_fragments]
            consumed.update(cell.ocr_fragments)
        graph.children.append(result.graph)
        requests.extend(result.ocr_requests)
    for cell in graph.cells:
        if consumed.intersection(cell.ocr_fragments):
            cell.ocr_fragments = [k for k in cell.ocr_fragments if k not in consumed]
            cell.text = "\n".join(fragments[k]["text"] for k in cell.ocr_fragments)
    return requests, uncertain, consumed


def decode_one(out, batch, i, config):
    tid, raw = batch["table_ids"][i], batch["raw_otsl"][i]
    indices = batch["cell_mask"][i].nonzero(as_tuple=True)[0].tolist()
    probabilities = out["cell_anomaly_logits"][i, indices].float().sigmoid().cpu().tolist()
    source_names = {value: key for key, value in BOX_SOURCES.items()}
    cells = [
        dict(cell_index=j, probability=p,
             bbox=batch["boxes"][i, j].float().cpu().tolist() if batch["bbox_valid"][i, j] else None,
             bbox_source=source_names[int(batch["bbox_source"][i, j])])
        for j, p in zip(indices, probabilities, strict=True)
    ]
    result = dict(
        table_id=tid, original_otsl=raw, parser_error=batch["parser_error_details"][i],
        anomaly_probability=float(out["anomaly_logits"][i].float().sigmoid()),
        confidence=float(out["confidence"][i]), abstain=bool(out["abstain"][i]),
        cell_anomaly_probability=probabilities, cell_anomalies=cells,
        missing_region_probability=out["missing_region_logits"][i].float().sigmoid().cpu().tolist(),
        visual_token_xy=out["visual_xy"].float().cpu().tolist(),
        ocr_requests=[], table=None, decision="retain_original",
    )
    if "cell_boxes" not in out or result["abstain"]:
        return result
    if result["anomaly_probability"] < config.proposal_threshold and not result["parser_error"]:
        return result
    fragments = batch["fragments"][i]
    grid = reconstruct(tid, out["cell_boxes"][i], out["cell_existence_logits"][i], out["cell_type_logits"][i],
                       out["relation_logits"][i], fragments, config.proposal_threshold)
    if grid.graph is None:
        return dict(result, abstain=True, decision="review", reason=grid.reason,
                    uncertain_cells=grid.uncertain_cells, unmatched_ocr=grid.unmatched_ocr)
    requests = list(grid.ocr_requests)
    uncertain = []
    if "nested_probability" in out:
        more, uncertain, consumed = nested_graphs(out, i, grid.graph, fragments, config.proposal_threshold)
        requests.extend(more)
        grid.unmatched_ocr = [f for f in grid.unmatched_ocr if all(f is not fragments[k] for k in consumed)]
    try:
        table = serialize_hierarchy(grid.graph)
    except ValueError as exc:
        return dict(result, abstain=True, decision="review", reason=str(exc), unmatched_ocr=grid.unmatched_ocr)
    return dict(result, decision="proposed_reconstruction", table=table, ocr_requests=requests,
                unmatched_ocr=grid.unmatched_ocr, uncertain_nested=uncertain)


class RepairPipeline:
    def __init__(self, model):
        self.model = model.eval()

    @torch.inference_mode()
    def __call__(self, samples):
        batch = move_batch(collate(samples, self.model.config), next(self.model.parameters()).device)
        out = self.model(batch)
        return [decode_one(out, batch, i, self.model.config) for i in range(len(samples))]
