"""Inference adapter for original Xiaomi OTSL, cell coordinates and text fragments."""

import torch

from .dataset import collate
from .geometry_encoder import BOX_SOURCES
from .grid_reconstruction import reconstruct
from .otsl_serializer import serialize_hierarchy
from .trainer import move_batch


def in_region(box, region):
    return (
        box is not None and region[0] <= box[0] < box[2] <= region[2]
        and region[1] <= box[1] < box[3] <= region[3]
    )


def nested_graphs(out, i, graph, fragments, threshold, *, next_level=None, visual=None, available=None):
    parents = {c.query_id: c for c in graph.cells if c.query_id is not None}
    requests, uncertain, consumed = [], [], set()
    available = range(len(fragments)) if available is None else available
    order = out["nested_probability"][i].argsort(descending=True).tolist()
    for j in order:
        if out["nested_probability"][i, j] < threshold:
            continue
        parent = parents.get(int(out["parent_cell_id"][i, j]))
        region = out["child_table_region"][i, j].float()
        if graph.nesting_depth:
            frame = region.new_tensor(graph.bbox)
            region = region * (frame[2:] - frame[:2]).repeat(2) + frame[:2].repeat(2)
        bbox = region.tolist()
        evidence = dict(table_id=graph.table_id, nested_query=j, bbox=bbox)
        if parent is None or not in_region(bbox, parent.bbox):
            uncertain.append(dict(evidence, reason="region is not contained by its selected cell"))
            continue
        offset = region[:2].repeat(2)
        scale = (region[2:] - region[:2]).repeat(2)
        boxes = out["child_boxes"][i, j].float() * scale + offset
        indexes = [k for k in available if k not in consumed and in_region(fragments[k].get("bbox"), bbox)]
        result = reconstruct(
            f"{graph.table_id}:child{j}", boxes, out["child_cell_logits"][i, j],
            out["child_type_logits"][i, j], out["child_relation_logits"][i, j],
            [fragments[k] for k in indexes], threshold,
        )
        if result.graph is None:
            uncertain.append(dict(evidence, reason=result.reason, cell_proposals=result.uncertain_cells))
            continue
        result.graph.parent_cell_id = parent.cell_id
        result.graph.nesting_depth = graph.nesting_depth + 1
        result.graph.bbox = tuple(bbox)
        for cell in result.graph.cells:
            cell.ocr_fragments = [indexes[k] for k in cell.ocr_fragments]
        if next_level is not None:
            deeper = next_level(visual[i:i + 1], out["child_embeddings"][i, j:j + 1])
            more, warnings, used = nested_graphs(deeper, 0, result.graph, fragments, threshold, available=indexes)
            consumed.update(used)
            requests.extend(more)
            uncertain.extend(warnings)
        for cell in result.graph.cells:
            consumed.update(cell.ocr_fragments)
        graph.children.append(result.graph)
        requests.extend(result.ocr_requests)
    for cell in graph.cells:
        if consumed.intersection(cell.ocr_fragments):
            cell.ocr_fragments = [k for k in cell.ocr_fragments if k not in consumed]
            cell.text = "\n".join(fragments[k]["text"] for k in cell.ocr_fragments)
    return requests, uncertain, consumed


def decode_one(out, batch, i, config, hierarchy=None):
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
        descend = hierarchy.decode_level if hierarchy is not None and config.max_nesting_depth == 2 else None
        more, uncertain, consumed = nested_graphs(
            out, i, grid.graph, fragments, config.proposal_threshold, next_level=descend,
            visual=out["features"]["visual"] if descend else None,
        )
        requests.extend(more)
        used = {id(fragments[k]) for k in consumed}
        grid.unmatched_ocr = [f for f in grid.unmatched_ocr if id(f) not in used]
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
        hierarchy = getattr(self.model, "hierarchy", None)
        return [decode_one(out, batch, i, self.model.config, hierarchy) for i in range(len(samples))]
