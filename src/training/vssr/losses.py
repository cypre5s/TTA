"""Independently reported losses with supervision and coordinate masks."""

from collections import defaultdict

import torch
from torch.nn import functional as F

from .cell_proposal import box_metrics, hungarian
from .constraint_layer import grid_constraints
from .dataset import checked_box


def positive_targets(cells, device):
    boxes = [checked_box(c.get("bbox")) for c in cells]
    types = [c.get("type") for c in cells]
    return (
        torch.tensor([b if b is not None else (0, 0, 0, 0) for b in boxes], device=device).reshape(-1, 4),
        torch.tensor([b is not None for b in boxes], device=device, dtype=torch.bool),
        torch.tensor([t if t is not None else 0 for t in types], device=device, dtype=torch.long),
        torch.tensor([t is not None for t in types], device=device, dtype=torch.bool),
    )


def binary(logits, target):
    return F.binary_cross_entropy_with_logits(logits.float(), torch.as_tensor(target, device=logits.device, dtype=torch.float32))


def localization(boxes, expected):
    _, giou = box_metrics(boxes.float(), expected.float())
    return F.l1_loss(boxes.float(), expected.float()) + (1 - giou).mean()


def set_losses(exists, boxes, types, cells, relation_logits, relation_target):
    target, box_ok, cls, cls_ok = positive_targets(cells, boxes.device)
    pi, ti = hungarian(exists, boxes, types, target, box_ok, cls, cls_ok)
    object_target = torch.zeros_like(exists)
    object_target[pi] = 1
    cell = binary(exists, object_target)
    zero = cell * 0
    bbox, relation = zero, zero
    if cls_ok[ti].any():
        yes = cls_ok[ti]
        cell = cell + F.cross_entropy(types[pi[yes]].float(), cls[ti[yes]])
    if box_ok[ti].any():
        yes = box_ok[ti]
        bbox = localization(boxes[pi[yes]], target[ti[yes]])
    if relation_target is not None and len(pi) > 1:
        expected = torch.as_tensor(relation_target, device=boxes.device)[ti][:, ti]
        off_diag = ~torch.eye(len(pi), device=boxes.device, dtype=torch.bool)
        valid = off_diag[..., None] & (expected >= 0)
        if valid.any():
            relation = binary(relation_logits[pi][:, pi][valid], expected[valid])
    return dict(cell=cell, bbox=bbox, relation=relation), dict(zip(ti.tolist(), pi.tolist())), target[box_ok]


def hierarchy_loss(out, i, children, parent_match):
    output_box = out["nested_region"][i]
    parent_logits = out["parent_logits"][i]
    existence = out["nested_logits"][i]
    targets, box_ok, _, _ = positive_targets(children, output_box.device)
    parents = [parent_match.get(c["parent"]) for c in children]
    parent_ok = torch.tensor([p is not None for p in parents], device=output_box.device, dtype=torch.bool)
    parent_ids = torch.tensor([p if p is not None else 0 for p in parents], device=output_box.device, dtype=torch.long)
    pi, ti = hungarian(existence, output_box, parent_logits, targets, box_ok, parent_ids, parent_ok)
    positive = torch.zeros_like(existence)
    positive[pi] = 1
    terms = [binary(existence, positive)]
    if box_ok[ti].any():
        mask = box_ok[ti]
        terms.append(localization(output_box[pi[mask]], targets[ti[mask]]))
    if parent_ok[ti].any():
        mask = parent_ok[ti]
        indices, matched = pi[mask], parent_ids[ti[mask]]
        terms.append(F.cross_entropy(parent_logits[indices].float(), matched))
        parent = out["cell_boxes"][i, matched].float()
        region = output_box[indices].float()
        terms.append((F.relu(parent[:, :2] - region[:, :2]) + F.relu(region[:, 2:] - parent[:, 2:])).mean())
    for p, t in zip(pi.tolist(), ti.tolist(), strict=True):
        child = children[t]
        if child.get("cells") is not None:
            child_terms, _, _ = set_losses(
                out["child_cell_logits"][i, p], out["child_boxes"][i, p], out["child_type_logits"][i, p],
                child["cells"], out["child_relation_logits"][i, p], child.get("relations"),
            )
            terms.append(sum(child_terms.values()))
    return torch.stack(terms).mean()


def supervised_anomaly(out, batch, i, labels):
    terms = []
    for key, field in (("anomaly", "anomaly_logits"), ("confidence", "confidence_logits")):
        if labels.get(key) is not None:
            terms.append(binary(out[field][i], labels[key]))
    cells = labels.get("cell_anomaly")
    if cells is not None:
        cells = cells[:batch["cell_mask"].shape[1]]
        expected = torch.as_tensor([v if v is not None else -1 for v in cells], device=out["anomaly_logits"].device)
        valid = (expected >= 0) & batch["cell_mask"][i, :len(expected)]
        if valid.any():
            terms.append(binary(out["cell_anomaly_logits"][i, :len(expected)][valid], expected[valid]))
    entries = labels.get("relation_anomaly")
    if entries:
        for a, b, value in entries:
            if 0 <= min(a, b) <= max(a, b) < batch["cell_mask"].shape[1] and batch["cell_mask"][i, a] and batch["cell_mask"][i, b]:
                terms.append(binary(out["relation_anomaly_logits"][i, a, b], value))
    regions = labels.get("missing_regions")
    if regions is not None:
        xy = out["visual_xy"].float()
        target = torch.zeros(len(xy), device=xy.device, dtype=torch.bool)
        for region in regions:
            box = xy.new_tensor(checked_box(region))
            target |= ((xy >= box[:2]) & (xy <= box[2:])).all(-1)
        terms.append(binary(out["missing_region_logits"][i], target))
    return terms


def compute_losses(out, batch, config):
    terms = defaultdict(list)
    zero = out["anomaly_logits"].float().sum() * 0
    for i, labels in enumerate(batch["targets"]):
        terms["anomaly"].extend(supervised_anomaly(out, batch, i, labels))
        for item in labels.get("alignment") or ():
            index = item["cell_index"]
            if not 0 <= index < batch["cell_mask"].shape[1] or not batch["cell_mask"][i, index]:
                continue
            visual = F.normalize(out["features"]["visual"][i].float(), dim=-1)
            cell = F.normalize(out["features"]["cells"][i, index].float(), dim=-1)
            xy = out["visual_xy"].float()
            expected = (xy - xy.new_tensor(item["point"])).square().sum(-1).argmin().reshape(1)
            terms["alignment"].append(F.cross_entropy((visual @ cell)[None] / 0.1, expected))
        if "cell_boxes" not in out:
            continue
        cells = labels.get("cells")
        match, annotated_boxes = {}, None
        if cells is not None:
            losses, match, annotated_boxes = set_losses(
                out["cell_existence_logits"][i], out["cell_boxes"][i], out["cell_type_logits"][i],
                cells, out["relation_logits"][i], labels.get("relations"),
            )
            for name, loss in losses.items():
                terms[name].append(loss)
        proposal = {key: out[key][i:i + 1] for key in ("cell_boxes", "cell_existence_logits", "relation_logits")}
        constraints = grid_constraints(proposal, annotated_boxes[None] if annotated_boxes is not None and len(annotated_boxes) else None)
        terms["constraint"].append(sum(constraints.values()))
        children = labels.get("nested")
        if "nested_logits" in out and children is not None:
            terms["hierarchy"].append(hierarchy_loss(out, i, children, match))
        repair = labels.get("repair_tokens")
        if repair is not None:
            logits = out["repair_logits"][i]
            target = torch.tensor([v if v is not None else -1 for v in repair[:len(logits)]], device=logits.device, dtype=torch.long)
            mask = (target >= 0) & batch["token_mask"][i, :len(target)]
            if mask.any():
                terms["repair"].append(F.cross_entropy(logits[:len(target)][mask].float(), target[mask]))
    losses = {name: torch.stack(terms[name]).mean() if terms[name] else zero for name in config.loss_weights}
    losses["total"] = sum(config.loss_weights[name] * losses[name] for name in config.loss_weights)
    return losses
