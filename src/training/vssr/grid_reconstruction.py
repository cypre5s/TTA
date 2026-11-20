"""Map decoded boxes and relations to a grid without generating text."""

from dataclasses import dataclass, field

import torch

from .otsl_serializer import CellNode, TableGraph


@dataclass
class Reconstruction:
    graph: TableGraph | None
    ocr_requests: list[dict] = field(default_factory=list)
    unmatched_ocr: list[dict] = field(default_factory=list)
    uncertain_cells: list[dict] = field(default_factory=list)
    reason: str | None = None


def boundaries(boxes, axis, relation, epsilon):
    coordinates = sorted((b[axis + 2 * edge], i, edge) for i, b in enumerate(boxes) for edge in (0, 1))
    clusters = []
    for x, i, edge in coordinates:
        current = clusters[-1] if clusters else []
        near = current and x - current[0][0] <= epsilon
        related = current and x - current[0][0] <= 2 * epsilon and any(
            edge == e and relation is not None and relation[i, j] > 0.5 for _, j, e in current
        )
        if (near or related) and all(j != i for _, j, _ in current):
            current.append((x, i, edge))
        else:
            clusters.append([(x, i, edge)])
    indices = {(i, edge): k for k, cluster in enumerate(clusters) for _, i, edge in cluster}
    limits = [sum(v for v, _, _ in c) / len(c) for c in clusters]
    return limits, indices


def add_text(nodes, fragments, table_id):
    unmatched = []
    for i, fragment in enumerate(fragments):
        if not fragment["text"]:
            continue
        bbox = fragment.get("bbox")
        candidates = []
        if bbox is not None:
            area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1e-8)
            for j, node in enumerate(nodes):
                a = node.bbox
                intersection = max(0, min(a[2], bbox[2]) - max(a[0], bbox[0])) * max(
                    0, min(a[3], bbox[3]) - max(a[1], bbox[1])
                )
                if intersection / area >= 0.5:
                    candidates.append((intersection / area, -j, node))
        elif fragment.get("grid_anchor") is not None:
            for j, node in enumerate(nodes):
                if (node.row, node.column, node.rowspan, node.colspan) == tuple(fragment["grid_anchor"]):
                    candidates.append((1.0, -j, node))
        if not candidates:
            unmatched.append(fragment)
            continue
        node = max(candidates, key=lambda item: item[:2])[2]
        node.ocr_fragments.append(i)
    requests = []
    for node in nodes:
        if node.ocr_fragments:
            node.text = "\n".join(fragments[i]["text"] for i in node.ocr_fragments)
        elif node.query_id is None or node.cell_type != 0:
            requests.append(dict(table_id=table_id, cell_id=node.cell_id, bbox=node.bbox, task="supplemental_ocr"))
    return requests, unmatched


@torch.no_grad()
def reconstruct(table_id, boxes, existence_logits, cell_types, relations=None, fragments=(), threshold=0.5, epsilon=0.015):
    keep = (existence_logits.float().sigmoid() >= threshold).nonzero(as_tuple=True)[0]
    coords = boxes[keep].float().cpu().tolist()
    query_ids = keep.cpu().tolist()
    if not coords:
        return Reconstruction(None, unmatched_ocr=list(fragments), reason="no confident cell proposals")
    if any(b[2] <= b[0] or b[3] <= b[1] for b in coords):
        return Reconstruction(None, unmatched_ocr=list(fragments), uncertain_cells=[dict(query_id=q, bbox=b) for q, b in zip(query_ids, coords)],
                              reason="degenerate proposal geometry")
    relation = relations[keep][:, keep].float().sigmoid().cpu() if relations is not None else None
    ys, y_map = boundaries(coords, 1, relation[..., 0] if relation is not None else None, epsilon)
    xs, x_map = boundaries(coords, 0, relation[..., 1] if relation is not None else None, epsilon)
    types = cell_types[keep].argmax(-1).cpu().tolist()
    rows, columns = len(ys) - 1, len(xs) - 1
    nodes, coverage = [], set()
    overlapping = False
    for i, (query, box, typ) in enumerate(zip(query_ids, coords, types, strict=True)):
        a, b = y_map[i, 0], y_map[i, 1]
        c, d = x_map[i, 0], x_map[i, 1]
        positions = {(r, col) for r in range(a, b) for col in range(c, d)}
        overlapping |= bool(coverage & positions)
        coverage |= positions
        nodes.append(CellNode(f"{table_id}:q{query}", a, c, b - a, d - c,
                              cell_type=typ, bbox=tuple(box), query_id=query))
    if overlapping:
        return Reconstruction(None, unmatched_ocr=list(fragments), uncertain_cells=[dict(query_id=q, bbox=b) for q, b in zip(query_ids, coords)],
                              reason="overlapping cell proposals need review")
    for r in range(rows):
        for c in range(columns):
            if (r, c) not in coverage:
                nodes.append(CellNode(f"{table_id}:missing:{r}:{c}", r, c,
                                      bbox=(xs[c], ys[r], xs[c + 1], ys[r + 1])))
    requests, unmatched = add_text(nodes, fragments, table_id)
    graph = TableGraph(table_id, rows, columns, sorted(nodes, key=lambda node: (node.row, node.column)),
                       bbox=(xs[0], ys[0], xs[-1], ys[-1]))
    return Reconstruction(graph, requests, unmatched)
