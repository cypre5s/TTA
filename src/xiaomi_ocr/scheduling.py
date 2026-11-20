"""Estimate decoding length solely to group work, preserving pixels and prompts."""

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=4096)
def estimate(table_id):
    crop = json.loads((Path("data/tables/crops") / (table_id + ".json")).read_text())
    base = Path("data/rendered") / crop["document_sha256"]
    page = json.loads((base / (crop["page_id"] + ".json")).read_text())
    layer = json.loads(Path(page["text_layer"]).read_text())
    x0, y0, x1, y1 = crop["bbox_px"]
    scale = page["dpi"] / 72
    a, b, c, d, e, f = layer["rotation_matrix"]
    chars = 0
    for left, top, right, bottom, word, *_ in layer["words"]:
        x = (left + right) / 2
        y = (top + bottom) / 2
        u = (a * x + c * y + e) * scale
        v = (b * x + d * y + f) * scale
        if x0 <= u <= x1 and y0 <= v <= y1:
            chars += len(word)
    area = (x1 - x0) * (y1 - y0) / (scale**2)
    return chars + 0.0008 * area


def order(paths, size):
    # Bound the pool so new, short requests cannot perpetually displace early long ones.
    window = sorted(paths, key=lambda p: (p.stat().st_mtime_ns, p.name))[: size * 8]
    target = min(window, key=lambda p: (p.stat().st_mtime_ns, p.name))
    length = estimate(target.stem)
    return sorted(window, key=lambda p: (abs(estimate(p.stem) - length), p.name))
