"""Rank adjacent-page tables for a human reviewer while retaining page-level tables."""

from difflib import SequenceMatcher
import re


def candidate(first, second, grid_a, grid_b):
    if (
        first["document_sha256"] != second["document_sha256"]
        or second["page_number"] != first["page_number"] + 1
    ):
        return None
    a = first["bbox_px"]
    b = second["bbox_px"]
    wa, ha = first["page_size_px"]
    wb, hb = second["page_size_px"]
    head = lambda g: " ".join(c.text for c in g.cells if c.row == 0)
    header = SequenceMatcher(None, head(grid_a), head(grid_b), autojunk=False).ratio()
    same = grid_a.columns == grid_b.columns
    alignment = max(0, 1 - abs(a[0] / wa - b[0] / wb) - abs(a[2] / wa - b[2] / wb))
    boundary = a[3] / ha > 0.72 and b[1] / hb < 0.28
    continuation = bool(re.search(r"续表|接上页|续页", second.get("caption", "")))
    if not (same and alignment > 0.85 and (boundary or continuation)):
        return None
    evidence = dict(
        same_column_count=same,
        normalized_edge_alignment=alignment,
        header_similarity=header,
        near_page_boundary=boundary,
        continuation_caption=continuation,
    )
    return dict(
        cross_page_link=[first["table_id"], second["table_id"]],
        link_confidence=round(
            0.30 * same
            + 0.25 * alignment
            + 0.15 * header
            + 0.15 * boundary
            + 0.15 * continuation,
            3,
        ),
        link_evidence=evidence,
        review_status="pending",
    )
