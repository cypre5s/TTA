import unittest

import pymupdf as pdf

from scripts.baseline_score import page_blocks, source_geometry
from scripts.freeze_pilot_repairs import verify_tree
from scripts.repair_structures import build, candidates


class PilotRepairTests(unittest.TestCase):
    def test_two_disconnected_child_tables_at_same_x_are_not_one_table(self):
        horizontal = [(y, 0, 100) for y in (0, 55, 110)]
        vertical = [(x, 0, 110) for x in (0, 20, 100)]
        for top in (10, 70):
            horizontal.extend((y, 27, 92) for y in (top, top + 12, top + 25))
            vertical.extend((x, top, top + 25) for x in (27, 54, 92))
        boxes, h, v = candidates(horizontal, vertical)
        tree = build(boxes[0], boxes[1:], h, v, [[(32, 17, "甲")], [(60, 88, "乙")]])
        self.assertEqual((tree["rows"], tree["columns"]), (2, 2))
        self.assertEqual(verify_tree(tree), 2)
        self.assertEqual(source_geometry(tree)[1], 2)
        self.assertEqual([len(c["children"]) for c in tree["cells"]], [0, 1, 0, 1])
        self.assertEqual(tree["cells"][1]["children"][0]["cells"][0]["text"], "甲")
        self.assertEqual(tree["cells"][3]["children"][0]["cells"][-1]["text"], "乙")

    def test_page_extraction_retains_unterminated_final_row(self):
        raw = "附表\n<fcel>甲<nl>\n<fcel>乙\n\n说明\n\n<fcel>丙<nl>"
        self.assertEqual(page_blocks(raw), ["<fcel>甲<nl>\n<fcel>乙", "<fcel>丙<nl>"])

    def test_missing_border_rows_form_true_span(self):
        h = [(0, 0, 100), (20, 30, 100), (40, 0, 100)]
        v = [(x, 0, 40) for x in (0, 30, 100)]
        tree = build(pdf.Rect(0, 0, 100, 40), [], h, v, [])
        self.assertEqual(tree["cells"][0]["rowspan"], 2)
        self.assertEqual(verify_tree(tree), 0)


if __name__ == "__main__":
    unittest.main()
