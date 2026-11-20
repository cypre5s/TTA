import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts import ground_truth as gt
from scripts.evaluate import score
from src.cross_page.candidates import candidate
from src.otsl.grid import parse


class GridTests(unittest.TestCase):
    def test_two_dimensional_merge(self):
        g = parse("<fcel>A<lcel><fcel>B<nl><ucel><xcel><ecel><nl>")
        self.assertEqual((g.rows, g.columns), (2, 3))
        self.assertEqual((g.cells[0].rowspan, g.cells[0].colspan), (2, 2))
        self.assertIn('rowspan="2" colspan="2"', g.html())

    def test_rejects_broken_spans_and_keeps_literal_text_safe(self):
        for raw in (
            "<lcel><nl>",
            "<fcel>A<nl><fcel>B<ecel><nl>",
            "<fcel>A<lcel><nl><ucel><fcel>B<nl>",
        ):
            with self.assertRaises(ValueError):
                parse(raw)
        self.assertIn("&lt;script&gt;", parse("<fcel><script><nl>").html())

    def test_metrics_separate_geometry_from_text(self):
        same = "<fcel>甲<lcel><nl>"
        other = "<fcel>乙<lcel><nl>"
        d = score(other, same)
        self.assertEqual(d["teds_s"], 1)
        self.assertLess(d["teds"], 1)
        self.assertEqual(d["char_edits"], 1)
        self.assertEqual(score("", same)["matched_cells"], 0)
        self.assertEqual(score(same, same)["exact"], 1)

    def test_adjacent_pages_require_more_evidence(self):
        a = dict(
            document_sha256="k",
            page_number=1,
            page_size_px=[100, 100],
            bbox_px=[10, 5, 80, 30],
            table_id="a",
        )
        b = dict(a, page_number=2, bbox_px=[10, 40, 80, 60], table_id="b")
        g = parse("<fcel>A<nl>")
        self.assertIsNone(candidate(a, b, g, g))
        a["bbox_px"][-1] = 96
        b["bbox_px"][1] = 4
        self.assertEqual(candidate(a, b, g, g)["review_status"], "pending")


class HumanReviewTests(unittest.TestCase):
    def test_human_gate_revision_and_project_split(self):
        wd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                Path("database").mkdir()
                db = sqlite3.connect("database/pilot.sqlite")
                db.execute("create table sources(sha256,site,project_id,project)")
                inbox = Path("data/annotation/packets")
                inbox.mkdir(parents=True)
                records = []
                for i in range(6):
                    tid = f"table{i}"
                    doc = f"doc{i}"
                    group = i // 2
                    db.execute(
                        "insert into sources values(?,?,?,?)",
                        (doc, "site", str(group), "name" + str(group)),
                    )
                    packet = inbox / (tid + ".json")
                    packet.write_bytes(
                        gt.jbytes(
                            dict(table_id=tid, document_sha256=doc, image_sha256="hash")
                        )
                    )
                    records.append(
                        dict(
                            table_id=tid,
                            reviewer_type="human",
                            reviewer="fixture",
                            reviewed_at="2020-01-01T00:00:00Z",
                            packet_sha256=gt.sha(packet.read_bytes()),
                            decision="approve",
                            corrected_otsl="<fcel>A<nl>",
                        )
                    )
                db.commit()
                db.close()
                path = Path("decisions.jsonl")
                path.write_bytes(gt.jbytes(dict(records[0], reviewer_type="model")))
                with self.assertRaises(ValueError):
                    gt.ingest(path)
                path.write_bytes(b"".join(gt.jbytes(row) for row in records))
                gt.ingest(path)
                gt.split("fixture", 42)
                parts = []
                for name in ("train", "validation", "test"):
                    rows = [
                        json.loads(s)
                        for s in (Path("data/splits/fixture") / (name + ".jsonl"))
                        .read_text()
                        .splitlines()
                    ]
                    parts.append({r["group_id"] for r in rows})
                    self.assertEqual(len(rows), 2)
                self.assertEqual(len(set.union(*parts)), 3)
                path.write_bytes(
                    gt.jbytes(dict(records[0], corrected_otsl="<fcel>changed<nl>"))
                )
                with self.assertRaises(ValueError):
                    gt.ingest(path)
            finally:
                os.chdir(wd)


if __name__ == "__main__":
    unittest.main()
