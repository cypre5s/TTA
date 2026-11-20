import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import cross_page_study as study
from scripts.ground_truth import jbytes, sha


class CrossPageStudyTests(unittest.TestCase):
    def test_score_requires_bound_human_decisions_and_preserves_page_parts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = {}
            gt = []
            for tid, page, text in [("a", 1, "货物"), ("b", 2, "货物"), ("c", 2, "编号")]:
                meta = dict(
                    table_id=tid, document_sha256="doc", page_number=page,
                    bbox_px=[10, 64, 90, 98] if page == 1 else [10, 3, 90, 60],
                    page_size_px=[100, 100], image_sha256="image-" + tid,
                )
                tables[tid] = meta
                raw = f"<fcel>{text}<nl>"
                path = root / (tid + ".gt.json")
                body = jbytes(dict(corrected_otsl=raw))
                path.write_bytes(body)
                gt.append(dict(table_id=tid, gt=str(path), gt_sha256=sha(body)))
                (root / (tid + ".json")).write_bytes(jbytes(dict(meta, raw_otsl=raw)))

            _, selected = study.select_pairs(tables, 10)
            pairs = [dict(pair_id=k, tables=p) for k, p in selected]
            protocol = root / "study.json"
            protocol.write_bytes(jbytes(dict(
                cohort_sha256="review-manifest", threshold=study.THRESHOLD,
                candidate_code_sha256=sha(study.LINKER.read_bytes()),
                tables=tables, pairs=pairs, eligible_pairs=2,
            )))
            decisions = {p["pair_id"]: "same_table" if p["tables"][-1] == "b" else "different_tables" for p in pairs}
            approval = root / "decisions.json"
            record = dict(
                decisions=decisions, reviewer="fixture", reviewer_type="model",
                reviewed_at="2026-11-20T00:00:00Z", study_sha256=sha(protocol.read_bytes()),
            )
            approval.write_bytes(jbytes(record))
            with self.assertRaisesRegex(ValueError, "human review"):
                study.review(approval, protocol)

            approval.write_bytes(jbytes(dict(record, reviewer_type="human")))
            body = b"".join(jbytes(r) for r in gt)
            (root / "test.jsonl").write_bytes(body)
            (root / "manifest.json").write_bytes(jbytes(dict(
                review_manifest_sha256="review-manifest", sha256={"test": sha(body)},
                cross_page=dict(study_sha256=sha(protocol.read_bytes()), human_review_sha256=sha(approval.read_bytes())),
            )))
            output = root / "results"
            study.score(root, protocol, approval, root, output)
            result = json.loads((output / "cross_page_metrics.json").read_bytes())
            self.assertEqual((result["confirmed_links"], result["scored"], result["link_f1"]), (1, 2, 1))
            chain, = json.loads((output / "logical_tables.json").read_bytes())
            self.assertEqual([p["table_id"] for p in chain["pages"]], ["a", "b"])
            self.assertEqual(study.parse(chain["merged_otsl"]).rows, 2)


if __name__ == "__main__":
    unittest.main()
