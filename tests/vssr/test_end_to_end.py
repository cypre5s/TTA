from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from PIL import Image

from scripts.vssr_smoke import fixture, reconstruction_fixture
from src.training.vssr import VSSR1, VSSR3, upgrade
from src.training.vssr.checkpoint import load_checkpoint, save_checkpoint
from src.training.vssr.dataset import TableSample, VSSRDataset, collate
from src.training.vssr.grid_reconstruction import reconstruct
from src.training.vssr.losses import compute_losses
from src.training.vssr.trainer import train_step
from test_modules import small_config
from scripts.ground_truth import jbytes, sha


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_forward_and_missing_supervision(self):
        cfg = small_config()
        sample, _ = fixture(cfg)
        sample.supervision = {"anomaly": 0.0}
        batch = collate([sample, TableSample("invalid", sample.image, "not otsl")], cfg)
        v1 = VSSR1(cfg).eval()
        v2 = upgrade(v1, "vssr-2")
        v3 = upgrade(v2, "vssr-3")
        with torch.inference_mode():
            out1, out2, out3 = v1(batch), v2(batch), v3(batch)
        self.assertTrue(torch.equal(out1["anomaly_logits"], out2["anomaly_logits"]))
        self.assertTrue(torch.equal(out2["cell_boxes"], out3["cell_boxes"]))
        self.assertEqual(out3["relation_logits"].shape, (2, 12, 12, 5))
        self.assertEqual(out3["parent_logits"].shape, (2, 3, 12))
        self.assertEqual(out3["child_boxes"].shape, (2, 3, 6, 4))
        self.assertEqual(out3["parser_error"].tolist(), [False, True])
        self.assertTrue(((out3["confidence"] >= 0) & (out3["confidence"] <= 1)).all())
        losses = compute_losses(out3, batch, cfg)
        self.assertGreater(float(losses["anomaly"]), 0)
        for name in ("cell", "bbox", "hierarchy", "alignment", "repair"):
            self.assertEqual(float(losses[name]), 0)

    def test_geometry_only_grid_does_not_hallucinate_text(self):
        cfg = small_config()
        sample, boxes = fixture(cfg)
        result = reconstruction_fixture(sample, boxes, True)
        self.assertEqual((result["rows"], result["columns"], result["rowspan"], result["colspan"]), (3, 3, 2, 2))
        self.assertEqual((result["child_tables"], result["depth"]), (2, 1))
        missing = reconstruct("gap", torch.tensor([[0., 0., .3, 1.], [.7, 0., 1., 1.]]),
                              torch.ones(2) * 9, torch.tensor([[0., 1., 0., 0.]]).repeat(2, 1))
        self.assertEqual(missing.graph.columns, 3)
        self.assertTrue(all(n.text == "" for n in missing.graph.cells))
        self.assertEqual(len(missing.ocr_requests), 3)
        overlap = reconstruct("overlap", torch.tensor([[0., 0., .6, .6], [.2, .2, 1., 1.]]),
                              torch.ones(2) * 9, torch.ones(2, 4))
        self.assertIsNone(overlap.graph)
        self.assertEqual(len(overlap.uncertain_cells), 2)

    def test_backward_and_checkpoint_round_trip(self):
        cfg = small_config()
        sample, _ = fixture(cfg)
        batch = collate([sample], cfg)
        model = VSSR3(cfg)
        before = model.repair.weight.detach().clone()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = train_step(model, batch, optim)
        self.assertGreater(losses["total"], 0)
        self.assertNotEqual(losses["hierarchy"], 0)
        self.assertFalse(torch.equal(before, model.repair.weight))
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.pt"
            save_checkpoint(p, model, optim, step=1)
            with self.assertRaises(FileExistsError):
                save_checkpoint(p, model)
            restored, meta = load_checkpoint(p, expected_config=cfg, expected_version="vssr-3")
            self.assertEqual((meta["step"], meta["config_sha256"]), (1, cfg.digest()))
            with torch.inference_mode():
                self.assertTrue(torch.equal(model.eval()(batch)["child_boxes"], restored(batch)["child_boxes"]))

    def test_jsonl_dataset_distinguishes_absent_and_empty_annotations(self):
        cfg = small_config()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            img, path = root / "crop.png", root / "samples.jsonl"
            Image.new("RGB", (32, 64), (12, 24, 36)).save(img)
            row = dict(table_id="fixture", image=img.name, image_sha256=sha(img.read_bytes()),
                       raw_otsl="<fcel>A<nl>", corrected_otsl="<fcel>B<nl>",
                       geometry=[None], supervision={"nested": []})
            path.write_bytes(jbytes(row) + jbytes(dict(row, table_id="unlabeled", corrected_otsl=None, supervision={})))
            data = VSSRDataset(path, cfg, root)
            batch = collate([data[0], data[1]], cfg)
            self.assertEqual(batch["images"].shape, (2, 3, 64, 64))
            self.assertFalse(batch["bbox_valid"].any())
            self.assertEqual(batch["targets"][0]["nested"], [])
            self.assertNotIn("nested", batch["targets"][1])
            self.assertEqual(batch["targets"][0]["anomaly"], 1)
            self.assertNotIn("anomaly", batch["targets"][1])
