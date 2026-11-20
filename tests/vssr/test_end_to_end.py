from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from PIL import Image

from scripts.train_vssr import loader, train_generation
from scripts.vssr_smoke import fixture, reconstruction_fixture
from src.training.vssr import VSSR1, VSSR3, upgrade
from src.training.vssr.checkpoint import load_checkpoint, save_checkpoint
from src.training.vssr.config import config_digest
from src.training.vssr.dataset import TableSample, VSSRDataset, collate, reviewed_cell_targets
from src.training.vssr.grid_reconstruction import reconstruct
from src.training.vssr.losses import compute_losses
from src.training.vssr.otsl_serializer import serialize_hierarchy
from src.training.vssr.repair_pipeline import nested_graphs
from src.training.vssr.trainer import batch_losses, train_step
from src.otsl.grid import parse
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
    def test_nested_grandchild_coordinates_and_ocr_ownership(self):
        def proposals(region):
            return dict(nested_probability=torch.ones(1, 1), parent_cell_id=torch.zeros(1, 1, dtype=torch.long),
                        child_table_region=torch.tensor([[region]]), child_boxes=torch.tensor([[[[0., 0., 1., 1.]]]]),
                        child_cell_logits=torch.ones(1, 1, 1) * 9, child_type_logits=torch.ones(1, 1, 1, 4),
                        child_relation_logits=torch.zeros(1, 1, 1, 1, 5), child_embeddings=torch.zeros(1, 1, 1, 32))

        fragments = [dict(text="root", bbox=(.01, .01, .05, .05)),
                     dict(text="child", bbox=(.15, .15, .19, .19)),
                     dict(text="grandchild", bbox=(.43, .43, .46, .46))]
        graph = reconstruct("root", torch.tensor([[0., 0., 1., 1.]]),
                            torch.tensor([9.]), torch.ones(1, 4), fragments=fragments).graph

        def descend(visual, parents):
            self.assertEqual(tuple(parents.shape), (1, 1, 32))
            return proposals((.4, .4, .7, .7))

        _, warnings, used = nested_graphs(proposals((.1, .1, .9, .9)), 0, graph, fragments, .5,
                                          next_level=descend, visual=torch.zeros(1, 3, 32))
        result = serialize_hierarchy(graph)
        self.assertFalse(warnings)
        self.assertEqual(result["nesting_depth"], 2)
        self.assertEqual(used, {1, 2})
        child, grandchild = result["child_tables"]
        self.assertEqual(result["relations"][1]["parent_table_id"], child["table_id"])
        self.assertAlmostEqual(grandchild["bbox"][0], .42, places=6)
        self.assertEqual([graph.cells[0].text, child["cells"][0]["text"], grandchild["cells"][0]["text"]],
                         ["root", "child", "grandchild"])

    def test_grandchild_supervision_runs_only_for_known_branches(self):
        cfg = replace(small_config(), max_nesting_depth=2)
        sample, _ = fixture(cfg)
        child = sample.supervision["nested"][0]
        child["nested"] = [dict(parent=0, bbox=(.03, .04, .25, .26), cells=[{"bbox": (.1, .1, .8, .8)}])]
        calls = []
        model = VSSR3(cfg).eval()
        handle = model.hierarchy.decode.register_forward_hook(lambda _, args, output: calls.append(len(args[0])))
        _, losses = batch_losses(model, collate([sample], cfg), bf16=False)
        self.assertEqual(calls, [1, 1])
        losses["hierarchy"].backward()
        self.assertTrue(torch.isfinite(model.hierarchy.child_box.weight.grad).all())
        del child["nested"]
        calls.clear()
        batch_losses(model, collate([sample], cfg))
        self.assertEqual(calls, [1])
        handle.remove()

    def test_reviewed_tree_uses_the_parent_table_coordinate_frame(self):
        def table(text, box, children):
            return dict(otsl=f"<fcel>{text}<nl>", bbox_pt=box, cells=[
                dict(row=0, column=0, rowspan=1, colspan=1, bbox_pt=box, children=children),
            ])

        tree = table("root", (0, 0, 100, 100), [
            table("child", (10, 20, 90, 100), [table("grandchild", (30, 40, 70, 80), [])]),
        ])
        targets = reviewed_cell_targets(parse(tree["otsl"]), tree, lambda box: tuple(v / 100 for v in box))
        child = targets["nested"][0]
        self.assertEqual(child["bbox"], (.1, .2, .9, 1))
        self.assertEqual(child["nested"][0]["bbox"], (.25, .25, .75, .75))
        self.assertEqual(child["nested"][0]["cells"][0]["bbox"], (0, 0, 1, 1))

    def test_old_checkpoints_preserve_depth_one(self):
        cfg = small_config()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            save_checkpoint(path, VSSR3(cfg))
            payload = torch.load(path, weights_only=True)
            del payload["config"]["max_nesting_depth"]
            payload["config_sha256"] = config_digest(payload["config"])
            torch.save(payload, path)
            model, _ = load_checkpoint(path, expected_config=cfg)
            self.assertEqual(model.config.max_nesting_depth, 1)

    def test_batched_finite_checks_retain_losses_and_optimizer_update(self):
        cfg = replace(small_config(), dropout=0.)
        sample, _ = fixture(cfg)
        batch = collate([sample], cfg)
        before = VSSR3(cfg)
        after = deepcopy(before)
        reference = torch.optim.AdamW(before.parameters(), lr=1e-3)
        optimized = torch.optim.AdamW(after.parameters(), lr=1e-3)
        losses = compute_losses(before(batch), batch, cfg)
        losses["total"].backward()
        reference.step()
        actual = train_step(after, batch, optimized)
        self.assertEqual(actual, {name: float(value.detach()) for name, value in losses.items()})
        for lhs, rhs in zip(before.parameters(), after.parameters(), strict=True):
            self.assertTrue(torch.equal(lhs, rhs))

    def test_training_stops_and_restores_the_best_epoch(self):
        cfg = replace(small_config(), dropout=0.)
        sample, _ = fixture(cfg)
        device = torch.device("cpu")
        batches = loader([sample], cfg, 1, 0, device)
        model = VSSR1(cfg)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            unchanged, receipt = train_generation(
                model, batches, batches, output, device, epochs=4, patience=2, min_delta=1e-4, lr=0., bf16=False,
            )
            self.assertTrue(receipt["stopped_early"])
            self.assertEqual(receipt["epochs_run"], 2)
            self.assertEqual(receipt["best_checkpoint"], "epoch-000.pt")
            for lhs, rhs in zip(unchanged.parameters(), model.parameters(), strict=True):
                self.assertTrue(torch.equal(lhs, rhs))
