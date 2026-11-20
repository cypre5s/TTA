from dataclasses import replace
import unittest

import torch

from src.training.vssr import VSSRConfig
from src.training.vssr.backbone import Backbone
from src.training.vssr.cell_proposal import hungarian
from src.training.vssr.cross_modal_fusion import CrossModalFusion
from src.training.vssr.dataset import TableSample, collate
from src.training.vssr.fpn import FPN
from src.training.vssr.geometry_encoder import GeometryEncoder
from src.training.vssr.relation_decoder import grid_edges
from src.training.vssr.structure_encoder import StructureEncoder
from src.otsl.grid import parse


def small_config():
    return VSSRConfig(backbone="resnet18", image_size=64, hidden=32, heads=4,
                      structure_ffn=64, visual_pools=(4, 3, 2, 1), cell_queries=12,
                      nested_queries=3, child_queries=6)


class ModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_backbone_strides_and_final_stage_freeze(self):
        cfg = replace(small_config(), unfreeze_last_stage=True)
        model = Backbone(cfg)
        model.train()
        features = model(torch.randn(2, 3, 64, 64))
        self.assertEqual([x.shape[-1] for x in features], [16, 8, 4, 2])
        self.assertEqual(model.reductions, (4, 8, 16, 32))
        trainable = [k for k, v in model.encoder.named_parameters() if v.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(k.startswith(model.last_stage_name + ".") for k in trainable))
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.encoder.get_submodule(model.last_stage_name).training)

    def test_fpn_keeps_all_scales(self):
        maps = [torch.randn(2, c, s, s) for c, s in [(8, 16), (12, 8), (16, 4), (24, 2)]]
        output = FPN((8, 12, 16, 24), 32)(maps)
        self.assertEqual([tuple(p.shape) for p in output], [(2, 32, s, s) for s in (16, 8, 4, 2)])

    def test_structure_handles_entirely_invalid_batch(self):
        cfg = small_config()
        batch = collate([TableSample("bad", torch.zeros(3, 64, 64), "<xcel><nl>")], cfg)
        tokens, cells = StructureEncoder(cfg)(batch)
        self.assertEqual((tokens.shape[-1], cells.shape[-1]), (32, 32))
        self.assertTrue(batch["parser_error"][0])
        self.assertFalse(batch["cell_mask"].any())
        self.assertTrue(torch.isfinite(tokens).all())

    def test_geometry_validity_and_source(self):
        enc = GeometryEncoder(32)
        source = torch.tensor([[0, 1, 2]])
        valid = torch.tensor([[False, True, True]])
        boxes = torch.tensor([[[float("nan")] * 4, [.1, .2, .6, .8], [.1, .2, .6, .8]]])
        a = enc(boxes, valid, source)
        boxes[0, 0] = 10
        b = enc(boxes, valid, source)
        self.assertTrue(torch.equal(a[0, 0], b[0, 0]))
        self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(torch.equal(a[0, 1], a[0, 2]))

    def test_fusion_changes_visual_and_structure_branches(self):
        fusion = CrossModalFusion(small_config()).eval()
        v, t, c = torch.randn(1, 6, 32), torch.randn(1, 3, 32), torch.randn(1, 2, 32)
        masks = (torch.ones(1, 3, dtype=torch.bool), torch.tensor([[True, False]]))
        one = fusion(v, t, c, torch.zeros_like(c), *masks)
        two = fusion(v, t + torch.randn_like(t), c, torch.zeros_like(c), *masks)
        three = fusion(v + torch.randn_like(v), t, c, torch.zeros_like(c), *masks)
        self.assertFalse(torch.equal(one["visual"], two["visual"]))
        self.assertFalse(torch.equal(one["tokens"], three["tokens"]))
        self.assertTrue(torch.equal(one["cells"][0, 1], torch.zeros(32)))

    def test_hungarian_ignores_unknown_geometry(self):
        boxes = torch.tensor([[.1, .1, .4, .4], [.6, .6, .9, .9]])
        ids, matches = hungarian(torch.zeros(2), boxes, None, torch.stack((torch.full((4,), float("nan")), boxes[0])),
                                 torch.tensor([False, True]))
        self.assertEqual(dict(zip(matches.tolist(), ids.tolist())), {0: 1, 1: 0})

    def test_typed_relation_edges_and_diagonal(self):
        cells = parse("<fcel>A<lcel><nl><fcel>B<fcel>C<nl>").cells
        edges = grid_edges(cells)
        self.assertTrue(edges[1, 2, 0])
        self.assertTrue(edges[0, 1, 3])
        self.assertTrue(edges[0, 1, 2])
        self.assertFalse(edges[0, 0].any())
