"""CPU regression checks for Hydra's EMA target, without dataset dependencies.

Load the actual loss definitions via AST so FFCV/torchvision are not needed.
Run: python tests/test_hydra_teacher.py
"""
import ast
import copy
import math
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fused_gated_contrast import fused_gated_contrast

names = {'Hydra', '_all_gather', '_sample_rows', '_label_partner',
         '_hydra_monitor', '_label_monitor'}
source = ast.parse((ROOT / 'main_ibot.py').read_text())
namespace = dict(torch=torch, nn=nn, F=F, dist=dist, copy=copy,
                 utils=types.SimpleNamespace(get_rank=lambda: 0),
                 fused_gated_contrast=fused_gated_contrast)
exec(compile(ast.Module(body=[n for n in source.body if getattr(n, 'name', None) in names],
                        type_ignores=[]), str(ROOT / 'main_ibot.py'), 'exec'), namespace)
Hydra = namespace['Hydra']


class TeacherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = Hydra(6, 8, 16, 0.2, 0.1, 1, 'sim')

    def test_gradients_bn_and_optimizer(self):
        m = self.model
        student = torch.randn(16, 8, requires_grad=True)
        teacher = torch.randn(16, 8, requires_grad=True)
        guide = torch.randn(8, 6, requires_grad=True)
        self.assertFalse(any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                             for branch in (m.projector, m.predictor) for layer in branch.modules()))
        self.assertIsInstance(m.key_bn, nn.BatchNorm1d)
        before = {k: p.clone() for k, p in m.teacher_projector.named_parameters()}
        buffers = m.key_bn.running_mean.clone()
        loss = m(student, teacher, guide)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(guide.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)
        for module in (m.projector, m.predictor, m.gate):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in m.teacher_projector.parameters()))
        self.assertNotIn('ins', m.monitor)
        self.assertFalse(torch.equal(buffers, m.key_bn.running_mean))
        with torch.no_grad():
            keys = m.key_bn(m.teacher_projector(teacher))
        torch.testing.assert_close(keys.mean(0), torch.zeros(8), atol=1e-5, rtol=0)
        self.assertTrue((keys.std(0, unbiased=False) > 0.9).all())
        torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.1).step()
        for k, p in m.teacher_projector.named_parameters():
            torch.testing.assert_close(p, before[k], rtol=0, atol=0)
        bn_before = m.key_bn.running_mean.clone()
        expected = [t * .9 + s * .1 for s, t in zip(m.projector.parameters(), m.teacher_projector.parameters())]
        m.update_teacher(.9)
        for p, value in zip(m.teacher_projector.parameters(), expected):
            torch.testing.assert_close(p, value)
        torch.testing.assert_close(m.key_bn.running_mean, bn_before)

    def test_teacher_similarity_filter_and_gate_metrics(self):
        m = Hydra(6, 8, 16, .2, .1, 4, 'sim', out_dim=12, filter_negatives=True)
        student = torch.randn(16, 8, requires_grad=True)
        teacher = torch.randn(16, 8)
        guide = torch.randn(8, 6)
        loss = m(student, teacher, guide)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(m.monitor['neg_filtered'], 0)
        self.assertLessEqual(m.monitor['neg_filtered'], 1)
        self.assertAlmostEqual(m.monitor['act'], m.monitor['gate_mean'] * 12, places=5)
        loss.backward()
        self.assertGreater(student.grad.abs().sum().item(), 0)


    def test_checkpoint_and_legacy_rejection(self):
        m = self.model
        with torch.no_grad():
            m.projector[0].weight.add_(.5)
        saved = copy.deepcopy(m.state_dict())
        new = Hydra(6, 8, 16, .2, .1, 1, 'sim')
        new.load_state_dict(saved, strict=True)
        for k, v in saved.items():
            torch.testing.assert_close(new.state_dict()[k], v)
        legacy = {k: v for k, v in saved.items() if not k.startswith('teacher_projector.')}
        with self.assertRaisesRegex(RuntimeError, "start this architecture from scratch"):
            new.load_state_dict(legacy, strict=False)

    def test_out_dim_is_independent_from_backbone_dim(self):
        m = Hydra(6, 8, 16, .2, .1, 1, 'sim', out_dim=12)
        h = torch.randn(16, 8, requires_grad=True)
        teacher = torch.randn(16, 8)
        guide = torch.randn(8, 6)
        self.assertEqual(m.projector(h).shape, (16, 12))
        self.assertEqual(m.predictor(m.projector(h)).shape, (16, 12))
        self.assertEqual(m.teacher_projector(teacher).shape, (16, 12))
        self.assertEqual(m.gate(guide).shape, (8, 12))
        self.assertEqual(m.key_bn.num_features, 12)
        loss = m(h, teacher, guide)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(h.grad.abs().sum().item(), 0)



class DiagnosticTests(unittest.TestCase):
    def test_baseline_matches_and_exclusion_keeps_positive(self):
        source = ast.parse((ROOT / 'smoke_hydra.py').read_text())
        namespace['Hydra'] = Hydra
        namespace['_all_gather'] = namespace['_all_gather']
        exec(compile(ast.Module(body=[n for n in source.body if getattr(n, 'name', None) == 'DiagnosticHydra'],
                               type_ignores=[]), 'smoke_hydra.py', 'exec'), namespace)
        diagnostic = namespace['DiagnosticHydra'](6, 8, 16, .2, .1, 1, 'sim')
        baseline = Hydra(6, 8, 16, .2, .1, 1, 'sim')
        # query_bn belongs only to the diagnostic variant.
        baseline.load_state_dict({name: value for name, value in diagnostic.state_dict().items()
                                  if not name.startswith('query_bn.')})
        qs = torch.randn(16, 8, requires_grad=True)
        kt = torch.randn(16, 8, requires_grad=True)
        refs = torch.randn(8, 6)
        torch.manual_seed(19)
        expected = baseline(qs, kt, refs)
        torch.manual_seed(19)
        actual = diagnostic(qs, kt, refs)
        torch.testing.assert_close(actual, expected)
        diagnostic.exclude_similar = True
        torch.manual_seed(19)
        loss = diagnostic(qs, kt, refs)
        self.assertTrue(torch.isfinite(loss))
        _, _, _, excluded, target = diagnostic.probe_state
        self.assertTrue(excluded.diagonal().all())
        self.assertFalse(excluded[torch.arange(8), target].any())
        self.assertLessEqual(diagnostic.monitor['candidates'], 7)
        loss.backward()
        self.assertIsNone(kt.grad)
        self.assertGreater(qs.grad.abs().sum().item(), 0)

    def test_multi_k_matches_and_shared_pairs_are_reused(self):
        source = ast.parse((ROOT / 'smoke_hydra.py').read_text())
        namespace['Hydra'] = Hydra
        exec(compile(ast.Module(body=[n for n in source.body if getattr(n, 'name', None) == 'DiagnosticHydra'],
                               type_ignores=[]), 'smoke_hydra.py', 'exec'), namespace)
        torch.manual_seed(71)
        diagnostic = namespace['DiagnosticHydra'](6, 8, 16, .2, .1, 4, 'sim')
        baseline = Hydra(6, 8, 16, .2, .1, 4, 'sim')
        # query_bn belongs only to the diagnostic variant.
        baseline.load_state_dict({name: value for name, value in diagnostic.state_dict().items()
                                  if not name.startswith('query_bn.')})
        qs = torch.randn(16, 8, requires_grad=True)
        kt = torch.randn(16, 8, requires_grad=True)
        refs = torch.randn(8, 6)
        torch.manual_seed(10)
        expected = baseline(qs, kt, refs)
        torch.manual_seed(10)
        actual = diagnostic(qs, kt, refs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(diagnostic.direction_partners[0].shape, (4, 8))
        diagnostic.share_partners = True
        loss = diagnostic(qs, kt, refs)
        self.assertTrue(torch.equal(*diagnostic.direction_partners))
        self.assertEqual(diagnostic.monitor['direction_partner_agreement'], 1.)
        loss.backward()
        self.assertIsNone(kt.grad)
        self.assertTrue(torch.isfinite(qs.grad).all())
        self.assertGreater(qs.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
