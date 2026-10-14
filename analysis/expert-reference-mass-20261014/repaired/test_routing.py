"""Independent probability oracle and exact-wheel seam checks; no model weights."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

import routing
from patch_runtime import candidate_source, MEMBER, SOURCE_SHA, WHEEL_SHA


def oracle(logits, active=5):
    probabilities = logits.double().softmax(-1)
    mass, ids = probabilities.topk(10, dim=-1)
    return (mass[:, :active] / mass.sum(-1, keepdim=True)).float(), ids[:, :active]


class RoutingTests(unittest.TestCase):
    def test_probability_reference_across_shapes_dtypes_and_concentrations(self):
        torch.manual_seed(713)
        for n in (0, 1, 7, 14, 56, 129):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for scale in (0.1, 1, 20):
                    logits = (torch.randn(n, 512) * scale).to(dtype)
                    weights, ids = routing.route_reference_mass(hidden_states=torch.zeros(n, 1), gating_output=logits, topk=5, renormalize=True)
                    expected, eid = oracle(logits)
                    # Equal logits may select different tied experts; compare selected probabilities.
                    actual_prob = logits.double().softmax(-1).gather(1, ids.long())
                    expected_prob = logits.double().softmax(-1).gather(1, eid)
                    torch.testing.assert_close(actual_prob, expected_prob, rtol=1e-6, atol=1e-12)
                    torch.testing.assert_close(weights, expected, rtol=2e-6, atol=2e-7)
                    self.assertEqual(ids.dtype, torch.int32)
                    self.assertTrue(ids.is_contiguous() and weights.is_contiguous())

    def test_native_top10_normalization_is_independent_reference(self):
        torch.manual_seed(55)
        x = torch.randn(14, 512)
        w, ids = oracle(x, active=10)
        with zipfile.ZipFile(os.environ['REFERENCE_WHEEL']) as archive:
            tree = ast.parse(archive.read('sglang/srt/layers/moe/topk.py'))
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fused_topk_torch_native')
        ns = {'torch':torch,'F':torch.nn.functional,'_RENORMALIZE_SUM_EPSILON':1e-20}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'exact-wheel-native','exec'),ns)
        native, nid = ns[fn.name](torch.zeros(14,1),x,10,True)
        torch.testing.assert_close(w,native,rtol=2e-6,atol=2e-7)
        self.assertTrue(torch.equal(ids,nid))

    @unittest.skipUnless(torch.cuda.is_available(), 'local CPU gate; exercised on run GPU')
    def test_cuda_graph_replays_changed_logits(self):
        x = torch.randn(14,512,device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): routing.route_reference_mass(hidden_states=None,gating_output=x,topk=5,renormalize=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): w, ids = routing.route_reference_mass(hidden_states=None,gating_output=x,topk=5,renormalize=True)
        for _ in range(3):
            x.copy_(torch.randn_like(x)*4)
            graph.replay()
            expected, eid = oracle(x.cpu())
            torch.testing.assert_close(w.cpu(),expected,rtol=2e-6,atol=2e-7)
            self.assertTrue(torch.equal(ids.cpu().long(),eid))
        print("REFERENCE_GRAPH_TEST_PASSED", flush=True)

    def test_missing_experts_are_not_redistributed(self):
        logits = torch.zeros(1, 512)
        w, _ = routing.route_reference_mass(hidden_states=None, gating_output=logits, topk=5, renormalize=True)
        torch.testing.assert_close(w, torch.full((1, 5), 0.1))
        self.assertAlmostEqual(w.sum().item(), 0.5)

    def test_extreme_logits_remain_finite(self):
        x = torch.arange(512, dtype=torch.float32).reshape(1, -1) * 100
        w, ids = routing.route_reference_mass(hidden_states=None, gating_output=x, topk=5, renormalize=True)
        self.assertTrue(torch.isfinite(w).all())
        self.assertEqual(ids[0, 0].item(), 511)

    def test_draft_is_untouched_and_candidate_contract_is_checked(self):
        draft = SimpleNamespace(num_experts_per_tok=10)
        with patch.dict(os.environ, {"ARC3_REFERENCE_TOPK": "10"}):
            self.assertIsNone(routing.reference_routing_for(draft, True, 0))
            cfg = SimpleNamespace(num_experts=512, num_experts_per_tok=5, norm_topk_prob=True)
            self.assertIs(routing.reference_routing_for(cfg, False, 0), routing.route_reference_mass)
            cfg.num_experts_per_tok = 10
            with self.assertRaises(ValueError): routing.reference_routing_for(cfg, False, 0)
        with self.assertRaises(ValueError):
            routing.route_reference_mass(hidden_states=None, gating_output=torch.zeros(1,512), topk=10, renormalize=True)

    def test_exact_wheel_patch_and_drift_rejection(self):
        wheel = Path(os.environ["REFERENCE_WHEEL"])
        self.assertEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), WHEEL_SHA)
        with zipfile.ZipFile(wheel) as archive: original = archive.read(MEMBER)
        self.assertEqual(hashlib.sha256(original).hexdigest(), SOURCE_SHA)
        changed = candidate_source(original)
        compile(changed, MEMBER, "exec")
        tree = ast.parse(changed)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func,ast.Name) and n.func.id == 'reference_routing_for']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(a) for a in calls[0].args], ['config','is_nextn','layer_id'])
        with self.assertRaises(ValueError): candidate_source(original+b'\n')


if __name__ == "__main__": unittest.main(verbosity=2)
