"""Drafter expert ids against the pruned target's expert map, on the exact Kaggle SGLang wheel.

Run with a torch interpreter and the extracted wheel on PYTHONPATH (see draft-map-tests.log).
Probe 1 died here: IndexError at logical id 448 while loading the full-512 MTP drafter.
"""
import hashlib
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.eplb import expert_location as el

import pruning

WHEEL_SOURCE_SHA = os.environ['SGLANG_EXPERT_LOCATION_SHA256']
STOCK = el.ExpertLocationMetadata.logical_to_all_physical


def target_map(num_logical=448, num_layers=48, physical_to_logical=None):
    """Build the map the target publishes, with the wheel's own construction helpers."""
    if physical_to_logical is None:
        physical_to_logical = torch.arange(num_logical).repeat(num_layers, 1)
    # Probe server args leave both MoE dispatch settings at their defaults.
    probe_exec = SimpleNamespace(moe=SimpleNamespace(ep_dispatch_algorithm=None, moe_a2a_backend='none'))
    with mock.patch('sglang.srt.runtime_context.get_exec', return_value=probe_exec):
        logical_to_all = el._compute_logical_to_all_physical_map(
            physical_to_logical_map=physical_to_logical, num_logical_experts=num_logical, ep_size=1, moe_ep_rank=0)
        return el.ExpertLocationMetadata._init_raw(
            ep_size=1, physical_to_logical_map=physical_to_logical, logical_to_all_physical_map=logical_to_all, moe_ep_rank=0)


class DraftMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(el.__file__).read_bytes()
        assert hashlib.sha256(source).hexdigest() == WHEEL_SOURCE_SHA, 'not the Kaggle wheel source'

    def tearDown(self):
        el.ExpertLocationMetadata.logical_to_all_physical = STOCK

    def test_stock_wheel_reproduces_probe_failure_and_silent_drop(self):
        metadata = target_map()
        with self.assertRaisesRegex(IndexError, 'index 448 is out of bounds for dimension 1 with size 448'):
            metadata.logical_to_all_physical(0, 448)
        # Global-expert params would get no slot at all: weights silently not loaded.
        self.assertEqual(metadata.logical_to_all_physical(0, 448, True), [])

    def test_guard_places_every_drafter_expert_in_its_own_slot(self):
        pruning.guard_draft_expert_map()
        pruning.guard_draft_expert_map()  # idempotent: no double wrap
        metadata = target_map()
        for require_global in (False, True):
            got = [metadata.logical_to_all_physical(0, i, require_global) for i in range(512)]
            self.assertEqual(got, [[i] for i in range(512)])

    def test_guard_leaves_target_lookups_to_the_wheel(self):
        metadata = target_map()
        expected = [[metadata.logical_to_all_physical(layer, i) for i in range(448)] for layer in (0, 47)]
        pruning.guard_draft_expert_map()
        self.assertEqual([[metadata.logical_to_all_physical(layer, i) for i in range(448)] for layer in (0, 47)], expected)

    def test_guard_refuses_non_identity_layout_and_out_of_bank_ids(self):
        pruning.guard_draft_expert_map()
        shuffled = torch.arange(448).roll(1).repeat(48, 1)
        with self.assertRaisesRegex(ValueError, 'not the one-rank identity'):
            target_map(physical_to_logical=shuffled).logical_to_all_physical(0, 448)
        with self.assertRaisesRegex(ValueError, 'outside the drafter bank'):
            target_map().logical_to_all_physical(0, 512)


if __name__ == '__main__':
    print('torch', torch.__version__, flush=True)
    unittest.main(verbosity=2)
