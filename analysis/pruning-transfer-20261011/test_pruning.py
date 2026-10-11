"""Loader contract: retained values follow router IDs; exclusions stay untouched."""
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pruning

HERE = Path(__file__).resolve().parent
MASK_PATH = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/prune/k448_mask_layers.json')


class PruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = json.loads(MASK_PATH.read_text())
        cls.index = json.loads((HERE / 'checkpoints/intel-model.safetensors.index.json').read_text())['weight_map']
        cls.router = np.broadcast_to(np.arange(512, dtype=np.uint16)[:, None], (512, 2560)).copy()
        cls.hashes = {str(i): hashlib.sha256(cls.router.tobytes()).hexdigest() for i in range(48)}

    def stream(self):
        for name in self.index:
            yield name, self.router if pruning.ROUTER.fullmatch(name) else name

    def test_real_checkpoint_complete_mapping_preserves_values_and_exclusions(self):
        stats = Counter(kept=0, dropped=0, routers=0)
        result = dict(pruning.filtered(self.stream(), self.mask, self.hashes, stats))
        self.assertEqual(dict(stats), {'kept': 193536, 'dropped': 27648, 'routers': 48})
        for layer in range(48):
            prefix = f'model.language_model.layers.{layer}.mlp.'
            # Independent expectation: row i must name the same original expert as tensor i.
            for new, old in enumerate(self.mask[str(layer)]):
                self.assertEqual(result[prefix + f'experts.{new}.gate_proj.qweight'],
                                 prefix + f'experts.{old}.gate_proj.qweight')
            np.testing.assert_array_equal(result[prefix + 'gate.weight'][:, 0], self.mask[str(layer)])
        self.assertTrue(all(result[n] == n for n in self.index if n.startswith('mtp.') or '.shared_expert.' in n))
        self.assertEqual(len(result), len(self.index) - 27648)

    def test_wrong_router_bytes_abort(self):
        hashes = dict(self.hashes, **{'0': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'Router identity'):
            list(pruning.filtered(self.stream(), self.mask, hashes, Counter(kept=0, dropped=0, routers=0)))

    def test_missing_tensor_aborts(self):
        stream = ((n, v) for n, v in self.stream() if n != 'model.language_model.layers.0.mlp.gate.weight')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            list(pruning.filtered(stream, self.mask, self.hashes, Counter(kept=0, dropped=0, routers=0)))

    def test_duplicate_and_unsupported_expert_abort(self):
        name = 'model.language_model.layers.0.mlp.experts.0.gate_proj.qweight'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            list(pruning.filtered([(name, 1), (name, 2)], self.mask, self.hashes, Counter()))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            pruning.remap_name(name.replace('qweight', 'unknown_scale'), {0: {0: 0}})

    def test_unpruned_path_is_identity(self):
        weights = [('anything', object()), ('mtp.layers.0.mlp.experts.511.gate_proj.weight', object())]
        self.assertEqual(list(pruning.filter_target_weights(SimpleNamespace(config=SimpleNamespace()), iter(weights))), weights)

    def test_view_only_changes_target_config_and_links_originals(self):
        report = json.loads((HERE / 'router-identity.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.mkdir()
            original = (HERE / 'checkpoints/intel-config.json').read_bytes()
            (source / 'config.json').write_bytes(original)
            (source / 'tensor.safetensors').write_bytes(b'unchanged checkpoint')
            view = pruning.make_view(source, root / 'view', self.mask, report)
            changed = json.loads((view / 'config.json').read_text())
            self.assertEqual(changed['text_config'].pop('retained_pruning')['mask'], self.mask)
            self.assertEqual(changed['text_config']['num_experts'], 448)
            changed['text_config']['num_experts'] = 512
            self.assertEqual(changed, json.loads(original))
            self.assertEqual((source / 'config.json').read_bytes(), original)
            self.assertEqual((view / 'tensor.safetensors').resolve(), source / 'tensor.safetensors')
            with self.assertRaises(FileExistsError):
                pruning.make_view(source, root / 'view', self.mask, report)


if __name__ == '__main__':
    unittest.main()
