"""Matched-pair budget: both arms must leave their notebooks with identical caps."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qualify import HERE, HybridSolver, HarnessSolver
from taaf.game_api import GameAPI

PUBLIC25 = ['ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59', 'lf52', 'lp85', 'ls20',
            'm0r0', 'r11l', 're86', 's5i5', 'sb26', 'sc25', 'sk48', 'sp80', 'su15', 'tn36', 'tr87',
            'tu93', 'vc33', 'wa30']


def donor_benchmark():
    """Donor state after its cell 16 under priority scheduling with all 25 games."""
    solver = HarnessSolver(analyzer_timeout=900.0)
    solver.concurrency = 25
    solver.max_runtime_s_per_game = 532*60 * solver.concurrency // 110
    solver.max_actions_per_game = None
    return SimpleNamespace(solver=solver, n_passes=1, game_weights=None,
                           games=[GameAPI(env_name=name+'-fixture') for name in PUBLIC25])


class PairRegressions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.out = Path(self.directory.name)/'pair'
        subprocess.run([sys.executable, '-B', str(HERE/'experiment/freeze_pair.py'),
                        '--out', str(self.out), '--seed', '7'], check=True, capture_output=True)

    def tearDown(self):
        self.directory.cleanup()

    def budget_block(self, arm):
        notebook = json.loads((self.out/arm/'kernel/notebook.ipynb').read_text())
        run = next(''.join(c['source']) for c in notebook['cells']
                   if '# Matched-pair budget' in ''.join(c['source']))
        start = run.index('# Matched-pair budget')
        return run[start:run.index('# Play the benchmark', start)]

    def test_both_arms_embed_the_same_budget_and_control_escapes_870_seconds(self):
        self.assertEqual(self.budget_block('control'), self.budget_block('hybrid'))
        bm = donor_benchmark()
        # At three games the donor formula would allot 532*60*3//110 = 870 s.
        bm.solver.concurrency = 3
        self.assertEqual(532*60*bm.solver.concurrency//110, 870)
        with tempfile.TemporaryDirectory() as root:
            namespace = {'bm': bm, 'json': json, 'datetime': datetime,
                         '_qualification_root': Path(root), '_qualification_started_epoch': 0.0}
            exec(self.budget_block('control'), namespace)
            config = json.loads((Path(root)/'pair-config.json').read_text())
        self.assertEqual(config['games'], ['ls20', 're86', 'sc25'])
        self.assertEqual(config['per_game_seconds'], 3600)
        self.assertEqual(config['concurrency'], 3)
        self.assertIsNone(config['per_game_action_cap'])
        self.assertEqual(bm.solver.max_runtime_s_per_game, 3600)
        self.assertEqual(namespace['soft_end'], datetime.fromtimestamp(5100))

    def test_hybrid_runner_compare_mode_rejects_drifted_budget(self):
        from hybrid import runner, pair
        for drift, expect_ok in [(None, True), (870, False)]:
            with tempfile.TemporaryDirectory() as directory, self.subTest(drift=drift):
                root = Path(directory)
                for name in ('benchmark.pkl', 'target.pkl'): (root/name).touch()
                async def run(**kwargs): pass
                bm = donor_benchmark()
                pair.apply(bm)
                if drift: bm.solver.max_runtime_s_per_game = drift
                bm = SimpleNamespace(**vars(bm), job_dir=root, game_runs=[], run=run,
                                     request_stop=lambda: None)
                with patch.object(sys, 'argv', ['runner', str(root/'benchmark.pkl'), str(root/'target.pkl')]), \
                     patch.object(runner.pickle, 'load', side_effect=[bm, SimpleNamespace(actual_run_as_submission=False)]), \
                     patch.dict(os.environ, HYBRID_RUN_MODE='compare', TYCHO_MODE='orchestrator',
                                TYCHO_NOOA_TREATMENT='control', ARC_LLM_MODEL='fixture', ARC3_MAX_ACTIVE_STREAMS='10'):
                    if expect_ok:
                        runner.main()
                        config = json.loads((root/'tycho-dfranzen/candidate-config.json').read_text())
                        self.assertEqual(config['per_game_seconds'], 3600)
                        self.assertEqual(sorted(n.split('-')[0] for n in config['game_names']), ['ls20', 're86', 'sc25'])
                        self.assertIsInstance(bm.solver, HybridSolver)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'pair budget mismatch'):
                            runner.main()

    def test_identity_records_order_draw_and_solver_only_difference(self):
        identity = json.loads((self.out/'pair-identity.json').read_text())
        self.assertEqual(sorted(identity['launch_order']), ['control', 'hybrid'])
        self.assertEqual(identity['arm_order_seed'], 7)
        self.assertEqual(set(identity['differing_donor_cells']), {'0', '4', '20', 'last'})
        self.assertEqual(identity['reserved_gpu_hours'], 3.0)


if __name__ == '__main__':
    unittest.main()
