"""Three-arm wave: identical caps in every arm, lean = port minus Builder/Scribe,
reservation once for all arms, at most two sessions per launch, never a retry."""
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

sys.path.insert(0, str(HERE/'experiment'))
import wave_run  # noqa: E402

WAVE_GAMES = ['cn04', 'dc22', 'ka59', 'ls20', 'm0r0', 'r11l', 're86', 'sc25', 'su15', 'vc33']


class WaveRegressions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.out = Path(self.directory.name)/'wave'
        subprocess.run([sys.executable, '-B', str(HERE/'experiment/freeze_wave.py'),
                        '--out', str(self.out), '--seed', '7'], check=True, capture_output=True)
        self.identity = json.loads((self.out/'pair-identity.json').read_text())

    def tearDown(self):
        self.directory.cleanup()

    def budget_block(self, arm):
        notebook = json.loads((self.out/arm/'kernel/notebook.ipynb').read_text())
        run = next(''.join(c['source']) for c in notebook['cells']
                   if '# Matched-pair budget' in ''.join(c['source']))
        start = run.index('# Matched-pair budget')
        return run[start:run.index('# Play the benchmark', start)]

    def test_every_arm_embeds_the_same_ten_game_budget(self):
        self.assertEqual(self.budget_block('control'), self.budget_block('port'))
        self.assertEqual(self.budget_block('control'), self.budget_block('lean'))
        bm = donor_benchmark()
        with tempfile.TemporaryDirectory() as root:
            namespace = {'bm': bm, 'json': json, 'datetime': datetime,
                         '_qualification_root': Path(root), '_qualification_started_epoch': 0.0}
            exec(self.budget_block('control'), namespace)
            config = json.loads((Path(root)/'pair-config.json').read_text())
        self.assertEqual(config['games'], WAVE_GAMES)
        self.assertEqual((config['per_game_seconds'], config['concurrency']), (3600, 10))
        self.assertIsNone(config['per_game_action_cap'])
        self.assertEqual(namespace['soft_end'], datetime.fromtimestamp(4500))

    def test_lean_differs_from_port_only_by_mode_and_treatment(self):
        port = json.loads((self.out/'port/kernel/notebook.ipynb').read_text())
        lean = json.loads((self.out/'lean/kernel/notebook.ipynb').read_text())
        diffs = [(''.join(a['source']), ''.join(b['source'])) for a, b in zip(port['cells'], lean['cells'])
                 if ''.join(a['source']).replace("'port'", "'X'").replace("-port'", "-X'")
                 != ''.join(b['source']).replace("'lean'", "'X'").replace("-lean'", "-X'")]
        self.assertEqual(len(diffs), 1)
        self.assertIn("'TYCHO_MODE': 'no_world_model', 'TYCHO_NOOA_TREATMENT': 'no_scribe'", diffs[0][1])
        self.assertEqual(self.identity['reserved_gpu_hours'], 4.0)
        self.assertEqual(sorted(self.identity['launch_order']), ['control', 'lean', 'port'])

    def test_compare_mode_accepts_wave_budget_and_rejects_drift(self):
        from hybrid import runner, pair
        for drift, expect_ok in [(None, True), (870, False)]:
            with tempfile.TemporaryDirectory() as directory, self.subTest(drift=drift):
                root = Path(directory)
                for name in ('benchmark.pkl', 'target.pkl'): (root/name).touch()
                async def run(**kwargs): pass
                bm = donor_benchmark()
                pair.apply(bm)
                if drift: bm.solver.max_runtime_s_per_game = drift
                bm = SimpleNamespace(**vars(bm), job_dir=root, game_runs=[], run=run, request_stop=lambda: None)
                with patch.object(sys, 'argv', ['runner', str(root/'benchmark.pkl'), str(root/'target.pkl')]), \
                     patch.object(runner.pickle, 'load', side_effect=[bm, SimpleNamespace(actual_run_as_submission=False)]), \
                     patch.dict(os.environ, HYBRID_RUN_MODE='compare', TYCHO_MODE='no_world_model',
                                TYCHO_NOOA_TREATMENT='no_scribe', ARC_LLM_MODEL='fixture', ARC3_MAX_ACTIVE_STREAMS='10'):
                    if expect_ok:
                        runner.main()
                        config = json.loads((root/'tycho-dfranzen/candidate-config.json').read_text())
                        self.assertEqual(sorted(n.split('-')[0] for n in config['game_names']), WAVE_GAMES)
                        self.assertIsInstance(bm.solver, HybridSolver)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'pair budget mismatch'):
                            runner.main()


class WaveLaunch(unittest.TestCase):
    def setUp(self):
        WaveRegressions.setUp(self)
        self.budget = Path(self.directory.name)/'budget.json'
        self.budget.write_text(json.dumps({'unreserved_gpu_hours': 4.5, 'run_reservations': []}))
        (self.out/'EXPERIMENT-CARD.md').write_text('fixture card')
        (self.out/'LOCAL-QUALIFICATION.json').write_text(json.dumps({'passed': True,
            'payload_sha256': self.identity['source_payload_sha256'],
            'dataset_readback_sha256': self.identity['source_payload_sha256']}))

    def tearDown(self):
        self.directory.cleanup()

    def api(self, calls):
        class API:
            def kernels_push(api, path, timeout):
                calls.append(Path(path).parent.name)
                return json.dumps({'versionNumber': 1, 'error': None})
        return API()

    def test_first_launch_reserves_all_arms_once_and_never_retries(self):
        calls = []
        with patch('wave_run.quota', return_value={'fixture': True}):
            wave_run.launch(self.out, self.api(calls), self.budget, ['control', 'port'])
            budget = json.loads(self.budget.read_text())
            self.assertEqual(len(budget['run_reservations']), 3)
            self.assertAlmostEqual(budget['unreserved_gpu_hours'], 0.5)
            wave_run.launch(self.out, self.api(calls), self.budget, ['lean'])
            self.assertAlmostEqual(json.loads(self.budget.read_text())['unreserved_gpu_hours'], 0.5)
            with self.assertRaises(AssertionError):
                wave_run.launch(self.out, self.api(calls), self.budget, ['port'])
        self.assertEqual(sorted(calls), ['control', 'lean', 'port'])

    def test_three_sessions_at_once_are_refused(self):
        with self.assertRaises(AssertionError):
            wave_run.launch(self.out, None, self.budget, ['control', 'port', 'lean'])
        self.assertEqual(json.loads(self.budget.read_text())['run_reservations'], [])

    def test_insufficient_budget_reserves_nothing(self):
        self.budget.write_text(json.dumps({'unreserved_gpu_hours': 3.9, 'run_reservations': []}))
        with patch('wave_run.quota', return_value={'fixture': True}), self.assertRaises(AssertionError):
            wave_run.launch(self.out, self.api([]), self.budget, ['control'])
        self.assertEqual(json.loads(self.budget.read_text())['run_reservations'], [])

    def test_collector_rejects_wrong_artifact_hash(self):
        class Remote:
            def __init__(self, *args): pass
            def fetch(self, name):
                if name == 'pair-evidence-index.json':
                    return json.dumps({'files': [{'path': 'benchmark.json', 'critical': True, 'sha256': 'wrong'}]}).encode()
                return b'{}'
        with patch('collect_outputs.KaggleOutput', Remote):
            wave_run.collect_arm(self.out, 'lean', 'fixture/kernel', 1)
        receipt = json.loads((self.out/'lean/collection-receipt.json').read_text())
        self.assertFalse(receipt['critical_complete'])
        self.assertIn('ValueError', [e['error_type'] for e in receipt['errors']])

    def test_watcher_collects_every_terminal_arm_once_after_transient_failure(self):
        for arm in ('control', 'port', 'lean'):
            (self.out/arm/'launch-response.json').write_text(json.dumps({'versionNumber': 1}))
        calls = []; collected = []
        class API:
            def kernels_status(api, kernel):
                calls.append(kernel)
                if len(calls) == 1: raise ConnectionError('transient status lookup')
                return SimpleNamespace(status='RUNNING' if len(calls) == 2 else 'COMPLETE')
        with patch('wave_run.quota', return_value={'fixture': True}), patch('wave_run.time.sleep'), \
             patch('wave_run.collect_arm', side_effect=lambda d, arm, kernel, version: collected.append(arm)):
            wave_run.watch(self.out, API())
        self.assertEqual(sorted(collected), ['control', 'lean', 'port'])
        self.assertEqual(json.loads((self.out/'monitor-result.json').read_text())['collected_arms'],
                         ['control', 'lean', 'port'])


class WaveTwo(unittest.TestCase):
    """wave-2 = control, lean_prune, lean_cap4: each lean arm is port with one env line changed."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.out = Path(self.directory.name)/'wave-2'
        subprocess.run([sys.executable, '-B', str(HERE/'experiment/freeze_wave.py'), '--wave-id', 'wave-2',
                        '--out', str(self.out), '--seed', '7'], check=True, capture_output=True)
        self.identity = json.loads((self.out/'pair-identity.json').read_text())

    def tearDown(self):
        self.directory.cleanup()

    def test_arms_kernels_and_budget(self):
        self.assertEqual(sorted(self.identity['arms']), ['control', 'lean_cap4', 'lean_prune'])
        self.assertEqual(self.identity['hybrid_arms'], ['lean_prune', 'lean_cap4'])
        self.assertEqual(self.identity['reserved_gpu_hours'], 4.0)
        for arm, slug in (('lean_prune', 'lean-prune'), ('lean_cap4', 'lean-cap4')):
            meta = json.loads((self.out/arm/'kernel/kernel-metadata.json').read_text())
            self.assertEqual(meta['id'], f'pranavagrawai/tycho-dfranzen-wave-2-{slug}')
            self.assertEqual(meta['title'], f'Tycho dfranzen wave-2 {slug}')  # Kaggle slugifies the title
            self.assertTrue((self.out/arm/'source-bundle/hybrid-runtime.zlib').is_file())
            self.assertEqual(WaveRegressions.budget_block(self, arm), WaveRegressions.budget_block(self, 'control'))

    def test_lean_arms_differ_only_by_treatment(self):
        prune = json.loads((self.out/'lean_prune/kernel/notebook.ipynb').read_text())
        cap = json.loads((self.out/'lean_cap4/kernel/notebook.ipynb').read_text())
        diffs = [(''.join(a['source']), ''.join(b['source'])) for a, b in zip(prune['cells'], cap['cells'])
                 if ''.join(a['source']).replace('lean_prune', 'X') != ''.join(b['source']).replace('lean_cap4', 'X')]
        self.assertEqual(diffs, [])
        self.assertTrue(any("'TYCHO_MODE': 'no_world_model', 'TYCHO_NOOA_TREATMENT': 'lean_cap4'" in ''.join(c['source'])
                            for c in cap['cells']))


if __name__ == '__main__':
    unittest.main()
