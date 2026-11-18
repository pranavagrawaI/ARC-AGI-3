"""Deadline accounting through real worker pipes, host actions and TAAF totals."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from qualify import HybridSolver, ExampleGame
from hybrid.accounting import final_usage
from tycho_nooa.protocol import WORKER_PREFIX, PROTOCOL_VERSION


def pending_worker(*, act_first=False, partial_tail=False):
    # Completed responses have an independently specified total of 31 tokens.
    # This worker never returns its last decision or its last usage over IPC.
    return f'''import json,sys,time
from pathlib import Path
prefix={WORKER_PREFIX!r}
version={PROTOCOL_VERSION!r}
def send(kind, **values):
    print(prefix+json.dumps(dict(protocol_version=version,type=kind,**values)),flush=True)
def completed(index, role, tokens):
    with Path('llm-trace.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(call_index=index,source=role,completion_tokens=tokens,prompt_tokens=100,cached_tokens=0))+'\\n')
sys.stdin.readline()
send('ready', metadata={{}})
sys.stdin.readline()
completed(1,'actor',7)
if {act_first!r}:
    stats=dict(calls=1,completion_tokens=7,prompt_tokens=100,cached_tokens=0,roles={{'actor':dict(calls=1,completion_tokens=7)}})
    send('decision',sequence=0,decision={{'action':'UP'}},hybrid=stats)
    sys.stdin.readline()
    send('observed',sequence=0,hybrid=stats)
    sys.stdin.readline()
completed(2,'builder',11)
completed(3,'actor',13)
if {partial_tail!r}:
    with Path('llm-trace.jsonl').open('a') as stream:
        stream.write('{{"call_index":4,')
time.sleep(90)
'''


class AccountingRegressions(unittest.TestCase):
    def run_pending(self, *, act_first=False, partial_tail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'worker_entry.py').write_text(pending_worker(act_first=act_first, partial_tail=partial_tail))
            game = ExampleGame(); game.start_game()
            solver = HybridSolver(runtime_root=str(root), worker_python=sys.executable,
                                  max_runtime_s_per_game=1, analyzer_timeout=10)
            solver.job_dir = root; solver.setup()
            try:
                solver._play(game, 0)
                path, = root.glob('hybrid-games/*')
                compute = json.loads((path / 'compute.json').read_text())
                receipt = json.loads((path / 'execution-receipt.json').read_text())
                return game.game_run, compute, receipt
            finally:
                solver.teardown()

    def test_completed_responses_survive_unreturned_decision(self):
        run, compute, receipt = self.run_pending()
        self.assertEqual(run.state, 'gave_up')
        self.assertEqual(len(run.history), 0)
        self.assertEqual(run.final_generated_tokens, 31)
        self.assertEqual(compute['completion_tokens'], 31)
        self.assertEqual(compute['calls'], 3)
        self.assertEqual(compute['roles'], {'actor': {'calls': 2, 'completion_tokens': 20},
                                           'builder': {'calls': 1, 'completion_tokens': 11}})
        self.assertEqual(receipt['stop_reason'], 'game_deadline')
        self.assertTrue(receipt['unresolved_provider_usage'])
        self.assertIsNotNone(receipt['worker_returncode'])

    def test_credited_actions_are_not_counted_twice_and_partial_tail_is_unknown(self):
        run, compute, receipt = self.run_pending(act_first=True, partial_tail=True)
        self.assertEqual(len(run.history), 1)
        self.assertEqual(run.history[0].generated_tokens, 7)
        self.assertEqual(run.final_generated_tokens, 24)
        self.assertEqual(compute['completion_tokens'], 31)
        self.assertEqual(receipt['usage_reconciliation']['incomplete_trace_tail'], True)
        self.assertTrue(receipt['unresolved_provider_usage'])

    def test_replays_all_preserved_q6_completed_usage(self):
        root = Path(__file__).resolve().parents[1] / 'experiment/gpu-q6/output/tycho-dfranzen/hybrid-games'
        if not root.exists():
            self.skipTest('q6 evidence unavailable')
        expected = {'ar25': (172, 225133, 3, 9043), 'sb26': (95, 235653, 5, 8591),
                    'ls20': (151, 232274, 0, 0)}
        for game, (calls, tokens, new_calls, new_tokens) in expected.items():
            with self.subTest(game=game):
                path, = root.glob(game + '-*')
                original = (path / 'compute.json').read_bytes()
                stats, receipt = final_usage(path, json.loads(original))
                self.assertEqual((stats['calls'], stats['completion_tokens']), (calls, tokens))
                self.assertEqual((receipt['unreported_completed_calls'], receipt['unreported_completion_tokens']),
                                 (new_calls, new_tokens))
                self.assertEqual((path / 'compute.json').read_bytes(), original)

    def test_invalid_committed_trace_and_regressing_usage_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / 'llm-trace.jsonl'
            trace.write_text('{invalid}\n')
            with self.assertRaises(ValueError):
                final_usage(root, {})
            trace.write_text(json.dumps(dict(call_index=1, source='actor', completion_tokens=7,
                                             prompt_tokens=100, cached_tokens=0)) + '\n')
            with self.assertRaisesRegex(ValueError, 'behind reported usage'):
                final_usage(root, {'calls': 1, 'completion_tokens': 9})

    def test_missing_trace_retains_ipc_usage_and_null_usage_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats, receipt = final_usage(root, {'calls': 1, 'completion_tokens': 13})
            self.assertEqual(stats, {'calls': 1, 'completion_tokens': 13})
            self.assertEqual(receipt['source'], 'worker_ipc_snapshot')
            (root / 'llm-trace.jsonl').write_text(json.dumps(dict(call_index=1, source='actor',
                completion_tokens=None, prompt_tokens=None, cached_tokens=None)) + '\n')
            stats, receipt = final_usage(root, {})
            self.assertEqual(stats['calls'], 1)
            self.assertEqual(receipt['unknown_usage_calls'], 1)
            (root / 'llm-trace.jsonl').write_text(json.dumps(dict(call_index=1, source='actor',
                completion_tokens=7, prompt_tokens=None, cached_tokens=0)) + '\n')
            stats, receipt = final_usage(root, {})
            self.assertEqual(stats['completion_tokens'], 7)
            self.assertEqual(receipt['unknown_usage_calls'], 1)

    def test_invalid_usage_or_role_cannot_be_counted_as_a_completed_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = dict(call_index=1, source='actor', completion_tokens=7, prompt_tokens=100, cached_tokens=0)
            for overrides in ({'call_index': 2}, {'source': ''}, {'completion_tokens': -1},
                              {'completion_tokens': True}, {'prompt_tokens': 1.5}):
                with self.subTest(overrides=overrides):
                    (root / 'llm-trace.jsonl').write_text(json.dumps({**good, **overrides}) + '\n')
                    with self.assertRaises(ValueError):
                        final_usage(root, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
