import ast
import datetime as dt
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import build
import evaluate
import execute
import install

HERE = Path(__file__).resolve().parent


class ContractTests(unittest.TestCase):
    def test_analyzer_reproduces_retained_run_and_rejects_missing_game(self):
        archived = HERE.parent / 'model-serving-20261011/wave-1/e2-c14/output'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / 'control/output'; (out / 'pruning').mkdir(parents=True)
            (root / 'control/collection.json').write_text(json.dumps({'status': 'COMPLETE'}))
            for name in ['serve.log', 'wave1-e2-c14-20261011.log']:
                (out / name).symlink_to(archived / name)
            shutil.copy2(archived / 'benchmark.json', out / 'benchmark.json')
            for name, value in {'target.json': {'complete': True, 'target_experts': 512}, 'draft.json': {'draft_experts': 512},
                                'server-info-start.json': {'max_running_requests': 14, 'context_length': 102400}}.items():
                (out / 'pruning' / name).write_text(json.dumps(value))
            with patch.object(evaluate, 'HERE', root):
                result = evaluate.arm('control', 512, 14)
                self.assertEqual(result['levels'], 107)
                self.assertAlmostEqual(result['score'], 45.77, places=2)
                self.assertEqual(result['throughput'], 815.1)
                doc = json.loads((out / 'benchmark.json').read_text()); doc['game_runs'].pop()
                (out / 'benchmark.json').write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    evaluate.arm('control', 512, 14)

    def test_admission_bounds_quota_slots_and_prior_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'probe').mkdir()
            run = {'run': 'probe', 'hard_cap_s': 3600, 'slug': 'owner/probe'}
            (root / 'runs.json').write_text(json.dumps({'runs': [run]}))
            quota = SimpleNamespace(time_used=dt.timedelta(hours=16), time_reserved=dt.timedelta(), total_time_allowed=dt.timedelta(hours=30))
            backend = SimpleNamespace(quota=lambda: SimpleNamespace(gpu_quota=quota), active_sessions=lambda owner: [], require_new_slug=lambda slug: None)
            with patch.object(execute, 'HERE', root), patch.object(execute, 'verify_freeze'):
                self.assertEqual(execute.admission(backend, [run])['new_cap_s'], 3600)
                quota.time_used = dt.timedelta(hours=30)
                with self.assertRaisesRegex(ValueError, 'quota'):
                    execute.admission(backend, [run])
                quota.time_used = dt.timedelta(hours=16)
                backend.active_sessions = lambda owner: [{'ref': 'a'}, {'ref': 'b'}]
                with self.assertRaisesRegex(ValueError, 'slots'):
                    execute.admission(backend, [run])
                backend.active_sessions = lambda owner: []
                (root / 'probe/launch-attempt.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    execute.admission(backend, [run])
                # A retired attempt keeps its cap committed against the six-hour ceiling.
                pair = [{'run': name, 'hard_cap_s': 9000, 'slug': 'owner/' + name} for name in ['control', 'candidate']]
                (root / 'runs.json').write_text(json.dumps({'retired': [run], 'runs': pair}))
                self.assertEqual(execute.admission(backend, pair)['committed_cap_s'], 3600)
                (root / 'probe2').mkdir(); (root / 'probe2/launch-attempt.json').write_text('{}')
                probe2 = dict(run, run='probe2', slug='owner/probe2')
                (root / 'runs.json').write_text(json.dumps({'retired': [run], 'runs': [probe2, *pair]}))
                self.assertEqual(execute.admission(backend, pair)['committed_cap_s'], 7200)  # 1 h + 1 h + 5 h = 7 h ceiling
                probe3 = dict(run, run='probe3', slug='owner/probe3'); (root / 'probe3').mkdir(); (root / 'probe3/launch-attempt.json').write_text('{}')
                (root / 'runs.json').write_text(json.dumps({'retired': [run, probe3], 'runs': [probe2, *pair]}))
                with self.assertRaisesRegex(ValueError, 'ceiling'):
                    execute.admission(backend, pair)
                (root / 'runs.json').write_text(json.dumps({'runs': pair}))
                self.assertEqual(execute.admission(backend, pair)['committed_cap_s'], 0)

    def test_collector_hashes_raw_data_and_rejects_changed_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root / 'probe'; (folder / 'kernel').mkdir(parents=True)
            (folder / 'launch-response.json').write_text('{"kernelId": 123}'); (folder / 'kernel/probe.ipynb').write_text('{}')
            run = {'run': 'probe', 'slug': 'owner/probe', 'metadata': {'code_file': 'probe.ipynb'}}
            def download(item, output):
                (output / item).write_text('raw evidence')
            backend = SimpleNamespace(status=lambda slug: {'status': 'COMPLETE'}, verify_remote=lambda *args: {'source': {}},
                                      output_page=lambda *args: (['evidence.txt'], None, 'kernel log'), download=download)
            with patch.object(execute, 'HERE', root):
                receipt = execute.collect(backend, run)
                self.assertEqual(receipt['hashes']['evidence.txt'], hashlib.sha256(b'raw evidence').hexdigest())
                self.assertEqual(execute.collect(backend, run), receipt)
                (folder / 'output/evidence.txt').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    execute.collect(backend, run)

    def test_exact_wheel_patch_compiles_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in install.HASHES:
                shutil.copy2(HERE / 'source/sglang/srt/models' / name, root / name)
            result = install.patch_sources(root)
            for name, digest in result.items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
                compile((root / name).read_text(), name, 'exec')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                install.patch_sources(root)

    def test_score_first_retention(self):
        base = {'levels': 100, 'score': 45., 'throughput': 800., 'retractions': 0}
        self.assertEqual(evaluate.decision(base, dict(base, levels=110, throughput=900)), 'RETAIN_CANDIDATE')
        self.assertEqual(evaluate.decision(base, dict(base, levels=110, throughput=900, score=44)), 'INCONCLUSIVE_RETAIN_CONTROL')
        self.assertEqual(evaluate.decision(base, dict(base, levels=100, throughput=1600)), 'INCONCLUSIVE_RETAIN_CONTROL')
        self.assertEqual(evaluate.decision(base, dict(base, levels=90, throughput=1600)), 'REJECT_CANDIDATE')
        self.assertEqual(evaluate.decision(base, dict(base, levels=110, throughput=900, retractions=1)), 'INCONCLUSIVE_RETAIN_CONTROL')

    def test_frozen_notebooks_compile_and_public_only(self):
        runs = json.loads((HERE / 'runs.json').read_text())['runs']
        for run in runs:
            notebook = json.loads((HERE / run['run'] / 'kernel' / run['metadata']['code_file']).read_text())
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                if source.startswith('%%writefile '):
                    continue  # data-only harness patch cell, identical to frozen c14
                source = '\n'.join('pass' if line.startswith('!') else line for line in source.splitlines())
                compile(source, run['run'], 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            self.assertIn('assert not TRUE_SUBMISSION', ''.join(notebook['cells'][4]['source']))
            self.assertEqual(run['metadata']['model_sources'], json.loads((build.BASE / 'kernel-metadata.json').read_text())['model_sources'])

    def test_probe_must_have_real_capacity_and_all_responses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / 'probe2/output/pruning'; out.mkdir(parents=True)
            smoke = {'passed': True, 'windows': [{'concurrency': n, 'responses': [1]*n} for n in [14, 18]],
                     'accept_length': [2.7], 'accept_floor': 2.0, 'answers_289': 30}
            docs = {'target.json': {'complete': True, 'target_experts': 448, 'counts': {'kept': 193536, 'dropped': 27648, 'routers': 48}},
                    'draft.json': {'draft_experts': 512}, 'SMOKE.json': smoke,
                    'server-info-start.json': {'internal_states': [{'effective_max_running_requests_per_dp': 18}]}}
            for name, doc in docs.items():
                (out / name).write_text(json.dumps(doc))
            with patch.object(execute, 'HERE', root):
                self.assertTrue(execute.probe_passed({'status': 'COMPLETE'}, 'probe2'))
                self.assertFalse(execute.probe_passed({'status': 'ERROR'}, 'probe2'))
                # A mis-placed drafter shows as near-1 acceptance; a broken target never reaches 289.
                for bad in [{'accept_length': [1.2]}, {'accept_length': []}, {'answers_289': 0}, {'accept_floor': 1.0, 'accept_length': [1.5]}]:
                    (out / 'SMOKE.json').write_text(json.dumps(dict(smoke, **bad)))
                    self.assertFalse(execute.probe_passed({'status': 'COMPLETE'}, 'probe2'), bad)
                (out / 'SMOKE.json').write_text(json.dumps(smoke))
                (out / 'server-info-start.json').write_text(json.dumps({'internal_states': [{'effective_max_running_requests_per_dp': 14}]}))
                self.assertFalse(execute.probe_passed({'status': 'COMPLETE'}, 'probe2'))

    def test_launch_marker_prevents_duplicate_push(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'probe').mkdir()
            (root / 'probe/launch-attempt.json').write_text('{}')
            backend = SimpleNamespace(push=lambda *args: self.fail('Duplicate remote push'))
            with patch.object(execute, 'HERE', root), patch.object(execute, 'verify_freeze'):
                with self.assertRaises(FileExistsError):
                    execute.launch(backend, {'run': 'probe'}, {})


if __name__ == '__main__':
    unittest.main()
