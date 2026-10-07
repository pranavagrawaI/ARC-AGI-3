"""Behavioral gates for the authorized three-baseline sweep. Stdlib unittest only."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analyze
import sweep
import build_notebook
import launch_watch
import verify_prelaunch


def state(baselines=(100, 100, 100), candidate=100, cv=0.01):
    labels = ('M3A', 'M3C', 'M3B', 'M1', 'M2', 'M4')
    values = (*baselines, 100, candidate, 100)
    return {'launches': {label: {'status': 'ok', 'summary': {
        'U': u, 'U_cv': cv, 'mean_R': 8.97, 'p_R_ge_9': 0.72,
        'early_stop_rate': 0.1, 'cached_fraction_block': 0.935,
        'errors': 0, 'warmup_criterion_met': True,
    }} for label, u in zip(labels, values)}}


class DecisionTests(unittest.TestCase):
    def test_null_keeps_incumbent(self):
        self.assertEqual(analyze.decide(state())['verdict'], 'KEEP M=3')

    def test_gain_beats_every_baseline_and_noise(self):
        result = analyze.decide(state((100, 102, 101), 110))
        self.assertEqual(result['candidates']['M2']['result'], 'CLEAN WIN')
        self.assertAlmostEqual(result['D'], 0.02)
        self.assertAlmostEqual(result['candidates']['M2']['threshold'], 0.03)
        self.assertIn('M2', result['verdict'])

    def test_fast_middle_baseline_cannot_be_ignored(self):
        result = analyze.decide(state((100, 115, 100), 110))
        self.assertNotEqual(result['candidates']['M2']['result'], 'CLEAN WIN')

    def test_missing_any_baseline_never_adopts(self):
        for label in ('M3A', 'M3C', 'M3B'):
            with self.subTest(label=label):
                s = state(candidate=150)
                del s['launches'][label]
                result = analyze.decide(s)
                self.assertNotIn('ADOPT', result['verdict'])
                self.assertIn(label, result['verdict'])

    def test_failed_middle_baseline_never_adopts(self):
        s = state(candidate=150)
        s['launches']['M3C']['status'] = 'error'
        self.assertIn('UNRESOLVED', analyze.decide(s)['verdict'])

    def test_monotonic_baselines_report_launch_order_and_flag(self):
        for values in ((100, 101, 102), (102, 101, 100)):
            result = analyze.decide(state(values))
            self.assertEqual([x['label'] for x in result['baselines_in_launch_order']],
                             ['M3A', 'M3C', 'M3B'])
            self.assertTrue(any(x.startswith('SYSTEMATIC_DRIFT') for x in result['flags']))
        self.assertFalse(analyze.decide(state((100, 102, 101)))['flags'])

    def test_drift_only_gain_remains_unresolved(self):
        result = analyze.decide(state((100, 110, 105), 113))
        self.assertEqual(result['candidates']['M2']['result'], 'UNRESOLVED (launch drift)')

    def test_planted_loss(self):
        self.assertEqual(analyze.decide(state(candidate=90))['candidates']['M2']['result'],
                         'CLEAN LOSS')

    def test_invalid_occupancy_cannot_adopt(self):
        s = state(candidate=150)
        s['launches']['M3A']['summary']['mean_R'] = 7
        self.assertIn('INVALID WORKLOAD', analyze.decide(s)['verdict'])

    def test_baseline_without_warmup_or_noise_measurement_is_unresolved(self):
        for field, value in [('warmup_criterion_met', False), ('U_cv', None), ('U', float('nan'))]:
            s = state(candidate=150)
            s['launches']['M3C']['summary'][field] = value
            self.assertIn('UNRESOLVED', analyze.decide(s)['verdict'])


class PlanTests(unittest.TestCase):
    def test_process_start_failure_is_preserved_without_retry(self):
        class Factory:
            starts = 0
            def start(self, *args, **kwargs):
                self.starts += 1
                raise OSError('cannot start server')
            def stop(self, proc):
                raise AssertionError('no process was created')
        factory = Factory()
        with tempfile.TemporaryDirectory() as tmp:
            rec = sweep.run_launch('M4', 4, factory=factory, base_url='unused', out=Path(tmp),
                                   workload=None, logs_dir=None, alpha=0.523, model='mock',
                                   measure_s=240, time_scale=20)
            self.assertEqual(rec['status'], 'error')
            self.assertEqual(factory.starts, 1)
            self.assertIn('cannot start server', json.loads((Path(tmp) / 'M4/launch.json').read_text())['error'])

    def test_order_and_baselines(self):
        todo, profile, _, cuts = sweep.plan_for_time(14400, set(), 'separate', 1)
        self.assertEqual([x[0] for x in todo], ['M3A', 'M1', 'M2', 'M3C', 'M4', 'M3B'])
        self.assertTrue(profile)
        self.assertFalse(cuts)
        self.assertEqual([x[0] for x in todo if x[2]], ['M3A', 'M3C', 'M3B'])

    def test_time_pressure_never_drops_baseline_c(self):
        todo, profile, _, cuts = sweep.plan_for_time(6000, set(), 'separate', 1)
        self.assertFalse(profile)
        self.assertIn('profile', cuts)
        self.assertEqual([x[0] for x in todo if x[2]], ['M3A', 'M3C', 'M3B'])
        self.assertIn('M1', [x[0] for x in todo])

    def test_resume_skips_completed_launches(self):
        todo, _, _, _ = sweep.plan_for_time(14400, {'M3A', 'M1'}, 'off', 1)
        self.assertEqual([x[0] for x in todo], ['M2', 'M3C', 'M4', 'M3B'])


class WatchTests(unittest.TestCase):
    def test_valid_quota_includes_reservation(self):
        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __init__(self): self.kernels = types.SimpleNamespace(kernels_api_client=self)
            def get_accelerator_quota_statistics(self):
                return json.dumps({'gpuQuota': {'timeUsed': '123.5s', 'timeReserved': '45s'}})
        api = types.SimpleNamespace(build_kaggle_client=Client)
        self.assertEqual(launch_watch.quota_used_s(api)[0], 168.5)

    def test_watcher_records_one_accepted_attempt_and_never_repeats_it(self):
        import requests
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api = unittest.mock.Mock()
            api.kernels_push.return_value = json.dumps({'versionNumber': 1, 'url': 'https://www.kaggle.com/code/owner/test'})
            with patch.object(launch_watch, 'EVID', root), patch.object(launch_watch, 'identity', return_value={'tag_commit': 'frozen'}), patch.object(launch_watch, 'priority_resolved', return_value=True), patch.object(launch_watch, 'refresh_token'), patch.object(launch_watch, 'KaggleApi', return_value=api), patch.object(launch_watch, 'active_sessions', return_value=[]), patch.object(launch_watch, 'quota_used_s', return_value=(0, {})), patch.object(requests.Session, 'send', requests.Session.send), patch.object(sys, 'argv', ['launch_watch.py']):
                launch_watch.main()
                receipt = json.loads((root / 'launch-attempt.json').read_text())
                self.assertEqual(receipt['state'], 'accepted')
                self.assertEqual(receipt['identity']['tag_commit'], 'frozen')
                with self.assertRaises(SystemExit):
                    launch_watch.main()
                api.kernels_push.assert_called_once()

    def test_dry_poll_failure_is_bounded_and_cannot_push(self):
        import requests
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api = unittest.mock.Mock()
            with patch.object(launch_watch, 'EVID', root), patch.object(launch_watch, 'identity', return_value={}), patch.object(launch_watch, 'priority_resolved', return_value=True), patch.object(launch_watch, 'refresh_token', side_effect=TimeoutError('poll timed out')), patch.object(launch_watch, 'KaggleApi', return_value=api), patch.object(requests.Session, 'send', requests.Session.send), patch.object(sys, 'argv', ['launch_watch.py', '--dry']):
                with self.assertRaises(SystemExit):
                    launch_watch.main()
                api.kernels_push.assert_not_called()
                self.assertFalse((root / 'launch-attempt.json').exists())

    def test_priority_hold_and_unaccepted_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / 'control-gate.json'
            resolved = Path(tmp) / 'resolved.json'
            with patch.object(launch_watch, 'GATE', gate), patch.object(launch_watch, 'WAVE2_RESOLVED', resolved):
                gate.write_text(json.dumps({'decision': 'hold'}))
                self.assertTrue(launch_watch.priority_resolved())
                gate.write_text(json.dumps({'decision': 'launch'}))
                self.assertFalse(launch_watch.priority_resolved())
                (Path(tmp) / 'control').mkdir()
                (Path(tmp) / 'control/launch-response.json').write_text(json.dumps({'versionNumber': 1}))
                self.assertTrue(launch_watch.priority_resolved())

    def test_recorded_hold_requires_matching_raw_evidence(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            readout = Path(tmp) / 'readout.json'
            readout.write_text('preserved readout')
            resolved = Path(tmp) / 'resolved.json'
            resolved.write_text(json.dumps({
                'decision': 'wave-2 control priority resolved: HOLD, control will not launch',
                'readout_json': str(readout),
                'readout_sha256': hashlib.sha256(readout.read_bytes()).hexdigest(),
            }))
            with patch.object(launch_watch, 'GATE', Path(tmp) / 'absent'), patch.object(launch_watch, 'WAVE2_RESOLVED', resolved):
                self.assertTrue(launch_watch.priority_resolved())
                readout.write_text('changed')
                self.assertFalse(launch_watch.priority_resolved())

    def test_unknown_remote_status_blocks_launch(self):
        class API:
            def kernels_list(self, **kwargs):
                return [type('K', (), {'ref': 'owner/kernel'})()]
            def kernels_status(self, ref):
                raise TimeoutError('network unavailable')
        self.assertEqual(launch_watch.active_sessions(API())[0][1], 'unknown (TimeoutError)')

    def test_known_terminal_status_does_not_hold_slot(self):
        class API:
            def kernels_list(self, **kwargs):
                return [type('K', (), {'ref': 'owner/kernel'})()]
            def kernels_status(self, ref):
                return type('S', (), {'status': 'COMPLETE'})()
        self.assertEqual(launch_watch.active_sessions(API()), [])

    def test_missing_quota_is_unknown_not_zero(self):
        class Client:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __init__(self):
                self.kernels = type('K', (), {'kernels_api_client': self})()
            def get_accelerator_quota_statistics(self):
                return '{}'
        class API:
            def build_kaggle_client(self):
                return Client()
        with self.assertRaises(ValueError):
            launch_watch.quota_used_s(API())


class FreezeTests(unittest.TestCase):
    def test_verifier_checks_code_readback_notebook_and_tag(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'kaggle/dataset'
            dataset.mkdir(parents=True)
            (root / 'kaggle/notebook').mkdir()
            (root / 'evidence').mkdir()
            (root / 'alpha.json').write_text(json.dumps({'alpha': 0.523}))
            (dataset / 'fixture.py').write_text('committed source')
            prov = {'code_commit': 'code-freeze', 'code_dirty': False, 'prelaunch_tag': verify_prelaunch.TAG}
            (dataset / 'provenance.json').write_text(json.dumps(prov))
            manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir()}
            (dataset / 'manifest.json').write_text(json.dumps(manifest))
            digest = hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest()
            readback = {'version': 2, 'manifest_sha256': digest, 'files': manifest, 'mismatches': []}
            (root / 'evidence/dataset-readback.json').write_text(json.dumps(readback))
            nb = {'cells': [{'cell_type': 'code', 'source': [digest, prov['code_commit']]}]}
            (root / f'kaggle/notebook/{build_notebook.KERNEL_SLUG}.ipynb').write_text(json.dumps(nb))
            (root / 'kaggle/notebook/kernel-metadata.json').write_text('{}')
            def git(*args):
                if args == ('rev-parse', '--show-prefix'): return 'analysis/mtp-depth-20261006/'
                if args[0] == 'rev-parse': return 'tag-freeze'
                return ''
            blob = types.SimpleNamespace(stdout=b'committed source')
            with patch.object(verify_prelaunch, 'HERE', root), patch.object(verify_prelaunch, 'git', side_effect=git), patch.object(build_notebook, 'CODE', ['fixture.py']), patch.object(verify_prelaunch.subprocess, 'run', return_value=blob):
                result = verify_prelaunch.verify()
                self.assertEqual(result['tag_commit'], 'tag-freeze')
                self.assertEqual(result['dataset_manifest_sha256'], digest)
                (dataset / 'fixture.py').write_text('changed source')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    verify_prelaunch.verify()
                (dataset / 'fixture.py').write_text('committed source')
                readback['manifest_sha256'] = 'stale'
                (root / 'evidence/dataset-readback.json').write_text(json.dumps(readback))
                with self.assertRaisesRegex(ValueError, 'readback'):
                    verify_prelaunch.verify()

    def test_builder_refuses_uncommitted_code(self):
        with patch.object(build_notebook.subprocess, 'run', return_value=types.SimpleNamespace(stdout=' M analyze.py')):
            with self.assertRaises(SystemExit):
                build_notebook.provenance(False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
