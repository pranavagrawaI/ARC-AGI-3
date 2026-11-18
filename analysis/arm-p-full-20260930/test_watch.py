"""Collector contract: only complete, collected runs can write a P retention decision.

Run with the installed Kaggle Python; all external calls are replaced at the boundary.
"""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import watch


class WatchTests(unittest.TestCase):
    def exercise(self, remote_status='COMPLETE', download_fails=False):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            for arm in ['p1', 'p2']:
                (here / (arm + '-launch.json')).write_text(json.dumps({'started_at': dt.datetime.now(dt.timezone.utc).isoformat()}))
            calls = []
            def command(args, log, timeout=60):
                calls.append(args)
                if 'output' in args and download_fails:
                    raise RuntimeError('download interrupted')
                if '--terminal' in args:
                    (here / 'assessment.json').write_text(json.dumps({'decision': {'status': 'PASS', 'candidate': 'P'}}))
            api = SimpleNamespace(kernels_status=lambda slug: SimpleNamespace(status=SimpleNamespace(name=remote_status), failure_message=''))
            with patch.object(watch, 'HERE', here), patch.object(watch, 'EVIDENCE', here), patch.object(watch, 'api', api), patch.object(watch, 'source_check'), patch.object(watch, 'command', command), patch.object(watch.subprocess, 'run'):
                if download_fails:
                    with self.assertRaises(RuntimeError):
                        watch.main()
                else:
                    watch.main()
            retention = json.loads((here / 'retention.json').read_text()) if (here / 'retention.json').exists() else None
            state = json.loads((here / 'watch-status.json').read_text())
            return retention, state, calls

    def test_complete_pair_is_collected_before_retention(self):
        retention, state, calls = self.exercise()
        self.assertEqual(retention['candidate'], 'P')
        self.assertFalse(retention['competition_submission_performed'])
        self.assertEqual(state['phase'], 'complete')
        self.assertEqual(sum('output' in c for c in calls), 2)
        self.assertFalse(any('push' in c for c in calls))

    def test_remote_failure_cannot_promote_even_with_passing_local_summary(self):
        retention, state, calls = self.exercise('ERROR')
        self.assertEqual(retention['candidate'], 'KF')
        self.assertEqual(retention['status'], 'INVALID')

    def test_interrupted_download_cannot_write_a_decision(self):
        retention, state, calls = self.exercise(download_fails=True)
        self.assertIsNone(retention)
        self.assertEqual(state['phase'], 'needs_attention')


if __name__ == '__main__':
    unittest.main()
