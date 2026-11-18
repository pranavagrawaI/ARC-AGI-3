"""Submission containment through main and daily_draw, mocking Kaggle only."""
import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import daily_draw, submit_draw

F1 = 'pranavagrawai/m2-f1-screen1-f1'
F1X = 'pranavagrawai/m2-f1-screen2-f1x'
CONTROL = 'dfranzen/dfranzen-m2-raw'
POLICY = {'schema_version': 1, 'entries': [
    {'kernel': F1, 'version': 1, 'reason': 'D9 COMPLETE zero; cause unresolved', 'evidence': ['receipt-d9.json']},
    {'kernel': F1X, 'version': 1, 'reason': 'D12 COMPLETE zero; D10 positive, cause unresolved', 'evidence': ['receipt-d12.json']},
]}
LISTING = ('ref,fileName,date,description,status,publicScore,privateScore\n'
           '1,submission.parquet,2026-10-20,Hidden draw D11 reap,SubmissionStatus.COMPLETE,23.32,\n'
           '2,submission.parquet,2026-10-21,Hidden draw D12 f1x,SubmissionStatus.COMPLETE,0.00,\n')


class FixedDatetime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 22, 11, tzinfo=dt.timezone.utc)


class SubmissionQuarantine(unittest.TestCase):
    def run_submit(self, kernel=F1X, version='1', policy=POLICY, extra=(), listing='', daily=False, entry=None):
        calls = []
        sdk = mock.MagicMock()
        sdk.api.build_kaggle_client.return_value.__enter__.return_value.kernels.kernels_api_client.get_accelerator_quota_statistics.return_value = '{}'
        modules = {
            'kagglesdk': types.SimpleNamespace(KaggleClient=sdk.client),
            'kagglesdk.kaggle_creds': types.SimpleNamespace(KaggleCredentials=sdk.credentials),
            'kaggle.api.kaggle_api_extended': types.SimpleNamespace(KaggleApi=lambda: sdk.api),
        }
        real_import = __import__

        def track_import(name, *args, **kwargs):
            if name in modules:
                sdk.import_seen(name)
            return real_import(name, *args, **kwargs)

        def external_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == daily_draw.KAGGLE_PY:
                with mock.patch.object(sys, 'argv', cmd[2:]):
                    return types.SimpleNamespace(returncode=submit_draw.main() or 0)
            output = 'Remaining today: 1' if 'submission-limits' in cmd else listing
            return types.SimpleNamespace(stdout=output, stderr='', returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'queue.json').write_text(json.dumps({'2026-10-22': entry} if entry else {}))
            if policy is not None:
                (folder / 'quarantine.json').write_text(policy if isinstance(policy, str) else json.dumps(policy))
            clock = types.SimpleNamespace(datetime=FixedDatetime, date=dt.date, timezone=dt.timezone)
            with mock.patch.object(submit_draw, 'HERE', folder), mock.patch.object(daily_draw, 'HERE', folder), \
                 mock.patch.object(submit_draw, 'dt', clock), mock.patch.object(daily_draw, 'dt', clock), \
                 mock.patch.dict(sys.modules, modules), mock.patch('builtins.__import__', track_import), \
                 mock.patch.object(sys, 'argv', ['submit_draw.py', 'D13', kernel, version, 'test', *extra]), \
                 mock.patch.object(submit_draw.subprocess, 'run', external_run), contextlib.redirect_stdout(io.StringIO()) as output:
                if daily:
                    with self.assertRaises(SystemExit) as result:
                        daily_draw.main()
                    code = result.exception.code
                else:
                    code = submit_draw.main()
            files = {path.name: path.read_text() for path in folder.iterdir()}
        return code or 0, output.getvalue(), calls, sdk, files

    def test_known_failed_version_blocks_before_kaggle(self):
        code, output, calls, sdk, files = self.run_submit()
        self.assertEqual(code, 5)
        self.assertIn('quarantined', output)
        self.assertIn('D12 COMPLETE zero', output)
        self.assertEqual(calls, [])
        self.assertEqual(sdk.mock_calls, [])
        self.assertNotIn('receipt-d13.json', files)
        self.assertNotIn('quota-before-d13.json', files)

    def test_missing_or_malformed_policy_fails_closed_before_kaggle(self):
        invalid_entry = {**POLICY['entries'][0], 'evidence': []}
        policies = [None, '{', [], {}, {'schema_version': True, 'entries': []},
                    {'schema_version': 2, 'entries': []}, {'schema_version': 1, 'entries': {}},
                    {'schema_version': 1, 'entries': [None]},
                    {'schema_version': 1, 'entries': [invalid_entry]},
                    {'schema_version': 1, 'entries': [*POLICY['entries'], invalid_entry]}]
        for field, value in [('kernel', None), ('kernel', 'missing-owner'), ('version', '1'),
                             ('version', True), ('version', 0), ('reason', ''), ('reason', 1),
                             ('evidence', 'receipt.json'), ('evidence', [None]), ('evidence', [''])]:
            policies.append({'schema_version': 1, 'entries': [{**POLICY['entries'][0], field: value}]})
        for policy in policies:
            with self.subTest(policy=policy):
                code, output, calls, sdk, files = self.run_submit(kernel=CONTROL, policy=policy)
                self.assertEqual(code, 1)
                self.assertIn('quarantine policy unavailable or invalid', output)
                self.assertEqual(calls, [])
                self.assertEqual(sdk.mock_calls, [])
                self.assertNotIn('receipt-d13.json', files)

    def test_invalid_later_entry_cannot_hide_behind_an_earlier_match(self):
        policy = {'schema_version': 1, 'entries': [*POLICY['entries'], {}]}
        code, output, calls, sdk, _ = self.run_submit(policy=policy)
        self.assertEqual(code, 1)
        self.assertIn('quarantine policy unavailable or invalid', output)
        self.assertEqual(calls, [])
        self.assertEqual(sdk.mock_calls, [])

    def test_quarantine_identity_cannot_be_bypassed_by_case_or_version_spelling(self):
        for kernel, version in [(F1, '1'), (F1X.upper(), '1'), (F1X, '01')]:
            with self.subTest(kernel=kernel, version=version):
                code, _, calls, sdk, _ = self.run_submit(kernel=kernel, version=version)
                self.assertEqual(code, 5)
                self.assertEqual(calls, [])
                self.assertEqual(sdk.mock_calls, [])

    def test_new_version_unrelated_kernel_and_valid_empty_policy_remain_eligible(self):
        for kernel, version, policy in [(F1X, '2', POLICY), (CONTROL, '1', POLICY),
                                        (F1X, '1', {'schema_version': 1, 'entries': []})]:
            with self.subTest(kernel=kernel, version=version, policy=policy):
                code, _, calls, sdk, files = self.run_submit(kernel=kernel, version=version, policy=policy)
                self.assertEqual(code, 0)
                submissions = [cmd for cmd in calls if 'submit' in cmd]
                self.assertEqual(len(submissions), 1)
                self.assertEqual(submissions[0][5:8], [kernel, '-v', version])
                self.assertIn('receipt-d13.json', files)
                sdk.credentials.load.return_value.refresh_access_token.assert_called_once()

    def test_existing_draw_remains_idempotent(self):
        code, output, calls, _, files = self.run_submit(kernel=CONTROL, listing='Hidden draw D13 control')
        self.assertEqual(code, 0)
        self.assertIn('already submitted', output)
        self.assertFalse(any('submit' in cmd for cmd in calls))
        self.assertNotIn('receipt-d13.json', files)

    def test_prerequisite_states_keep_their_exit_codes(self):
        for listing, expected in [(LISTING, 3), (LISTING.replace('COMPLETE,0.00', 'PENDING,'), 4), ('401 Unauthorized', 1)]:
            with self.subTest(expected=expected):
                code, _, calls, _, _ = self.run_submit(kernel=CONTROL, extra=('--requires', 'D11,D12'), listing=listing)
                self.assertEqual(code, expected)
                self.assertFalse(any('submit' in cmd for cmd in calls))

    def test_daily_empty_queue_submits_the_control_fallback_once(self):
        code, _, calls, _, files = self.run_submit(daily=True)
        self.assertEqual(code, 0)
        submissions = [cmd for cmd in calls if 'submit' in cmd]
        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0][5:8], [daily_draw.FALLBACK[0], '-v', daily_draw.FALLBACK[1]])
        self.assertIn('receipt-d13.json', files)

    @mock.patch.object(daily_draw, 'FALLBACK', (F1X, '1', 'quarantined fallback'))
    def test_daily_empty_queue_and_quarantined_queue_never_submit_or_fall_back(self):
        for entry in [None, {'kernel': F1, 'version': 1, 'message': 'quarantined queued draw', 'requires': ['D12']}]:
            with self.subTest(entry=entry):
                code, _, calls, sdk, _ = self.run_submit(daily=True, entry=entry)
                self.assertEqual(code, 5)
                self.assertEqual(len(calls), 1)  # only the local Python invocation
                self.assertEqual(sdk.mock_calls, [])

    @mock.patch.object(daily_draw, 'FALLBACK', (F1X, '1', 'no new candidate queued'))
    def test_failed_prerequisite_fallback_is_blocked_at_submission_boundary(self):
        entry = {'kernel': 'pranavagrawai/m2-f1x-reap14-screen1', 'version': 1,
                 'message': 'merge', 'requires': ['D11', 'D12']}
        code, output, calls, _, files = self.run_submit(daily=True, entry=entry, listing=LISTING)
        self.assertEqual(code, 5)
        self.assertIn('requires D11,D12: failed', output)
        self.assertIn('quarantined ' + F1X, output)
        local = [cmd for cmd in calls if cmd[0] == daily_draw.KAGGLE_PY]
        self.assertEqual([cmd[4] for cmd in local], [entry['kernel'], F1X])
        self.assertFalse(any('submit' in cmd for cmd in calls))
        self.assertNotIn('receipt-d13.json', files)

    def test_daily_healthy_queued_control_still_submits(self):
        entry = {'kernel': CONTROL, 'version': 1, 'message': 'healthy control'}
        code, _, calls, _, files = self.run_submit(daily=True, entry=entry)
        self.assertEqual(code, 0)
        self.assertEqual(len([cmd for cmd in calls if 'submit' in cmd]), 1)
        self.assertEqual(json.loads(files['receipt-d13.json'])['kernel'], CONTROL)

    def test_checked_in_policy_contains_the_failed_versions_and_allows_successors(self):
        policy = json.loads((HERE / 'quarantine.json').read_text())
        self.assertEqual([(entry['kernel'], entry['version']) for entry in policy['entries']], [(F1, 1), (F1X, 1)])
        for entry in policy['entries']:
            for evidence in entry['evidence']:
                self.assertTrue((HERE.parents[1] / evidence).is_file(), evidence)
        for kernel, version, expected in [(F1, '1', 5), (F1X, '1', 5),
                                           (F1, '2', 0), (F1X, '2', 0), (CONTROL, '1', 0)]:
            with self.subTest(kernel=kernel, version=version):
                code, _, calls, _, _ = self.run_submit(kernel=kernel, version=version, policy=policy)
                self.assertEqual(code, expected)
                self.assertEqual(len([cmd for cmd in calls if 'submit' in cmd]), 0 if expected == 5 else 1)


if __name__ == '__main__':
    unittest.main()
