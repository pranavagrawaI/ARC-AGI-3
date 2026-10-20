"""Queue entries with "requires": submit only after the named draws are valid, else yield the day to the fallback.

python3 -B -m unittest analysis/hidden-draws-20261010/test_requires.py
"""
import datetime as dt, json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import daily_draw, submit_draw  # noqa: E402

HEAD = 'ref,fileName,date,description,status,publicScore,privateScore\n'


def row(draw, status, score):
    return f'1,submission.parquet,2026-10-20,"Hidden draw {draw} (candidate, X): text, with commas",SubmissionStatus.{status},{score},\n'


class RequirementState(unittest.TestCase):
    def state(self, rows, draws=('D11', 'D12')):
        return submit_draw.requirement_state(HEAD + ''.join(rows), list(draws))

    def test_met_when_all_complete_above_zero(self):
        self.assertEqual(self.state([row('D11', 'COMPLETE', '27.10'), row('D12', 'COMPLETE', '30.00')]), 'met')

    def test_zero_score_is_a_failure(self):  # D9 F1 scored 0.00 with status COMPLETE
        self.assertEqual(self.state([row('D11', 'COMPLETE', '0.00'), row('D12', 'COMPLETE', '30.00')]), 'failed')

    def test_error_fails_even_while_another_is_pending(self):
        self.assertEqual(self.state([row('D11', 'ERROR', ''), row('D12', 'PENDING', '')]), 'failed')

    def test_running_or_missing_is_pending(self):
        self.assertEqual(self.state([row('D11', 'COMPLETE', '27.10'), row('D12', 'PENDING', '')]), 'pending')
        self.assertEqual(self.state([row('D11', 'COMPLETE', '27.10')]), 'pending')

    def test_draw_tags_do_not_prefix_match(self):  # D1 must not match D11
        self.assertEqual(self.state([row('D1', 'COMPLETE', '26.24'), row('D10', 'COMPLETE', '30.88')], draws=('D11',)), 'pending')

    def test_unreadable_listing_is_unknown_not_pending(self):  # pending at >= 10Z would hand the day to the fallback
        self.assertEqual(submit_draw.requirement_state('', ['D11']), 'unknown')
        self.assertEqual(submit_draw.requirement_state('401 Unauthorized\n', ['D11']), 'unknown')

    def test_text_before_the_csv_header_is_skipped(self):
        listing = 'Warning: Looks like you are using an outdated API Version\n' + HEAD + row('D11', 'COMPLETE', '27.1')
        self.assertEqual(submit_draw.requirement_state(listing, ['D11']), 'met')


class DailyFlow(unittest.TestCase):
    """daily_draw.main with submit_draw stubbed: which kernels does the day try, in what order?"""

    def run_day(self, codes, requires=('D11', 'D12')):
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        calls = []

        def fake_run(cmd, cwd):
            calls.append(cmd)
            return mock.Mock(returncode=codes[len(calls) - 1])
        with tempfile.TemporaryDirectory() as d:
            entry = {'kernel': 'pranavagrawai/merge', 'version': 1, 'message': 'm', **({'requires': list(requires)} if requires else {})}
            Path(d, 'queue.json').write_text(json.dumps({today: entry}))
            Path(d, 'draws.log').write_text(f'x D0: submitted\n')
            with mock.patch.object(daily_draw, 'HERE', Path(d)), mock.patch.object(daily_draw.subprocess, 'run', fake_run):
                with self.assertRaises(SystemExit) as ex:
                    daily_draw.main()
        return [c[4] for c in calls], calls, ex.exception.code

    def test_requirements_met_submits_the_candidate_only(self):
        kernels, calls, code = self.run_day([0])
        self.assertEqual(kernels, ['pranavagrawai/merge'])
        self.assertEqual(calls[0][-2:], ['--requires', 'D11,D12'])
        self.assertEqual(code, 0)

    def test_failed_requirement_falls_back_at_once(self):
        kernels, calls, code = self.run_day([3, 0])
        self.assertEqual(kernels, ['pranavagrawai/merge', daily_draw.FALLBACK[0]])
        self.assertIn('pranavagrawai/merge held', calls[1][6])  # the description says why the fallback ran
        self.assertNotIn('no new candidate queued', calls[1][6])
        self.assertEqual(code, 0)

    def test_unreadable_requirements_never_fall_back(self):
        kernels, _, code = self.run_day([1])
        self.assertEqual(kernels, ['pranavagrawai/merge'])
        self.assertEqual(code, 1)

    def test_pending_waits_until_fallback_hour(self):
        kernels, _, code = self.run_day([4, 0])
        late = dt.datetime.now(dt.timezone.utc).hour >= daily_draw.FALLBACK_HOUR
        self.assertEqual(kernels, ['pranavagrawai/merge'] + ([daily_draw.FALLBACK[0]] if late else []))
        self.assertEqual(code, 0)

    def test_entry_without_requires_is_unchanged(self):
        kernels, calls, _ = self.run_day([0], requires=())
        self.assertEqual(kernels, ['pranavagrawai/merge'])
        self.assertNotIn('--requires', calls[0])

    def test_fall_back_rule(self):
        self.assertTrue(daily_draw.fall_back(3, 0))
        self.assertFalse(daily_draw.fall_back(4, daily_draw.FALLBACK_HOUR - 1))
        self.assertTrue(daily_draw.fall_back(4, daily_draw.FALLBACK_HOUR))
        self.assertFalse(daily_draw.fall_back(0, 23))
        self.assertFalse(daily_draw.fall_back(1, 23))  # a real submit error is not a reason to burn the fallback


if __name__ == '__main__':
    unittest.main()
