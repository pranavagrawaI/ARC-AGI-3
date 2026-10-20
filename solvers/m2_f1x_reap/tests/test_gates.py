"""The screen's readout and queue gates, run over synthetic passing and failing outputs built from real runs.

python3 -B -m unittest solvers/m2_f1x_reap/tests/test_gates.py

PASS fixture: F1x screen-2 play (76 levels, preload/out16k evidence) + REAP candidate2's pruning receipts and
serve.log with running requests capped at 14 + a kernel log carrying the 14-stream gate line.
"""
import datetime as dt, importlib.util, json, re, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
F1X_OUT = ROOT / 'solvers/m2_f1/experiment/screen-2/f1x/output'
F1_OUT = ROOT / 'solvers/m2_f1/experiment/screen-1/f1/output'  # 62 levels: below the 64 floor
REAP_OUT = ROOT / 'analysis/pruning-transfer-20261011/candidate2/output'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


analyze = load('merge_analyze', HERE / 'experiment/analyze.py')
queue_candidate = load('merge_queue', HERE / 'experiment/queue_candidate.py')


def fixture(d, play=F1X_OUT, running_cap=14, gate=14, receipts=True):
    d.mkdir(parents=True)
    for f in play.iterdir():
        if f.name.endswith('_requests.jsonl') or f.name == 'benchmark.json':
            (d / f.name).symlink_to(f)
    serve = (REAP_OUT / 'serve.log').read_text(errors='replace')
    if running_cap:
        serve = re.sub(r'#running-req: (\d+)', lambda m: f'#running-req: {min(int(m.group(1)), running_cap)}', serve)
    (d / 'serve.log').write_text(serve)
    (d / 'kernel.log').write_text(f'priority gate active: {gate} concurrent streams\nSERVER READY at 595s from notebook start\n')
    if receipts:
        (d / 'pruning').symlink_to(REAP_OUT / 'pruning')
    return d


class Readout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = Path(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()

    def test_passing_run_passes(self):
        r = analyze.readout(fixture(self.d / 'a'))
        self.assertEqual(r['verdict'], 'PASS', r['mechanism'])
        self.assertEqual(r['no_harm']['merge_levels'], 76)
        self.assertEqual(r['no_harm']['floor'], 64)

    def test_unpruned_ten_stream_serving_fails_mechanism(self):
        d = self.d / 'b'
        fixture(d, running_cap=None, gate=10)
        (d / 'serve.log').unlink(); (d / 'serve.log').symlink_to(F1X_OUT / 'serve.log')
        r = analyze.readout(d)
        self.assertTrue(r['verdict'].startswith('MECHANISM_FAIL'))
        for k in ('gate_14_streams', 'pool_above_f1x', 'throughput_up_10pct'):
            self.assertFalse(r['mechanism'][k], k)

    def test_eighteen_streams_fail_the_gate(self):
        r = analyze.readout(fixture(self.d / 'c', running_cap=None, gate=18))
        self.assertFalse(r['mechanism']['gate_14_streams'])

    def test_low_levels_fail_no_harm_only(self):
        r = analyze.readout(fixture(self.d / 'd', play=F1_OUT))
        self.assertTrue(all(r['mechanism'].values()), r['mechanism'])
        self.assertTrue(r['verdict'].startswith('NO_HARM_FAIL'))

    def test_retractions_count_events_not_server_args(self):
        d = fixture(self.d / 'e')
        self.assertEqual(analyze.readout(d)['serving']['retractions'], 0)
        with (d / 'serve.log').open('a') as f:
            f.write('[2026-10-20 10:00:00] KV cache pool is full. Retract requests. #retracted_reqs: 2\n' * 6)
        r = analyze.readout(d)
        self.assertEqual(r['serving']['retractions'], 6)
        self.assertFalse(r['mechanism']['retractions_le_5'])


class Queue(unittest.TestCase):
    def run_queue(self, verdict, queue):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / 'screen-1/merge').mkdir(parents=True)
            r = analyze.readout(fixture(t / 'out'))
            r['verdict'] = verdict
            (t / 'screen-1/RESULT.json').write_text(json.dumps(r))
            (t / 'screen-1/merge/launch-response.json').write_text(json.dumps(
                {'ref': '/code/pranavagrawai/m2-f1x-reap14-screen1', 'versionNumber': 1}))
            qfile = t / 'queue.json'; qfile.write_text(json.dumps(queue))
            with mock.patch.object(queue_candidate, 'HERE', t), mock.patch.object(queue_candidate, 'QUEUE', qfile):
                queue_candidate.main()
                queue_candidate.main()  # idempotent
            return json.loads(qfile.read_text())

    def test_pass_queues_once_with_requirements_on_a_free_day(self):
        earliest = max(dt.date(2026, 10, 22), dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1))
        taken = {earliest.isoformat(): {'kernel': 'pranavagrawai/other', 'version': 1, 'message': 'x'}}
        q = self.run_queue('PASS', taken)
        mine = {k: v for k, v in q.items() if v['kernel'] == 'pranavagrawai/m2-f1x-reap14-screen1'}
        self.assertEqual(len(mine), 1)
        [(day, entry)] = mine.items()
        self.assertEqual(day, (earliest + dt.timedelta(days=1)).isoformat())
        self.assertEqual(entry['requires'], ['D11', 'D12'])
        self.assertEqual(entry['version'], 1)
        self.assertEqual(q[earliest.isoformat()]['kernel'], 'pranavagrawai/other')

    def test_fail_queues_nothing(self):
        for verdict in ('MECHANISM_FAIL: inspect', 'NO_HARM_FAIL: decide'):
            self.assertEqual(self.run_queue(verdict, {}), {})


if __name__ == '__main__':
    unittest.main()
