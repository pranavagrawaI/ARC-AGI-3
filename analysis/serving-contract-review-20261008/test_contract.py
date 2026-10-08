"""Independent runtime-contract regressions. No GPU or launch side effects.

Run: uv run --no-project --with prometheus-client==0.26.0 python <this file>
Pinned native methods run unmodified against real Prometheus counters. Tests of
trace validation use labelled fixtures and do not establish actual CUDA capture.
"""
import ast
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / 'serving-headroom-v2-20261008'
sys.path[:0] = [str(TARGET), str(HERE.parent / 'mtp-depth-20261006')]
import probe


def native(relative, classname, method):
    manifest = json.loads((TARGET / 'native-contract.json').read_text())
    path = TARGET / 'native-source' / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][relative]
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == classname)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[method]


def runtime():
    registry = CollectorRegistry()
    collector = NS(labels={})
    collector.realtime_tokens_total = Counter('sglang:realtime_tokens_total', 'native decode', ['mode'], registry=registry)
    increment = native('srt/observability/metrics_collector.py', 'SchedulerMetricsCollector', 'increment_realtime_tokens')
    collector.increment_realtime_tokens = lambda **kw: increment(collector, **kw)
    for mode in ('prefill_compute', 'prefill_cache'):
        collector.increment_realtime_tokens(dp_cooperation_info=None, **{mode + '_tokens': 100})
    tokenizer = NS()
    for key in ('prompt_tokens_total', 'generation_tokens_total', 'num_requests_total'):
        value = Counter('sglang:' + key, 'native completion', ['model_name', 'is_streaming'], registry=registry)
        value.labels(model_name='fixture', is_streaming='false')
        setattr(tokenizer, key, value)
    tokenizer.histogram_e2e_request_latency = Histogram('e2e', 'fixture', ['model_name','is_streaming'], registry=registry)
    for key in ('prompt_tokens_histogram', 'uncached_prompt_tokens_histogram', 'generation_tokens_histogram'):
        setattr(tokenizer, key, Histogram(key, 'fixture', ['model_name'], registry=registry))
    finished = native('srt/observability/metrics_collector.py', 'TokenizerMetricsCollector', 'observe_one_finished_request')
    reporter = NS(current_scheduler_metrics_enabled=True, metrics_collector=collector,
                  enable_mfu_metrics=False, scheduler_status_logger=None,
                  forward_ct_decode=1, decode_log_interval=40,
                  scheduler=NS(running_batch=NS(batch_size=lambda: 10, dp_cooperation_info=None)))
    decode = native('srt/managers/scheduler_components/metrics_reporter.py', 'SchedulerMetricsReporter', 'report_decode_stats')
    for name, value in [('num_running_reqs', 10), ('num_queue_reqs', 0), ('num_retracted_requests_total', 0)]:
        Gauge('sglang:' + name, 'fixture', registry=registry).set(value)
    return registry, collector, lambda: decode(reporter, True, num_correct_drafts=20), lambda: finished(tokenizer, {'model_name':'fixture'}, 1000, 4096, 0, 1., False)


def parse(raw):
    if hasattr(probe, 'parse_metrics'):
        return probe.parse_metrics(raw.encode())
    if hasattr(probe, 'parse_sample'):
        return probe.parse_sample(raw)
    with patch.object(probe, 'http', return_value=raw.encode()):
        return probe.sample()


def rows():
    return [dict(t=float(i), pending=10, running=10, queue=0, decode_tokens=i*300.,
                 prefill_compute=100., prefill_cache=100., gen_tokens=0., scrape_seconds=.01)
            for i in range(40)]


class RuntimeContract(unittest.TestCase):
    def test_real_native_inflight_counter_with_real_prometheus_export(self):
        registry, collector, decode, finished = runtime()
        samples = []
        for i in range(40):
            decode()
            value = parse(generate_latest(registry).decode())
            self.assertEqual(value['decode_tokens'], (i+1)*30)
            self.assertEqual(value['gen_tokens'], 0)
            samples.append(dict(value, t=float(i), pending=10, scrape_seconds=.01))
        result = probe.plateau(samples, 10)
        self.assertTrue(result['valid'])
        self.assertEqual(result['U'], 30)
        finished()
        value = parse(generate_latest(registry).decode())
        self.assertEqual(value['gen_tokens'], 4096)
        self.assertEqual(value['decode_tokens'], 1200)

    def test_native_finished_requests_still_contribute_delayed_decode_accounting(self):
        resolve = native('srt/managers/scheduler_components/batch_result_processor.py',
                         'SchedulerBatchResultProcessor', '_resolve_spec_v2_tokens')
        requests = [NS(is_retracted=False, finished=lambda: True, grammar=None) for _ in range(10)]
        tensor = lambda value: NS(is_cpu=True, tolist=lambda: value)
        result = NS(next_token_ids=tensor(list(range(40))), accept_lens=tensor([4]*10),
                    block_accept_lens=None, cap_lens=None, speculative_num_draft_tokens=4)
        processor = NS(model_worker=NS(on_verify_complete_cpu=lambda *a, **kw: None),
                       advance_grammar_fsm=lambda *a: None)
        predict = resolve(processor, result, NS(reqs=requests))
        # Native finished-request branch skips token commitment but still reports
        # the accepted lengths to the reporter. This is no proof of a4*n total bound.
        self.assertEqual(result.num_correct_drafts, 30)
        self.assertEqual(sum(map(len, predict)), 40)
        self.assertTrue(all(not hasattr(r, 'kv_committed_len') for r in requests))

    def test_prefill_is_never_added_to_decode(self):
        registry, collector, decode, finished = runtime()
        decode()
        collector.increment_realtime_tokens(dp_cooperation_info=None, prefill_compute_tokens=1000000, prefill_cache_tokens=2000000)
        value = parse(generate_latest(registry).decode())
        self.assertEqual(value['decode_tokens'], 30)
        self.assertEqual(value['prefill_compute'], 1000100)
        self.assertEqual(value['prefill_cache'], 2000100)

    def test_duplicate_scheduler_decode_series_rejected(self):
        raw = ('sglang:realtime_tokens_total{mode="decode",worker="0"} 100\n'
               'sglang:realtime_tokens_total{mode="decode",worker="1"} 100\n')
        with self.assertRaises((ValueError, RuntimeError)):
            parse(raw)

    def test_decode_counter_reset_invalidates_entire_block(self):
        sample = rows()
        for row in sample[20:]: row['decode_tokens'] -= 6000
        self.assertFalse(probe.plateau(sample, 10)['valid'])

    def test_missing_metric_cannot_be_skipped_to_salvage_block(self):
        sample = rows()
        del sample[20]['decode_tokens']
        self.assertFalse(probe.plateau(sample, 10)['valid'])

    def test_profile_trigger_does_not_depend_on_positive_throughput(self):
        sample = rows()
        for row in sample: row['decode_tokens'] = 0.
        self.assertFalse(probe.plateau(sample, 10)['valid'])
        self.assertTrue(probe.profile_ready(sample, 10))

    def test_recorded_effective12_is_rejected_for_requested16(self):
        info = json.loads((HERE.parent/'serving-headroom-readout-20261008/raw/headroom/server-info-before.json').read_text())
        self.assertEqual(info['max_running_requests'], 16)
        self.assertEqual(info['internal_states'][0]['effective_max_running_requests_per_dp'], 12)
        with self.assertRaisesRegex(RuntimeError, 'effective max-running'):
            probe.capacity(info, [{'recorded_prompt': 100}] * 10)

    def test_http_success_is_not_a_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'no trace'): probe.verify_trace('PROFILE10', Path(tmp))
            trace = Path(tmp)/'traces/PROFILE10-TP-0.trace.json.gz'
            trace.parent.mkdir()
            with gzip.open(trace,'wt') as stream: json.dump({'traceEvents':[{'cat':'cpu_op','ph':'X','dur':10}]},stream)
            with self.assertRaisesRegex(RuntimeError,'no CUDA'): probe.verify_trace('PROFILE10', Path(tmp))
            with gzip.open(trace,'wt') as stream: json.dump({'traceEvents':[{'cat':'kernel','ph':'X','dur':10,'ts':0,'name':'fixture'}]},stream)
            self.assertEqual(probe.verify_trace('PROFILE10', Path(tmp))[0]['kernel_events'], 1)


if __name__ == '__main__': unittest.main(verbosity=2)
