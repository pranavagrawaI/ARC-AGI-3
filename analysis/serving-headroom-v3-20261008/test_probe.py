import gzip
import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mtp-depth-20261006'))
sys.path.insert(0, str(HERE))
import probe
import build
import native_fixture as native


class Tests(unittest.TestCase):
    def samples(self, n=10):
        return [dict(t=float(i), running=n, queue=0, pending=n, gen_tokens=0,
                     prefill_compute=100, prefill_cache=0, decode_tokens=(i // 3) * 300)
                for i in range(30)]

    def test_native_inflight_decode_counts_accepted_tokens(self):
        reporter, collector = native.decoder(10)
        rows = []
        for i in range(30):
            reporter.report_decode_stats(True, num_correct_drafts=15)
            raw = ('sglang:num_running_reqs 10\nsglang:num_queue_reqs 0\n'
                   'sglang:generation_tokens_total 0\n' + collector.realtime_tokens_total.scrape()).encode()
            rows.append(dict(t=float(i), pending=10, **probe.parse_metrics(raw)))
        self.assertEqual(collector.realtime_tokens_total.values['decode'], 750)
        self.assertTrue(probe.plateau(rows, 10)['valid'])
        self.assertEqual(probe.plateau(rows, 10)['U'], 25)
        self.assertEqual(rows[-1]['gen_tokens'], 0)

    def test_bursty_denominator_prefill_and_reset_exclusion(self):
        got = probe.plateau(self.samples(), 10)
        self.assertTrue(got['valid'])
        self.assertEqual((got['seconds'], got['U']), (24, 100))
        for mutation in ('prefill_compute', 'prefill_cache', 'decode_tokens', 'pending', 'queue', 'running'):
            rows = self.samples()
            for r in rows:
                if mutation.startswith('prefill'): r[mutation] += r['t']
                elif mutation == 'decode_tokens': r[mutation] = -r['t']
                elif mutation == 'queue': r[mutation] = 1
                else: r[mutation] = 9
            self.assertFalse(probe.plateau(rows, 10)['valid'], mutation)
        rows = self.samples()
        rows[12]['decode_tokens'] = 0  # otherwise-valid windows after reset cannot rescue a block
        self.assertFalse(probe.plateau(rows, 10)['valid'])
        for value in (math.nan, math.inf):
            rows = self.samples(); rows[12]['decode_tokens'] = value
            self.assertFalse(probe.plateau(rows, 10)['valid'])
        rows = self.samples()
        for r in rows: del r['decode_tokens']
        self.assertFalse(probe.plateau(rows, 10)['valid'])
        rows = self.samples(); del rows[12]['decode_tokens']
        self.assertFalse(probe.plateau(rows, 10)['valid'])

    def test_metric_modes_duplicates_and_raw_receipts(self):
        raw = b'sglang:realtime_tokens_total{dp_rank="0",mode="decode"} 123\nsglang:realtime_tokens_total{mode="prefill_compute"} 1000\nsglang:realtime_tokens_total{mode="prefill_cache"} 2000\n'
        got = probe.parse_metrics(raw)
        self.assertEqual(got, {'decode_tokens':123, 'prefill_compute':1000, 'prefill_cache':2000})
        with self.assertRaisesRegex(RuntimeError, 'multiple scheduler'):
            probe.parse_metrics(raw + b'sglang:realtime_tokens_total{mode="decode",dp_rank="1"} 456\n')
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe, 'http', return_value=raw):
            got = probe.sample(Path(tmp))
            self.assertEqual((Path(tmp) / got['raw_metrics']).read_bytes(), raw)
            self.assertEqual(len(got['raw_sha256']), 64)

    def test_profile_trigger_independent_of_decode_rate(self):
        rows = self.samples()
        for r in rows: r['decode_tokens'] = 0
        self.assertFalse(probe.plateau(rows, 10)['valid'])
        self.assertTrue(probe.profile_ready(rows, 10))
        rows[-1]['prefill_cache'] = 1
        self.assertFalse(probe.profile_ready(rows, 10))

    def test_pinned_admission_and_actual_old_receipt(self):
        self.assertEqual(native.admission(60), (5, 12))
        self.assertEqual(native.admission(80), (5, 16))
        old = json.loads((HERE.parent / 'serving-headroom-readout-20261008/raw/headroom/server-info-before.json').read_text())
        with self.assertRaisesRegex(RuntimeError, 'effective max-running'):
            probe.capacity(old, [{'recorded_prompt': 100}] * 10)
        info = {'max_running_requests':16, 'internal_states':[{'effective_max_running_requests_per_dp':10,
                'max_mamba_cache_size':60, 'memory_usage':{'token_capacity':1000000}}]}
        jobs = [{'recorded_prompt':100}] * 10
        self.assertEqual(probe.capacity(info,jobs)['required_tokens'], 42240)
        info['internal_states'][0]['memory_usage']['token_capacity'] = 42239
        with self.assertRaisesRegex(RuntimeError, 'insufficient KV'):
            probe.capacity(info,jobs)

    def test_native_profiler_step_stop_export_and_missing_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); manager = native.profiler()
            class Events:
                def emit(self,*args,**kw): pass
            def wire(path,body,**kw):
                self.assertEqual(path,'/start_profile')
                return native.start_profile(manager,body)
            with patch.object(probe,'http',side_effect=wire):
                probe.profile_block('PROFILE1',out,Events())
                probe.profile_block('PROFILE10',out,Events())
            self.assertFalse(manager.profile_in_progress)
            self.assertEqual(manager.forward,130)
            rec=probe.verify_trace('PROFILE10',out)
            self.assertEqual(rec[0]['kernel_events'],1)
            with self.assertRaisesRegex(RuntimeError,'no trace'): probe.verify_trace('MISSING',out)
            with gzip.open(out/rec[0]['path'],'wt') as f:json.dump({'traceEvents':[]},f)
            with self.assertRaisesRegex(RuntimeError,'no CUDA kernels'):probe.verify_trace('PROFILE10',out)
            with patch.object(probe,'http',return_value=b'{"success":false}'):
                with self.assertRaisesRegex(RuntimeError,'unexpected profiler'):probe.profile_block('BAD',out,Events())

    def test_batch_http_rehearsal_profiles_and_preserves_failure(self):
        clock=[1000.]; finished=threading.Event(); manager=native.profiler()
        reporter,collector=native.decoder(1)
        state={'done':True, 'gen':0}
        class Handler(BaseHTTPRequestHandler):
            write_trace=True
            def log_message(self,*args):pass
            def do_GET(self):
                clock[0]+=1
                if not state['done']:reporter.report_decode_stats(True,num_correct_drafts=99)
                raw=(f'sglang:num_running_reqs {0 if state["done"] else 1}\nsglang:num_queue_reqs 0\nsglang:generation_tokens_total {state["gen"]}\nsglang:realtime_tokens_total{{mode="prefill_compute"}} 0\n'
                     +collector.realtime_tokens_total.scrape()).encode()
                self.send_response(200);self.end_headers();self.wfile.write(raw)
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/start_profile':
                    if self.write_trace:native.start_profile(manager,body)
                    raw=b'Start profiling.\n'
                else:
                    base_decode=collector.realtime_tokens_total.values.get('decode',0)
                    state['done']=False
                    finished.wait(.5)
                    collector.realtime_tokens_total.values['decode']=base_decode+body['max_tokens']
                    state['gen']+=body['max_tokens'];state['done']=True
                    raw=json.dumps({'usage':{'prompt_tokens':100,'completion_tokens':body['max_tokens'],
                        'prompt_tokens_details':{'cached_tokens':95}}}).encode()
                self.send_response(200);self.end_headers();self.wfile.write(raw)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        # Advancing virtual scrape time leaves actual wire/thread behavior intact.
        def tick(seconds):
            if clock[0]-base[0]>24:finished.set()
            threading.Event().wait(.002)
        job={'game':'fixture','body':{'ignore_eos':True,'seed':20261008}}
        try:
            with tempfile.TemporaryDirectory() as tmp,patch.object(probe,'PORT',server.server_port), \
                 patch.object(probe.time,'monotonic',side_effect=lambda:clock[0]),patch.object(probe.time,'sleep',side_effect=tick):
                out=Path(tmp);base=[clock[0]]
                result=probe.batch('PROFILE1',[job],out,4096)
                self.assertTrue(result['profile_attempted'])
                self.assertTrue(result['plateau']['valid'])
                self.assertEqual(result['plateau']['U'],100)
                self.assertEqual(result['traces'][0]['kernel_events'],1)
                self.assertTrue((out/'PROFILE1/fixture-response.json').is_file())
                events=[json.loads(x) for x in (out/'PROFILE1/events.jsonl').read_text().splitlines()]
                self.assertEqual(sum(x['kind']=='profile_request' for x in events),1)
                self.assertTrue(all((out/'PROFILE1'/x['raw_metrics']).is_file() for x in events if x['kind']=='metrics'))
                Handler.write_trace=False;finished.clear();base[0]=clock[0]
                with self.assertRaisesRegex(RuntimeError,'no trace'):probe.batch('PROFILE10',[job],out,4096)
                self.assertEqual(json.loads((out/'PROFILE10/summary.json').read_text())['status'],'failed')
        finally:finished.set();server.shutdown();server.server_close()

    def test_capacity_failure_sends_no_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(probe,'cohort',return_value=[{'recorded_prompt':10}] * 10), \
                 patch.object(probe,'http',return_value=b'{"max_running_requests":16,"internal_states":[{"effective_max_running_requests_per_dp":12}]}'), \
                 patch.object(probe,'batch') as send:
                with self.assertRaisesRegex(RuntimeError,'effective max-running'):probe.run_probe(out,out,99999)
                send.assert_not_called()
                self.assertIn('stop',json.loads((out/'state.json').read_text()))

    def test_accounting_requires_wire_completion_and_decode_agreement(self):
        before={'decode_tokens':100,'gen_tokens':20}
        after={'decode_tokens':4195,'gen_tokens':4116}
        self.assertEqual(probe.accounting(before,[],after,4096,1)['decode_delta'],4095)
        for key,value in [('gen_tokens',4115),('decode_tokens',5000),('retractions',1)]:
            wrong=dict(after,**{key:value})
            with self.assertRaisesRegex(RuntimeError,'reconciliation'):probe.accounting(before,[],wrong,4096,1)
        with self.assertRaisesRegex(RuntimeError,'counter lifetime'):
            probe.accounting(before,[{}],after,4096,1)

    def test_notebook_preserves_serving_configuration(self):
        build.build()
        nb=json.loads((HERE/'kernel'/(build.SLUG+'.ipynb')).read_text())
        parent=json.loads((build.PARENT/'kernel/mtp-depth-sweep-retry1-20261007.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')
        self.assertEqual(nb['cells'][4]['source'],parent['cells'][4]['source'])
        self.assertEqual(nb['cells'][3]['source'][:len(parent['cells'][3]['source'])],parent['cells'][3]['source'])
        self.assertIn('verify_bundle', ''.join(nb['cells'][3]['source']))
        self.assertIn('verify_installed',''.join(nb['cells'][5]['source']))
        self.assertIn('3300',''.join(nb['cells'][2]['source']))
        self.assertIn(probe.RUN_ID,''.join(nb['cells'][-1]['source']))
        self.assertEqual(probe.PLAN,[('PROFILE1',1),('PROFILE10',10),('N10A',10),('N10B',10)])
        meta=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
        self.assertEqual(meta['competition_sources'],['arc-prize-2026-arc-agi-3'])
        self.assertEqual(meta['machine_shape'],'NvidiaRtxPro6000')
        self.assertTrue(meta['is_private'])

if __name__=='__main__':unittest.main()
