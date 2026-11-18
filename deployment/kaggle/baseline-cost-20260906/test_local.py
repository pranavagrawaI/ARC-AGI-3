import ast
import base64
import http.server
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import build
import runner


class Checks(unittest.TestCase):
    def test_notebook_identity_and_competition(self):
        files=build.build()
        for name,data in files.items(): self.assertEqual((build.HERE/name).read_bytes(),data)
        meta=json.loads(files['kernel-metadata.json'])
        self.assertEqual(meta['competition_sources'],['arc-prize-2026-arc-agi-3'])
        self.assertEqual(meta['machine_shape'],'NvidiaRtxPro6000')
        self.assertFalse(meta['enable_internet'])
        nb=json.loads(files['baseline_cost.ipynb'])
        # Check the executable final asset binding, not just the loose source.
        source=''.join(nb['cells'][2]['source'])
        decoded=[]
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode':
                if node.args and isinstance(node.args[0],ast.Constant):
                    decoded.append(base64.b64decode(node.args[0].value))
        self.assertIn((build.HERE/'runner.py').read_bytes(),decoded)

    def test_denominator_and_unknown_cache(self):
        value=runner.decode_stats({'completion_tokens':101,'decode_throughput':50})
        self.assertEqual(value['server_tokenizer_decode_span_s'],2)
        self.assertEqual(value['server_tokenizer_ms_per_token'],20)
        self.assertIsNone(runner.cache_stats({'prompt_tokens':100})['new_input_tokens'])
        self.assertEqual(runner.cache_stats({'prompt_tokens':100,'cached_tokens':80})['new_input_tokens'],20)

    def test_all_experiment_branches_and_preserved_chat(self):
        corpus=json.loads((build.DONOR/'corpus.json').read_text())
        events=[]; fixed=[]; requests=[]; selected=iter([0,5,12,13,12,13])
        def post(request):
            i=next(selected);requests.append(request)
            expected=corpus['records'][i]['request']
            self.assertEqual({k:v for k,v in request.items() if k not in ['model','return_meta_info','return_token_ids','return_prompt_token_ids']},
                             {k:v for k,v in expected.items() if k!='model'})
            n=corpus['records'][i]['captured_usage']['prompt_tokens']
            events.append(('chat',i))
            return {'choices':[{'prompt_token_ids':[i]*n,'meta_info':{'prompt_tokens':n,'cached_tokens':0}}], 'usage':{}},.1
        def http(route,body=None):
            events.append((route,body))
            if route=='/start_profile':
                (Path(body['output_dir'])/'fixture.trace.json').write_text(json.dumps({'traceEvents':[{'cat':'kernel','name':'fixture-only'}]}))
        def generate(name,body):
            fixed.append((name,body));events.append(('generate',name))
            return {'name':name,'meta_info':{'completion_tokens':body['sampling_params']['max_new_tokens']}}
        with tempfile.TemporaryDirectory() as folder, patch.object(runner,'OUT',Path(folder)), patch.object(runner,'ROWS',[]), patch.object(runner,'http',http),patch.object(runner,'generate',generate),patch.object(runner,'guarded',lambda fn:fn()):
            runner.experiments(corpus,Path('/model'),SimpleNamespace(post=post,inspect_response=lambda *a:{}))
            self.assertEqual(len(fixed),42)
            self.assertEqual(len(requests),6)
            for _,body in fixed:
                self.assertEqual(body['sampling_params']['top_k'],20)
                self.assertEqual(body['sampling_params']['temperature'],1)
            pair=events.index(('chat',12),events.index(('chat',13))+1)
            self.assertEqual(events[pair+1],('chat',13)) # no flush in receive gap
            batches=[json.loads(p.read_text()) for p in Path(folder).glob('batch-*.json')]
            self.assertEqual(len(batches),6)
            sets=[]
            for repeat in [0,1]:
                for c in [1,2,4]:
                    sets.append([b['input_ids'] for n,b in sorted(fixed,key=lambda row:row[0]) if n.startswith(f'batch-{c}-{repeat}-')])
            self.assertTrue(all(s==sets[0] for s in sets))

    def test_concurrency_is_real_and_finite(self):
        barrier=threading.Barrier(4,timeout=2)
        def call(name,body):
            barrier.wait()
            return {'name':name,'meta_info':{'completion_tokens':2}}
        result=runner.finite_batch([(str(i),{}) for i in range(4)],4,call)
        self.assertEqual(len(result['requests']),4)
        self.assertAlmostEqual(result['aggregate_generated_tok_s']*result['wall_s'],8)

    def test_profile_failure_aborts(self):
        def fail(route,body=None):
            if route=='/start_profile':raise RuntimeError('profiler unavailable')
        with tempfile.TemporaryDirectory() as folder,patch.object(runner,'OUT',Path(folder)),patch.object(runner,'http',fail):
            with self.assertRaisesRegex(RuntimeError,'profiler unavailable'):runner.profile(1024,[1])
            self.assertTrue((Path(folder)/'trace-1024'/'UNAVAILABLE.json').is_file())

    def test_failure_does_not_admit_queued_replacements(self):
        admissions=[]
        def call(name,body):
            admissions.append(name)
            raise RuntimeError('unknown retirement')
        with self.assertRaisesRegex(RuntimeError,'unknown retirement'):
            runner.finite_batch([(str(i),{}) for i in range(4)],1,call)
        self.assertEqual(admissions,['0'])

    def test_real_http_terminal_checks(self):
        state={'truncate':False}
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                meta={'prompt_tokens':len(body['input_ids'])-int(state['truncate']), 'completion_tokens':4,
                      'finish_reason':{'type':'length'},'decode_throughput':50}
                data=('data: '+json.dumps({'meta_info':meta})+'\n\ndata: [DONE]\n\n').encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        watchdog=runner.module('watchdog',str(build.HERE.parent/'mtp-sampling-repair-20260906'/'watchdog.py'))
        try:
            with tempfile.TemporaryDirectory() as folder,patch.object(runner,'OUT',Path(folder)),patch.object(runner,'BASE',f'http://127.0.0.1:{server.server_port}'),patch.object(runner,'PROCESS',SimpleNamespace(poll=lambda:None)):
                result=runner.generate('test',runner.payload([1,2,3],4))
                self.assertEqual(result['server_tokenizer_ms_per_token'],20)
                state['truncate']=True
                with self.assertRaisesRegex(RuntimeError,'truncated'):runner.generate('bad',runner.payload([1,2,3],4))
        finally: server.shutdown();server.server_close();thread.join()

    def test_dead_engine_stops_before_wait_timeout(self):
        runner.module('watchdog',str(build.HERE.parent/'mtp-sampling-repair-20260906'/'watchdog.py'))
        with patch.object(runner,'PROCESS',SimpleNamespace(poll=lambda:17)):
            with self.assertRaisesRegex(RuntimeError,'Engine exited'):runner.guarded(lambda:None)


if __name__=='__main__': unittest.main()
