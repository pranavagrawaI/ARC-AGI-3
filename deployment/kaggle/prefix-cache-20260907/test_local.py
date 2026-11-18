import ast
import base64
import http.server
import json
import lzma
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import build
import runner
import assess

HERE=Path(__file__).resolve().parent

class LocalChecks(unittest.TestCase):
    def test_embedded_identity(self):
        files=build.build()
        for name,data in files.items():self.assertEqual((HERE/name).read_bytes(),data)
        code=''.join(json.loads(files['prefix_cache.ipynb'])['cells'][2]['source'])
        # Assets are LZMA-compressed to fit Kaggle's 1 MiB kernel-source limit;
        # the inherited donor preamble still embeds plain base64. Accept both, so
        # this still asserts byte-identity of the artifact actually written.
        raw=[base64.b64decode(n.args[0].value) for n in ast.walk(ast.parse(code))
             if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode' and n.args and isinstance(n.args[0],ast.Constant)]
        decoded=[]
        for blob in raw:
            try:decoded.append(lzma.decompress(blob))
            except lzma.LZMAError:decoded.append(blob)
        for name in ['runner.py','baseline.py','corpus.json','assess.py']:
            self.assertIn((HERE/name).read_bytes(),decoded)
        self.assertEqual(json.loads(files['manifest.json'])['request_cap'],runner.REQUEST_CAP)

    def test_request_cap_is_hard(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'OUT',Path(d)),patch.object(runner,'REQUEST_COUNT',147):
            runner.admit_request()
            with self.assertRaisesRegex(RuntimeError,'cap reached'):runner.admit_request()
            self.assertEqual(json.loads((Path(d)/'request-count.json').read_text())['admitted'],148)

    def test_four_arms_total_and_canonical_inputs(self):
        corpus=json.loads((HERE/'corpus.json').read_text())
        count={'generate':0,'chat':0}; rows={}
        def chat(name,record,*args):
            count['chat']+=1
            value=int(record['name'].split('-')[-1])
            return [42]*20+[value]
        def gen(name,body):
            count['generate']+=1
            rows.setdefault(name,[]).append(body)
            return {'name':name,'meta_info':{'prompt_tokens':len(body['input_ids']),
                    'completion_tokens':body['sampling_params']['max_new_tokens'],'cached_tokens':0},
                    'cached_input_tokens':0,'server_forward_to_prefill_finished_s':1.0}
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'OUT',Path(d)),patch.object(runner,'REQUEST_COUNT',0),patch.object(runner.base,'OUT',Path(d)),patch.object(runner.base,'http'),patch.object(runner.base,'chat',chat),patch.object(runner.base,'generate',gen):
            rendered=None
            for i in range(4):rendered=runner.experiments(corpus,Path('/model'),None,rendered,reverse=bool(i%2))
            self.assertEqual(count,{'generate':112,'chat':36})
            self.assertEqual(runner.REQUEST_COUNT,148)
            self.assertTrue(all(all(x==values[0] for x in values) for values in rows.values()))

    def test_rendered_drift_stops(self):
        corpus=json.loads((HERE/'corpus.json').read_text())
        expected={r['name']:[0] for p in corpus['pairs'] for r in [p['seed'],p['successor']]}
        expected.update({r['name']:[0] for r in corpus['batch_records']})
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'OUT',Path(d)),patch.object(runner.base,'OUT',Path(d)),patch.object(runner,'REQUEST_COUNT',0),patch.object(runner.base,'http'),patch.object(runner.base,'chat',return_value=[1]):
            with self.assertRaisesRegex(RuntimeError,'Cross-arm'):runner.render_records(corpus,Path('/model'),None,expected)

    def test_real_stream_and_bad_cache(self):
        missing=[False]
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                n=body['sampling_params']['max_new_tokens']
                m={'prompt_tokens':len(body['input_ids']),'completion_tokens':n,
                   'finish_reason':{'type':'length'},'decode_throughput':40,
                   'forward_entry_time':1.,'prefill_finished_time':2.}
                if not missing[0]:m['cached_tokens']=0
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                self.wfile.write(('data: '+json.dumps({'meta_info':m,'output_ids':[7]*n,'text':'fixture'})+'\n\ndata: [DONE]\n\n').encode())
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d,patch.object(runner,'OUT',Path(d)),patch.object(runner,'REQUEST_COUNT',0),patch.object(runner.base,'OUT',Path(d)),patch.object(runner.base,'BASE',f'http://127.0.0.1:{server.server_port}'),patch.object(runner.base,'guarded',lambda f:f()):
                result=runner.checked_generate('valid',runner.fixed_payload([1,2]))
                self.assertEqual(result['cached_input_tokens'],0)
                self.assertIsNotNone(result['output_fingerprint']['token_ids_sha256'])
                missing[0]=True
                with self.assertRaisesRegex(RuntimeError,'cached-token'):runner.checked_generate('invalid',runner.fixed_payload([1,2]))
                self.assertTrue((Path(d)/'invalid-stream.jsonl').exists())
        finally:server.shutdown();server.server_close();thread.join()

    def test_assessment_rejects_partial_run(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'arm-0-chunk-2048';path.mkdir()
            (path/'lifecycle.json').write_text(json.dumps({'status':'passed'}))
            with self.assertRaisesRegex(ValueError,'Incomplete arm'):assess.assess(d)

    def test_assessment_surfaces_cache_tradeoff_without_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i,c in enumerate(runner.CHUNKS):
                path=root/f'arm-{i}-chunk-{c}';path.mkdir()
                def save(name,obj):
                    (path/name).write_text(json.dumps(obj))
                save('lifecycle.json',{'status':'passed'})
                save('resolved-policy.json',{'chunked_prefill_size':c})
                save('natural-results.json',[{'name':str(n),'usage':{'completion_tokens':n+1}} for n in range(9)])
                for n in range(6):
                    name=f'case{n}'
                    successor={'name':name,'cached_input_tokens':8 if c==2048 else 16,
                               'client_e2e_s':3.,'server_forward_to_prefill_finished_s':1.}
                    save(name+'-condition.json',{'successor':successor,'condition_wall_s':5.})
                    save(name+'-request.json',runner.fixed_payload([1,2,3]))
                for n in range(4):
                    save(f'batch-{n}.json',{'input_sha256':{'case':'fixture'},'wall_s':6.,
                        'requests':[{'name':str(j),'ready_to_completed_s':j+1.,'cached_input_tokens':0} for j in range(4)],
                        'completed_within_180s_from_ready':4})
            report=assess.assess(root)
            self.assertFalse(report['promotion'])
            self.assertEqual(report['conditions'][0]['cached_tokens_abba'],[8,16,16,8])
            self.assertFalse(report['conditions'][0]['cache_counts_match'])
            self.assertEqual(report['conditions'][0]['output_equality'],'unknown')
            bad=root/'arm-3-chunk-2048/case0-request.json'
            bad.write_text(json.dumps(runner.fixed_payload([9])))
            with self.assertRaisesRegex(ValueError,'input drift'):assess.assess(root)

if __name__=='__main__':unittest.main()
