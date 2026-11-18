import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import transport
import http_error_capture as capture

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent/'packet-r2/serving-candidate-feas5-20261023.ipynb'


class Qualification(unittest.TestCase):
    def test_exact_bytes_and_all_embedded_sources(self):
        raw=ORIGINAL.read_bytes();wrapped=transport.wrap(raw)
        self.assertEqual(transport.decode(wrapped),raw)
        values=transport.literals(transport.cell_source(json.loads(transport.decode(wrapped))))
        expected=json.loads((HERE.parent/'review-packet.json').read_text())
        self.assertIsNone(values['ASSIGNMENT'])
        self.assertEqual(hashlib.sha256(json.dumps(values['CONFIG'],sort_keys=True).encode()).hexdigest(),expected['config_sha256'])
        hashes={}
        for name,record in values['PAYLOAD'].items():
            decoded=base64.b64decode(record['base64'],validate=True)
            self.assertEqual(hashlib.sha256(decoded).hexdigest(),record['sha256'])
            hashes[name]=record['sha256']
        self.assertEqual(hashes,expected['files'])
        self.assertEqual(hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),expected['sources_sha256'])
        self.assertLess(len(wrapped),1_000_000)  # local size target; not a claim about Kaggle's current limit

    def test_actual_unassigned_wrapper_refuses_before_extraction(self):
        source=transport.cell_source(json.loads((HERE/'transport-unassigned.ipynb').read_bytes()))
        harness='''import pathlib, subprocess, sys
from unittest.mock import patch
source=sys.stdin.read()
with patch.object(pathlib.Path,'mkdir',side_effect=AssertionError('extraction attempted')), patch.object(pathlib.Path,'open',side_effect=AssertionError('extraction attempted')), patch.object(pathlib.Path,'write_text',side_effect=AssertionError('extraction attempted')), patch.object(subprocess,'Popen',side_effect=AssertionError('process attempted')):
    try: exec(compile(source,'<actual-wrapper>','exec'))
    except RuntimeError as e:
        if str(e) != 'PREPARATION ONLY: no reviewed serialized assignment; do not execute GPU work': raise
        print(str(e))
    else: raise AssertionError('unassigned wrapper did not refuse')
'''
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,'-c',harness],input=source,text=True,capture_output=True,cwd=directory,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PREPARATION ONLY',result.stdout)

    def test_corrupted_transport_fails_digest(self):
        wrapped=(HERE/'transport-unassigned.ipynb').read_bytes()
        original=transport.sha(ORIGINAL.read_bytes()).encode()
        with self.assertRaises(ValueError): transport.decode(wrapped.replace(original,b'0'*64))

    def test_armed_source_is_not_preparation_input(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'UNASSIGNED'):
                transport.build_unassigned(HERE.parent/'kernel/serving-candidate-feas5-20261023.ipynb',Path(directory)/'bad.ipynb')
            self.assertFalse((Path(directory)/'bad.ipynb').exists())

    def test_actual_http400_retains_body_and_status_without_credentials(self):
        import threading
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        import requests
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                body=b'{"error":{"code":400,"message":"notebook rejected"}}'
                self.send_response(400);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)));self.send_header('Set-Cookie','private=credential')
                self.send_header('X-Request-Id','fixture-request');self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            try:
                r=requests.get('http://127.0.0.1:'+str(server.server_port),headers={'Authorization':'Bearer sensitive-fixture'},timeout=2)
                r.raise_for_status();self.fail('expected HTTPError')
            except requests.HTTPError as error: record=capture.exception_record(error)
        finally:server.shutdown();server.server_close();thread.join()
        self.assertEqual(record['status'],400)
        self.assertEqual(json.loads(record['body'])['error']['message'],'notebook rejected')
        self.assertEqual(record['headers']['X-Request-Id'],'fixture-request')
        self.assertFalse(record['body_truncated'])
        self.assertNotIn('private=credential',json.dumps(record));self.assertNotIn('sensitive-fixture',json.dumps(record))

    def test_bounded_response_redaction_and_no_response(self):
        import requests
        response=requests.Response();response.status_code=400
        response._content=b'{"token":"secret-fixture","url":"https://host/path?secret=private"}'+b'x'*70000
        record=capture.exception_record(requests.HTTPError('Bearer private-fixture',response=response))
        self.assertTrue(record['body_truncated']);self.assertEqual(record['body_bytes'],len(response.content))
        for secret in ('secret-fixture','private-fixture','secret=private'):self.assertNotIn(secret,json.dumps(record))
        self.assertEqual(capture.exception_record(RuntimeError('local failure'))['http_response'],'UNAVAILABLE')


if __name__=='__main__':
    manifest=transport.build_unassigned(ORIGINAL,HERE/'transport-unassigned.ipynb')
    (HERE/'transport-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    unittest.main(verbosity=2)
