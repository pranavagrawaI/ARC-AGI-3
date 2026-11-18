import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'sources'))
import client


SUCCESS=dict(id='actual-response',usage=dict(prompt_tokens=120095,completion_tokens=5),
             choices=[dict(finish_reason='tool_calls',message=dict(tool_calls=[dict(function=dict(name='python',arguments='{"code":"print(1)"}'))]))])


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.body=self.root/'body.json';self.body.write_text('{"model":"flashnext"}')
        self.case=dict(key='su15:35',images=50,request_path=str(self.body),
                       request_sha256=hashlib.sha256(self.body.read_bytes()).hexdigest(),local_prompt_tokens=120095)

    def tearDown(self):self.tmp.cleanup()

    def server(self,status=200,raw=None,trickle=False):
        body=json.dumps(SUCCESS).encode() if raw is None else raw
        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler):
                handler.rfile.read(int(handler.headers['Content-Length']))
                handler.send_response(status);handler.send_header('Content-Length',str(len(body)));handler.end_headers()
                try:
                    if trickle:
                        for byte in body:
                            handler.wfile.write(bytes([byte]));handler.wfile.flush();time.sleep(.05)
                    else:handler.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
            def log_message(*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        return 'http://127.0.0.1:'+str(server.server_port)

    def test_actual_success_keeps_raw_usage_and_no_rate_claim(self):
        endpoint=self.server();got=client.run([self.case],endpoint,self.root/'out',time.monotonic()+3)
        self.assertEqual(got['status'],'FIT_FUNCTIONAL_ONLY')
        self.assertEqual(got['results'][0]['usage']['completion_tokens'],5)
        self.assertIsNone(got['rate']);self.assertFalse(got['adoption'])

    def test_http_error_preserves_exact_status_and_body(self):
        raw=b'{"error":{"message":"actual OOM","detail":"full retained"}}'
        got=client.run([self.case],self.server(400,raw),self.root/'out',time.monotonic()+3)
        self.assertEqual(got['status'],'FEASIBILITY_FAILED')
        p=self.root/'out/su15-35';self.assertEqual((p/'response.bin').read_bytes(),raw)
        self.assertEqual(json.loads((p/'http.json').read_text())['status'],400)

    def test_absolute_cutoff_reaps_trickling_success_and_error(self):
        for status in (200,500):
            with self.subTest(status=status):
                out=self.root/str(status);began=time.monotonic()
                got=client.run([self.case],self.server(status,b'x'*100,trickle=True),out,began+.35,per_request=.3)
                self.assertLess(time.monotonic()-began,1.5)
                self.assertEqual(got['status'],'FEASIBILITY_FAILED')
                self.assertTrue((out/'su15-35/cutoff.json').is_file())

    def test_corrupted_raw_body_cannot_reuse_declared_wire_hash(self):
        endpoint=self.server();root=self.root/'run'
        baseline=client.run([self.case],endpoint,root,time.monotonic()+3)
        self.assertEqual(baseline['status'],'FIT_FUNCTIONAL_ONLY')
        out=root/'su15-35'
        (out/'request.json').write_text('{"changed":true}')
        got=client.assess(out,self.case)
        self.assertFalse(got['ok'])

    def test_missing_or_zero_usage_fails_closed(self):
        value=dict(SUCCESS,usage=None)
        got=client.run([self.case],self.server(raw=json.dumps(value).encode()),self.root/'out',time.monotonic()+3)
        self.assertEqual(got['status'],'FEASIBILITY_FAILED')

    def test_short_prompt_receipt_cannot_qualify_long_input_fit(self):
        value=dict(SUCCESS,usage=dict(prompt_tokens=1,completion_tokens=5))
        got=client.run([self.case],self.server(raw=json.dumps(value).encode()),self.root/'out',time.monotonic()+3)
        self.assertEqual(got['status'],'FUNCTIONAL_PROMPT_MISMATCH')
        self.assertFalse(got['long_input_fit_qualified'])

    def test_valid_json_in_truncated_http_body_is_incomplete(self):
        body=json.dumps(SUCCESS).encode()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler):
                handler.rfile.read(int(handler.headers['Content-Length']))
                handler.send_response(200);handler.send_header('Content-Length',str(len(body)+100));handler.end_headers()
                handler.wfile.write(body);handler.wfile.flush();handler.close_connection=True
            def log_message(*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        got=client.run([self.case],'http://127.0.0.1:'+str(server.server_port),self.root/'out',time.monotonic()+3)
        self.assertEqual(got['status'],'FEASIBILITY_FAILED')

    def test_first_http_failure_stops_new_case_dispatch(self):
        cases=[dict(self.case,key='case:'+str(i)) for i in range(4)]
        got=client.run(cases,self.server(500,b'{"error":"stop"}'),self.root/'out',time.monotonic()+3)
        self.assertEqual(got['status'],'FEASIBILITY_FAILED')
        self.assertEqual(set(got['unattempted']),{'case:2','case:3'})
        self.assertFalse((self.root/'out/case-2').exists())

    def test_error_headers_stop_admission_while_error_body_drains(self):
        received=[];error_body=b'{"error":"delayed full error body"}'
        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler):
                handler.rfile.read(int(handler.headers['Content-Length']))
                rid=handler.headers['X-Request-Id'];received.append(rid)
                failed=rid.endswith('-0')
                value=dict(SUCCESS,id=rid)
                body=error_body if failed else json.dumps(value).encode()
                handler.send_response(500 if failed else 200)
                handler.send_header('Content-Length',str(len(body)));handler.end_headers();handler.wfile.flush()
                if failed:time.sleep(1)
                try:handler.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
            def log_message(*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        cases=[dict(self.case,key='case:'+str(i)) for i in range(6)]
        got=client.run(cases,'http://127.0.0.1:'+str(server.server_port),self.root/'out',time.monotonic()+3,per_request=2)
        self.assertEqual(set(received),{'serving-feas6-case-0','serving-feas6-case-1'})
        self.assertEqual(set(got['unattempted']),{'case:2','case:3','case:4','case:5'})
        self.assertEqual((self.root/'out/case-0/response.bin').read_bytes(),error_body)
        self.assertEqual(got['status'],'FEASIBILITY_FAILED')


if __name__=='__main__':unittest.main()
