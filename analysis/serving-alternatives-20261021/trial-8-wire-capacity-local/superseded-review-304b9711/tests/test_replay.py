import http.server
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import io
import catalog
import replay
import reduce_api

class Server(http.server.ThreadingHTTPServer):
    daemon_threads=True
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        body=self.rfile.read(int(self.headers['Content-Length']));value=json.loads(body)
        rid=self.headers['X-Request-Id'];self.server.seen.append((rid,body,time.monotonic()))
        idx=int(rid.rsplit('-',1)[1]);mode=self.server.mode
        data=json.dumps({'id':rid,'usage':{'prompt_tokens':999,'completion_tokens':5,'total_tokens':1004,'prompt_tokens_details':{'image_tokens':0}},'choices':[{'finish_reason':'length','message':{'content':'hello'}}]}).encode()
        status=500 if mode=='error' and idx==0 else 200
        if mode=='error' and idx!=0:self.server.error_headers.wait(1)
        self.send_response(status);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.flush()
        if status==500:self.server.error_headers.set();time.sleep(.35)
        if mode=='slow':
            for byte in data:
                try:self.wfile.write(bytes([byte]));self.wfile.flush();time.sleep(.02)
                except (OSError,ValueError):break
        else:
            try:self.wfile.write(data)
            except (OSError,ValueError):pass

class ReplayTests(unittest.TestCase):
    def run_fixture(self,mode,seconds=(.2,.4)):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        bodies=root/'bodies';bodies.mkdir();index={};records={};streams={}
        for g in ('a','b','c'):
            streams[g]=[]
            for i in range(4):
                k=f'{g}:{i}';raw=catalog.canonical({'model':'flashnext','messages':[{'role':'user','content':k}],'max_tokens':5})
                name=f'{g}-{i}.json';(bodies/name).write_bytes(raw);index[k]={'file':name,'sha256':catalog.digest(raw)}
                streams[g].append(k);records[k]={'extends_previous':True}
        (bodies/'index.json').write_text(json.dumps(index))
        cat={'initial_holders':['a','b'],'initial_queue':['c'],'records':records,'streams':streams}
        server=Server(('127.0.0.1',0),Handler);server.seen=[];server.mode=mode;server.error_headers=threading.Event()
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        obj=replay.Replay(cat,bodies,root/'out',f'http://127.0.0.1:{server.server_port}','control',seconds=seconds,identity={'session_id':'CPU_ONLY'},slots=2)
        start=time.monotonic();ok=obj.run();elapsed=time.monotonic()-start
        events=list(map(json.loads,(root/'out/events.jsonl').read_text().splitlines()))
        return root,server,obj,events,ok,elapsed
    def test_actual_wire_and_fixed_cutoff(self):
        root,server,obj,events,ok,elapsed=self.run_fixture('ok')
        self.assertTrue(ok);self.assertLess(elapsed,1.5)
        for event in events:
            if event['kind']=='send':
                actual=reduce_api.transport((root/'out'/event['transport_ref']).read_bytes(),event['rid'])
                planned=json.loads((root/'out'/event['wire_ref']).read_bytes())['body']
                self.assertEqual(actual,planned);self.assertLess(event['t'],obj.end)
        self.assertEqual([e['kind'] for e in events].count('measure_start'),1)
        self.assertEqual(events[-1]['threads_alive'],0)
    def test_error_headers_stop_admission_while_body_drains(self):
        root,server,obj,events,ok,elapsed=self.run_fixture('error',(.1,.8))
        self.assertFalse(ok)
        self.assertEqual(len(server.seen),2)
        errors=[e for e in events if e['kind']=='error'];self.assertTrue(errors)
        self.assertFalse(any(e['kind']=='send' and e['t']>errors[0]['t'] for e in events))
        rid=next(r for r,_,_ in server.seen if r.endswith('-0'))
        receipt=json.loads((root/'out'/rid/'http.json').read_text())
        self.assertEqual(receipt['status'],500);self.assertTrue(receipt['complete'])
        self.assertTrue((root/'out'/rid/'body.bin').read_bytes())
    def test_trickle_body_cutoff_no_late_success(self):
        root,server,obj,events,ok,elapsed=self.run_fixture('slow',(.1,.15))
        self.assertTrue(ok);self.assertFalse(obj.failed)
        self.assertLess(elapsed,1.5);self.assertEqual(events[-1]['threads_alive'],0)
        self.assertFalse(any(e['kind']=='response' for e in events))
        self.assertEqual(len([e for e in events if e['kind']=='cutoff_pending']),2)
        for path in (root/'out').glob('*/http.json'):
            receipt=json.loads(path.read_text())
            self.assertEqual(receipt['termination'],'INTENTIONAL_CUTOFF')
            self.assertEqual(receipt['error_kind'],'INTENTIONAL_CUTOFF')
            self.assertIsNone(receipt['body_completed_at'])
            reduce_api.pending_http(receipt,obj.end,lambda ref:(root/'out'/ref).read_bytes(),receipt['rid'])
    def test_fail_closed_truncated_wire(self):
        with self.assertRaises(ValueError):reduce_api.transport(b'broken','x')
    def test_delayed_connect_cannot_write_after_cutoff(self):
        import http.client
        class Socket:
            def __init__(self):self.writes=[]
            def sendall(self,data):self.writes.append(time.monotonic())
        sock=Socket()
        def connect(connection):time.sleep(.05);connection.sock=sock
        conn=replay.CaptureConnection('localhost',trace=io.BytesIO(),deadline=time.monotonic()+.02)
        with patch.object(http.client.HTTPConnection,'connect',connect):
            with self.assertRaises(TimeoutError):conn.send(b'hello')
        self.assertEqual(sock.writes,[])
    def test_socket_write_crossing_cutoff_is_invalid_and_retained(self):
        class Socket:
            def sendall(self,data):time.sleep(.04)
        conn=replay.CaptureConnection('localhost',trace=io.BytesIO(),deadline=time.monotonic()+.02)
        conn.sock=Socket()
        with self.assertRaisesRegex(TimeoutError,'crossed cutoff'):conn.send(b'hello')
        self.assertEqual(conn.writes[0]['bytes'],5)
        self.assertGreaterEqual(conn.writes[0]['ended'],conn.deadline)
