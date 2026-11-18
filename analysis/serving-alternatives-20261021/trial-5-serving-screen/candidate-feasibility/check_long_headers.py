"""One real 62s delayed-header check, separate from fast transport tests."""
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import client
from test_client import SUCCESS


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']));time.sleep(62)
        raw=json.dumps(SUCCESS).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers()
        try:self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):pass
    def log_message(*args):pass


with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);request=root/'request.json';request.write_text('{}')
    case=dict(key='long:0',images=1,request_path=str(request),
              request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),local_prompt_tokens=120095)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    began=time.monotonic()
    try:
        got=client.run([case],'http://127.0.0.1:'+str(server.server_port),root/'out',began+68,per_request=66)
        print(json.dumps(dict(status=got['status'],elapsed_s=time.monotonic()-began,results=got['results'])),flush=True)
        assert got['status']=='FIT_FUNCTIONAL_ONLY','premature socket timeout before absolute request cap'
    finally:server.shutdown();server.server_close()
