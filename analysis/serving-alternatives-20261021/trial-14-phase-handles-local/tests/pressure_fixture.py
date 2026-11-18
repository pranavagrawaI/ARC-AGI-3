"""Loopback HTTP readiness and owned sleeping frontend, no native/model imports."""
import json,os,subprocess,sys,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
operation=sys.argv[1];path=Path(sys.argv[2])
if operation=='http':
    class Handler(BaseHTTPRequestHandler):
        count=0
        def do_GET(self):
            Handler.count+=1;self.send_response(200 if Handler.count>=524 else 503)
            self.send_header('Content-Length','0');self.end_headers()
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',0),Handler);path.write_text(json.dumps(dict(port=server.server_port,pid=os.getpid())))
    server.serve_forever()
elif operation=='wrapper':
    child=subprocess.Popen([sys.executable,__file__,'frontend',str(path)])
    raise SystemExit(child.wait())
elif operation=='frontend':
    path.write_text(json.dumps(dict(pid=os.getpid())));time.sleep(300)
else:raise ValueError(operation)
