"""Loopback load-to-health fixture; never a model or inference request."""
import json,os,sys,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
out=Path(os.environ['SCREEN_OUTPUT']);out.mkdir(exist_ok=True)
if sys.argv[1]!='serve':
 (out/(sys.argv[1]+'.json')).write_text(json.dumps(dict(CPU_ONLY=True,operation=sys.argv[1])));raise SystemExit(0)
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
 def log_message(self,*args):pass
HTTPServer(('127.0.0.1',int(sys.argv[2])),Handler).serve_forever()
