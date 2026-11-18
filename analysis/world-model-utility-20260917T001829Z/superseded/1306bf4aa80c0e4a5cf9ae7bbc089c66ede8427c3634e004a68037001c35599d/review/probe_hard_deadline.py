"""No-auth loopback reproduction: slow provider body outlives supervisor deadline."""
import sys,json,time,threading,http.server
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'engine')]
from run_utility import supervise
from transport import Transport,initial_ledger
from fixtures import response
from actor import SYSTEM,ACT
out=Path(__file__).resolve().parent/'hard-deadline-probe';out.mkdir()
body={'model':'qwen/qwen3.8-27b','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':'local deadline fixture'}],'tools':[ACT],'tool_choice':'required','provider':{},'max_tokens':1024,'reasoning':{'enabled':False}}
raw=json.dumps(response('act',{'action':'RIGHT','notes':'fixture'},20)).encode()
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']))
  self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers()
  # Every read makes progress well inside the per-socket timeout, but total exceeds hard.
  width=(len(raw)+47)//48
  for i in range(0,len(raw),width):
   time.sleep(.5);self.wfile.write(raw[i:i+width]);self.wfile.flush()
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
ledger=out/'ledger.json';initial_ledger(ledger)
t=Transport('control',out/'raw',ledger,time.monotonic()+20.5,fixture_url=f'http://127.0.0.1:{server.server_port}/chat')
code='import json,sys,time; print('+repr(json.dumps({'request':body}))+',flush=True); sys.stdin.readline(); time.sleep(30)'
try:
 r=supervise('control',out/'supervised','-','-',t,hard=21,command=[sys.executable,'-B','-c',code])
 print(json.dumps({'fixture_only':True,'declared_hard_seconds':21,'receipt':r,'exceeded_hard_seconds':r['elapsed_seconds']-21,'ledger':json.loads(ledger.read_text())},indent=2))
finally:server.shutdown();server.server_close()
