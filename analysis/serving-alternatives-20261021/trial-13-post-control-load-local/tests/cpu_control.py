"""Trusted CPU-only foreground registered control fixture; no GPU/model/imports."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import registration
ROOT=HERE.parent/'shutdown-release-prototype-r2-local-20261023'
out=Path(os.environ['SCREEN_OUTPUT']);run=os.environ['SCREEN_RUN_ID'];operation=sys.argv[1]
if operation=='prepare':
    out.mkdir(exist_ok=True);entry=out/'cpu-entry';log=out.parent/'native'/run/'native.stderr'
    entry.write_text(f'''#!{sys.executable}
import json,sys,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
sys.path.insert(0,{str(ROOT)!r})
import native_fixture
log=Path({str(log)!r})
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  rows=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
  self.send_response(200 if any(r['event']=='frontend_ready' for r in rows) else 503)
  self.send_header('Content-Length','0');self.end_headers()
 def log_message(self,*args):pass
server=HTTPServer(('127.0.0.1',{int(sys.argv[2])}),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
time.sleep(.15)
native_fixture.frontend(str(log))
''');entry.chmod(0o700)
    (out/'control-prepared.json').write_text(json.dumps(dict(python=sys.executable,argv=[str(entry)])))
    source={str(entry):registration.sha(entry),str(ROOT/'native_fixture.py'):registration.sha(ROOT/'native_fixture.py')}
    (out/'cpu-sources.json').write_text(json.dumps(source))
elif operation=='identity':(out/'cpu-identity.json').write_text(json.dumps(dict(CPU_ONLY=True)))
elif operation=='serve':
    data=json.loads((out/'control-prepared.json').read_text());child=subprocess.Popen(data['argv'],stdin=subprocess.DEVNULL,start_new_session=False)
    registration.emit(child.pid,out/'control-prepared.json',json.loads((out/'cpu-sources.json').read_text()),run,out/'control-frontend-registration.json',time.monotonic()+2)
    raise SystemExit(child.wait())
else:raise ValueError(operation)
