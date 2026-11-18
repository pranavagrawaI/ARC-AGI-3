"""Explicit CPU phase/loopback fixture. No model, GPU or external network."""
import json,os,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
HERE=Path(__file__).resolve().parents[1]
out=Path(os.environ['SCREEN_OUTPUT']);out.mkdir(exist_ok=True)
phase=sys.argv[1]
if phase=='prepare':
    plan=json.loads((HERE/'candidate-plan.json').read_text())
    for name,value in [('installed-source-identity.json',plan['candidate_frontend_source_sha']),('model-source-identity.json',dict(version=plan['model']['model_version'],asset_sha256={key:value['sha256'] for key,value in plan['model']['assets'].items()})),('prepared.json',dict(CPU_ONLY=True)),('candidate-launch.json',dict(argv=[sys.executable,str(Path(__file__).resolve()),'serve'],image_max_pixels=409600,image_limit=58,sequences=10,native_overlay_enabled=False))]:
        (out/name).write_text(json.dumps(value))
    if os.environ.get('CPU_FAIL')=='1':raise SystemExit(9)
elif phase=='start':os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),'serve'])
elif phase=='serve':
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',8001),Handler).serve_forever()
elif phase in ('identity','corpus','replay','qualify'):pass
else:raise ValueError('CPU fixture mode')
