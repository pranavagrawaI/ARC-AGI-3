"""Real HTTP boundary fixtures: receipts, budgets, errors and absolute deadlines."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,tempfile,threading,time,signal
import worker

class Handler(BaseHTTPRequestHandler):
    mode='success'
    def log_message(self,*_):pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.mode.startswith('slow'):
            self.send_response(400 if self.mode=='slow_error' else 200)
            self.send_header('Content-Length','100');self.end_headers()
            try:
                for _ in range(100):self.wfile.write(b' ');self.wfile.flush();time.sleep(.05)
            except (BrokenPipeError,ConnectionResetError):pass
            return
        result={'choices':[{'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}],
                'usage':{'completion_tokens':min(2,body['max_tokens']),'prompt_tokens':1}}
        if self.mode=='missing_usage':result.pop('usage')
        data=json.dumps(result).encode()
        self.send_response(400 if self.mode=='error' else 200)
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
url=f'http://127.0.0.1:{server.server_port}/v1'
results=[]
for mode in ['success','error','missing_usage','slow_success','slow_error']:
    Handler.mode=mode
    path=Path(tempfile.mkdtemp(prefix='continuation-transport-'))
    transport=worker.Transport(path,url,1701,token_cap=3,request_cap=3,episode_seconds=4,request_seconds=.3)
    start=time.monotonic();status='ok'
    try:
        transport.post(url+'/chat/completions',json={'max_tokens':100},timeout=2)
        if mode=='success':
            transport.post(url+'/chat/completions',json={'max_tokens':100},timeout=2)
            assert transport.tokens==3
            transport.post(url+'/chat/completions',json={'max_tokens':100},timeout=2)
    except worker.Stop as e:status=e.status
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    elapsed=time.monotonic()-start
    want={'success':'token_budget','error':'http_error','missing_usage':'invalid_usage',
          'slow_success':'request_deadline','slow_error':'request_deadline'}[mode]
    assert status==want,(mode,status)
    if mode.startswith('slow'):assert elapsed<1.0 and transport.unresolved
    lines=[json.loads(l) for l in (path/'wire.jsonl').read_text().splitlines()]
    assert len([x for x in lines if x['event']=='request'])==transport.calls
    if mode!='success':assert any(x['event']=='failure' and x['latency_seconds']>=0 for x in lines)
    results.append({'case':mode,'status':status,'seconds':elapsed,'calls':transport.calls,'tokens':transport.tokens,'unresolved':transport.unresolved})
worker.write(worker.restore.HERE/'evidence/transport-qualification.json',results)
print(json.dumps(results,indent=2))
server.shutdown()
