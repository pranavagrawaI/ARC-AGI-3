"""Real constructor -> replay -> outbound HTTP -> sandbox -> native level completion."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import copy,json,subprocess,sys,threading
import worker

HERE=Path(__file__).resolve().parent
route=json.loads((HERE/'evidence/fixture-code.json').read_text())
class Handler(BaseHTTPRequestHandler):
    mode='success';bodies=[]
    def log_message(self,*_):pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));self.bodies.append(body)
        reply={'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':'cpu_fixture','index':0,
               'type':'function','function':{'name':'python','arguments':json.dumps({'code':"print('restored helper', callable(growS))" if len(self.bodies)==1 else route})}}]},'finish_reason':'tool_calls'}],
               'usage':{'prompt_tokens':100,'completion_tokens':1,'total_tokens':101}}
        if self.mode=='error':reply={'error':{'message':'intentional CPU qualification failure'}}
        raw=json.dumps(reply).encode();self.send_response(200 if self.mode=='success' else 400)
        self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
results=[]
for mode in ['success','error']:
    Handler.mode=mode;Handler.bodies=[]
    out=HERE/'evidence'/f'http-{mode}-v2'
    cmd=[sys.executable,'-B',str(HERE/'worker.py'),'--checkpoint','b1','--arm','counterexample','--mode','live',
         '--output',str(out),'--url',f'http://127.0.0.1:{server.server_port}/v1']
    done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    (HERE/'evidence'/f'http-{mode}-v2.log').write_bytes(done.stdout)
    r=json.loads((out/'result.json').read_text())
    assert done.returncode==0 and len(Handler.bodies)==(2 if mode=='success' else 1),(r,done.stdout[-1000:])
    expected=copy.deepcopy(json.loads((HERE/'inputs/b1.json').read_text())['branch']['messages'])
    feedback=json.loads((HERE/'inputs/feedback.json').read_text())['b1']['counterexample']
    expected.append({'role':'user','content':feedback})
    # Original transport strips this internal harness marker before sending.
    for message in expected:message.pop('_arc3_control',None)
    assert Handler.bodies[0]['messages']==expected
    assert Handler.bodies[0]['seed']==1701
    if mode=='success':
        assert r['status']=='solved' and r['levels_completed']==2 and r['total_actions']==73
        assert r['generated_tokens']==2 and r['unresolved'] is None
    else:
        assert r['status']=='http_error' and r['generated_tokens'] is None and r['unresolved']
        assert r['total_actions']==42
    results.append({'case':mode,'status':r['status'],'wire_messages_exact':True,'calls':r['new_model_calls'],
                   'native_actions':r['total_actions'],'generated_tokens':r['generated_tokens']})
worker.write(HERE/'evidence/e2e-qualification.json',{'cpu_fixtures_only':True,'results':results})
print(json.dumps(results,indent=2));server.shutdown()
