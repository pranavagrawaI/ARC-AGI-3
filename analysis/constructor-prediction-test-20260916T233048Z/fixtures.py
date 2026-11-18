"""MANUFACTURED FIXTURES ONLY. No model, source solution or target in real prompt."""
import json,threading
from http.server import BaseHTTPRequestHandler,HTTPServer
WORK={'decision':'fixture prediction','obstacle':'fixture uncertainty','operation':'test','claim':'fixture observable','evidence_refs':[1],'dependencies':['grid from observation'],'completion_checks':[1],'remaining_scope':'unobserved dynamics','reasoning_need':'needs_computation'}
IDENTITY='''def initialize(observation):
 return {"grid":observation["grid"]}
def predict(state,action):
 return [{"state":state,"commitments":{"type":"grid","predicted_grid":state["grid"]}}]
def assimilate(predicted_state,observation,verdict):
 return {"grid":observation["grid"]}
'''
def source_for(commitment_expression):
 return IDENTITY.replace('{"type":"grid","predicted_grid":state["grid"]}', '('+commitment_expression+') if action=="DOWN" else {"type":"grid","predicted_grid":state["grid"]}')
def reply(name,args,n):
 return {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'index':0,'id':'fixture-'+str(n),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}],'usage':{'prompt_tokens':100,'completion_tokens':100}}
def replies(source,inspection='import json\nprint(json.load(open("/evidence/case.json"))["frontier"])'):
 return [reply('execute_python',{'code':inspection},1),reply('submit_candidate',{'work_item':WORK,'candidate_source':source.splitlines()},2)]
class Server:
 def __init__(self,responses):self.responses=list(responses);self.requests=[]
 def __enter__(self):
  outer=self
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    outer.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
    raw=json.dumps(outer.responses.pop(0)).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
  self.server=HTTPServer(('127.0.0.1',0),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url=f'http://127.0.0.1:{self.server.server_port}/fixture';return self
 def __exit__(self,*a):self.server.shutdown();self.server.server_close();self.thread.join()
