"""Manufactured provider responses, never empirical model/game efficacy evidence."""
import copy,json,threading
from http.server import BaseHTTPRequestHandler,HTTPServer
from common import MODEL
WORK={'decision':'fixture next action','obstacle':'fixture unknown','operation':'test','claim':'typed fixture','evidence_refs':[1,8],'dependencies':['grid from observation'],'completion_checks':[1,8],'remaining_scope':'unknown dynamics','reasoning_need':'needs_computation'}
SOURCE='''def initialize(observation):
 return {"grid":observation["grid"]}
def predict(state,action):
 return [{"state":{"grid":state["grid"],"actual_action":action},"commitments":{"type":"grid","predicted_grid":state["grid"]}}]
def assimilate(predicted_state,observation,verdict):
 return {"grid":observation["grid"],"last_action":predicted_state["actual_action"]}
'''
DETAILS=[{'type':'reasoning.text','text':'fixture deliberation','signature':'opaque','id':'r0','format':'unknown','index':0}]
def response(name,args,n=1,phase='control'):
 msg={'role':'assistant','content':None,'tool_calls':[{'index':0,'id':f'fixture-{n}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
 if phase=='constructor':msg['reasoning_details']=copy.deepcopy(DETAILS)
 return {'id':f'fixture-{n}','provider':'Reka','model':MODEL,'choices':[{'index':0,'finish_reason':'tool_calls','message':msg}],'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150,'completion_tokens_details':{'reasoning_tokens':10 if phase=='constructor' else 0},'cost':0.0001489},'openrouter_metadata':{'requested':MODEL,'strategy':'direct','attempt':1,'pipeline':[],'endpoints':{'available':[{'provider':'Reka','model':MODEL,'selected':True}]}}}
def construction(source=SOURCE,repair=False):
 a=[response('execute_python',{'code':'import json\nc=json.load(open("/evidence/case.json"))\nprint(c["frontier"])'},1,'constructor')]
 if repair:a.append(response('submit_candidate',{'work_item':'wrong type','candidate_source':source.splitlines()},2,'constructor'))
 a.append(response('submit_candidate',{'work_item':WORK,'candidate_source':source.splitlines()},len(a)+1,'constructor'));return a
class Server:
 def __init__(self,replies):self.replies=list(replies);self.requests=[]
 def __enter__(self):
  outer=self
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    outer.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
    reply=outer.replies.pop(0) if outer.replies else (500,{'error':'fixture exhausted'})
    status,reply=reply if isinstance(reply,tuple) else (200,reply)
    raw=reply if isinstance(reply,bytes) else json.dumps(reply).encode()
    self.send_response(status);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
  self.server=HTTPServer(('127.0.0.1',0),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url=f'http://127.0.0.1:{self.server.server_port}/chat';return self
 def __exit__(self,*a):self.server.shutdown();self.server.server_close();self.thread.join()
