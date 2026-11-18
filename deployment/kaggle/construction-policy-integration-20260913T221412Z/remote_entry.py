import json,os,sys,urllib.request
from pathlib import Path
root=Path(os.environ['CONSTRUCTION_ASSET_ROOT']);sys.path.insert(0,str(root))
from integration import FixtureTransport,IntegrationHost
from real_trace import load_case
import integration,integration_boundary,constructor,request_contract,real_trace
origins={m.__name__:str(Path(m.__file__).resolve()) for m in (integration,integration_boundary,constructor,request_contract,real_trace)}
if any(Path(p).parent!=root for p in origins.values()): raise RuntimeError('module origin drift:'+repr(origins))
C='''\ndef _o(o):\n g=o["grid"];q=[(r,c) for r,x in enumerate(g) for c,v in enumerate(x) if v==12];return [min(r for r,c in q),min(c for r,c in q)],next(c for c,v in enumerate(g[61]) if v==11)\ndef initialize(o): b,m=_o(o);return {"block":b,"bar_left":m}\ndef predict(s,a):\n d={"UP":(-5,0),"DOWN":(5,0),"LEFT":(0,-5),"RIGHT":(0,5)}.get(a,(0,0));n={"block":[s["block"][0]+d[0],s["block"][1]+d[1]],"bar_left":s["bar_left"]+1};return [{"state":n,"commitments":dict(n),"declared_dependencies":["block","bar_left"]}]\ndef assimilate(s,o,v): b,m=_o(o);return {"block":b,"bar_left":m}\n'''
W={'decision':'next move','obstacle':'transition uncertainty','operation':'test','claim':'joint transition','evidence_refs':[1,2],'dependencies':['block','bar_left'],'completion_checks':[1,2],'remaining_scope':'outside trace','reasoning_need':'needs_computation'}
def call(i,n,a):return {'id':'call-'+str(i),'type':'function','function':{'name':n,'arguments':json.dumps(a)}}
def response(call):return {'id':'fixture','provider':'fixture/provider','choices':[{'message':{'role':'assistant','content':None,'tool_calls':[call]}}],'usage':{'prompt_tokens':100,'completion_tokens':30,'completion_tokens_details':{'reasoning_tokens':0}}}
if os.environ.get('CONSTRUCTION_FIXTURE')=='1':
 transport=FixtureTransport([response(call(1,'execute_python',{'code':"import json;print(len(json.load(open('/work/prefix.json'))['visible_transitions']))"})),response(call(2,'execute_python',{'code':"import json;print(json.load(open('/work/prefix.json'))['frontier'])"})),response(call(3,'submit_candidate',{'work_item':W,'candidate_source':C}))])
else:
 class HTTP:
  def __call__(self,body):
   request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer local-model-no-key'})
   with urllib.request.urlopen(request,timeout=90) as r:return json.loads(r.read())
 transport=HTTP()
out=Path('/kaggle/working/construction-evidence');out.mkdir(parents=True,exist_ok=False);(out/'module-origins.json').write_text(json.dumps(origins,sort_keys=True)+'\n')
result=IntegrationHost(load_case(),transport,out).run();print(json.dumps(result,sort_keys=True))
