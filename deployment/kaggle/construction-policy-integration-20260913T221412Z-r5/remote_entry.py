import argparse,hashlib,json,os,sys
from pathlib import Path

root=Path(os.environ['CONSTRUCTION_ASSET_ROOT']);sys.path.insert(0,str(root))
from integration import FixtureTransport,IntegrationHost,write_json
from constructor import ConstructionLimits
from local_transport import LocalHTTPTransport
from real_trace import load_case
from request_contract import LOCAL_MODEL
import integration,integration_boundary,constructor,local_transport,request_contract,real_trace

origins={m.__name__:str(Path(m.__file__).resolve()) for m in (integration,integration_boundary,constructor,local_transport,request_contract,real_trace)}
if any(Path(path).parent!=root for path in origins.values()): raise RuntimeError('module origin drift:'+repr(origins))
C='''\ndef _o(o):\n g=o["grid"];q=[(r,c) for r,x in enumerate(g) for c,v in enumerate(x) if v==12];return [min(r for r,c in q),min(c for r,c in q)],next(c for c,v in enumerate(g[61]) if v==11)\ndef initialize(o): b,m=_o(o);return {"block":b,"bar_left":m}\ndef predict(s,a):\n d={"UP":(-5,0),"DOWN":(5,0),"LEFT":(0,-5),"RIGHT":(0,5)}.get(a,(0,0));n={"block":[s["block"][0]+d[0],s["block"][1]+d[1]],"bar_left":s["bar_left"]+1};return [{"state":n,"commitments":dict(n),"declared_dependencies":["block","bar_left"]}]\ndef assimilate(s,o,v): b,m=_o(o);return {"block":b,"bar_left":m}\n'''
W={'decision':'next move','obstacle':'transition uncertainty','operation':'test','claim':'joint transition','evidence_refs':[1,2],'dependencies':['block','bar_left'],'completion_checks':[1,2],'remaining_scope':'outside trace','reasoning_need':'needs_computation'}
def call(i,n,a):return {'id':i,'type':'function','function':{'name':n,'arguments':json.dumps(a)}}
def response(item):return {'id':'chatcmpl-local','object':'chat.completion','created':1,'model':LOCAL_MODEL,'system_fingerprint':'sglang-fixture','choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'reasoning_content':None,'tool_calls':[{**item,'index':0}]}}],'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130,'completion_tokens_details':{'reasoning_tokens':0}}}

def solver_sessions(evidence,expected):
 rows=[]
 for index in range(expected):
  wire=evidence/'local-wire'/f'{index:02d}.body';receipt=evidence/'http-receipts'/f'{index:02d}.json';raw=evidence/'raw-http'/f'{index:02d}.body'
  if not wire.is_file() or not receipt.is_file() or not raw.is_file(): raise RuntimeError('durable local HTTP evidence is incomplete')
  fact=json.loads(receipt.read_text())
  if fact.get('status')!=200: raise RuntimeError('durable local HTTP response was not successful')
  rows.append({'sequence':str(index),'request_sha256':hashlib.sha256(wire.read_bytes()).hexdigest(),'response_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'receipt':str(receipt)})
 return {'schema':'nooa-join-sessions-v4','game_count':1,'total_actor_calls':expected,'construction_mapping':'actor_call_count is the donor field for actual construction-model HTTP requests; no gameplay occurred','games':{'construction':{'model_evidence':{'actor_call_count':expected,'actor_calls_by_sequence':{row['sequence']:1 for row in rows},'request_role':'construction_model_request_not_gameplay_actor','transport_receipts':rows}}}}

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--base-url',required=True);parser.add_argument('--model',required=True)
 parser.add_argument('--workspace-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--timeout-s',type=float,required=True);args=parser.parse_args()
 evidence=args.workspace_root/'construction-evidence'
 args.output.parent.mkdir(parents=True,exist_ok=True)
 try:
  if args.model!=LOCAL_MODEL: raise ValueError('official launcher model identity drift')
  if os.environ.get('CONSTRUCTION_FIXTURE')=='1':
   transport=FixtureTransport([response(call('call_inspect_A','execute_python',{'code':"import json;print(len(json.load(open('/evidence/case.json'))['visible_transitions']))"})),response(call('opaque-frontier-7','execute_python',{'code':"import json;print(json.load(open('/evidence/case.json'))['frontier'])"})),response(call('tool_submit_xyz','submit_candidate',{'work_item':W,'candidate_source':C}))])
  else: transport=LocalHTTPTransport(base_url=args.base_url,model=args.model,output=evidence,timeout_s=min(90,args.timeout_s))
  evidence.parent.mkdir(parents=True,exist_ok=True)
  write_json(evidence/'module-origins.json',origins)
  result=IntegrationHost(load_case(),transport,evidence,limits=ConstructionLimits(wall_seconds=min(900,max(21,int(args.timeout_s))))).run()
  sessions=None if os.environ.get('CONSTRUCTION_FIXTURE')=='1' else solver_sessions(evidence,result['model_requests'])
  launcher={'schema':'construction-official-launcher-result-v2','status':'passed','candidate_classification':result['classification'],'construction_result':result,'evidence_directory':str(evidence),'gameplay_result':None}
  if sessions is not None: launcher['solver_sessions']=sessions
  write_json(args.output,launcher)
  print(json.dumps(launcher,sort_keys=True));return 0
 except (Exception, KeyboardInterrupt, SystemExit) as error:
  launcher={'schema':'construction-official-launcher-result-v2','status':'failed','candidate_classification':None,'failure_type':type(error).__name__,'failure':str(error),'evidence_directory':str(evidence),'gameplay_result':None}
  write_json(args.output,launcher);print(json.dumps(launcher,sort_keys=True));return 1
if __name__=='__main__': raise SystemExit(main())
