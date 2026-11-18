"""No-key, no-paid endpoint pipeline. Separate prefix construction, prediction, score."""
import contextlib,copy,dataclasses,json,os,sys,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'engine')]
from evidence import read,put,digest,journal
from prediction_contract import branch,grid,score
from prefix import project_transitions
from constructor import ConstructionLimits,ConstructionResult,run_construction
from host import bound_constructor
from integration_boundary import CandidateRPC,SealedBoundary
import containment_base as cb

@contextlib.contextmanager
def audit(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 import tempfile
 old_tmp=tempfile.tempdir;tempfile.tempdir=str(output)
 old=cb.bounded
 def capture(command,**kwargs):
  receipt=old(command,**kwargs)
  journal(output.parent/'sealed-rpc.jsonl',{'command':command,'receipt':receipt})
  return receipt
 cb.bounded=capture
 try:yield
 finally:cb.bounded=old;tempfile.tempdir=old_tmp

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k):return None

class FixtureHTTP:
 """Only loopback HTTP, no auth loader, headers or remote transport path."""
 def __init__(self,url,output):
  u=urllib.parse.urlsplit(url)
  if u.scheme!='http' or u.hostname!='127.0.0.1' or not u.port or u.username or u.password:raise ValueError('loopback fixture URL only')
  self.url=url;self.output=Path(output);self.output.mkdir();self.n=0
 def __call__(self,body):
  self.n+=1;stem=f'{self.n:03}';put(self.output/(stem+'.request.json'),body)
  req=urllib.request.Request(self.url,data=json.dumps(body,allow_nan=False).encode(),headers={'Content-Type':'application/json'},method='POST')
  started=time.monotonic();raw=None;status=None
  try:
   with urllib.request.build_opener(NoRedirect).open(req,timeout=30) as r:raw=r.read(2_000_001);status=r.status
   if len(raw)>2_000_000:raise ValueError('fixture response size')
   (self.output/(stem+'.response.raw')).write_bytes(raw)
   return json.loads(raw)
  finally:put(self.output/(stem+'.receipt.json'),{'status':status,'latency_seconds':time.monotonic()-started,'response_sha256':digest(raw) if raw else None,'fixture':True,'paid_call':False})

def _construct_child(prefix,output,fixture_url):
 output=Path(output);output.mkdir(exist_ok=False)
 put(output/'CONSUMED.json',{'operation':'local fixture construction','prefix_sha256':digest(Path(prefix).read_bytes())})
 case=project_transitions(prefix);result=ConstructionResult(status='WORKING')
 def progress(r):
  # Incremental immutable snapshots, not overwriting previous forensic evidence.
  n=len(list(output.glob('progress-*.json')));put(output/f'progress-{n:03}.json',dataclasses.asdict(r))
 try:
  with audit(output/'tmp'),bound_constructor():
   result=run_construction(case,transport=FixtureHTTP(fixture_url,output/'raw'),limits=ConstructionLimits(max_model_requests=6,wall_seconds=240),result=result,progress_sink=progress,event_sink=lambda e:journal(output/'tool-events.jsonl',e))
  put(output/'CONSTRUCTION.json',dataclasses.asdict(result))
  if result.status=='CANDIDATE_READY':
   assert result.evaluation['evidence_claim']=='TYPED_VISIBLE_REPLAY_ONLY_NOT_PREDICTIVE_SUPPORT'
   put(output/'FROZEN.json',{'schema':'typed-constructor-freeze-v1','candidate_source':result.candidate_source,'source_digest':digest(result.candidate_source.encode()),'prefix_sha256':digest(Path(prefix).read_bytes()),'evidence_claim':'TYPED_VISIBLE_REPLAY_ONLY_NOT_PREDICTIVE_SUPPORT','fixture_constructor':True,'predictive_support':False})
  return result
 except Exception as e:
  put(output/'FAILURE.json',{'type':type(e).__name__,'error':str(e)});raise


def construct(prefix,output,fixture_url,hard_seconds=245):
    """Inherited bounded process supervisor, with root promotion only on clean exit."""
    u=urllib.parse.urlsplit(fixture_url)
    if u.scheme!='http' or u.hostname!='127.0.0.1' or not u.port:
        raise ValueError('loopback fixture URL only')
    output=Path(output);output.mkdir(exist_ok=False)
    put(output/'CONSUMED.json',{'operation':'supervised local fixture constructor','prefix_sha256':digest(Path(prefix).read_bytes())})
    receipt=cb.bounded([sys.executable,'-B',str(HERE/'constructor_worker.py'),str(Path(prefix).resolve()),str((output/'child').resolve()),fixture_url],timeout=min(245,hard_seconds),output_cap=1_000_000)
    put(output/'SUPERVISION.json',receipt)
    if receipt['status']!='completed' or receipt['returncode']!=0:
        put(output/'FAILURE.json',{'error':'child incomplete; no root freeze promotion','status':receipt['status']})
        raise RuntimeError('constructor failed or hard deadline; child files forensic only')
    r=read(output/'child/CONSTRUCTION.json')
    result=ConstructionResult(**r)
    put(output/'CONSTRUCTION.json',r)
    if result.status=='CANDIDATE_READY':
        freeze=read(output/'child/FROZEN.json')
        if digest(freeze['candidate_source'].encode())!=freeze['source_digest']:
            raise ValueError('child freeze source mismatch')
        put(output/'FROZEN.json',freeze)
    return result


def freeze_artifact(artifact,prefix,output):
    """Offline import of a provided source artifact, rechecked under typed ABI."""
    from evaluate_visible import evaluate_visible_abi
    output=Path(output);output.mkdir(exist_ok=False)
    artifact=Path(artifact);payload=read(artifact)
    source=payload['candidate_source']
    if not isinstance(source,str) or digest(source.encode())!=payload['source_digest']:
        raise ValueError('provided artifact source digest mismatch')
    put(output/'CONSUMED.json',{'artifact':str(artifact.resolve()),'artifact_sha256':digest(artifact.read_bytes()),'prefix_sha256':digest(Path(prefix).read_bytes())})
    with audit(output/'tmp'):
        evaluation=evaluate_visible_abi(source,project_transitions(prefix))
    put(output/'EVALUATION.json',evaluation)
    if evaluation['visible']['status']!='passed':
        raise ValueError('provided artifact fails typed visible replay; no conversion or repair')
    put(output/'FROZEN.json',{'schema':'typed-constructor-freeze-v1','candidate_source':source,'source_digest':digest(source.encode()),'prefix_sha256':digest(Path(prefix).read_bytes()),'evidence_claim':'TYPED_VISIBLE_REPLAY_ONLY_NOT_PREDICTIVE_SUPPORT','fixture_constructor':bool(payload.get('fixture_constructor',False)),'predictive_support':False,'provided_artifact_sha256':digest(artifact.read_bytes())})


def predict_once(frozen,prefix,query,output,registry=None):
 """Never accepts an outcome path and never reads a heldout observation."""
 output=Path(output);output.mkdir(exist_ok=False)
 frozen,prefix,query=map(Path,(frozen,prefix,query))
 freeze=read(frozen);q=read(query);prefix_sha=digest(prefix.read_bytes())
 if q['prefix_sha256']!=prefix_sha or freeze.get('prefix_sha256')!=prefix_sha:raise ValueError('prefix binding mismatch')
 if freeze.get('schema')!='typed-constructor-freeze-v1':raise ValueError('typed prefix-bound freeze required')
 source=freeze['candidate_source']
 if digest(source.encode())!=freeze['source_digest']:raise ValueError('source digest mismatch')
 if q['prefix_step']!=1 or q['step']!=2 or q['action']=='RESET':raise ValueError('query chronology/reset')
 case=project_transitions(prefix)
 if case.frontier!=1:raise ValueError('exact step-1 prefix required')
 binding={'schema':'prediction-binding-v1','freeze_sha256':digest(frozen.read_bytes()),'source_sha256':freeze['source_digest'],'prefix_sha256':prefix_sha,'query_sha256':digest(query.read_bytes()),'session':q['session'],'step':q['step'],'action':q['action'],'outcome_sha256':q['outcome_sha256']}
 registry=Path(registry) if registry is not None else HERE/'consumption'
 registry.mkdir(parents=True,exist_ok=True)
 key=digest((binding['source_sha256']+binding['prefix_sha256']+binding['query_sha256']).encode())
 put(registry/(key+'.json'),{'binding':binding,'output':str(output.resolve()),'one_attempt':True})
 put(output/'CONSUMED.json',binding);journal(output/'events.jsonl',{'event':'query_consumed','time_ns':time.time_ns()})
 try:
  with audit(output/'tmp'):
   module=CandidateRPC(SealedBoundary(),source)
   state=module.initialize({'grid':case.visible[0]['before']['grid']})
   if not isinstance(state,dict):raise ValueError('initialize must return object')
   for row in case.visible:
    observed_shape=grid(row['before']['grid'])
    visible_prediction=module.predict(state,row['action']);branch(visible_prediction,observed_shape)
    state=module.assimilate(visible_prediction[0]['state'],{'grid':row['after']['grid']},{'status':'unscored'})
    if not isinstance(state,dict):raise ValueError('assimilate must return object')
   put(output/'PREFIX-REPLAY.json',{'rpc_methods':module.calls,'state_sha256':digest(json.dumps(state,sort_keys=True,allow_nan=False).encode())})
   # Separate bounded invocation; the outcome never enters arguments/files/environment.
   final=CandidateRPC(SealedBoundary(),source)
   value=final.predict(state,q['action'])
   put(output/'RAW-PREDICTION.json',{'value':value,'rpc_methods':final.calls})
   commitment=branch(value,grid(case.visible[-1]['after']['grid']))
  pred={'binding':binding,'commitment':commitment,'prediction_coverage':int(commitment['type']=='grid'),'model_evidence':False,'candidate_origin': 'fixture' if freeze.get('fixture_constructor') else 'provided artifact; model origin not independently certified'}
  pred_sha=put(output/'PREDICTION.json',pred)
  binding.update(prediction_sha256=pred_sha,raw_prediction_sha256=digest((output/'RAW-PREDICTION.json').read_bytes()))
  binding_sha=put(output/'BOUND.json',binding)
  journal(output/'events.jsonl',{'event':'prediction_bound','binding_sha256':binding_sha,'time_ns':time.time_ns()})
  return binding_sha
 except Exception as e:
  put(output/'FAILURE.json',{'type':type(e).__name__,'error':str(e),'no_target_read':True});raise


def score_bound(output,frozen,prefix,query,outcome,binding_sha256):
 """Only this phase can open the target; external digest protects binding file."""
 output=Path(output);frozen,prefix,query,outcome=map(Path,(frozen,prefix,query,outcome))
 raw=(output/'BOUND.json').read_bytes()
 if digest(raw)!=binding_sha256:raise ValueError('binding digest changed')
 b=json.loads(raw)
 for path,key in [(frozen,'freeze_sha256'),(prefix,'prefix_sha256'),(query,'query_sha256'),(output/'PREDICTION.json','prediction_sha256'),(output/'RAW-PREDICTION.json','raw_prediction_sha256')]:
  if digest(path.read_bytes())!=b[key]:raise ValueError('bound source/prefix/query/prediction drift')
 freeze=read(frozen);q=read(query);pred=read(output/'PREDICTION.json');case=project_transitions(prefix)
 if digest(freeze['candidate_source'].encode())!=b['source_sha256']:raise ValueError('source drift')
 if any(q[k]!=b[k] for k in ('session','step','action','outcome_sha256')):raise ValueError('query mismatch')
 expected={k:v for k,v in b.items() if k not in ('prediction_sha256','raw_prediction_sha256')}
 if pred['binding']!=expected:raise ValueError('prediction context mismatch')
 commitment=branch([{'state':None,'commitments':pred['commitment']}],grid(case.visible[-1]['after']['grid']))
 put(output/'SCORING-CONSUMED.json',{'binding_sha256':binding_sha256,'time_ns':time.time_ns(),'no_retry':True})
 journal(output/'events.jsonl',{'event':'target_read_begins_after_binding','time_ns':time.time_ns()})
 try:
  raw=outcome.read_bytes()
  if digest(raw)!=b['outcome_sha256']:raise ValueError('target provenance drift')
  target=json.loads(raw)
  if any(target[k]!=b[k] for k in ('session','step','action')):raise ValueError('target episode/action/step mismatch')
  metrics=score(case.visible[-1]['after']['grid'],target['grid'],commitment)
  receipt={'binding_sha256':binding_sha256,'prediction_sha256':b['prediction_sha256'],'outcome_sha256':digest(raw),'metrics':metrics,'fixture_constructor':freeze.get('fixture_constructor',False),'kind':'retrospective chronological holdout','heldout_queries':1,'heldout_assimilations':0,'target_feedback_to_constructor':False,'retry':False}
  put(output/'SCORE.json',receipt);journal(output/'events.jsonl',{'event':'score_persisted','time_ns':time.time_ns()});return receipt
 except Exception as e:
  put(output/'SCORING-FAILURE.json',{'type':type(e).__name__,'error':str(e)});raise
