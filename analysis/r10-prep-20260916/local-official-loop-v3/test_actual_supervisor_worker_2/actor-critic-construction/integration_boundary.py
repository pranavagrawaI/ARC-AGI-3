"""CandidateRPC ABI over the exact qualified chroot/seccomp runner."""
import copy, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import containment_base as cb

HERE=Path(__file__).resolve().parent
RUNNER=HERE/'qualified_runner.py'
INSPECTION_RUNNER=HERE/'inspection_runner.py'

def _json(value,label):
 raw=json.dumps(value,allow_nan=False,sort_keys=True)
 if len(raw.encode())>200_000: raise ValueError(label+' exceeds JSON size limit')
 return json.loads(raw)

def _wrapped(source):
 return source+'''\n\n# Trusted adapter is appended after candidate definitions. Fresh process per call.\n_candidate_initialize=initialize\n_candidate_predict=predict\n_candidate_assimilate=assimilate\ndef run(method,args):\n if method not in ("initialize","predict","assimilate"): raise ValueError("bad method")\n return {"initialize":_candidate_initialize,"predict":_candidate_predict,"assimilate":_candidate_assimilate}[method](*args)\n'''

class SealedBoundary:
 def __init__(self, evaluation_seconds=15, call_seconds=3, inspection_seconds=8):
  self.deadline=time.monotonic()+evaluation_seconds;self.call_seconds=call_seconds;self.inspection_seconds=inspection_seconds
 def _invoke(self,source,mode,args,prefix=None):
  if time.monotonic()>=self.deadline: raise TimeoutError('boundary deadline')
  with tempfile.TemporaryDirectory(prefix='construction-policy-sealed-') as d:
   root=Path(d)/'root'
   manifest=cb.rootfs(root,source,mode,args,require_root=True)
   if mode=='inspection':
    p=root/'evidence/case.json'; p.parent.chmod(0o755); p.chmod(0o644)
    p.write_text(json.dumps(prefix,separators=(',',':'))); p.chmod(0o444); p.parent.chmod(0o555)
    manifest['evidence/case.json']=cb.digest(p)
   cb.validate_tree(root,manifest,require_root=True)
   remaining=self.deadline-time.monotonic()
   if remaining<=0: raise TimeoutError('boundary deadline')
   receipt=cb.bounded([sys.executable,'-I','-S','-B',str(RUNNER),str(root),str(os.getpid())],timeout=min(self.call_seconds,remaining),output_cap=1_000_000)
   if receipt['status']!='completed' or receipt['returncode']!=0 or not cb.valid_seal(receipt['stderr']):
    raise RuntimeError('sealed invocation failed closed: '+receipt['status']+':'+receipt['stderr'][:2000])
   value=json.loads(receipt['stdout'])
   cb.validate_tree(root,manifest,require_root=True)
   return _json(value,'call result')
 def invoke(self,source,method,args):
  if method not in {'initialize','predict','assimilate'}: raise AttributeError(method)
  return self._invoke(_wrapped(source),'candidate',[method,args])
 def inspect(self,code,prefix):
  if not isinstance(code,str) or not code.strip(): raise ValueError('inspection code must be nonempty text')
  if len(code.encode())>32_000: raise ValueError('inspection code exceeds 32,000 bytes')
  with tempfile.TemporaryDirectory(prefix='construction-policy-inspect-') as d:
   root=Path(d)/'root';manifest=cb.rootfs(root,code,'inspection',[],require_root=True)
   p=root/'evidence/case.json';p.parent.chmod(0o755);p.chmod(0o644);p.write_text(json.dumps(prefix,separators=(',',':')));p.chmod(0o444);p.parent.chmod(0o555);manifest['evidence/case.json']=cb.digest(p)
   cb.validate_tree(root,manifest,require_root=True)
   read_fd,write_fd=os.pipe()
   try:
    receipt=cb.bounded([sys.executable,'-I','-S','-B',str(INSPECTION_RUNNER),str(root),str(os.getpid()),str(write_fd)],timeout=self.inspection_seconds,output_cap=20_000,extra_fds=(write_fd,))
    fd,write_fd=write_fd,None
    os.close(fd)
    startup=os.read(read_fd,8192).decode(errors='replace')
   finally:
    try:
     if write_fd is not None: os.close(write_fd)
    finally:
     os.close(read_fd)
   if not cb.valid_seal(startup): raise RuntimeError('sealed inspection startup failed closed')
   cb.validate_tree(root,manifest,require_root=True)
   if receipt['status']=='timeout': raise TimeoutError('candidate execution time limit')
   if receipt['status']=='output_limit': raise RuntimeError('candidate output limit exceeded')
   stdout=receipt['stdout'][:16_000];stderr=receipt['stderr'][:4_000]
   return {'status':'completed' if receipt['returncode']==0 else 'failed','returncode':receipt['returncode'],'stdout':stdout,'stderr':stderr,'output_truncated':len(receipt['stdout'])>len(stdout) or len(receipt['stderr'])>len(stderr)}

class CandidateRPC:
 def __init__(self,boundary,source): self.boundary,self.source,self.calls=boundary,source,[]
 def __getattr__(self,method):
  if method not in {'initialize','predict','assimilate'}: raise AttributeError(method)
  def invoke(*args):
   self.calls.append({'ordinal':len(self.calls)+1,'method':method,'args_sha256':hashlib.sha256(json.dumps(args,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
   return self.boundary.invoke(self.source,method,list(args))
  return invoke

def observed(state):
 grid=state['grid']; cells=[(r,c) for r,row in enumerate(grid) for c,v in enumerate(row) if v==12]
 if not cells: raise ValueError('observer found no block')
 bar=next((c for c,v in enumerate(grid[61]) if v==11),None)
 if bar is None: raise ValueError('observer found no meter')
 return {'block':[min(r for r,c in cells),min(c for r,c in cells)],'bar_left':bar}

def branches_for(module,branches,action,cap):
 out=[]
 for branch in branches:
  produced=module.predict(copy.deepcopy(branch['state']),action)
  if not isinstance(produced,list) or not produced: raise ValueError('predict must return nonempty list')
  for i,item in enumerate(produced):
   if not isinstance(item,dict) or not {'state','commitments'}<=set(item): raise ValueError('bad branch')
   commitments=_json(item['commitments'],'commitments'); deps=item.get('declared_dependencies',[])
   if not isinstance(commitments,dict) or not isinstance(deps,list) or not all(isinstance(x,str) for x in deps): raise ValueError('bad branch fields')
   out.append({'ancestry':branch['ancestry']+[i],'state':_json(item['state'],'state'),'commitments':commitments,'dependencies':sorted(set(deps)),'history':branch['history']+[commitments]})
   if len(out)>cap: raise RuntimeError('branch_cap_exceeded')
 return out

def compatible(branch,measurement,required): return all(branch['commitments'].get(k)==measurement[k] for k in required)

def replay_visible(module,fixture):
 rows=fixture['visible']; initial={'grid':rows[0]['before']['grid']}; state=_json(module.initialize(copy.deepcopy(initial)),'initial')
 branches=[{'ancestry':[0],'state':state,'commitments':{},'dependencies':[],'history':[]}]; missing=set(); deps=[]; checks=[]
 for row in rows:
  forecasts=branches_for(module,branches,row['action'],fixture['branch_cap']); measurement=observed(row['after'])
  for f in forecasts: missing.update(set(fixture['required_targets'])-set(f['commitments'])); deps.append(f['dependencies'])
  survivors=[f for f in forecasts if compatible(f,measurement,fixture['required_targets'])]
  checks.append({'step':row['step'],'action':row['action'],'measurement':measurement,'forecast_branches':len(forecasts),'compatible_branches':len(survivors),'status':'passed' if survivors else 'contradicted'})
  if not survivors:return [],{'status':'contradicted','checks':checks},missing,deps
  branches=[{**s,'state':_json(module.assimilate(copy.deepcopy(s['state']),{'grid':row['after']['grid']},{'status':'compatible','measurement':measurement}),'assimilated')} for s in survivors]
 return branches,{'status':'passed','checks':checks},missing,deps

def sealed_rollout(module,start,fixture):
 # Build every prediction layer before reading any suffix observation. Never assimilate.
 branches=copy.deepcopy(start); layers=[]; deps=[]
 for row in fixture['suffix']:
  branches=branches_for(module,branches,row['action'],fixture['branch_cap']); layers.append(copy.deepcopy(branches)); deps += [b['dependencies'] for b in branches]
 survivors=None; audit=[]; depth=0
 for i,row in enumerate(fixture['suffix']):
  candidates=layers[i]
  if survivors is not None:
   parents={tuple(x['ancestry']) for x in survivors}; candidates=[x for x in candidates if tuple(x['ancestry'][:-1]) in parents]
  measurement=observed(row['after']); survivors=[x for x in candidates if compatible(x,measurement,fixture['required_targets'])]
  audit.append({'step':row['step'],'action':row['action'],'measurement':measurement,'compatible_paths':len(survivors)})
  if not survivors:return {'status':'contradicted','compatible_depth':depth,'first_contradiction_step':row['step'],'audit':audit},deps
  depth+=1
 return {'status':'passed','compatible_depth':depth,'audit':audit},deps

def evaluate_candidate(source,case,include_suffix=False,boundary=None):
 fixture=case.evaluator_payload(include_suffix=include_suffix); module=CandidateRPC(boundary or SealedBoundary(),source)
 start,visible,missing,vdeps=replay_visible(module,fixture)
 suffix,rdeps=sealed_rollout(module,start,fixture) if visible['status']=='passed' else ({'status':'not_run','compatible_depth':0},[])
 digest=hashlib.sha256(source.encode()).hexdigest()
 return {'model_sha256':digest,'belief_model_sha256':digest,'rpc_methods':module.calls,'visible':visible,'suffix':suffix,'coverage':{'required':fixture['required_targets'],'missing_required':sorted(missing)},'dependencies':{'declared_on_visible_cases':sorted({x for g in vdeps for x in g}),'encountered_on_proposed_rollout':sorted({x for g in rdeps for x in g}),'static_completeness_established':False}}

def execute_inspection(code,case):
 boundary=SealedBoundary()
 return boundary.inspect(code,case.inspection_view())
