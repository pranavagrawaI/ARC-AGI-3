"""Sealed model: hypothetical queries are read-only; actual transitions own state."""
import contextlib,tempfile,time
from pathlib import Path
from common import read,put,journal,hash_file,digest,encoded
from prediction_contract import grid,branch,score
from integration_boundary import CandidateRPC,SealedBoundary
import containment_base as cb

@contextlib.contextmanager
def audit(out):
 (out/'tmp').mkdir(exist_ok=True);old_tmp=tempfile.tempdir;tempfile.tempdir=str(out/'tmp');old=cb.bounded
 def capture(command,**kwargs):
  receipt=old(command,**kwargs);journal(out/'sealed-rpc.jsonl',{'command':command,'receipt':receipt});return receipt
 cb.bounded=capture
 try:yield
 finally:cb.bounded=old;tempfile.tempdir=old_tmp

class Model:
 def __init__(self,freeze,prefix,out):
  self.out=out;self.active=False;self.queries=[];self.query_count=0;self.state=None;self.prepared=None;self.source=None
  if not freeze or not Path(freeze).is_file():self.reason='constructor did not freeze';return
  f=read(freeze)
  if f['prefix_sha256']!=hash_file(prefix) or digest(f['candidate_source'].encode())!=f['source_digest']:raise ValueError('freeze/prefix binding')
  self.source=f['candidate_source'];self.source_sha=f['source_digest'];self.reason='active'
  rows=[__import__('json').loads(x) for x in Path(prefix).read_text().splitlines()]
  try:
   self.state=self.call('initialize',{'grid':rows[0]['before']['grid']})
   if not isinstance(self.state,dict):raise ValueError('initialize object required')
   for row in rows:
    produced=self.predict(row['action'],row['before']);self.state=self.call('assimilate',produced[0]['state'],{'grid':row['after']['grid']},{'status':'unscored'})
    if not isinstance(self.state,dict):raise ValueError('assimilate object required')
   self.active=True
  except Exception as e:self.disable('prefix replay '+type(e).__name__)
 def call(self,method,*args):
  rpc=CandidateRPC(SealedBoundary(),self.source)
  result=getattr(rpc,method)(*args)
  journal(self.out/'model-calls.jsonl',{'method':method,'args':args,'result':result,'source_sha256':self.source_sha,'rpc':rpc.calls})
  return result
 def predict(self,action,obs):
  result=self.call('predict',self.state,action);branch(result,grid(obs['grid']));return result
 def disable(self,reason):
  self.active=False;self.reason=reason;self.prepared=None
  journal(self.out/'model-events.jsonl',{'event':'disabled','reason':reason})
 def query(self,action,obs,step,forced=False):
  if self.query_count>=20:raise ValueError('query execution cap')
  self.query_count+=1
  q={'id':self.query_count,'step':step,'action':action,'forced':forced,'time_ns':time.time_ns(),'before_sha256':digest(encoded(obs)),'source_sha256':getattr(self,'source_sha',None)}
  if not self.active:q.update(status='unavailable',reason=self.reason)
  else:
   try:q.update(status='predicted',value=self.predict(action,obs))
   except Exception as e:q.update(status='error',error=type(e).__name__);self.disable('query '+type(e).__name__)
  sha=put(self.out/f'QUERY-{self.query_count:03}.json',q);q['binding_sha256']=sha
  self.queries.append(q)
  from game import compact
  result={k:v for k,v in q.items() if k!='value'}
  if q['status']=='predicted':
   c=q['value'][0]['commitments'];result['commitment']=({'type':'grid','prediction':compact({'grid':c['predicted_grid']})} if c['type']=='grid' else c)
  return result
 def before_actual(self,action,obs,step):
  self.prepared=None
  if not self.active:return
  if action=='RESET':self.disable('reset committed');return
  # A different hypothetical action's branch is never assimilated.
  try:
   matching=[q for q in self.queries if q['step']==step and q['action']==action and q['status']=='predicted' and q['before_sha256']==digest(encoded(obs))]
   value=matching[-1]['value'] if matching else self.predict(action,obs)
   self.prepared={'step':step,'action':action,'value':value,'query_ids':[q['id'] for q in matching]}
   put(self.out/f'ACTUAL-BOUND-{step:03}.json',self.prepared)
  except Exception as e:self.disable('actual prediction '+type(e).__name__)
 def after_actual(self,row):
  pending=self.prepared;self.prepared=None
  if not self.active or not pending:return
  if pending['step']+1!=row['step'] or pending['action']!=row['action']:raise ValueError('actual transition binding mismatch')
  if row['after']['levels_completed']!=row['before']['levels_completed'] or row['after']['state']!='NOT_FINISHED':self.disable('level/terminal boundary');return
  # Diagnostics are never passed to source, never used for constructor repair.
  for q in self.queries:
   if q['id'] in pending['query_ids']:
    put(self.out/f'QUERY-SCORE-{q["id"]:03}.json',{'query_sha256':q['binding_sha256'],'step':row['step'],'metrics':score(row['before']['grid'],row['after']['grid'],q['value'][0]['commitments'])})
  try:
   self.state=self.call('assimilate',pending['value'][0]['state'],{'grid':row['after']['grid']},{'status':'unscored'})
   if not isinstance(self.state,dict):raise ValueError('assimilate object required')
  except Exception as e:self.disable('actual assimilation '+type(e).__name__)
