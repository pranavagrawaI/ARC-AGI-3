"""Host-only orchestration for the construction-policy integration."""
import copy, hashlib, json, os
from dataclasses import asdict
from pathlib import Path
from constructor import ConstructionLimits, ConstructionResult, run_construction
from integration_boundary import SealedBoundary, evaluate_candidate
from real_trace import load_case

class ProtocolError(RuntimeError): pass

def write_json(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def append(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f: f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

class FixtureTransport:
 """Provider-shaped deterministic responses. It never interprets tool calls."""
 def __init__(self,responses): self.responses=copy.deepcopy(responses);self.index=0;self.executed_tools=0
 def __call__(self,body):
  if self.index>=len(self.responses): raise ProtocolError('fixture responses exhausted after malformed arguments')
  r=copy.deepcopy(self.responses[self.index]);self.index+=1;return r

class IntegrationHost:
 def __init__(self,case,transport,output):
  self.case,self.transport,self.output=case,transport,Path(output);self.boundary=SealedBoundary();self.requests=0
 def next_request(self,messages):
  # Produced by the actual constructor so the existing parser/contract remains authoritative.
  captured=[]
  class Stop(Exception): pass
  def transport(body): captured.append(body);raise Stop()
  try: run_construction(self.case,transport=transport,limits=ConstructionLimits(max_model_requests=1),result=ConstructionResult(status='WORKING'))
  except Stop: pass
  return captured[0]
 def _transport(self,body):
  from request_contract import validate_request
  validate_request(body); i=self.requests;self.requests+=1
  write_json(self.output/'requests'/f'{i:02d}.json',body); append(self.output/'lifecycle.jsonl',{'event':'request_admitted','index':i})
  response=self.transport(body); write_json(self.output/'responses'/f'{i:02d}.json',response);append(self.output/'lifecycle.jsonl',{'event':'response_received','index':i})
  return response
 def run(self):
  self.output.mkdir(parents=True,exist_ok=True)
  def tool_event(event): append(self.output/'tool-events.jsonl',event);append(self.output/'lifecycle.jsonl',{'event':'tool_dispatch','tool':event['call']['function']['name'],'id':event['call']['id']})
  result=run_construction(self.case,transport=self._transport,event_sink=tool_event,progress_sink=lambda r:write_json(self.output/'progress.json',asdict(r)))
  write_json(self.output/'transcript.json',result.transcript)
  suffix=None;suffix_count=0
  if result.status=='CANDIDATE_READY':
   source=result.candidate_source; digest=hashlib.sha256(source.encode()).hexdigest(); path=self.output/'frozen_candidate.py';path.write_text(source);path.chmod(0o444)
   write_json(self.output/'FROZEN.json',{'model_sha256':digest,'source_path':'frozen_candidate.py','visible_evaluation':result.evaluation,'sealed_suffix_sha256':self.case.sealed_suffix_sha256})
   append(self.output/'lifecycle.jsonl',{'event':'frozen','model_sha256':digest})
   # Host alone owns suffix payload/observer/verdict; frozen candidate receives only each current ABI call.
   sealed_full=evaluate_candidate(source,self.case,include_suffix=True,boundary=self.boundary);suffix=sealed_full['suffix'];suffix_count=1
   write_json(self.output/'sealed_suffix_full_evaluation.json',sealed_full)
   write_json(self.output/'sealed_suffix_evaluation.json',suffix);append(self.output/'lifecycle.jsonl',{'event':'suffix_assessment','model_sha256':digest})
   # Frozen v1 prelaunch baseline: incumbent compatible_depth=1. Preserve the
   # original strict improvement criterion rather than treating any support as improvement.
   incumbent={'compatible_depth':1};write_json(self.output/'incumbent_suffix_evaluation.json',incumbent)
   classification='INTEGRATION_VALID_CANDIDATE_SUPPORTED' if suffix.get('compatible_depth',0)>incumbent['compatible_depth'] else 'INTEGRATION_VALID_CANDIDATE_NOT_SUPPORTED'
  elif result.status=='NEEDS_EVIDENCE': classification='INTEGRATION_VALID_NEEDS_EVIDENCE'
  elif result.status=='BUDGET_EXHAUSTED': classification='INTEGRATION_VALID_BUDGET_EXHAUSTED'
  else: classification='INTEGRATION_INVALID'
  summary={'classification':classification,'construction_status':result.status,'model_requests':result.model_requests,'inspections':result.inspections,'submissions':result.submissions,'usage':result.usage,'suffix_assessments':suffix_count,'sealed_suffix_evaluation':suffix,'scope':'one fixed LS20 construction episode; no competence claim'}
  write_json(self.output/'RESULT.json',summary);return summary
