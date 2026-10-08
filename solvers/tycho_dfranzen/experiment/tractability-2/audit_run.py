"""Independently reconcile request exposure, final reservations and engine outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();run=args.run
output=run/'output'; cases=[]; failures=[]
def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
receipt=read(run/'collection-receipt.json')
for f in receipt['files']:
 path=output/f['path']
 if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:failures.append('Collected hash drift: '+f['path'])
for path in sorted((output/'tractability-output').glob('block-*/*/result.json')):
 result=read(path);root=path.parent;known=unknown=0;protected=4096;final_calls=[];requests=sorted((root/'requests').glob('*.json'))
 for req_path in requests:
  req=read(req_path); maximum=int(req['parameters']['max_tokens']); final=[t['name'] for t in req['tools']]==['take_actions']
  remaining=protected-maximum if final else protected
  if remaining<0 or known+unknown+maximum+remaining>25000:failures.append('Admission exceeds protected cap: '+str(req_path.relative_to(output)))
  if final:
   protected=remaining
   if maximum!=2048 or req['parameters']['extra_body']['chat_template_kwargs']['enable_thinking'] is not False:failures.append('Final request policy drift: '+str(req_path))
  resp=root/'responses'/req_path.name
  if resp.exists():
   usage=read(resp)['usage']; completion=usage.get('completion_tokens')
   if completion is None:unknown+=maximum
   else:known+=int(completion)
  else:unknown+=maximum
  if final:final_calls.append({'call':req['index'],'elapsed_seconds':req['elapsed_seconds'],'response_received':resp.exists()})
 if (known,unknown,len(requests),protected)!=(result['completed_completion_tokens'],result['unknown_usage_reserved_tokens'],result['calls_started'],result['unspent_final_token_reservation']):failures.append('Ledger mismatch: '+str(path.relative_to(output)))
 if known+unknown>25000:failures.append('Completed exposure exceeds cap: '+str(path))
 admissions=lines(root/'admission.jsonl');lifecycle=lines(root/'actor-lifecycle.jsonl')
 if any(a['action_demand'] for a in admissions):failures.append('Final action admission rejected: '+str(path))
 engine_path=root/'engine-evaluation.json';engine=read(engine_path) if engine_path.exists() else None
 if result['action'] is not None and not (engine or {}).get('prefix_verified'):failures.append('Committed batch lacks verified prefix: '+str(path))
 engine_summary={k:v for k,v in engine.items() if k not in ('transitions','proposed_batch')} if engine else None
 cases.append({**result,'path':str(path.relative_to(output)),'engine':engine_summary,'final_calls':final_calls,'local_rejections':admissions,'lifecycle':lifecycle})
for_stalls=defaultdict(lambda:{'cases':0,'actions':0,'levels_completed':0,'final_admission_rejections':0,'cases_with_final_requests':0,'unknown_usage_cases':0})
controls=defaultdict(lambda:{'cases':0,'level_completions':0})
for c in cases:
 e=c['engine'] or {};level=bool(e.get('prefix_verified') and e.get('level_completed'))
 if c['packet']=='re86-success':
  row=controls[c['arm']];row['cases']+=1;row['level_completions']+=level
 else:
  row=for_stalls[c['arm']];row['cases']+=1;row['actions']+=c['action'] is not None;row['levels_completed']+=level
  row['final_admission_rejections']+=sum(a['action_demand'] for a in c['local_rejections']);row['cases_with_final_requests']+=bool(c['final_calls']);row['unknown_usage_cases']+=c['unknown_usage_reserved_tokens']>0
suite_path=output/'tractability-output/suite-status.json'; suite=read(suite_path) if suite_path.exists() else {}
if len(cases)!=24 or suite.get('status')!='complete' or not receipt['critical_complete']:failures.append('Incomplete planned evidence')
report={'valid':not failures,'cases':len(cases),'collected_files_hash_checked':len(receipt['files']),'failures':failures,'stalls':dict(for_stalls),'controls':dict(controls),'case_details':cases,'claim_ceiling':'First committed batch from a saved request; progress requires verified engine outcomes; historical deltas are descriptive under changed common admission'}
(run/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='case_details'},indent=2))
