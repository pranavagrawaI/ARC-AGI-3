import copy,hashlib,json,shutil,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
P=BASE/'trial-15-wire-capacity-local'
sys.path.insert(0,str(P))
import capacity_evidence as e,artifacts
from core import resources
SRC=P/'qualified-r2/cases/cpu-evidence-joins'
DEST=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
DEST.mkdir(parents=True,exist_ok=True)
root=DEST/'fixture'
if root.exists():raise SystemExit('exclusive fixture already exists')
shutil.copytree(SRC,root)
read=lambda name:json.loads((root/name).read_text())
trace=[json.loads(x) for x in (root/'supervisor.jsonl').read_text().splitlines()]
final=read('candidate-release.json');work=trace[0]['work_deadline']
def verify(value):
 if isinstance(value,dict):
  if {'path','sha256','bytes','complete'}<=value.keys():assert artifacts.verify(root/'query-artifacts',value,time.monotonic()+60)
  else:
   for x in value.values():verify(x)
 elif isinstance(value,list):
  for x in value:verify(x)
e.final_resource_evidence(final,trace,work);verify(final)
results={'baseline_final_helper_and_all_raw_references':'PASS','scope':'Actual preserved CPU lifecycle; exercises real final-resource helper plus same raw artifact verification used by provenance. Not a full production pair/reducer success fixture.'}
# Emulate producer-parser regression: actual raw GPU evidence is ineligible,
# while reduced eligible/used_mib fields erroneously retain their prior values.
changed=copy.deepcopy(final)
ref=changed['samples'][-1]['probes']['gpu']['command']['artifacts']['stdout'];path=root/'query-artifacts'/ref['path']
old=path.read_text();parts=old.strip().split(',');parts[1]='5000';raw=(','.join(parts)+'\n').encode();path.write_bytes(raw)
(DEST/'gpu.before.raw').write_text(old);(DEST/'gpu.after.raw').write_bytes(raw)
def update(v):
 if isinstance(v,dict):
  if v.get('path')==ref['path'] and 'sha256' in v:v.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
  else:
   for z in v.values():update(z)
 elif isinstance(v,list):
  for z in v:update(z)
update(changed)
actual=resources.gpu(raw.decode(),parts[2].strip());assert not actual['eligible']
e.final_resource_evidence(changed,trace,work);verify(changed)
(root/'candidate-release.raw-gpu5000-summary0.json').write_text(json.dumps(changed,indent=2))
results['raw_gpu5000_summary0']={'helper':'PASS','artifact_verification':'PASS','actual_raw_reduction':actual,'stored_used_mib':changed['samples'][-1]['probes']['gpu']['used_mib']}
# Separate receipt-only cases; restore raw before comparing unmodified receipt.
path.write_text(old)
widen=copy.deepcopy(final);events=copy.deepcopy(trace);widen['absolute_deadline']=work
for v in events:
 if v['event']=='candidate_cleanup_start':v['phase_deadline']=work
 if v['event']=='candidate_resources_pass':v['absolute_deadline']=work
e.final_resource_evidence(widen,events,work);verify(widen)
results['widened_final_deadline']={'helper':'PASS','fixture_configured_seconds':2,'original_budget_seconds':final['absolute_deadline']-final['start'],'accepted_budget_seconds':work-final['start']}
(root/'candidate-release.widened-deadline.json').write_text(json.dumps(widen,indent=2));(root/'supervisor.widened-deadline.json').write_text(json.dumps(events,indent=2))
# A self-consistent cleanup receipt for a different group is not joined to
# candidate-server-binding.json by this helper or its caller.
wrong=copy.deepcopy(final);events=copy.deepcopy(trace);actualpgid=wrong['server_pgid'];wrong['server_pgid']=actualpgid+1000000
for v in events:
 if v['event']=='candidate_cleanup_start':v['server_pgid']=wrong['server_pgid']
 if v['event']=='cleanup' and v['pgid']==actualpgid:v['pgid']=wrong['server_pgid']
e.final_resource_evidence(wrong,events,work)
results['other_group_cleanup']={'helper':'PASS','actual_candidate_pgid':actualpgid,'accepted_cleanup_pgid':wrong['server_pgid']}
(DEST/'RESULTS.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
