from pathlib import Path
import json,hashlib,sys,datetime,base64
from decimal import Decimal
b=Path('/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021');p=b/'trial-17-candidate-first-pair-r2-local';r=p/'remote-readback';c=b/'coordination-20261024'
sys.path[:0]=[str(p/'collection_tools'),str(p)]
def audit(event,args):
 if event.startswith('socket.connect'):raise RuntimeError('offline closure forbids network')
sys.addaudithook(audit)
import primary,primary_bundle,decimal_quota,reconciliation,transport
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
raw=(r/'primary-current-once/PRIMARY.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='55e9968c1f6c9de907dbf67093165ea708e704be0047d1c7ac77e07e37018e97'
a=primary.verify(raw,r/'poll-20261024T062113126081Z');z=primary.verify(raw,r/'after-primary-exact');assert a==z
returned=json.loads((r/'primary-cli-returned.json').read_bytes());ex=json.loads((r/'primary-cli-exit.json').read_bytes())
assert ex['observed_exit_code']==0 and (r/'primary-cli.stderr').read_bytes()==b''
assert sha(r/'primary-cli.stdout')==ex['stdout_sha256'] and sha(r/'primary-cli-returned.json')==ex['returned_sha256']
assert json.loads((r/'primary-cli.stdout').read_bytes())==returned
assert returned['complete'] and returned['primary_bundle_sha256']==hashlib.sha256(raw).hexdigest()
assert returned['start']<=returned['clocks']['receipt_persistence_completed']<=returned['absolute_deadline']==returned['start']+120
assert all(returned[k]==v for k,v in a.items())
policy=reconciliation.load_policy();values=[];receipts=[]
for name in ('poll-20261024T062113126081Z','after-primary-exact'):
 rb=json.loads((r/name/'READBACK.json').read_bytes());assert rb['source_exact'] and rb['inner_sha256']=='87a588dfaa0b3f421d469e0274d9cecb33b843ee8637b12e1425a8f55f02751a';assert (rb['kernel_id'],rb['version'],rb['status']['status'])==(140530941,1,'ERROR')
 q=rb['typed_quota'];v=decimal_quota.validate(q,policy,now=datetime.datetime.fromisoformat(q['utc']));assert v['used_s']==Decimal('4842.678') and v['reserved_s']==0
 values.append(v);receipts.append({'path':str(r/name/'READBACK.json'),'sha256':sha(r/name/'READBACK.json'),'quota_utc':q['utc']})
assert values[0]==values[1]
result=a['result'];assert result['pass_'] is False and result['ratio'] is None and result['adoption'] is False and not result['candidate_health200']
cleanup={k:result[k] for k in ('cleanup_errors','phase_cleanup_errors','surviving_phase_groups','surviving_native_group','retained_queries','query_cleanup')};assert all(v==[] for v in cleanup.values())
bundle=primary_bundle.verify(raw);events=[json.loads(line) for line in base64.b64decode(bundle['files']['supervisor.jsonl']['base64']).splitlines()]
assert [x['seq'] for x in events]==list(range(len(events)))
assert not [x for x in events if x.get('phase') in ('candidate.replay','control.prepare','control.replay') and x['event']=='spawn']
f=json.loads((p/'freeze.json').read_bytes());assert sha(p/'freeze.json')=='34befabf21e93d7753e251ae6aa3e1131ab224f4395779f60f4845bf9a1f32aa'
for path,want in [(p/n,h) for n,h in f['files'].items()]+[(Path(n),h) for n,h in f['references'].items()]:
 with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==want,str(path)
closure={'decision':'ATTEMPT_CLOSED_EXCLUSIVE_SLOT_RELEASED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'assignment':json.loads((p/'slot-confirmation.json').read_text())['serialized_assignment'],'kernel_ref':'pranavagrawai/serving-candidate-first-capacity17-20261024','kernel_id':140530941,'version':1,'platform_status':'ERROR','freeze_sha256':sha(p/'freeze.json'),'bindings_verified':22075,'primary_sha256':hashlib.sha256(raw).hexdigest(),'primary_bytes':len(raw),'primary_frozen_verify_before_after_equal':True,'primary_CLI_postwrite_completion_verified':True,'readbacks':receipts,'terminal_quota_exact_wire_seconds':{k:str(v) for k,v in values[1].items()},'baseline_used_s':'2714.079','quota_used_delta_s':str(Decimal('4842.678')-Decimal('2714.079')),'result':result,'cleanup':cleanup,'no_candidate_replay_or_control_dispatch_in_primary_trace':True,'missing_primary_files':a['missing'],'truncated_primary_files':a['truncated'],'scope':'Operational closure from exact terminal/current-source/primary producer cleanup and quota. No independent full raw resource, actual postpublication result or paired scientific qualification.','ratio':None,'retry_authorized':False,'GPU_authorized':False,'required_GETs_authorized':False,'archive_authorized':False,'next_local_scope':'Finish local frozen offline plan/gap and primary provenance handoff, disable followup. Separate bounded read-only startup trace/source/readout diagnosis against accepted trial16; no timeout/order/cache/loader/threshold changes or successor packet.'}
out=c/'trial17-attempt-closed.json'
with out.open('x') as s:json.dump(closure,s,indent=2);s.write('\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'decision':closure['decision'],'delta':closure['quota_used_delta_s'],'result':result,'missing_count':len(a['missing'])},indent=2))
