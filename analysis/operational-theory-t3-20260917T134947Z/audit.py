import copy,datetime,hashlib,json,os,shutil,sys,tarfile
from decimal import Decimal
from pathlib import Path
root=Path.cwd();candidate=root/'analysis/operational-theory-t1r3-20260917';out=Path(Path('/tmp/arc-t3-operator/output-path').read_text());sys.path.insert(0,str(candidate))
from identity import verify,sha
from run_pair import reconcile
from transport import encode,digest

def save(name,value):
 (out/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
assert json.loads((out/'LAUNCH-END.json').read_text())['exit_code']==0
ledger=json.loads((out/'live/ledger.json').read_text());account=reconcile(out/'live/ledger.json')
assert account['unresolved_requests']==0 and account['unresolved_exposure_usd']=='0'
results={a:json.loads((out/'live'/a/'RESULT.json').read_text()) for a in ('control','treatment')}
first={};receipts=[];all_nonmessage=[];timestamps=[]
for arm in results:
 paths=sorted((out/'live'/f'{arm}-raw').glob('*.request.json'))
 assert len(paths)==ledger['arms'][arm]['requests']<=6
 for p in paths:
  stem=p.name.split('.')[0];r=json.loads(p.read_bytes());receipt_path=p.with_name(stem+'.receipt.json');receipt=json.loads(receipt_path.read_text());response_path=p.with_name(stem+'.response.raw');response=json.loads(response_path.read_bytes())
  assert sha(p)==receipt['request_sha256']
  assert response['usage']['prompt_tokens']==receipt['usage']['prompt_tokens']
  assert response['usage']['completion_tokens']==receipt['usage']['completion_tokens']
  assert Decimal(str(response['usage']['cost']))==Decimal(receipt['actual_cost_usd'])
  assert response['model'] in ('qwen/qwen3.8-27b','qwen/qwen3.8-27b-20260814') and response['provider']=='Reka'
  assert receipt.get('stop') is None
  assert 'Authorization' not in r and 'headers' not in r
  all_nonmessage.append({k:v for k,v in r.items() if k!='messages'})
  receipts.append(receipt)
  if stem=='001':first[arm]=copy.deepcopy(r)
  for f in (p,response_path,receipt_path):timestamps.append({'path':str(f.relative_to(out)),'filesystem_persisted_mtime_ns':f.stat().st_mtime_ns})
 assert json.loads((out/'live'/arm/'SUPERVISION.json').read_text())['reaped']
 assert len(json.loads((out/'live'/arm/'EVIDENCE.json').read_text()))==1
 assert not results[arm]['proposals']
 assert not results[arm]['metrics']['construction_completion']
shared=first['treatment'];marker='\n\n<operational_cycle>';content=shared['messages'][0]['content'];assert content.count(marker)==1 and content.endswith('</operational_cycle>');shared['messages'][0]['content']=content.split(marker)[0]
assert shared==first['control']
assert all(x==all_nonmessage[0] for x in all_nonmessage)
save('PARITY.json',{'initial_request_equal_after_removing_only_treatment_block':True,'shared_request_sha256':digest(encode(shared)),'all_12_nonmessage_parameter_sets_identical':True,'later_model_authored_messages_expected_to_differ':True})
save('TIMESTAMPS.json',{'request_timing':'Per-request receipt elapsed_seconds plus filesystem persistence timestamps; UTC run start/end retained separately. Filesystem timestamps are not exact dispatch times.','artifacts':timestamps})
counts={}
for name,expected in json.loads((out/'SOURCE-BEFORE.json').read_text()).items():
 folder=root/name;actual={str(f.relative_to(folder)):sha(f) for f in folder.rglob('*') if f.is_file()};assert actual==expected,name;counts[name]=len(actual)
assert verify()=='992cc66ae2e8577f69f852f00a5330f8f0fb2269155c435f4d1d35ba2931c53c'
orphans=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes()
  if b'bwrap\x00' in args and str(out).encode() in args:orphans.append(int(proc.name))
 except (OSError,PermissionError):pass
assert not orphans
save('INTEGRITY.json',{'source_manifest_sha256':verify(),'all_candidate_and_predecessor_inventory_files_unchanged':counts,'matching_bwrap_processes':orphans,'both_workers_reaped':True,'receipt_counts_match_ledger':True,'raw_usage_costs_match_receipts':True,'secrets_in_request_headers_persisted':False})
with tarfile.open(out/'preflight-fixtures.tar.gz','w:gz') as archive:archive.add('/tmp/arc-t3-operator/preflight-fixtures',arcname='fixtures')
(out/'launch-console.txt').write_bytes(Path('/tmp/arc-t3-operator/launch-console.txt').read_bytes())
# Common rubric: do not replace unresolved artifact review with treatment schema scores.
rubric={}
for arm in results:
 asset=out/'live'/arm/'actor/team_nemo/shared/knowledge'
 refs=[{'path':str(f.relative_to(out)),'sha256':sha(f)} for f in asset.glob('*.md')]
 rubric[arm]={'entered_observation':'present in executed analysis; canonical entry absent for treatment','retained_claim_with_tested_scope_and_open_conditions':'unresolved; inspect native notes' if refs else 'absent published asset','explicit_reconciliation':'unresolved' if refs else 'absent','decision_relevant_uncertainty_or_procedure':'unresolved; inspect native hypotheses' if refs else 'absent published asset','declared_mode':'absent explicit accepted mode','complete_prospective_record':'absent; zero proposals','post_observation_revision':'absent; second recorded transition not revealed','reusable_next_policy_without_transcript':'unresolved; notes alone not scored sufficient' if refs else 'absent published asset','raw_native_assets':refs,'common_workflow_completion':False,'counterevidence_opportunity':'second recorded continuation not reached','review_status':'independent T4 adjudication pending'}
save('WORKFLOW-REVIEW.json',rubric)
tokens={'input':sum(a['prompt_tokens'] for a in account['arms'].values()),'output':sum(a['completion_tokens'] for a in account['arms'].values())};tokens['total']=sum(tokens.values())
score={'task':'T3','status':'COMPLETE','operational_status':'one bounded pair completed; both constructions incomplete at six-request cap','model':'qwen/qwen3.8-27b','endpoint':'reka/fp8','authorization_sha256':sha(out/'T3-AUTHORIZATION.json'),'requests':{**{a:account['arms'][a]['requests'] for a in results},'total':account['requests']},'usage':{'tokens':tokens,'usd':account['total_cost_usd'],'unresolved_requests':0,'reasoning_tokens':sum(a['reasoning_tokens'] for a in account['arms'].values())},'arms':results,'delta':'Both arms have zero prospective proposals and incomplete common workflow; no demonstrated construction advantage. No efficacy inference.','source_identity_unchanged':True,'provider_retries':0,'metadata_get_requests':2,'scientific_scope':'recorded-prefix construction completion only','t4_audit':'pending'}
save('RESULT.json',score)
(out/'REPORT.md').write_text(f'''# T3: bounded pair completed; both constructions incomplete

Exact OpenRouter model `qwen/qwen3.8-27b`, endpoint `reka/fp8`. Fresh catalog had
one exact match and the endpoint passed the frozen metadata gate. User authorization
was persisted before dispatch. No substitutions, retries, repairs or reruns occurred.

| Arm | Requests | Input tokens | Output tokens | Accepted proposals | Construction complete |
|---|---:|---:|---:|---:|---|
| Control | 6 | 93,174 | 1,951 | 0 | No |
| Treatment | 6 | 95,638 | 538 | 0 | No |

Total: **188,812 input + 2,489 output = 191,301 tokens**, **$0.037821390**.
All twelve usage/cost receipts are resolved; reasoning tokens and unresolved exposure
are zero. The pair took 70.49 seconds. Each arm ended at its request ceiling.

Control inspected the grid and retained step/hypothesis notes. Treatment inspected
the grid, called the workflow skill, and inspected the Operational Theory schema;
its canonical cycle remained idle with no accepted theory versions. A schema-read
cell raised NameError for `_json`. No manual repair or calls beyond the six-request
allowance were made.
Neither arm submitted a prospective action, so the recorded second transition was
never revealed. No gameplay was executed. There is no demonstrated construction
advantage in this pair, and no evidence here about gameplay competence or efficacy.

`PARITY.json` verifies exact first-request equality after removing only the declared
treatment block, and identical nonmessage parameters on all twelve requests.
`INTEGRITY.json` verifies receipts, complete frozen inventories, reaping of both
workers and absence of matching remaining bwrap processes. `WORKFLOW-REVIEW.json`
keeps unresolved native-asset rubric judgments for independent T4 review; it does
not treat treatment-only schema metrics as a comparative effect statistic.

The frozen driver was invoked once. Its process-local HERE was relocated to this
external directory to satisfy the user's output-location requirement; the pinned
selected metadata was copied byte-for-byte for its gate. No frozen file changed.
`OUTPUT-RELOCATION.json`, operator source and authorization preserve this detail.
Raw request bodies, response bytes, receipts, host events, native assets, empty
canonical lineage state, accounting and supervision remain under `live/`. Headers
containing credentials were never persisted. Metadata GET evidence and preflight
fixtures are retained separately. Timestamp provenance is stated in TIMESTAMPS.json.

Next boundary: independent T4 post-run audit of these exact artifacts. No further
launch or replication is authorized. RESULT.json reports operational completion,
not successful construction or a scientific efficacy result.
''')
print(json.dumps(score,indent=2))
