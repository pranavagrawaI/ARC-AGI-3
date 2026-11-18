from pathlib import Path
import hashlib,json,ast,base64
HERE=Path(__file__).resolve().parent;PREP=HERE.parent;ROOT=PREP.parents[1];P=PREP/'r10a-package'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
for name in ['input-hashes.json','prior-evidence-hashes.json']:
 assert all(sha(ROOT/n)==h for n,h in json.loads((HERE/name).read_text()).items()),name+' changed'
actual={str(f.relative_to(P)):sha(f) for f in P.rglob('*') if f.is_file()};assert actual==json.loads((PREP/'R10A-PACKAGE-INVENTORY.json').read_text())['files']
assert all(actual[n]==h for n,h in json.loads((P/'ASSET-HASHES.json').read_text()).items())
assert (P/'EXPERIMENT-CARD.md').read_bytes()==(PREP/'R10A-EXPERIMENT-CARD.md').read_bytes()==(PREP/'r10a-cd82-20260916/EXPERIMENT-CARD.md').read_bytes()
old=json.loads((PREP/'R9-SOURCE-HASHES.json').read_text());assert all(sha(ROOT/n)==h for n,h in old.items())
fixture=HERE/'package-basetemp'
null=next(fixture.glob('test_admitted_null_state_next_0/probe.json'))
late=next(fixture.glob('test_post_child_write_hard_kil0/probe.json'))
n=json.loads(null.read_text());d=json.loads(late.read_text())
assert n['first_submit']==n['next_submit']=='submitted'
assert n['repeat_query']=='PREDICT_WITHOUT_ASSIMILATE' and n['wrong_step']==n['wrong_assimilation']=='REJECT_MISMATCH'
assert n['pending_after_rejection']==['CLICK 0 0',1]
a=[x for x in n['events'] if x['event']=='assimilate_frozen' and x['status']=='COMPLETED'];assert len(a)==1 and json.loads(a[0]['tool_result'])['verdict']=='unscored'
assert d['service']['construction_status']=='FAILED' and d['service']['worker_alive'] is False
assert d['job_frozen'] is True and d['root_frozen'] is False and d['audit_freeze'] is None and d['summary']['frozen'] is False
root=fixture/'test_actual_supervisor_worker_0/runtime-output/solver-workspaces'
r=json.loads((root/'raw/RESULTS.json').read_text());assert r['status']=='passed' and r['local_requests']['complete']
assert [g['actions'] for g in r['games']]==[3,3] and all(g['official']['reconciled'] for g in r['games'])
g=r['games'][1];session=root/g['workspace']/g['sessions'][0]['session_dir'];events=rows(session/'host-events.jsonl');transitions=rows(session/'ipc/transitions.jsonl');ctor=json.loads((session/'RESULT.json').read_text())
assert (ctor['model_requests'],ctor['inspections'],ctor['submissions'])==(6,2,4)
frozen=json.loads((session/'FROZEN.json').read_text());digest=hashlib.sha256(frozen['candidate_source'].encode()).hexdigest();assert frozen['source_digest']==digest
q=[e for e in events if e['event']=='query_frozen_predict' and e['status']=='COMPLETED'];a=[e for e in events if e['event']=='assimilate_frozen' and e['status']=='COMPLETED']
assert [e['action'] for e in q]==['CLICK 31 41','DOWN'] and len(a)==1
for e in q:
 matches=[t for t in transitions if t['action']==e['action'] and t['step']==e['step']+1];assert len(matches)==1 and e['source_digest']==digest
 assert [[int(c,16) for c in row] for row in e['observation']['grid_rows']]==matches[0]['before']['grid']
assert a[0]['executed_transition']==transitions[1] and json.loads(a[0]['tool_result'])['verdict']=='unscored'
assert not any((PREP/'r10a-cd82-20260916'/n).exists() for n in ['manifest.json','authorization.json','launch-review.json','hermes-release.json','CONSUMED.json','launch.json'])
result={'status':'RECONCILED','inventory_sha256':sha(PREP/'R10A-PACKAGE-INVENTORY.json'),'notebook_sha256':sha(P/'construction.ipynb'),'input_hashes_sha256':sha(HERE/'input-hashes.json'),'gate_hashes_record_sha256':sha(PREP/'repair-20260916/GATE-HASHES.json'),'prior_evidence_unchanged':True,'predecessor_files_unchanged':len(old),'null_probe':str(null.relative_to(ROOT)),'late_kill_probe':str(late.relative_to(ROOT)),'official_session':str(session.relative_to(ROOT)),'constructor_counts':[6,2,4],'queries':2,'assimilations':1,'final_pending':1,'production_unfrozen':True}
(HERE/'reconciliation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
