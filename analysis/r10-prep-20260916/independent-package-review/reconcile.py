import ast,base64,difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREP=HERE.parent
ROOT=PREP.parents[1]
P=PREP/'r10a-package'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
before=json.loads((HERE/'input-hashes.json').read_text())
assert all(sha(ROOT/n)==h for n,h in before.items()),'review inputs changed'
actual={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()}
assert actual==json.loads((PREP/'R10A-PACKAGE-INVENTORY.json').read_text())['files']
nb=json.loads((P/'construction.ipynb').read_text());tree=ast.parse(''.join(nb['cells'][1]['source']))
vals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('ASSETS','EXPECTED')}
assert set(vals['ASSETS'])==set(actual)-{'construction.ipynb','kernel-metadata.json'}
for name,data in vals['ASSETS'].items():
 b=base64.b64decode(data,validate=True);assert b==(P/name).read_bytes();assert hashlib.sha256(b).hexdigest()==vals['EXPECTED'][name]
old=json.loads((PREP/'R9-SOURCE-HASHES.json').read_text());assert all(sha(ROOT/n)==h for n,h in old.items())
changed=[];diff=''
for f in sorted(P.rglob('*.py')):
 prior=ROOT/'deployment/kaggle/actor-critic-construction-kaggle-r9-cd82'/f.relative_to(P)
 if prior.read_bytes()!=f.read_bytes():
  changed.append(str(f.relative_to(P)))
  diff+=''.join(difflib.unified_diff(prior.read_text().splitlines(True),f.read_text().splitlines(True),fromfile=str(prior.relative_to(ROOT)),tofile=str(f.relative_to(ROOT))))
(HERE/'recomputed-source-diff.patch').write_text(diff)
assert diff==(PREP/'R10A-SOURCE-DIFF.patch').read_text(),'source diff does not match'
fixtures=[]
for directory in ('basetemp-main','basetemp-delayed'):
 root=HERE/directory/'test_actual_supervisor_worker_0/runtime-output/solver-workspaces'
 result=json.loads((root/'raw/RESULTS.json').read_text());assert result['status']=='passed' and result['local_requests']['complete']
 assert [g['actions'] for g in result['games']]==[3,3] and all(g['official']['reconciled'] for g in result['games'])
 game=result['games'][1];session=root/game['workspace']/game['sessions'][0]['session_dir']
 events=rows(session/'host-events.jsonl');transitions=rows(session/'ipc/transitions.jsonl');frozen=json.loads((session/'FROZEN.json').read_text());digest=hashlib.sha256(frozen['candidate_source'].encode()).hexdigest();assert digest==frozen['source_digest']
 ctor=json.loads((session/'RESULT.json').read_text());assert (ctor['model_requests'],ctor['inspections'],ctor['submissions'])==(6,2,4)
 assert ctor['construction_repair']=={'accepted_after_repair':True,'failed_submissions':3}
 queries=[e for e in events if e['event']=='query_frozen_predict' and e['status']=='COMPLETED'];assims=[e for e in events if e['event']=='assimilate_frozen' and e['status']=='COMPLETED']
 assert [q['action'] for q in queries]==['CLICK 31 41','DOWN']
 for q in queries:
  matched=[r for r in transitions if r['action']==q['action'] and r['step']==q['step']+1];assert len(matched)==1 and q['source_digest']==digest
  grid=[[int(c,16) for c in row] for row in q['observation']['grid_rows']];assert grid==matched[0]['before']['grid']
 assert len(assims)==1 and assims[0]['executed_transition']==transitions[1] and json.loads(assims[0]['tool_result'])['verdict']=='unscored'
 for rel in game['sessions'][0]['constructor_files']:assert (root/game['workspace']/rel).is_file()
 activation=next(e for e in events if e['event']=='request_construction')
 fixtures.append({'fixture':directory,'session':str(session.relative_to(ROOT)),'constructor_wall_seconds':activation['wall_seconds'],'queries':2,'assimilated':1,'pending_at_stop':1,'calls':result['local_requests']['roles'],'repair':ctor['construction_repair']})
assert fixtures[1]['constructor_wall_seconds']>60
result={'input_snapshot_sha256':sha(HERE/'input-hashes.json'),'inventory_sha256':sha(PREP/'R10A-PACKAGE-INVENTORY.json'),'package_files':len(actual),'embedded_assets':len(vals['ASSETS']),'predecessor_files_unchanged':len(old),'changed_python_files':changed,'diff_matches':True,'inputs_unchanged':True,'fixtures':fixtures}
(HERE/'reconciliation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
