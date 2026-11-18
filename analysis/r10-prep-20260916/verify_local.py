"""Read-only R10a evidence audit, no verdict substitution or remote calls."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
P=HERE/'r10a-package'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def main():
    inventory=json.loads((HERE/'R10A-PACKAGE-INVENTORY.json').read_text())['files']
    actual={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()}
    assert inventory==actual
    assert not any('__pycache__' in x or x.endswith('.pyc') for x in actual)
    old=json.loads((HERE/'R9-SOURCE-HASHES.json').read_text())
    assert all(sha(ROOT/name)==value for name,value in old.items())
    nb=json.loads((P/'construction.ipynb').read_text())
    tree=ast.parse(''.join(nb['cells'][1]['source']))
    embedded=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ASSETS'))
    for name,value in embedded.items():
        raw=base64.b64decode(value,validate=True)
        assert (P/name).read_bytes()==raw
        if name.endswith('.py'):compile(raw,name,'exec')
    meta=json.loads((P/'kernel-metadata.json').read_text())
    assert meta['is_private'] is True and meta['enable_internet'] is False
    assert meta['competition_sources']==['arc-prize-2026-arc-agi-3']
    assert (P/'EXPERIMENT-CARD.md').read_bytes()==(HERE/'R10A-EXPERIMENT-CARD.md').read_bytes()
    sys.path.insert(0,str(P));import audit
    results=[]
    for directory in ('local-final','local-long-constructor-v2'):
        root=HERE/directory/'test_actual_supervisor_worker_0/runtime-output/solver-workspaces'
        result=json.loads((root/'raw/RESULTS.json').read_text())
        assert result['status']=='passed' and result['local_requests']['complete']
        assert result['local_requests']['roles']['constructor']==6
        assert [g['actions'] for g in result['games']]==[3,3]
        assert all(g['official']['reconciled'] for g in result['games'])
        game=result['games'][1];session=root/game['workspace']/game['sessions'][0]['session_dir']
        events=rows(session/'host-events.jsonl');transitions=rows(session/'ipc/transitions.jsonl')
        freeze=json.loads((session/'FROZEN.json').read_text());digest=hashlib.sha256(freeze['candidate_source'].encode()).hexdigest()
        assert freeze['source_digest']==digest
        assert audit.provider_constructed_source(root/'raw',digest)
        ctor=json.loads((session/'RESULT.json').read_text())
        assert ctor['model_requests']==6 and ctor['inspections']==2 and ctor['submissions']==4
        assert ctor['construction_repair']=={'accepted_after_repair':True,'failed_submissions':3}
        queries=[e for e in events if e['event']=='query_frozen_predict' and e['status']=='COMPLETED']
        assims=[e for e in events if e['event']=='assimilate_frozen' and e['status']=='COMPLETED']
        assert [q['action'] for q in queries]==['CLICK 31 41','DOWN']
        for q in queries:
            matches=[r for r in transitions if r['action']==q['action'] and r['step']==q['step']+1]
            assert len(matches)==1 and q['source_digest']==digest
        assert len(assims)==1 and assims[0]['action']=='CLICK 31 41'
        assert json.loads(assims[0]['tool_result'])['verdict']=='unscored'
        assert assims[0]['executed_transition']==transitions[1]
        assert audit.bound_use(queries,assims,transitions)
        wrong=[{**assims[0],'action':'CLICK 30 41'}]
        assert not audit.bound_use(queries,wrong,transitions), 'coordinates must not collapse'
        for entry in game['sessions'][0]['constructor_files']:
            assert (root/game['workspace']/entry).is_file()
        assert len(game['sessions'][0]['constructor_files'])>=4
        activation=next(e for e in events if e['event']=='request_construction')
        if directory=='local-long-constructor-v2':assert activation['wall_seconds']>=66
        results.append({'fixture':directory,'session':str(session.relative_to(ROOT)),'constructor_requests':6,'constructor_wall_seconds':activation['wall_seconds'],'repairs':ctor['construction_repair'],'bound_queries':len(queries),'bound_assimilations':len(assims),'pending_at_stop':1,'semantic_accuracy':'unscored','official_actions':[g['actions'] for g in result['games']]})
    launch_names=('manifest.json','authorization.json','launch-review.json','CONSUMED.json','launch.json')
    assert not any((HERE/'r10a-cd82-20260916'/n).exists() for n in launch_names)
    print(json.dumps({'status':'LOCAL_EVIDENCE_VERIFIED','scope':'scripted loopback/official public game; no target model/GPU or remote outcome','package_files':len(actual),'embedded_assets':len(embedded),'preserved_r9_files':len(old),'fixtures':results,'remote_calls':0,'freeze':False,'launch':False},indent=2))
if __name__=='__main__':main()
