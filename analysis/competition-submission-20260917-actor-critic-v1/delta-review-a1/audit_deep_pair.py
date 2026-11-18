import json,hashlib,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent;ROOT=R.parents[1];P=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
report={}
for label,pkg,expected in [('deep-before-reproduced',R/'checks/repair-a1/before-package',14),('deep-current',P,13)]:
 b=O/label;q=b/'notebook/working/qualification';nb=json.loads((pkg/'submission.ipynb').read_text())
 assert (b/'execute-notebook.py').read_text()=='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
 for n,h in json.loads((pkg/'ASSETS.json').read_text()).items():assert hashlib.sha256((b/'notebook'/n).read_bytes()).hexdigest()==h
 req=rows(b/'model-requests.jsonl');ledger=[v for p in q.glob('request-budget-*.jsonl') for v in rows(p)];calls=[v for p in q.rglob('model_calls.jsonl') for v in rows(p)]
 deep=list(q.rglob('deep_investigations.jsonl'));lr=json.loads((q/'launcher-result.json').read_text());res=json.loads((b/'RESULT.json').read_text())
 assert len(req)==expected and len(ledger)==len(calls)==13
 assert len(deep)==expected-13
 assert all(x['state']=='WIN' and x['levels_completed']==2 for x in lr['games']) and len(lr['games'])==2
 refusal='deep investigation is disabled under competition request accounting' in json.dumps(req)
 assert refusal==(expected==13)
 extra=[v for v in req if not v['request'].get('tools') and v['request'].get('max_tokens')!=192]
 assert len(extra)==expected-13
 assert res['status']==('passed' if expected==13 else 'reconciliation_failed')
 origins=json.loads((b/'dependency-origins.json').read_text());assert origins['python'].startswith('3.12') and all('site-packages' in v for v in origins['origins'].values())
 report[label]=dict(server=expected,ledger=13,ipc_calls=13,deep_files=len(deep),deep_requests=len(extra),refusal_observed=refusal,result=res['status'],official_completed_games=2,materialized_assets_match=True,origins=origins)
# Independent supervisor re-reconciliation of fresh repaired 110 raw run.
b=R/'checks/repair-a1/gateway-110-cp312';q=b/'notebook/working/qualification'
s=importlib.util.spec_from_file_location('outer_audit',b/'notebook/official_launcher_runtime.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
lr=json.loads((q/'launcher-result.json').read_text());count=len(rows(b/'model-requests.jsonl'));assert count==660
facts=m.reconcile_launcher_request_count(lr,dict(chat_completion_accesses=1,successful_chat_completion_accesses=1),dict(chat_completion_accesses=661,successful_chat_completion_accesses=661,successful_freeze_gc_accesses=1,ready_marker_count=1));assert facts['whole_model_request_count_verified'];report['110_recomputed_supervisor']=facts
(O/'deep-pair-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
