"""Read-only independent byte/provenance/metric audit. Prints JSON; no candidate imports."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
J=lambda p:json.loads(Path(p).read_bytes())
inv=J(P/'INVENTORY.json'); checked={}
for n,h in inv['files'].items():
 assert H(P/n)==h,n
 checked[n]=h
prior=J(P/'PRIOR-HASHES.json')
for n,h in prior.items(): assert H(n)==h,n
local=J(P/'LOCAL-DEMO-HASHES.json')
for n,h in local.items(): assert H(P/n)==h,n
pres=J(P/'PRESERVATION-CHECK.json')
r10=P.parent/'r10-prep-20260916/r10a-cd82-20260916/manifest.json'
assert H(r10)==pres['r10_manifest_sha256']
for n,h in J(r10)['sources'].items():assert H(n)==h,n
changed=[]
for n,v in J(P/'SOURCE-BINDINGS.json').items():
 assert H(v['source'])==v['sha256'],n
 if H(P/n)!=v['sha256']:changed.append(n)
assert sorted(changed)==['engine/constructor_contract.py','engine/evaluate_visible.py']
s=J(P/'recorded/SELECTION.json');raw=Path(s['source']).read_bytes();assert hashlib.sha256(raw).hexdigest()==s['source_sha256']
a,b=[json.loads(x) for x in raw.splitlines() if x.strip()][:2]
assert (P/'recorded/prefix.jsonl').read_bytes()==raw.splitlines(keepends=True)[0]
assert [a['step'],b['step'],a['turn'],b['turn']]==[1,2,0,1]
assert a['after']['grid']==b['before']['grid']
assert all(x['state']=='NOT_FINISHED' and x['levels_completed']==0 for x in [a['before'],a['after'],b['before'],b['after']])
y=J(P/'recorded/heldout/outcome.json');q=J(P/'recorded/query.json')
assert y=={'session':s['session'],'step':2,'action':b['action'],'grid':b['after']['grid']}
assert H(P/'recorded/heldout/outcome.json')==q['outcome_sha256']==s['outcome_sha256']
assert H(P/'EXPERIMENT-CARD.md')==s['card_sha256']
prov=J(P/'RECORDED-PROVENANCE.json');assert H(prov['session_mapping_path'])==prov['session_mapping_sha256']
d=P/'local-recorded-identity';p=d/'prediction';bound=J(p/'BOUND.json');f=d/'constructor/FROZEN.json'
for path,k in [(f,'freeze_sha256'),(P/'recorded/prefix.jsonl','prefix_sha256'),(P/'recorded/query.json','query_sha256'),(p/'PREDICTION.json','prediction_sha256'),(p/'RAW-PREDICTION.json','raw_prediction_sha256')]:assert H(path)==bound[k],k
assert hashlib.sha256(J(f)['candidate_source'].encode()).hexdigest()==bound['source_sha256']
assert J(d/'predict.stdout')['binding_sha256']==H(p/'BOUND.json')
c=[v for row in a['after']['grid'] for v in row];target=[v for row in y['grid'] for v in row];pred=J(p/'PREDICTION.json');forecast=[v for row in pred['commitment']['predicted_grid'] for v in row]
assert forecast==c
N=len(c);D=[i for i in range(N) if c[i]!=target[i]];U=[i for i in range(N) if c[i]==target[i]]
m={'cells':N,'changed_cells':len(D),'unchanged_cells':len(U),'coverage':1,'exact_match':forecast==target,'cell_accuracy':sum(x==z for x,z in zip(forecast,target))/N,'changed_cell_accuracy':sum(forecast[i]==target[i] for i in D)/len(D),'false_changes':sum(forecast[i]!=c[i] for i in U),'paired_cell_accuracy_difference':0.0}
score=J(p/'SCORE.json')
for k in ['cells','changed_cells','unchanged_cells','coverage','paired_cell_accuracy_difference']:assert score['metrics'][k]==m[k]
for k in ['exact_match','cell_accuracy','changed_cell_accuracy','false_changes']:assert score['metrics']['prediction'][k]==m[k]
assert score['fixture_constructor'] and not score['metrics']['predictive_success'] and not pred['model_evidence']
events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
assert [x['event'] for x in events]==['query_consumed','prediction_bound','target_read_begins_after_binding','score_persisted']
assert [x['time_ns'] for x in events]==sorted(x['time_ns'] for x in events)
assert J(p/'RAW-PREDICTION.json')['rpc_methods'][0]['method']=='predict' and len(J(p/'RAW-PREDICTION.json')['rpc_methods'])==1
print(json.dumps({'status':'PASS','inventory_sha256':H(P/'INVENTORY.json'),'inventory_files_verified':len(checked),'prior_files_verified':len(prior),'r10_sources_verified':len(J(r10)['sources']),'r10_manifest_sha256':H(r10),'demo_files_verified':len(local),'changed_inherited_files':changed,'source_trace_sha256':s['source_sha256'],'actions':[a['action'],b['action']],'mapping_sha256':prov['session_mapping_sha256'],'demo_binding_sha256':H(p/'BOUND.json'),'independent_metrics':m,'reviewed_input_hashes':checked},indent=2))
