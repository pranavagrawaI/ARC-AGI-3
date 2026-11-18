"""Mechanical audit only: labels are not epistemic correctness grades."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'analysis/local-probes/20260911-critic-deliberation-v1'
x=json.loads((OUT/'EXECUTION.json').read_text())
cases={c['id']:c for c in json.loads((OUT/'CASES.json').read_text())}
f=json.loads((OUT/'FROZEN.json').read_text())
result={'status':x['status'],'cost_usd':x['cost_usd'],
 'frozen_hashes_match':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in f['files'].items()),
 'requests':len(list(OUT.glob('*.request.json'))),'responses':len(list(OUT.glob('*.response.json'))),'arms':{}}
for arm in ['direct','thinking','explicit']:
    rows=[r for r in x['calls'] if r['arm']==arm and 'usage' in r]
    if not rows:continue
    result['arms'][arm]={'responses':len(rows),'delivery_valid':sum(r['status']=='valid' for r in rows),
       'mean_seconds':statistics.mean(r['seconds'] for r in rows),
       'completion_tokens':sum(r['usage']['completion_tokens'] for r in rows),
       'reasoning_tokens':sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in rows),
       'parsed_labels_matching_expected':sum(isinstance(r.get('parsed_critique'),dict) and r['parsed_critique'].get('assessment')==cases[r['case']]['expected'] for r in rows),
       'delivered_labels_matching_expected':sum(r['status']=='valid' and r['parsed_critique']['assessment']==cases[r['case']]['expected'] for r in rows)}
(OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
