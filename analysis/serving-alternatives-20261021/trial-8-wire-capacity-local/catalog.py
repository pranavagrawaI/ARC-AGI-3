"""New API-capacity catalog. Correct local counts without changing any wire input."""
import copy
import json
from pathlib import Path
from corpus_base import canonical, digest, image_observations, payload_for, verify_corpus, wire_hash

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent
OLD_SHA='dfc1654de8a324f8142306e6459fcc3c47cee6ef2768f91f8a10dd3d0aeb2985'
AUDIT_SHA='1b657726e09f2eed4e7de631f108d1ebb478eb519b686b64b78f02df6a883fc0'

def corrected(old, rows):
    audit={r['key']:r for r in rows}
    if len(audit)!=1369 or len(rows)!=1369 or set(audit)!=set(old['records']):
        raise ValueError('audit keyset/uniqueness')
    result=copy.deepcopy(old)
    result.update(schema='wire-package-api-capacity-v1', old_catalog_sha=OLD_SHA, corrected_audit_sha=AUDIT_SHA)
    for key, record in result['records'].items():
        row=audit[key]
        counts=row['normalized_prompt'];renders=row['normalized_render_sha'];ids=row['normalized_token_ids_sha']
        if not (counts['control']==counts['candidate']==record['local_prompt']+11==row['old_prompt']+11
                and row['cap']==record['cap'] and row['images']==len(record['images'])
                and renders['control']==renders['candidate'] and ids['control']==ids['candidate']):
            raise ValueError('corrected count/source join '+key)
        record.update(old_local_prompt=record['local_prompt'],old_render_sha=record['render_sha'],
                      local_prompt=counts['control'],render_sha=renders['control'],token_ids_sha=ids['control'])
        if record['local_prompt']+record['cap']>139264:raise ValueError('context bound')
    return result

def build():
    raw=(PRIOR/'trial-4-saturated-local/catalog.json').read_bytes()
    audit=(PRIOR/'trial-5-serving-screen/prompt-audit/records.jsonl').read_bytes()
    if digest(raw)!=OLD_SHA or digest(audit)!=AUDIT_SHA:raise ValueError('reference identity')
    old=json.loads(raw);out=corrected(old,list(map(json.loads,audit.splitlines())))
    unchanged=('game','index','file','offset','prompt','cap','cached','images','wire_sha','extends_previous')
    proof=dict(records=len(out['records']),old_catalog_sha=OLD_SHA,audit_sha=AUDIT_SHA,
               keyset_unchanged=set(old['records'])==set(out['records']),
               order_unchanged=all(old[k]==out[k] for k in ('streams','initial_holders','initial_queue','slots')),
               source_wire_image_caps_unchanged=all(all(a[k]==out['records'][key][k] for k in unchanged) for key,a in old['records'].items()),
               every_count_delta_11=all(out['records'][k]['local_prompt']-a['local_prompt']==11 for k,a in old['records'].items()),
               catalog_sha=digest(canonical(out)))
    if not all(proof[k] for k in ('keyset_unchanged','order_unchanged','source_wire_image_caps_unchanged','every_count_delta_11')):raise ValueError('catalog drift')
    return out,proof

def load(path=HERE/'catalog.json'):
    raw=Path(path).read_bytes();proof=json.loads((HERE/'catalog-proof.json').read_text())
    if digest(raw)!=proof['catalog_sha']:raise ValueError('catalog identity')
    return json.loads(raw)

if __name__=='__main__':
    value,proof=build();(HERE/'catalog.json').write_bytes(canonical(value))
    (HERE/'catalog-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
