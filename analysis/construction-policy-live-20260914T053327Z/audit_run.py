#!/usr/bin/env python3
"""Fail-closed post-run audit and trace-local classification."""
from decimal import Decimal
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; PACKAGE=REPO/'deployment/kaggle'/HERE.name; E=HERE/'evidence'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def require(ok,label,errors):
    if not ok: errors.append(label)
def json_files(root):
    for p in root.rglob('*.json'):
        try: yield p,load(p)
        except Exception: continue
def recurse(value):
    if isinstance(value,dict):
        yield value
        for v in value.values(): yield from recurse(v)
    elif isinstance(value,list):
        for v in value: yield from recurse(v)
def main():
    errors=[]; details={}
    manifest_sha=sha(HERE/'MANIFEST.json'); card_sha=sha(HERE/'EXPERIMENT-CARD.md')
    auth=load(HERE/'AUTHORIZATION.json'); review=load(HERE/'REVIEW-RESOLVED.json'); consumed=load(HERE/'LAUNCH-CONSUMED.json'); launch=load(E/'LAUNCH-RESULT.json')
    require(auth.get('manifest_sha256')==manifest_sha and auth.get('card_sha256')==card_sha and auth.get('used') is False,'AUTH_BINDING',errors)
    require(review=={'status':'RESOLVED','identity':HERE.name,'manifest_sha256':manifest_sha,'card_sha256':card_sha,'notebook_sha256':sha(PACKAGE/'construction.ipynb')},'REVIEW_BINDING',errors)
    require(consumed.get('manifest_sha256')==manifest_sha and consumed.get('no_retry') is True,'CONSUMPTION_BINDING',errors)
    require(launch.get('pushes')==1 and launch.get('retries')==0 and launch.get('returned_identity')==load(PACKAGE/'kernel-metadata.json')['id'] and launch.get('version')==1,'LAUNCH_IDENTITY',errors)
    status=load(E/'status-exact-final.json'); require(status.get('status') in {'COMPLETE','ERROR','CANCELLED'},'EXACT_TERMINAL_STATUS',errors)
    readback=E/'server-readback'; metas=list(readback.rglob('kernel-metadata.json')); notebooks=list(readback.rglob('construction.ipynb'))
    require(len(metas)==1,'SERVER_METADATA_UNIQUE',errors); require(len(notebooks)==1,'SERVER_SOURCE_UNIQUE',errors)
    if len(metas)==1:
        server=load(metas[0]); expected=load(HERE/'SERVER-REFERENCE.json')
        for k,v in expected.items(): require(server.get(k)==v,'SERVER_METADATA:'+k,errors)
    if len(notebooks)==1:
        local=load(PACKAGE/'construction.ipynb'); remote=load(notebooks[0]); require(local==remote,'SERVER_NOTEBOOK_COMPLETE_JSON',errors)
        require([c.get('source') for c in local.get('cells',[]) if c.get('cell_type')=='code']==[c.get('source') for c in remote.get('cells',[]) if c.get('cell_type')=='code'],'SERVER_EXECUTABLE_BYTES',errors)
        details['server_raw_notebook_sha256']=sha(notebooks[0]); details['local_notebook_sha256']=sha(PACKAGE/'construction.ipynb')
    retrieval=load(E/'RETRIEVAL.json'); require(retrieval.get('exact_ref')==f"{launch.get('returned_identity')}/{launch.get('version')}",'RETRIEVAL_PROVENANCE',errors)
    require(not any(any(word in x['path'].lower() for word in ('.env','credential','secret','token','.git')) for x in retrieval.get('retained',[])),'RAW_ALLOWLIST',errors)
    remote=E/'remote'; results=[(p,v) for p,v in json_files(remote) if p.name=='RESULT.json' and isinstance(v,dict) and 'construction_status' in v]
    require(len(results)==1,'CONSTRUCTION_RESULT_UNIQUE',errors); result=results[0][1] if len(results)==1 else {}
    evidence=results[0][0].parent if len(results)==1 else remote
    requests=list((evidence/'requests').glob('*.json')) if (evidence/'requests').exists() else []
    responses=list((evidence/'responses').glob('*.json')) if (evidence/'responses').exists() else []
    receipts=list((evidence/'http-receipts').glob('*.json')) if (evidence/'http-receipts').exists() else []
    wire=list((evidence/'local-wire').glob('*.json')) if (evidence/'local-wire').exists() else []
    raw=list((evidence/'raw-http').glob('*.body')) if (evidence/'raw-http').exists() else []
    require(len(requests)==result.get('model_requests',-1),'REQUEST_ACCOUNTING',errors)
    require(len(responses)==len(receipts)==len(raw)==result.get('model_requests',-1),'RESPONSE_RECEIPT_ACCOUNTING',errors)
    require(len(wire)>=len(requests),'WIRE_ACCOUNTING',errors)
    usage=result.get('usage',{}); require(isinstance(usage,dict) and {'prompt_tokens','completion_tokens'}<=set(usage),'USAGE_PRESENT',errors)
    require(result.get('model_requests',99)<=16 and result.get('inspections',99)<=10 and result.get('submissions',99)<=4,'POLICY_COUNTS',errors)
    require(usage.get('completion_tokens',999999)<=16384 and usage.get('prompt_tokens',999999)<=300000,'TOKEN_CAPS',errors)
    lifecycle=(evidence/'lifecycle.jsonl').read_text().splitlines() if (evidence/'lifecycle.jsonl').exists() else []
    events=[json.loads(x) for x in lifecycle]
    suffix_attempts=sum(x.get('event')=='suffix_attempt' for x in events); suffix_assessments=sum(x.get('event')=='suffix_assessment' for x in events)
    require(suffix_attempts==result.get('suffix_attempts') and suffix_attempts<=1 and suffix_assessments<=1,'SUFFIX_ACCOUNTING',errors)
    if suffix_attempts:
        require((evidence/'FROZEN.json').is_file() and (evidence/'frozen_candidate.py').is_file() and (evidence/'SUFFIX-ATTEMPT.json').is_file(),'FREEZE_BEFORE_SUFFIX_FILES',errors)
        if (evidence/'FROZEN.json').is_file() and (evidence/'frozen_candidate.py').is_file(): require(load(evidence/'FROZEN.json').get('model_sha256')==sha(evidence/'frozen_candidate.py'),'FROZEN_HASH',errors)
        names=[x.get('event') for x in events]; require(names.index('frozen')<names.index('suffix_attempt'),'FREEZE_BEFORE_SUFFIX_ORDER',errors)
    launchers=[v for _,v in json_files(remote) if isinstance(v,dict) and v.get('schema')=='construction-official-launcher-result-v2']
    require(len(launchers)==1,'OFFICIAL_LAUNCHER_RESULT_UNIQUE',errors)
    if launchers: require(launchers[0].get('gameplay_result') is None,'NO_GAMEPLAY',errors)
    docs=[d for _,v in json_files(remote) for d in recurse(v)]
    gpu_values=[str(d.get(k)) for d in docs for k in ('observed_accelerator','gpu_name','device_name') if d.get(k)]
    require(any('RTX' in x.upper() and '6000' in x for x in gpu_values),'ACTUAL_GPU_NOT_ESTABLISHED',errors)
    cleanup=[d[k] for d in docs for k in ('cleanup_complete','package_cleanup_complete','closed','reaped') if isinstance(d.get(k),bool)]
    require(bool(cleanup) and all(cleanup),'CLEANUP_OR_REAPING_NOT_ESTABLISHED',errors)
    quotas=[]
    for name in ('quota-before.json','quota-after.json'):
        receipt=load(E/name); require(receipt.get('returncode')==0,'QUOTA_RECEIPT:'+name,errors)
        try: quotas.append(next(x for x in json.loads(receipt['stdout']) if x.get('resource')=='GPU'))
        except Exception: errors.append('QUOTA_PARSE:'+name)
    if len(quotas)==2:
        try: delta=Decimal(quotas[1]['used'][:-1])-Decimal(quotas[0]['used'][:-1]); require(Decimal('0')<=delta<=Decimal('1.0'),'GPU_HOUR_CAP',errors); details['displayed_gpu_hours']=str(delta)
        except Exception: errors.append('GPU_QUOTA_UNITS')
    if errors: classification='INFRASTRUCTURE_INVALID'; construction_completed=False; prospective=False
    elif result.get('classification')=='INTEGRATION_VALID_CANDIDATE_SUPPORTED': classification='CONSTRUCTION_COMPLETED_PROSPECTIVELY_SUPPORTED_ON_THIS_TRACE_ONLY'; construction_completed=True; prospective=True
    elif result.get('classification')=='INTEGRATION_VALID_CANDIDATE_NOT_SUPPORTED': classification='CANDIDATE_NOT_SUPPORTED'; construction_completed=True; prospective=False
    elif result.get('classification')=='INTEGRATION_VALID_BUDGET_EXHAUSTED': classification='BUDGET_EXHAUSTED_NO_CANDIDATE'; construction_completed=False; prospective=False
    elif result.get('classification')=='INTEGRATION_VALID_NEEDS_EVIDENCE': classification='NEEDS_EVIDENCE'; construction_completed=False; prospective=False
    else: classification='INFRASTRUCTURE_INVALID'; errors.append('UNMAPPED_RESULT'); construction_completed=False; prospective=False
    out={'classification':classification,'machinery_valid':not errors,'construction_completed':construction_completed,'prospectively_supported_on_this_trace_only':prospective,'general_competence_established':False,'terminal_status':status.get('status'),'pushes':1,'retries':0,'errors':errors,**details}
    (E/'AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True)); return 0 if not errors else 2
if __name__=='__main__': raise SystemExit(main())
