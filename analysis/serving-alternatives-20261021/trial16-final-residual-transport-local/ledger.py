"""Three actual pass receipts and per-GET retry records, never a synthetic ZIP/pass."""
import datetime as dt,importlib.util,json,sys
from collections import Counter
from pathlib import Path
from settings import *

def prepare():
    require(sha(REVIEW)==REVIEW_SHA and sha(SECOND/'COLLECTION.json')==SECOND_SHA,'root/second receipt drift')
    review=read(REVIEW);first=prior_bridge.prepare();state=read(SECOND/'COLLECTION.json')
    allowed2={r['name']:r for r in first['remaining']}
    files2,attempts2=prior_bridge.ledger(SECOND,state,allowed2,prior_context.CAPS,False)
    driver2=prior_bridge.driver(SECOND,600,'RESIDUAL_DRIVER_COMPLETION_V1');before2=prior_bridge.readback(SECOND,state,'before')
    require(len(files2)==19 and len(attempts2)==20 and state['downloaded_bytes']==3751799,'fixed second ledger')
    require(state['error']['type']=='ConnectTimeout' and state['safe_error']['http_response']=='UNAVAILABLE' and 'after' not in state and not (SECOND/'after').exists(),'second failure/absent after')
    require(state['partial']['name']==attempts2[-1]==FAILED and state['partial']['bytes_persisted']==0 and not (SECOND/'members'/FAILED).exists(),'second partial boundary')
    require(set(attempts2)-set(files2)=={FAILED} and not set(files2)&set(first['first_files']),'second completion gap/duplicate retained file')
    from run_residual import authorize
    authority=read(SECOND/'AUTHORITY.json')
    require(authority['approval_sha256']==sha(authority['approval_path']) and authority['approval']==authorize(Path(authority['approval_path'])),'second authority binding')
    gaps=SECOND/'stopped-proof/REMAINING-GAPS.json';require(sha(gaps)==GAPS_SHA==review['remaining_gaps_sha256'],'fixed residual167 gap proof')
    remaining=[r for n,r in sorted(allowed2.items()) if n not in files2]
    require([{k:v for k,v in r.items() if k!='attempt_state'} for r in read(gaps)['files']]==remaining and len(remaining)==167,'exact final residual rows')
    retained=first['first_files']|files2;counts=Counter(first['first_attempts']+attempts2)
    require(len(retained)==1419 and review['retained_files']==1419 and review['residual_names']==167 and review['proposed_bounds']==POLICY,'final count/cap contract')
    require(counts[REPEAT75]==2 and counts[FAILED]==1 and sum(counts.values())==1421 and all(v==1 for n,v in counts.items() if n!=REPEAT75),'prior lifetime attempt counts')
    return dict(first=first,second=state,second_files=files2,second_attempts=attempts2,second_driver=driver2,second_before=before2,retained_files=retained,remaining=remaining,allowed=first['allowed'],waivers=first['waivers'],prior_attempt_counts=counts)


def transport_journal(directory,state,files):
    records=[json.loads(x) for x in (directory/'transport-attempts.jsonl').read_bytes().splitlines()]
    require(sha(directory/'transport-attempts.jsonl')==state['transport_journal_sha256'],'transport journal hash')
    require(len(records)==2*state['transport_attempts'] and len(files)<=state['transport_attempts']<=min(334,2*len(files)),'transport record/attempt cap')
    seen=Counter();completed=set();pending=None;errors=0
    for offset in range(0,len(records),2):
        start,outcome=records[offset:offset+2];number=offset//2+1;name=start['name']
        require(name in files and start['event']=='START' and name not in completed,'unexpected/repeated successful transport name')
        require(all(r['collection_pass']==3 and r['global_attempt']==number and r['name']==name for r in (start,outcome)),'transport pass/name/index')
        seen[name]+=1;require(start['attempt_in_pass']==outcome['attempt_in_pass']==seen[name]<=2,'per-file transport attempt cap')
        if pending:
            require(name==pending['name'] and seen[name]==2 and start['at']>=pending['at']+1,'retry name/delay contract');pending=None
        else:require(seen[name]==1,'unjustified second attempt')
        require(start['at']<=outcome['at'],'attempt chronology')
        if outcome['event']=='ERROR':
            require(seen[name]==1 and outcome['phase']=='GET' and outcome['type']=='ConnectTimeout' and outcome['response_absent'] is True and outcome['received_bytes']==outcome['persisted_bytes']==0 and outcome['retry_eligible'] is True,'non-retryable transport error')
            pending=outcome;errors+=1
        else:
            require(outcome['event']=='RESPONSE_CONTEXT_EXITED' and outcome['received_bytes']==outcome['persisted_bytes']==files[name]['bytes'] and outcome['retry_eligible'] is False,'transport completion byte accounting')
            completed.add(name)
    require(pending is None and completed==set(files),'transport dependency gap')
    return seen,errors


def lifetime_counts(plan,third_counts):
    require(set(third_counts)=={r['name'] for r in plan['remaining']} and all(1<=v<=2 for v in third_counts.values()),'final residual transport partition')
    counts=plan['prior_attempt_counts']+third_counts
    require(counts[REPEAT75]==2 and 2<=counts[FAILED]<=3,'named lifetime attempt exception')
    require(all(v<=2 for n,v in counts.items() if n not in (FAILED,REPEAT75)),'unapproved lifetime repeat')
    require(not set(third_counts)&set(plan['retained_files']),'redownload of retained successful artifact')
    return counts


def validate(third):
    plan=prepare();third=Path(third);state=read(third/'COLLECTION.json');allowed={r['name']:r for r in plan['remaining']}
    files,names=prior_bridge.ledger(third,state,allowed,CAPS,True)
    driver=prior_bridge.driver(third,900,'FINAL_RESIDUAL_DRIVER_COMPLETION_V1')
    before=prior_bridge.readback(third,state,'before');after=prior_bridge.readback(third,state,'after')
    times=[dt.datetime.fromisoformat(r['utc']) for r in (plan['first']['first_before'],plan['second_before'],before,after)]
    require(times==sorted(times),'three-pass identity chronology')
    require(state['collection_pass']==3 and state['transport_attempts']<=334,'third-pass schema/cap')
    counts,retries=transport_journal(third,state,files);lifetime=lifetime_counts(plan,counts)
    records=[json.loads(x) for x in (third/'transport-attempts.jsonl').read_bytes().splitlines()]
    require(all(driver['started']<=r['at']<=driver['receipt_persisted_end']<driver['deadline'] for r in records),'transport events outside shared driver window')
    from run_final import authorize
    a=read(third/'AUTHORITY.json');require(a['approval_sha256']==sha(a['approval_path']) and a['approval']==authorize(Path(a['approval_path'])),'third authority binding')
    require(set(plan['retained_files'])|set(files)|set(plan['waivers'])==set(plan['allowed']),'original1596 dependency gap')
    report=dict(schema='TRIAL16_THREE_PASS_REQUIRED_FILES_V1',first_complete=False,second_complete=False,first_after_available=False,second_after_available=False,third_complete=True,
        first=dict(receipt_sha256=FIRST_SHA,driver=plan['first']['first_driver'],caps=read(APPROVED/'MISSING-MANIFEST.json')['bounds'],files=1400,attempts=1401,bytes=248479141),
        second=dict(receipt_sha256=SECOND_SHA,driver=plan['second_driver'],caps=prior_context.CAPS,files=19,attempts=20,bytes=3751799),
        third=dict(receipt_sha256=sha(third/'COLLECTION.json'),driver=driver,caps=POLICY,files=167,file_invocations=167,GET_attempts=state['transport_attempts'],eligible_retries=retries,bytes=state['downloaded_bytes']),
        downloaded_files=1586,conditional_waivers=list(plan['waivers'].values()),total_GET_attempts=sum(lifetime.values()),lifetime_counts=dict(lifetime),
        transport_journal_sha256=state['transport_journal_sha256'],original_archive_complete=False,central_directory_verified=False,no_further_automatic_pass=True,
        clock_scope='three separate cooperative driver clocks; no synthetic complete single pass')
    return plan,files,after,report


def assemble(third,destination,plan,third_files):
    spec=importlib.util.spec_from_file_location('approved_audit_r2',AUDIT_R2/'audit.py');module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(AUDIT_R2));spec.loader.exec_module(module)
    first=plan['first'];evidence,origins=module.merge_inputs(FIRST,destination,first['first_files'],plan['allowed'])
    for name in first['first_files']:
        for row in origins[name]:row.update(collection_pass=1,collection_receipt_sha256=FIRST_SHA)
    prior_bridge.append_second(SECOND,evidence,origins,plan['second_files'])
    append_third(third,evidence,origins,third_files)
    return evidence,origins

def append_third(third,evidence,origins,third_files):
    prior_bridge.append_second(third,evidence,origins,third_files)
    for name in third_files:
        for row in origins[name]:row['collection_pass']=3
    return evidence,origins
