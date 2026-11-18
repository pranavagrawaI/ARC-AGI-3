"""Audit recoverable trial9 responses with the ORIGINAL API/accounting contract.

No full-log, paired, rate or archive qualification is asserted.
"""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'trial-9-readiness-local'
sys.path.insert(0,str(PRIOR))
import accounting,catalog,reduce_api

def screen():
    recovered=HERE/'evidence/partial-members'
    recovery=json.loads((recovered/'RECOVERY.json').read_text())
    root=recovered/'members/serving-wire-capacity9/output/control'
    events=list(map(json.loads,(PRIOR/'remote-readback/priority-events/serving-wire-capacity9/output/control/events.jsonl').read_text().splitlines()))
    sends={e['rid']:e for e in events if e['kind']=='send'}
    end=next(e['boundary'] for e in events if e['kind']=='measure_end')
    cat=catalog.load();rows=[];missing=[];seen=set()
    for event in (e for e in events if e['kind']=='response'):
        rid=event['rid'];directory=root/rid
        needed=[directory/n for n in ('body.bin','http.json','wire.http','wire.json')]
        if not all(p.exists() for p in needed):missing.append(dict(rid=rid,missing=[p.name for p in needed if not p.exists()]));continue
        row=dict(rid=rid,key=sends[rid]['key'],issues=[],response_event=event)
        try:
            body_raw=(directory/'body.bin').read_bytes();body=json.loads(body_raw)
            http=json.loads((directory/'http.json').read_text())
            actual=reduce_api.transport((directory/'wire.http').read_bytes(),rid)
            planned=json.loads((directory/'wire.json').read_text())['body'];record=cat['records'][row['key']]
            assert actual==planned and catalog.wire_hash(actual)==record['wire_sha'],'actual/planned/corpus wire'
            assert catalog.image_observations(actual)==record['images'],'actual PNG hashes/count'
            assert http['rid']==rid and http['key']==row['key'],'HTTP join'
            assert http['complete'] is True and http['status']==200 and http['error'] is None and http['termination']=='COMPLETE' and http['error_kind'] is None,'HTTP completion'
            lengths=[v for k,v in http['headers'] if k.lower()=='content-length']
            assert len(lengths)==1 and int(lengths[0])==len(body_raw)==http['received_bytes']==http['persisted_bytes'],'body bytes'
            assert http['body_completed_at']==http['ended'] and sends[rid]['t']<=http['started']<=http['ended']<=event['t']<end,'body/event clocks'
            assert event['t']-http['ended']<=.25,'publication lag'
            assert http['writes'] and all(w['t']<=w['ended']<end for w in http['writes']),'socket write clocks'
            assert sum(w['bytes'] for w in http['writes'])==len((directory/'wire.http').read_bytes()),'socket write bytes'
            response_id=body['id'];assert isinstance(response_id,str) and response_id and response_id not in seen,'response identity'
            seen.add(response_id)
            row['accounting']=accounting.observe('control',record,body)
            row['finish_reason']=body['choices'][0]['finish_reason']
        except (AssertionError,ValueError,KeyError,TypeError) as error:row['issues'].append(str(error))
        rows.append(row)
    result=dict(scope='RECOVERED_CRC_VERIFIED_COMPLETE_RESPONSE_SUBSET_ONLY',trial9_stays_invalid=True,ratio=None,
                complete_archive=False,central_directory_verified=False,remote_calls=0,
                recovery_receipt_sha256=hashlib.sha256((recovered/'RECOVERY.json').read_bytes()).hexdigest(),
                recovered_local_members=recovery['verified_members'],completed_response_events=541,
                joined_response_subset=len(rows),subset_pass=sum(not r['issues'] for r in rows),
                subset_fail=sum(bool(r['issues']) for r in rows),missing_complete_joins=len(missing),
                original_source_sha256={n:hashlib.sha256((PRIOR/n).read_bytes()).hexdigest() for n in ('reduce_api.py','accounting.py','catalog.py','catalog.json')},
                limitations=accounting.LIMITATIONS+['Recovered member subset selected by archive order; not representative sampling.',
                            'No whole-log/load/sample/stability/mix/length/paired/rate qualification.'],rows=rows,missing=missing)
    path=HERE/'evidence/partial-response-screen.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','missing')},indent=2))

if __name__=='__main__':screen()
