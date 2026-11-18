"""One final pass: exact external authority, shared900s and no automatic successor."""
import argparse,json,sys,time
from pathlib import Path
import settings,ledger
from settings import HERE,CAPS,POLICY,FIRST_SHA,SECOND_SHA,sha,read,require
from collector import collect
from retry_transport import RetryRemote
DECISION='TRIAL16_FINAL_RESIDUAL_TRANSPORT_REMOTE_APPROVED'
RETRY_POLICY=dict(exception='requests.ConnectTimeout',response_absent=True,received_bytes=0,persisted_bytes=0,additional_attempts=1,delay_seconds=1,connect_timeout_seconds=10,read_timeout_seconds=30,no_further_automatic_pass=True)

def approval_fields(value):
    require(value.get('decision')==DECISION and value.get('remote_authorized') is True,'no final transport remote authority')
    require((value.get('kernel_id'),value.get('version'))==(140493086,1),'final run identity')
    require(value.get('bounds')==POLICY and value.get('retry_policy')==RETRY_POLICY,'exact final bounds/retry policy')
    require(value.get('first_collection_sha256')==FIRST_SHA and value.get('second_collection_sha256')==SECOND_SHA,'failed-pass authority bindings')

def authorize(path):
    settings.verify_local();value=read(path);approval_fields(value)
    require(value.get('source_pins_sha256')==sha(HERE/'SOURCE-PINS.json') and value.get('manifest_sha256')==sha(HERE/'FINAL-MANIFEST.json'),'final source/manifest authority')
    return value

def persist(destination,state,started,clock=time.monotonic):
    state['driver_elapsed_seconds']=clock()-started;path=destination/'COLLECTION.json';path.write_text(json.dumps(state,indent=2)+'\n')
    d=dict(schema='FINAL_RESIDUAL_DRIVER_COMPLETION_V1',started=started,deadline=started+900,receipt_persisted_end=clock(),receipt_sha256=sha(path))
    (destination/'DRIVER-COMPLETION.json').write_text(json.dumps(d,indent=2)+'\n')
    if clock()>=started+900:
        state.update(complete=False,error=dict(type='Stop',reason='final driver persistence exceeded cooperative deadline'));path.write_text(json.dumps(state,indent=2)+'\n')
    return state

def main():
    started=time.monotonic();p=argparse.ArgumentParser();p.add_argument('authorization',type=Path);p.add_argument('destination',type=Path);args=p.parse_args()
    approval=authorize(args.authorization);plan=ledger.prepare();manifest=read(HERE/'FINAL-MANIFEST.json')
    require(manifest['files']==plan['remaining'] and manifest['bounds']==POLICY and manifest['retry_policy']==RETRY_POLICY,'exact final manifest')
    require(not args.destination.exists(),'fresh exclusive destination required')
    from space import check
    require(check(args.destination.parent,'archive')['eligible'],'13GiB simultaneous reserve unavailable')
    with (HERE/'FINAL-INVOKED.json').open('x') as stream:json.dump(dict(approval_sha256=sha(args.authorization),destination=str(args.destination.resolve()),started=started),stream)
    from remote import Remote
    api=RetryRemote(Remote(),{r['name'] for r in plan['remaining']},args.destination,started+900)
    remaining=started+900-time.monotonic();require(remaining>0,'final preflight exhausted cooperative cap')
    state=collect(plan['remaining'],args.destination,api,limits=CAPS|dict(wall_seconds=remaining))
    state.update(collection_pass=3,transport_attempts=api.attempts,transport_journal_sha256=sha(args.destination/'transport-attempts.jsonl') if (args.destination/'transport-attempts.jsonl').exists() else None)
    (args.destination/'AUTHORITY.json').write_text(json.dumps(dict(approval=approval,approval_path=str(args.authorization.resolve()),approval_sha256=sha(args.authorization)),indent=2)+'\n')
    state=persist(args.destination,state,started)
    print(json.dumps({k:state.get(k) for k in ('complete','download_attempts','transport_attempts','downloaded_bytes','error')}))
    return 0 if state['complete'] else 1
if __name__=='__main__':sys.exit(main())
