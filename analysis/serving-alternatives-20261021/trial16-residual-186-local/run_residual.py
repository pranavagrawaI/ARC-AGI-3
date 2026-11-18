"""One separate residual invocation; remote authority must be issued after local review."""
import argparse,json,sys,time
from pathlib import Path
import context,bridge
from context import HERE,CAPS,REPEAT,FIRST_SHA,sha,read,require
from collector import collect
DECISION='TRIAL16_RESIDUAL_186_REMOTE_COLLECTION_APPROVED'


def approval_fields(receipt):
    require(receipt.get('decision')==DECISION and receipt.get('remote_authorized') is True,'no exact residual remote authority')
    require((receipt.get('kernel_id'),receipt.get('version'))==(140493086,1),'residual authority identity')
    require(receipt.get('bounds')==CAPS and receipt.get('repeat_connection_name')==REPEAT,'residual bounds/only repeated connection')
    require(receipt.get('first_collection_sha256')==FIRST_SHA,'first-pass authority binding')


def authorize(path):
    context.verify_local();receipt=read(path);approval_fields(receipt)
    require(receipt.get('source_pins_sha256')==sha(HERE/'SOURCE-PINS.json') and receipt.get('manifest_sha256')==sha(HERE/'RESIDUAL-MANIFEST.json'),'residual authority source/manifest')
    return receipt


def persist(destination,result,started,clock=time.monotonic):
    destination=Path(destination);deadline=started+600
    result['driver_elapsed_seconds']=clock()-started
    path=destination/'COLLECTION.json';path.write_text(json.dumps(result,indent=2)+'\n')
    completion=dict(schema='RESIDUAL_DRIVER_COMPLETION_V1',started=started,deadline=deadline,receipt_persisted_end=clock(),receipt_sha256=sha(path))
    (destination/'DRIVER-COMPLETION.json').write_text(json.dumps(completion,indent=2)+'\n')
    if clock()>=deadline:
        result.update(complete=False,error=dict(type='Stop',reason='residual driver persistence exceeded cooperative deadline'))
        path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    started=time.monotonic();p=argparse.ArgumentParser();p.add_argument('authorization',type=Path);p.add_argument('destination',type=Path);args=p.parse_args()
    approval=authorize(args.authorization);plan=bridge.prepare();manifest=read(HERE/'RESIDUAL-MANIFEST.json')
    require(manifest['files']==plan['remaining'] and manifest['bounds']==CAPS,'residual manifest drift')
    require(not args.destination.exists(),'fresh exclusive destination required')
    from space import check
    require(check(args.destination.parent,'archive')['eligible'],'13GiB simultaneous reserve unavailable')
    with (HERE/'RESIDUAL-INVOKED.json').open('x') as marker:
        marker.write(json.dumps(dict(approval_sha256=sha(args.authorization),destination=str(args.destination.resolve()),started=started))+'\n')
    remaining=600-(time.monotonic()-started);require(remaining>0,'preflight exhausted residual driver cap')
    from remote import Remote
    result=collect(plan['remaining'],args.destination,Remote(),limits=CAPS|{'wall_seconds':remaining})
    (args.destination/'AUTHORITY.json').write_text(json.dumps(dict(approval=approval,approval_path=str(args.authorization.resolve()),approval_sha256=sha(args.authorization)),indent=2)+'\n')
    result=persist(args.destination,result,started)
    print(json.dumps({k:result.get(k) for k in ('complete','list_pages','download_attempts','downloaded_bytes','error')}))
    return 0 if result['complete'] else 1
if __name__=='__main__':sys.exit(main())
