"""Remote execution entry point: requires a separately issued exact approval."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from binding import HERE,ROOT,OVERLAY
from collector import CAPS,collect,require

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify_local():
    pins=json.loads((HERE/'SOURCE-PINS.json').read_bytes())
    for name,expected in pins['files'].items():require(sha(HERE/name)==expected,'adapter source pin drift')
    for name,expected in pins['references'].items():require(sha(name)==expected,'collection input/source reference drift')
    freeze=json.loads((ROOT/'freeze.json').read_bytes())
    for name,expected in freeze['files'].items():require(sha(ROOT/name)==expected,'original frozen file drift')
    for name,expected in freeze['references'].items():require(sha(name)==expected,'original frozen reference drift')
    require(freeze['bindings']==13118,'original binding count')
    return pins

def authorize(path):
    verify_local();receipt=json.loads(Path(path).read_bytes())
    require(receipt.get('decision')=='TRIAL16_MISSING_RAW_REMOTE_COLLECTION_APPROVED' and receipt.get('remote_authorized') is True,'no exact remote authorization')
    require(receipt.get('source_pins_sha256')==sha(HERE/'SOURCE-PINS.json') and receipt.get('manifest_sha256')==sha(HERE/'MISSING-MANIFEST.json'),'approval source/manifest mismatch')
    require((receipt.get('kernel_id'),receipt.get('version'))==(140493086,1),'approval run identity mismatch')
    require(receipt.get('bounds')==CAPS,'approval bounded contract mismatch')
    return receipt

def persist_driver_result(destination,result,started,clock=time.monotonic):
    destination=Path(destination);deadline=started+1800;path=destination/'COLLECTION.json'
    result['driver_elapsed_seconds']=clock()-started
    path.write_text(json.dumps(result,indent=2)+'\n')
    completion=dict(schema='COLLECTION_DRIVER_COMPLETION_V1',started=started,deadline=deadline,
        receipt_persisted_end=clock(),receipt_sha256=sha(path))
    (destination/'DRIVER-COMPLETION.json').write_text(json.dumps(completion,indent=2)+'\n')
    if clock()>=deadline:
        result.update(complete=False,error=dict(type='Stop',reason='driver persistence completed after cooperative deadline'))
        path.write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('authorization',type=Path);parser.add_argument('destination',type=Path);args=parser.parse_args()
    approval=authorize(args.authorization)
    from remote import Remote
    manifest=json.loads((HERE/'MISSING-MANIFEST.json').read_bytes())
    require(manifest['bounds']==CAPS and len(manifest['files'])==1596,'exact manifest/cap contract')
    require(not args.destination.exists(),'exclusive destination already exists')
    from space import check
    require(check(args.destination.parent,'archive')['eligible'],'13GiB simultaneous collection reserve unavailable')
    remaining=1800-(time.monotonic()-started)
    require(remaining>0,'preflight exhausted cooperative wall budget')
    result=collect(manifest['files'],args.destination,Remote(),limits={'wall_seconds':remaining})
    (args.destination/'AUTHORITY.json').write_text(json.dumps(dict(approval=approval,approval_sha256=sha(args.authorization),
        source_pins_sha256=sha(HERE/'SOURCE-PINS.json'),manifest_sha256=sha(HERE/'MISSING-MANIFEST.json')),indent=2)+'\n')
    result=persist_driver_result(args.destination,result,started)
    print(json.dumps({key:result.get(key) for key in ('complete','list_pages','download_attempts','downloaded_bytes','error')}))
    return 0 if result['complete'] else 1
if __name__=='__main__':sys.exit(main())
