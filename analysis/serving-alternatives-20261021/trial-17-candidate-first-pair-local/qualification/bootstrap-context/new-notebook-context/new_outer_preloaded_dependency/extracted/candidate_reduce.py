"""One candidate arm only. No historical or paired-rate decision."""
import argparse,json
from pathlib import Path
import api_arm
import catalog


def arm(root):
    return api_arm.arm(Path(root)/'candidate',catalog.load(),'candidate')


def main():
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--arm',choices=['candidate','control'],default='candidate');a=p.parse_args()
    value=api_arm.arm(Path(a.output)/a.arm,catalog.load(),a.arm)
    Path(a.output,a.arm+'-arm.json').write_text(json.dumps(value,indent=2)+'\n')
    # Qualification failure is a returned observation, not a lost subprocess result.
    print(json.dumps(value),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())


def complete_api_failure(root):
    """A complete bad HTTP/API reply is negative; missing/cancelled raw is not."""
    import hashlib
    import accounting
    import load_contract
    folder=Path(root)/'candidate';cat=catalog.load()
    events=[json.loads(line) for line in (folder/'events.jsonl').read_bytes().splitlines()]
    failure_log(events)
    if events[-1]['threads_alive']!=0:return None
    sent=[row for row in events if row['kind']=='send']
    if not 0<len(sent)<=1369 or len({row['rid'] for row in sent})!=len(sent):return None
    failures=[]
    for row in sent:
        rid=row['rid'];raw=(folder/row['transport_ref']).read_bytes()
        body=api_arm.transport(raw,rid);record=cat['records'][row['key']]
        catalog.checked_body(raw.split(b'\r\n\r\n',1)[1],record)
        wire=json.loads((folder/row['wire_ref']).read_bytes())
        if body!=wire['body'] or not wire['prepared_body_sha256']==wire['outbound_body_sha256']==record['prepared_body_sha256']:return None
        http=json.loads((folder/rid/'http.json').read_bytes())
        writes=http['writes']
        if not writes or sum(value['bytes'] for value in writes)!=len(raw):return None
        if http['complete'] is not True or http['error'] is not None or http['termination']!='COMPLETE':continue
        reply=(folder/rid/'body.bin').read_bytes()
        lengths=[value for key,value in http['headers'] if key.lower()=='content-length']
        if len(lengths)!=1 or int(lengths[0])!=len(reply) or len(reply)!=http['received_bytes'] or len(reply)!=http['persisted_bytes']:return None
        if http['body_completed_at']!=http['ended'] or not http['started']<=http['ended']<=events[-1]['t']:return None
        if http['status']!=200:failures.append(dict(kind='COMPLETE_HTTP_FAILURE',rid=rid,status=http['status'],body_sha256=hashlib.sha256(reply).hexdigest(),raw_verified=True))
        else:
            try:accounting.observe('candidate',record,json.loads(reply))
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
                failures.append(dict(kind='COMPLETE_API_CONTRACT_FAILURE',rid=rid,error=str(error),body_sha256=hashlib.sha256(reply).hexdigest(),raw_verified=True))
    return failures[0] if failures else None


def failure_log(events):
    """Failure receipts need structural identity/clock completeness, not a passed window.

    The accepted-rate path still uses the unchanged strict heartbeat/window gate.
    Failed HTTP capture can spend its declared five-second drain without heartbeat.
    Parent phase, actual child and cleanup clocks are independently joined outside.
    """
    import load_contract as contract
    contract.require(events and events[0]['kind']=='replay_start' and events[-1]['kind']=='capture_end','failure capture boundaries')
    first=events[0];last=first['t']
    contract.require(contract.integer(first['pid'],1) and isinstance(first['boot_id'],str) and first['boot_id'],'failure capture identity')
    for index,row in enumerate(events):
        contract.require(row['seq']==index and type(row['seq']) is int and row['pid']==first['pid'] and row['boot_id']==first['boot_id'],'failure capture sequence/identity')
        contract.require(contract.number(row['t']) and last<=row['t'],'failure capture clock')
        contract.require(index in (0,len(events)-1) or row['kind'] not in ('replay_start','capture_end'),'interior failure capture boundary')
        last=row['t']
    contract.require(type(events[-1]['failed']) is bool and events[-1]['threads_alive']==0,'failure capture workers unresolved')
    return first['boot_id']
