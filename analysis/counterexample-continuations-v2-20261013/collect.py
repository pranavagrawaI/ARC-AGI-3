"""Read-only monitor and one terminal collection. Never launches or retries a run."""
from pathlib import Path
import datetime,json,sys,time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mtp-depth-20261006'))
import launch_watch as lw
lw.EVID=HERE/'evidence'

def main():
    launch=json.loads((lw.EVID/'launch-attempt.json').read_text())
    assert launch['state']=='accepted'
    ref=json.loads((HERE/'identity.json').read_text())['metadata']['id']
    while True:
        api=lw.KaggleApi();api.authenticate()
        try:
            result=api.kernels_status(ref);status=str(result.status)
            row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'failure':getattr(result,'failure_message',None)}
        except Exception as exc:row={'status':'UNKNOWN','error':repr(exc)}
        with (lw.EVID/'status-history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if any(x in row['status'] for x in ['COMPLETE','ERROR','CANCEL']):break
        time.sleep(45)
    marker=lw.EVID/'collection-attempt.json'
    with marker.open('x') as f:json.dump(row,f)
    output=HERE/'output';output.mkdir(exist_ok=True)
    api.kernels_output(ref,path=str(output),force=True,quiet=True)
    snapshots=list(output.rglob('continuations/results.json'))
    if snapshots:
        episodes=json.loads(snapshots[0].read_text())
        summary={'terminal_status':row['status'],'episodes_returned':len(episodes),'solves':sum(r['status']=='solved' for r in episodes),
                 'known_tokens':sum(r.get('known_receipted_tokens',r.get('generated_tokens')) or 0 for r in episodes),
                 'unresolved_exposure':any(r.get('unresolved') for r in episodes),'episodes':episodes}
    else:summary={'terminal_status':row['status'],'episodes_returned':0,'status':'OUTPUTS_REQUIRE_INSPECTION'}
    (lw.EVID/'collection.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
