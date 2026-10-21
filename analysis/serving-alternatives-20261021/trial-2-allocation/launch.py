"""Launch exactly once after explicit coordinator slot confirmation; never start serving."""
import datetime as dt
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

HERE=Path(__file__).resolve().parent


def main():
    confirmation=HERE/'slot-confirmation.json'
    assert confirmation.exists(), 'Coordinator has not confirmed the serialized slot'
    clearance=json.loads(confirmation.read_text())
    freeze=json.loads((HERE/'freeze.json').read_text())
    assert clearance['coordinator_thread_id']=='d74e17d0-679e-4189-a52b-bfcc9e6895be'
    assert clearance['approved_freeze_sha256']==hashlib.sha256((HERE/'freeze.json').read_bytes()).hexdigest()
    assert clearance['hard_timeout_s']==180
    assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(clearance['confirmed_utc'])).total_seconds()<600
    assert all(hashlib.sha256((HERE/k).read_bytes()).hexdigest()==v for k,v in freeze['files'].items())
    assert not (HERE/'launch-attempt.json').exists(), 'one attempt already recorded; no retry'
    assert json.loads((HERE/'request-capture.json').read_text())['network_calls']==0
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
    used=q.gpu_quota.time_used.total_seconds();reserved=q.gpu_quota.time_reserved.total_seconds()
    quota={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'used_s':used,'reserved_s':reserved}
    (HERE/'quota-prelaunch.json').write_text(json.dumps(quota,indent=2)+'\n')
    assert reserved==0 and used+180<=30*3600, 'GPU slot occupied or insufficient quota'
    # Coordinator serializes this against m2_guard; recheck recent account sessions as well.
    sessions=[]
    for k in api.kernels_list(mine=True,user='pranavagrawai',sort_by='dateRun',page_size=20):
        ref=getattr(k,'ref',None)
        if not ref:continue
        state=str(api.kernels_status(ref).status);sessions.append({'ref':ref,'status':state})
    (HERE/'sessions-prelaunch.json').write_text(json.dumps(sessions,indent=2)+'\n')
    assert all('RUNNING' not in x['status'] and 'QUEUED' not in x['status'] for x in sessions)
    receipt={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'hard_timeout_s':180,'clearance':clearance}
    with (HERE/'launch-attempt.json').open('x') as f:json.dump(receipt,f,indent=2)
    try:
        response=api.kernels_push(str(HERE/'kernel'),timeout='180')
        (HERE/'launch-response.json').write_text(str(response)+'\n');print(str(response))
    except Exception as e:
        (HERE/'launch-error.json').write_text(json.dumps({'error':repr(e),'no_retry':True})+'\n');raise


if __name__=='__main__':main()
