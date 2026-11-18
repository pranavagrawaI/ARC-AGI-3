"""Future one-shot launch gate. Missing fresh coordinator assignment stops locally.

This module is preparation only; do not run it as part of local verification.
"""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
COORDINATOR='d74e17d0-679e-4189-a52b-bfcc9e6895be'
PREPARATION_FREEZE='freeze-terminal-r3.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_terminal_sessions(sessions):
    if not sessions or any(row['status'] not in ('COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED') for row in sessions):
        raise ValueError('recent account session not established terminal')


def clearance(now=None):
    confirmation=HERE/'slot-confirmation.json'
    if not confirmation.exists():raise ValueError('no reviewed serialized assignment; no launch')
    slot=json.loads(confirmation.read_text());freeze=HERE/PREPARATION_FREEZE
    if slot.get('launch_authorized') is not True:raise ValueError('explicit launch authorization missing/false')
    if slot['coordinator_thread_id']!=COORDINATOR or slot['approved_freeze_sha256']!=sha(freeze):
        raise ValueError('coordinator/freeze assignment mismatch')
    now=now or dt.datetime.now(dt.timezone.utc)
    age=(now-dt.datetime.fromisoformat(slot['confirmed_utc'])).total_seconds()
    if not 0<=age<600 or slot['hard_timeout_s']!=5400 or slot['supervisor_timeout_s']!=5100:
        raise ValueError('expired assignment or cap mismatch')
    frozen=json.loads(freeze.read_text())
    for name,expected in frozen['files'].items():
        if sha(HERE/name)!=expected:raise ValueError('reviewed source changed: '+name)
    packet=json.loads((HERE/'review-packet.json').read_text())
    if slot['sources_sha256']!=packet['sources_sha256'] or slot['config_sha256']!=packet['config_sha256']:
        raise ValueError('assignment source/config mismatch')
    if not slot.get('serialized_assignment'):raise ValueError('serialized ownership identity missing')
    if (HERE/'launch-attempt.json').exists():raise ValueError('one attempt already recorded; no retry')
    return slot,packet


def main():
    slot,packet=clearance()
    # Arming changes only the reviewed assignment data. The builder verifies the
    # entire embedded source digest/config/cap before creating the executable notebook.
    sys.path.insert(0,str(HERE.parent/'runner'));import build
    assignment=dict(launch_authorized=True,serialized_assignment=slot['serialized_assignment'],
                    total_seconds=5100,sources_sha256=packet['sources_sha256'],config_sha256=packet['config_sha256'])
    spec=json.loads((HERE/'packet-spec.json').read_text())
    build.build(spec,HERE/'kernel',assignment=assignment)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as c:quota=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
    used=quota.gpu_quota.time_used.total_seconds();reserved=quota.gpu_quota.time_reserved.total_seconds()
    (HERE/'quota-prelaunch.json').write_text(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),used_s=used,reserved_s=reserved)))
    if not all(math.isfinite(v) for v in (used,reserved)) or used<0 or reserved!=0 or used+5400>108000:
        raise ValueError('unknown/occupied/insufficient GPU quota')
    sessions=[]
    for kernel in api.kernels_list(mine=True,user='pranavagrawai',sort_by='dateRun',page_size=20):
        state=api.kernels_status(kernel.ref).status
        name=getattr(state,'name',str(state).split('.')[-1]).upper()
        sessions.append(dict(ref=kernel.ref,status=name))
    (HERE/'sessions-prelaunch.json').write_text(json.dumps(sessions,indent=2))
    require_terminal_sessions(sessions)
    # Recheck freshness after read-only calls and immediately before the attempt.
    clearance()
    with (HERE/'launch-attempt.json').open('x') as stream:
        json.dump(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),slot=slot,hard_timeout_s=5400,
                       armed_notebook_sha=sha(HERE/'kernel'/spec['metadata']['code_file'])),stream,indent=2)
    try:
        result=api.kernels_push(str(HERE/'kernel'),timeout='5400')
        (HERE/'launch-response.json').write_text(str(result)+'\n')
    except Exception as error:
        (HERE/'launch-error.json').write_text(json.dumps(dict(error=repr(error),retry_allowed=False)))
        raise


if __name__=='__main__':main()
