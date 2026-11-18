"""One-shot successor launch gate. No authority file is shipped with this packet."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import packet
import reconciliation
from collection_tools import space as collection_space
from http_error_capture import exception_record

HERE=Path(__file__).resolve().parent
COORDINATOR='d74e17d0-679e-4189-a52b-bfcc9e6895be'
OWNER='51ac6b55-d9b8-4c8f-8b0d-4c425754a2bd'
KERNEL_REF='pranavagrawai/serving-wire-capacity15-20261024'
PREPARATION_FREEZE='freeze.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_terminal_sessions(sessions):
    if not sessions or any(row['status'] not in ('COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED') for row in sessions):
        raise ValueError('recent account session not established terminal')


def clearance(now=None):
    confirmation=HERE/'slot-confirmation.json'
    if not confirmation.exists():raise ValueError('no reviewed serialized assignment; no launch')
    slot=json.loads(confirmation.read_text());freeze=HERE/PREPARATION_FREEZE
    if slot.get('launch_authorized') is not True or slot.get('local_only') is not False:
        raise ValueError('explicit real launch authorization missing/false; local fixture forbidden')
    if slot.get('coordinator_thread_id')!=COORDINATOR or slot.get('launch_owner_thread_id')!=OWNER:
        raise ValueError('coordinator/owner assignment mismatch')
    if slot.get('kernel_ref')!=KERNEL_REF or slot.get('approved_freeze_sha256')!=sha(freeze):
        raise ValueError('successor identity/freeze assignment mismatch')
    now=now or dt.datetime.now(dt.timezone.utc)
    age=(now-dt.datetime.fromisoformat(slot['confirmed_utc'])).total_seconds()
    if not 0<=age<600 or slot['hard_timeout_s']!=10500 or slot['supervisor_timeout_s']!=10200 or slot.get('work_timeout_s')!=9900 or slot.get('cleanup_timeout_s')!=300:
        raise ValueError('expired assignment or cap mismatch')
    if slot.get('maximum_push_attempts')!=1 or slot.get('maximum_model_loads')!=2 or slot.get('maximum_replay_requests')!=2738:
        raise ValueError('bounded scope mismatch')
    if slot.get('retry_authorized') is not False or slot.get('adoption_authorized') is not False or slot.get('post_control_candidate_load_authorized') is not False or slot.get('capacity_screen_authorized') is not True or slot.get('experimental_nonhost_after_replay_authorized') is not True:
        raise ValueError('retry/rate/adoption scope mismatch')
    frozen=json.loads(freeze.read_text())
    for name,expected in frozen['files'].items():
        if sha(HERE/name)!=expected:raise ValueError('reviewed source changed: '+name)
    for name,expected in frozen.get('references',{}).items():
        if sha(name)!=expected:raise ValueError('reviewed reference changed: '+name)
    report=json.loads((HERE/'review-packet.json').read_text())
    if slot['sources_sha256']!=report['sources_sha256'] or slot['config_sha256']!=report['config_sha256']:
        raise ValueError('assignment source/config mismatch')
    if not slot.get('serialized_assignment') or not slot['serialized_assignment'].startswith(KERNEL_REF.split('/')[1]+'-') or 'LOCAL_ONLY' in slot['serialized_assignment']:
        raise ValueError('real serialized ownership identity missing')
    if slot.get('reconciliation_policy_sha256')!=sha(HERE/'prior-reconciliation-policy.json'):
        raise ValueError('trial15-specific operational reconciliation not explicitly reviewed')
    if (HERE/'launch-attempt.json').exists():raise ValueError('one attempt already recorded; no retry')
    return slot,report


def arm(slot,report,destination):
    assignment=dict(launch_authorized=True,serialized_assignment=slot['serialized_assignment'],
                    total_seconds=10200,sources_sha256=report['sources_sha256'],config_sha256=report['config_sha256'],
                    kernel_ref=KERNEL_REF,platform_hard_timeout_s=10500,confirmed_utc=slot['confirmed_utc'],local_only=False,
                    approved_freeze_sha256=slot['approved_freeze_sha256'])
    manifest=packet.prepare(destination,assignment)
    if manifest['sources_sha256']!=report['sources_sha256'] or manifest['config_sha256']!=report['config_sha256']:
        raise ValueError('armed successor source/config mismatch')
    return manifest


def quota_readback(api):
    with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
    return dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),used_s=q.gpu_quota.time_used.total_seconds(),reserved_s=q.gpu_quota.time_reserved.total_seconds())


def fresh_checks(api):
    local=collection_space.check(HERE)
    (HERE/'collection-space-prelaunch.json').write_text(json.dumps(local,indent=2))
    if not local['eligible']:raise ValueError('local simultaneous archive/extraction/derived space unavailable; no API')
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    quota=quota_readback(api);used=quota['used_s'];reserved=quota['reserved_s']
    (HERE/'quota-prelaunch.json').write_text(json.dumps(quota))
    if not all(math.isfinite(v) for v in (used,reserved)) or used<0 or reserved!=0 or used+10500>108000:
        raise ValueError('unknown/occupied/insufficient GPU quota')
    sessions=[]
    for kernel in api.kernels_list(mine=True,user='pranavagrawai',sort_by='dateRun',page_size=20):
        if not kernel.ref:
            sessions.append(dict(ref='',status='REDACTED_UNKNOWN',raw=type(kernel).to_dict(kernel)));continue
        state=api.kernels_status(kernel.ref).status
        sessions.append(dict(ref=kernel.ref,status=getattr(state,'name',str(state).split('.')[-1]).upper()))
    (HERE/'sessions-prelaunch.json').write_text(json.dumps(sessions,indent=2))
    predecessor=api.kernels_status(reconciliation.PREDECESSOR_REF).status
    predecessor_status=dict(status=getattr(predecessor,'name',str(predecessor).split('.')[-1]).upper())
    (HERE/'predecessor-status-prelaunch.json').write_text(json.dumps(dict(kernel_ref=reconciliation.PREDECESSOR_REF,utc=dt.datetime.now(dt.timezone.utc).isoformat(),**predecessor_status)))
    require_terminal_sessions([predecessor_status])
    if all(row['ref'] for row in sessions):
        require_terminal_sessions(sessions)
        return None
    # A generic redacted row is forbidden. The frozen, trial15-only proposed policy must
    # independently establish every reconciliation predicate; slot cannot override.
    policy=reconciliation.load_policy()
    evidence=dict(started_utc=started,prior_kernel_ref=reconciliation.PRIOR_REF,predecessor_kernel_ref=reconciliation.PREDECESSOR_REF,predecessor_status=predecessor_status,quota_before=quota,recent=sessions)
    try:
        for key,search in (('exact_search','serving-candidate-feas5-20261023'),('negative_search','no-such-kernel-feas5-reconciliation-20261023')):
            result=api.kernels_list_with_response(mine=True,user='pranavagrawai',sort_by='dateRun',page_size=20,search=search)
            evidence[key]=type(result).to_dict(result)
        try:
            api.kernels_status(reconciliation.PRIOR_REF)
            evidence['prior_status']={'status':None,'body':'status unexpectedly succeeded'}
        except Exception as error:
            evidence['prior_status']=exception_record(error)
        evidence['quota_after']=quota_readback(api)
        (HERE/'quota-prelaunch.json').write_text(json.dumps(evidence['quota_after']))
        evidence['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        evidence['judgment']=reconciliation.evaluate(evidence,policy)
    finally:
        (HERE/'failed-save-reconciliation-prelaunch.json').write_text(json.dumps(evidence,indent=2))
    return evidence


def push_once(api,slot,manifest):
    spec=json.loads((HERE/'packet-spec.json').read_text())
    with (HERE/'launch-attempt.json').open('x') as stream:
        json.dump(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),slot=slot,hard_timeout_s=10500,
                       kernel_ref=KERNEL_REF,inner_notebook_sha=manifest['inner_notebook_sha256'],
                       armed_notebook_sha=sha(HERE/'kernel'/spec['metadata']['code_file']),
                       armed_outer_notebook_bytes=manifest['outer_notebook_bytes']),stream,indent=2)
    try:
        result=api.kernels_push(str(HERE/'kernel'),timeout='10500')
        (HERE/'launch-response.json').write_text(str(result)+'\n')
    except Exception as error:
        (HERE/'launch-error.json').write_text(json.dumps(exception_record(error),indent=2))
        raise


def main():
    slot,report=clearance()
    manifest=arm(slot,report,HERE/'kernel')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    evidence=fresh_checks(api)
    clearance()  # Live authority freshness/source recheck immediately before the attempt marker.
    if evidence is not None:reconciliation.evaluate(evidence,reconciliation.load_policy())
    push_once(api,slot,manifest)


if __name__=='__main__':main()
