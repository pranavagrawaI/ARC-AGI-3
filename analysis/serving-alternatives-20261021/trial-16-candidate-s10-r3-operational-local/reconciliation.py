"""Trial16-only PROPOSED operational reconciliation of the preserved failed trial5 save.

This never labels the redacted shell terminal or absent. No remote mutations.
"""
import datetime as dt
import hashlib
import json
import math,re
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR_REF='pranavagrawai/serving-candidate-feas5-20261023'
PREDECESSOR_REF='pranavagrawai/serving-wire-capacity15-20261024'
HISTORICAL_PREDECESSOR_USED_S=77433.978
REVIEWED_QUOTA=dict(used_s=0,reserved_s=0,total_allowed_s=21600,minimum_allowed_s=21600,refresh_time_utc='2026-10-31T00:00:00+00:00')
SUCCESSOR_REF='pranavagrawai/serving-candidate-s10-16-20261024'
STUB={'title':'[Private Notebook]','lastRunTime':'2010-04-01T00:00:00.000Z'}
TERMINAL={'COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'}


def require(condition,message):
    if not condition:raise ValueError('failed-save reconciliation: '+message)


def load_policy():
    policy=json.loads((HERE/'prior-reconciliation-policy.json').read_text())
    require(policy['prior_kernel_ref']==PRIOR_REF and policy['successor_kernel_ref']==SUCCESSOR_REF,'wrong prior/successor identity')
    for name,expected in policy['files'].items():
        require(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,'prior binding changed '+name)
    attempt=json.loads((HERE/policy['attempt']).read_text());closure=json.loads((HERE/policy['closure']).read_text())
    require(attempt['slot']['serialized_assignment']==closure['assignment']==policy['prior_assignment'],'wrong prior attempt/closure identity')
    require(attempt['slot']['approved_freeze_sha256']==policy['prior_freeze_sha256'],'wrong attempted freeze')
    require(closure['outcome']=='SAVE_KERNEL_HTTP400_NO_ACCEPTED_RUN_RECEIPT' and closure['slot_released'] is True and closure['push_attempts']==1,'prior closure not established')
    predecessor=json.loads((HERE/policy['predecessor_closure']).read_text())
    require(policy['predecessor_kernel_ref']==PREDECESSOR_REF and policy['predecessor_kernel_id']==140466291 and policy['predecessor_version']==1,'wrong predecessor policy identity')
    require(predecessor['kernel_ref']==PREDECESSOR_REF and predecessor['kernel_id']==140466291 and predecessor['version']==1,'wrong predecessor closure identity')
    require(predecessor['decision']=='ATTEMPT_CLOSED_EXCLUSIVE_SLOT_RELEASED' and predecessor['terminal_status']=='ERROR'
        and predecessor['observed_cleanup_errors']==[] and predecessor['raw_result']['qualification']=='FEASIBILITY_ONLY'
        and predecessor['raw_result']['candidate_health200'] is False and predecessor['rate_qualified'] is False
        and predecessor['native_log_archive_equality_join']=='PENDING_SINGLE_VERSION_BOUND_ARCHIVE',
        'predecessor operational closure/partial collection limitations not retained')
    require(predecessor['typed_quota']['used_s']==HISTORICAL_PREDECESSOR_USED_S and predecessor['typed_quota']['reserved_s']==0,'wrong predecessor terminal quota')
    require(policy['predecessor_freeze_sha256']=='a0eec61df3dfc4be357f14cce41272c994fcaa9e1a5cdb0ef3328d70304b19ef'
        and hashlib.sha256((HERE/policy['predecessor_freeze']).read_bytes()).hexdigest()==policy['predecessor_freeze_sha256'],'wrong predecessor freeze')
    accepted=json.loads((HERE/policy['predecessor_acceptance']).read_text())
    require(accepted['kernelId']==140466291 and accepted['versionNumber']==1 and accepted['ref']=='/code/'+PREDECESSOR_REF
        and accepted.get('error') is None,'wrong predecessor acceptance')
    readback=json.loads((HERE/policy['predecessor_readback']).read_text())
    require(readback['kernel_ref']==PREDECESSOR_REF and readback['kernel_id']==140466291 and readback['version']==1
        and readback['source_exact'] is True and readback['status']['status']=='ERROR'
        and readback['typed_quota']['used_s']==HISTORICAL_PREDECESSOR_USED_S and readback['typed_quota']['reserved_s']==0,'wrong predecessor source/terminal readback')
    require(policy['historical_predecessor_used_s']==HISTORICAL_PREDECESSOR_USED_S,'historical terminal charge changed')
    require(policy['current_quota']==REVIEWED_QUOTA,'unreviewed current quota contract')
    return policy


def quota_values(raw):
    gpu=raw['gpuQuota']
    def seconds(key):
        value=gpu[key]
        require(isinstance(value,str) and re.fullmatch(r'\d+(?:\.\d+)?s',value) is not None,'missing/malformed quota duration '+key)
        value=float(value[:-1]);require(math.isfinite(value),'nonfinite quota duration');return value
    refresh=dt.datetime.fromisoformat(raw['quotaRefreshTime'].replace('Z','+00:00'))
    require(refresh.tzinfo is not None and refresh.utcoffset()==dt.timedelta(0),'quota refresh must be explicit UTC')
    return dict(used_s=seconds('timeUsed'),reserved_s=seconds('timeReserved'),total_allowed_s=seconds('totalTimeAllowed'),
                minimum_allowed_s=seconds('minimumTimeAllowed'),refresh_time_utc=refresh.astimezone(dt.timezone.utc).isoformat())


def quota_receipt(response):
    receipt=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),raw_response=type(response).to_dict(response))
    try:
        gpu=response.gpu_quota;refresh=response.quota_refresh_time
        # Installed SDK DateTimeSerializer strips the wire Z and returns naive UTC.
        if refresh.tzinfo is None:refresh=refresh.replace(tzinfo=dt.timezone.utc)
        typed=dict(used_s=gpu.time_used.total_seconds(),reserved_s=gpu.time_reserved.total_seconds(),
                   total_allowed_s=gpu.total_time_allowed.total_seconds(),minimum_allowed_s=gpu.minimum_time_allowed.total_seconds(),
                   refresh_time_utc=refresh.astimezone(dt.timezone.utc).isoformat())
        require(typed==quota_values(receipt['raw_response']),'SDK raw/typed quota disagreement');receipt.update(typed)
    except Exception as error:receipt['parse_error']=repr(error)
    return receipt


def validate_quota(receipt,policy,now=None,*,prelaunch=True):
    expected=policy['current_quota'];require(expected==REVIEWED_QUOTA,'unreviewed current quota contract')
    require(not receipt.get('parse_error'),'invalid measured quota')
    raw=quota_values(receipt['raw_response'])
    require(all(receipt[key]==value for key,value in raw.items()),'raw/typed quota disagreement')
    for key in ('used_s','reserved_s','total_allowed_s','minimum_allowed_s'):
        require(type(receipt[key]) in (int,float) and math.isfinite(receipt[key]) and receipt[key]>=0,'invalid quota numeric field')
    require(all(receipt[key]==expected[key] for key in ('refresh_time_utc','total_allowed_s','minimum_allowed_s')),'quota window/allowance changed')
    now=now or dt.datetime.now(dt.timezone.utc);stamp=dt.datetime.fromisoformat(receipt['utc'])
    require(0<policy['freshness_seconds']<=120 and 0<=(now-stamp).total_seconds()<=policy['freshness_seconds'],'stale/future quota')
    refresh=dt.datetime.fromisoformat(receipt['refresh_time_utc'])
    require(stamp<refresh and receipt['used_s']>=expected['used_s'] and receipt['used_s']+receipt['reserved_s']<=receipt['total_allowed_s'],'quota outside current window/allowance')
    if prelaunch:
        require(receipt['used_s']==expected['used_s'] and receipt['reserved_s']==expected['reserved_s']==0,'quota moved or reserved')
        require(receipt['used_s']+5400<=receipt['total_allowed_s'] and now+dt.timedelta(seconds=5400)<refresh,'insufficient reviewed GPU allowance/window')
    return raw


def evaluate_quota_pair(evidence,policy,now=None):
    now=now or dt.datetime.now(dt.timezone.utc)
    for field in ('started_utc','finished_utc'):
        require(0<=(now-dt.datetime.fromisoformat(evidence[field])).total_seconds()<=policy['freshness_seconds'],'stale/future evidence')
    start=dt.datetime.fromisoformat(evidence['started_utc']);finish=dt.datetime.fromisoformat(evidence['finished_utc'])
    stamps=[]
    for field in ('quota_before','quota_after'):
        validate_quota(evidence[field],policy,now);stamps.append(dt.datetime.fromisoformat(evidence[field]['utc']))
    require(start<=stamps[0]<=stamps[1]<=finish,'quota/evidence clock order')


def evaluate(evidence,policy,now=None):
    require(policy['prior_kernel_ref']==PRIOR_REF and policy['successor_kernel_ref']==SUCCESSOR_REF,'wrong prior/successor identity')
    require(evidence['prior_kernel_ref']==PRIOR_REF,'wrong queried prior identity')
    require(policy['predecessor_kernel_ref']==PREDECESSOR_REF and evidence['predecessor_kernel_ref']==PREDECESSOR_REF,'wrong predecessor identity')
    require(evidence['predecessor_status']['status'] in TERMINAL,'predecessor not freshly terminal')
    now=now or dt.datetime.now(dt.timezone.utc)
    require(0<policy['freshness_seconds']<=120,'invalid freshness policy')
    for field in ('started_utc','finished_utc'):
        age=(now-dt.datetime.fromisoformat(evidence[field])).total_seconds()
        require(0<=age<=policy['freshness_seconds'],'stale/future evidence')
    require(evidence['started_utc']<=evidence['finished_utc'],'evidence clock order')
    evaluate_quota_pair(evidence,policy,now)
    unknown=[row for row in evidence['recent'] if not row.get('ref')]
    named=[row for row in evidence['recent'] if row.get('ref')]
    require(len(unknown)==1 and unknown[0]['raw']==STUB,'extra/changed redacted stub')
    require(bool(named) and all(row['status'] in TERMINAL for row in named),'named session active or unknown')
    require(evidence['exact_search']=={'kernels':[STUB]},'exact search does not identify the one saved stub')
    require(evidence['negative_search'] in ({},{'kernels':[]}),'negative control not empty')
    status=evidence['prior_status']
    require(status.get('status')==404,'exact prior status not HTTP404')
    parsed=json.loads(status.get('body','null'))
    require(isinstance(parsed,dict) and isinstance(parsed.get('error'),dict) and parsed['error'].get('message')=='No runs found for this kernel.' and parsed['error'].get('code')==404,'missing exact parsed no-runs message')
    return dict(classification='RECONCILED_FAILED_SAVE_NO_RUN_OBSERVED',terminal_identity_proven=False,
                kernel_absence_proven=False,prior_kernel_ref=PRIOR_REF,
                judgment='Coordinator-scoped operational exception for this one preserved failed SaveKernel shell only.',
                residual_identity_gap='Redacted stub has no kernel/version/source identity; not classified TERMINAL or ABSENT.')
