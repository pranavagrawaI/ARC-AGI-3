"""Trial16-only PROPOSED operational reconciliation of the preserved failed trial5 save.

This never labels the redacted shell terminal or absent. No remote mutations.
"""
import datetime as dt
import hashlib
import json
import math
import quota_wire
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR_REF='pranavagrawai/serving-candidate-feas5-20261023'
PREDECESSOR_REF='pranavagrawai/serving-candidate-s10-16-20261024'
HISTORICAL_PREDECESSOR_USED_S=77433.978
REVIEWED_QUOTA=dict(used_s='2714.079',reserved_s=0,total_allowed_s=108000,minimum_allowed_s=108000,refresh_time_utc='2026-10-31T00:00:00+00:00')
SUCCESSOR_REF='pranavagrawai/serving-candidate-first-capacity17-20261024'
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
    require((policy['predecessor_kernel_ref'],policy['predecessor_kernel_id'],policy['predecessor_version'])==(PREDECESSOR_REF,140493086,1),'wrong predecessor policy identity')
    require((predecessor['kernel_ref'],predecessor['kernel_id'],predecessor['version'],predecessor['platform_status'])==(PREDECESSOR_REF,140493086,1,'COMPLETE'),'wrong predecessor terminal identity')
    require(predecessor['decision']=='ATTEMPT_CLOSED_EXCLUSIVE_SLOT_RELEASED' and all(value==[] for value in predecessor['cleanup'].values()),'predecessor cleanup/closure')
    require(predecessor['terminal_quota_exact_wire_seconds']['used_s']=='2714.079' and predecessor['terminal_quota_exact_wire_seconds']['reserved_s']=='0','predecessor exact terminal quota')
    require(hashlib.sha256((HERE/policy['predecessor_freeze']).read_bytes()).hexdigest()==policy['predecessor_freeze_sha256'],'predecessor freeze')
    accepted=json.loads((HERE/policy['predecessor_acceptance']).read_text())
    require(accepted['kernelId']==140493086 and accepted['versionNumber']==1 and accepted['ref']=='/code/'+PREDECESSOR_REF and accepted.get('error') is None,'predecessor acceptance')
    require(policy['historical_predecessor_used_s']==HISTORICAL_PREDECESSOR_USED_S,'historical terminal charge changed')
    require(policy['current_quota']==REVIEWED_QUOTA,'unreviewed current quota contract')
    return policy


# Wire parser retained at the receipt boundary; no SDK to_dict authority.
quota_values=quota_wire.quota_values

def validate_quota(receipt,policy,now=None,*,prelaunch=True):
    import decimal_quota
    from decimal import Decimal
    values=decimal_quota.validate(receipt,policy,now)
    if prelaunch:
        require(values['used_s']==Decimal(str(REVIEWED_QUOTA['used_s'])) and values['reserved_s']==0,'quota moved or reserved')
        stamp=now or dt.datetime.now(dt.timezone.utc)
        require(values['used_s']+10500<=values['total_allowed_s'] and stamp+dt.timedelta(seconds=10500)<dt.datetime.fromisoformat(receipt['refresh_time_utc']),'insufficient reviewed GPU allowance/window')
    return values


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
