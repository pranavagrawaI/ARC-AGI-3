"""Trial7-only operational reconciliation of the preserved failed trial5 save.

This never labels the redacted shell terminal or absent. No remote mutations.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR_REF='pranavagrawai/serving-candidate-feas5-20261023'
PREDECESSOR_REF='pranavagrawai/serving-candidate-feas6-20261023'
EXPECTED_QUOTA=59275.88
SUCCESSOR_REF='pranavagrawai/serving-candidate-feas7-20261023'
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
    require(policy['predecessor_kernel_ref']==PREDECESSOR_REF and policy['predecessor_kernel_id']==140341756 and policy['predecessor_version']==1,'wrong predecessor policy identity')
    require(predecessor['kernel_ref']==PREDECESSOR_REF and predecessor['kernel_id']==140341756 and predecessor['version']==1,'wrong predecessor closure identity')
    require(predecessor['status']=='ERROR' and predecessor['surviving_groups']==[] and predecessor['serialized_slot']=='RELEASED','predecessor not closed/cleaned/released')
    require(predecessor['quota']['used_s']==EXPECTED_QUOTA and predecessor['quota']['reserved_s']==0,'wrong predecessor terminal quota')
    return policy


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
    require(policy['quota_used_s']==EXPECTED_QUOTA,'wrong bound quota identity')
    for field in ('quota_before','quota_after'):
        require(evidence[field]['used_s']==EXPECTED_QUOTA and evidence[field]['reserved_s']==0,'quota moved or reserved')
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
