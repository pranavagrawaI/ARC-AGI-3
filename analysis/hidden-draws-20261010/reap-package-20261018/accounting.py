"""Read-only terminal accounting; shared quota deltas are not per-run billing."""
import datetime as dt
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'analysis/marlin-fixed-replay-r3-20261011'))
from kaggle_backend import KaggleBackend


def main():
    collection=json.loads((HERE/'collection.json').read_text())
    assert collection['status'] in {'COMPLETE','ERROR','CANCELED','CANCELLED'}
    before=json.loads((HERE/'admission-launch.json').read_text())
    quota=KaggleBackend().quota().gpu_quota
    after=dict(used_s=quota.time_used.total_seconds(),reserved_s=quota.time_reserved.total_seconds(),
               allowed_s=quota.total_time_allowed.total_seconds())
    raw=HERE/'output/kernel-log.raw.json'
    duration=None
    if raw.exists():
        rows=json.loads(raw.read_text())
        if isinstance(rows,list):
            duration=max((r.get('time',0) for r in rows),default=None)
    result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),status=collection['status'],
                kernel_id=collection['kernel_id'],version=collection['version'],
                launches=1,hard_gpu_cap_s=2700,notebook_log_elapsed_s=duration,
                quota_before={k:before[k] for k in ['used_s','reserved_s','allowed_s']},
                quota_after=after,shared_quota_delta_s=after['used_s']-before['used_s'],
                other_gpu_sessions_at_launch=before['gpu_active'],
                quota_attribution='Shared quota delta includes concurrent activity; not isolated REAP billing.')
    (HERE/'RUN-ACCOUNTING.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
