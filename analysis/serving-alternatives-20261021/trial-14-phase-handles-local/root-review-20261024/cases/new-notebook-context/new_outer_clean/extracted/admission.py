"""Separate feasibility admission; never changes original host/resource reports."""
import math

def nonhost(sample,end):
    """Original complete observation, all nonhost checks; host must be measured."""
    try:
        def clock(a,b):return type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and sample['start']<=a<=b<=sample['end']<=end
        if not clock(sample['start'],sample['end']) or sample['probe_worker_alive'] or sample['ipc_error'] is not None:return False
        process=sample['process'];probes=sample['probes']
        if set(probes)!={'gpu','apps','cgroup','port','disk','native'} or process['eligible'] is not True or process['remaining']!=[] or not clock(process['start'],process['end']):return False
        for name,record in probes.items():
            if 'error' in record or not clock(record['start'],record['end']):return False
            if name!='cgroup' and record['eligible'] is not True:return False
        host=probes['cgroup']
        return (type(host['eligible']) is bool and type(host['computed_bytes']) is int and host['computed_bytes']>=0
                and host['threshold_bytes']==140*1024**3 and host['files_stable_during_read'] is True)
    except (KeyError,TypeError,ValueError):return False


def require_candidate(sample,now,maximum_age,owned_native,retained_queries,phase_groups):
    if (not nonhost(sample,now) or not 0<=now-sample['end']<=maximum_age
            or owned_native or retained_queries or phase_groups):
        raise ValueError('candidate admission: incomplete/stale/nonhost/unknown ownership')
