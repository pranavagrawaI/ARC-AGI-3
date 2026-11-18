"""Independent raw release/fresh/resource/OOM joins for the capacity protocol."""
import hashlib,json,math
from pathlib import Path
import admission

def require(ok,message):
    if not ok:raise ValueError('capacity evidence: '+message)
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def release_evidence(release,fresh,events):
    require(release['pass'] is True and release['admission_policy']=='FIRST_FRESH_NONHOST'
        and release['error'] is None and release['cleanup_error'] is None and release['controller_query_failure'] is None
        and release['remaining']==[],'unsuccessful FIRST_FRESH_NONHOST release')
    start,end,deadline=(release[k] for k in ('start','end','absolute_deadline'))
    require(all(finite(v) for v in (start,end,deadline)) and start<=end<=deadline,'release original deadline')
    samples=release['samples'];require(bool(samples),'release observations missing')
    accepted=[];previous=start
    for index,sample in enumerate(samples):
        require(all(finite(sample[k]) for k in ('start','end')) and previous<=sample['start']<=sample['end']<=end,'release observation clocks')
        if admission.nonhost(sample,deadline):accepted.append(index)
        previous=sample['end']
    require(accepted==[len(samples)-1],'release did not stop at first accepted nonhost observation')
    beginnings=[e for e in events if e['event']=='release_start'];results=[e for e in events if e['event']=='release_result']
    require(len(beginnings)==len(results)==1 and beginnings[0]['deadline']==deadline and beginnings[0]['t']<=start
        and end<=results[0]['t']<=deadline,'release trace/deadline join')
    require(all(results[0].get(k)==v for k,v in release.items()),'release result contradicts raw release')
    sample=fresh['sample']
    require(fresh['policy']=='FIRST_FRESH_NONHOST' and fresh['no_numeric_host_admission'] is True
        and finite(fresh['admitted_at']) and end<=sample['start']<=sample['end']<=fresh['admitted_at']<=deadline
        and sample['end']<=sample['ipc_deadline']<=deadline,'fresh observation outside original release deadline')
    require(admission.nonhost(sample,deadline) and fresh['admitted_at']-sample['end']<=2,'raw fresh admission gate')
    admitted=[e for e in events if e['event']=='candidate_admitted']
    require(len(admitted)==1 and all(admitted[0].get(k)==v for k,v in fresh.items()),'fresh admission trace join')

def oom_evidence(boundaries,rows,series,events,root):
    require(series['pass_'] is True and series['error'] is None and rows and boundaries,'observer/boundary evidence missing')
    require(series['samples']==len(rows) and series['bytes']==(root/'host-series.jsonl').stat().st_size
        and sha(root/'host-series.jsonl')==series['series_sha256'],'observer series count/bytes/hash')
    baseline=None
    for group in (boundaries,rows):
        previous=-math.inf
        for row in group:
            require(all(finite(row[k]) for k in ('start','end')) and previous<=row['start']<=row['end'],'host observation clocks')
            require(row.get('valid',True) is True and not row.get('invalidity',[]),'invalid boundary observation')
            raw={k:int(v) for k,v in (line.split() for line in row['raw']['events'].splitlines())}
            require(raw==row['oom_events'] and all(type(raw[k]) is int and raw[k]>=0 for k in ('oom','oom_kill')),'raw/stored OOM counters contradict')
            counts=tuple(raw[k] for k in ('oom','oom_kill'))
            if baseline is None:baseline=counts
            require(counts==baseline,'boundary/series OOM counter increase or reset')
            previous=row['end']
    labels=['before_control_preparation', 'corpus.prepare.before', 'corpus.prepare.after', 'control.prepare.before', 'control.prepare.after', 'control.identity.before', 'control.identity.after', 'control.health200', 'control.replay.before', 'control.replay.after', 'control.qualify.before', 'control.qualify.after', 'control.released.before_admission', 'candidate.prepare.after', 'candidate.identity.after', 'candidate.server.spawned', 'candidate.health200', 'candidate.replay.before', 'candidate.replay.after', 'candidate.qualify.before', 'candidate.qualify.after', 'candidate.cleaned', 'final_cleanup']
    require([row['label'] for row in boundaries]==labels,'phase boundary coverage')
    recorded=[e for e in events if e['event']=='host_boundary']
    require(len(recorded)==len(boundaries) and all((r['label'],r['start'],r['end'])==(e['label'],e['start'],e['end']) for r,e in zip(boundaries,recorded)),'host boundary trace join')

def final_resource_evidence(receipt,events,work_end):
    require(receipt['pass_'] is True and receipt['error'] is None,'failed final candidate resource phase')
    start,end,deadline=(receipt[k] for k in ('start','end','absolute_deadline'))
    require(all(finite(v) for v in (start,end,deadline)) and start<=end<=deadline<=work_end,'candidate cleanup/resource clocks')
    starts=[e for e in events if e['event']=='candidate_cleanup_start']
    require(len(starts)==1 and starts[0]['phase_deadline']==deadline and starts[0]['server_pgid']==receipt['server_pgid'] and start<=starts[0]['t']<=deadline,'cleanup start/deadline join')
    cleaned=[e for e in events if e['event']=='cleanup' and e['pgid']==receipt['server_pgid']]
    require(len(cleaned)==1 and cleaned[0]['remaining']==[] and start<=cleaned[0]['t']<=deadline,'candidate whole-group cleanup late/incomplete')
    accepted=[];previous=cleaned[0]['t']
    for index,sample in enumerate(receipt['samples']):
        require(previous<=sample['start']<=sample['end']<=end,'post-cleanup resource clocks')
        if admission.nonhost(sample,deadline):accepted.append(index)
        previous=sample['end']
    require(bool(receipt['samples']) and accepted==[len(receipt['samples'])-1],'final resources did not stop on first eligible nonhost sample')
    passes=[e for e in events if e['event']=='candidate_resources_pass']
    require(len(passes)==1 and all(passes[0][k]==receipt[k] for k in ('start','end','absolute_deadline')) and end<=passes[0]['t']<=deadline,'final resources trace join')
