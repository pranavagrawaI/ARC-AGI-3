"""Offline exact archive/binding/raw feasibility readout. No SDK/network/GPU."""
import argparse,hashlib,json,math,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import artifacts,transport,admission

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def refs(v):
    if isinstance(v,dict):
        if {'path','sha256','bytes','complete'}<=v.keys():yield v
        else:
            for item in v.values():yield from refs(item)
    elif isinstance(v,list):
        for item in v:yield from refs(item)

def require(ok,message):
    if not ok:raise ValueError(message)

def finite(value):return type(value) in (int,float) and math.isfinite(value)

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
    labels=['before_control_preparation','control.prepare.before','control.prepare.after','control.identity.before','control.identity.after',
        'control.health200','control.released.before_admission','candidate.prepare.after','candidate.identity.after','candidate.server.spawned','candidate.health200','candidate.cleaned','final_cleanup']
    require([row['label'] for row in boundaries]==labels,'phase boundary coverage')
    recorded=[e for e in events if e['event']=='host_boundary']
    require(len(recorded)==len(boundaries) and all((r['label'],r['start'],r['end'])==(e['label'],e['start'],e['end']) for r,e in zip(boundaries,recorded)),'host boundary trace join')

def collection_reporting(readback):
    state=str(readback.get('status',{}).get('status','')).upper().split('.')[-1]
    quota=readback.get('typed_quota');terminal=state in {'COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'}
    typed=isinstance(quota,dict) and all(finite(quota.get(k)) for k in ('used_s','reserved_s'))
    complete=bool(terminal and typed and quota['used_s']>=70064.281 and quota['reserved_s']==0)
    return dict(terminal_collection_complete=complete,terminal_typed_quota=quota if complete else None,
        observed_typed_quota=quota,observed_platform_status=state,slot_release_claim=False,
        collection_gaps=[] if complete else ['platform nonterminal or complete terminal typed quota unavailable; no closure/release claim'])

def main():
    p=argparse.ArgumentParser();p.add_argument('archive_directory',type=Path);p.add_argument('readback_directory',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();a.destination.mkdir(exist_ok=False)
    spec=json.loads((HERE/'packet-spec.json').read_text());out=dict(status='NOT_QUALIFIED',ratio=None,rate_claim=False,adoption=False,issues=[],terminal_collection_complete=False,terminal_typed_quota=None,slot_release_claim=False,collection_gaps=['terminal collection not established'])
    try:
        freeze=json.loads((HERE/'freeze.json').read_text())
        for name,h in freeze['files'].items():
            if sha(HERE/name)!=h:raise ValueError('frozen input drift '+name)
        for name,h in freeze['references'].items():
            if sha(name)!=h:raise ValueError('frozen reference drift '+name)
        receipt=json.loads((a.archive_directory/'ARCHIVE.json').read_text());readback=json.loads((a.readback_directory/'READBACK.json').read_text())
        response=json.loads((HERE/'launch-response.json').read_text())
        if not receipt['complete'] or not receipt['CRC_verified'] or sha(a.archive_directory/'output.zip')!=receipt['archive_sha256']:raise ValueError('archive incomplete/hash')
        if receipt['kernel_id']!=response['kernelId'] or receipt['version']!=response['versionNumber'] or receipt['kernel_ref']!=spec['metadata']['id']:raise ValueError('archive version identity')
        if readback.get('error') or not readback.get('source_exact') or readback['kernel_id']!=receipt['kernel_id'] or readback['version']!=receipt['version']:raise ValueError('readback gap')
        inner=transport.decode((a.readback_directory/'source.ipynb').read_bytes());attempt=json.loads((HERE/'launch-attempt.json').read_text())
        if transport.sha(inner)!=attempt['inner_notebook_sha']:raise ValueError('armed inner source gap')
        values=transport.literals(transport.cell_source(json.loads(inner)));manifest=json.loads((HERE/'review-packet.json').read_text())
        if values['CONFIG']!=spec['config'] or {name:r['sha256'] for name,r in values['PAYLOAD'].items()}!=manifest['files']:raise ValueError('actual payload/config mismatch')
        inventory=[]
        with zipfile.ZipFile(a.archive_directory/'output.zip') as zipped:
            for info in zipped.infolist():
                if info.is_dir():continue
                data=zipped.read(info);name=Path(info.filename)
                if name.is_absolute() or '..' in name.parts:raise ValueError('archive path')
                file=a.archive_directory/'members'/name
                if file.read_bytes()!=data:raise ValueError('archive/extracted bytes mismatch')
                inventory.append(dict(name=str(name),bytes=len(data),CRC=info.CRC,sha256=hashlib.sha256(data).hexdigest()))
        root=a.archive_directory/'members'/Path(spec['config']['cwd']).name/'output'
        result=json.loads((root/'result.json').read_text());release=json.loads((root/'release.json').read_text())
        checked=0
        for name in ('pre-release.json','release.json','query-compatibility.json','candidate-admission.json'):
            file=root/name
            if not file.exists():out['issues'].append('not reached/missing '+name);continue
            for ref in refs(json.loads(file.read_text())):artifacts.verify(root/'query-artifacts',ref,time.monotonic()+600);checked+=1
        events=[json.loads(s) for s in (root/'supervisor.jsonl').read_text().splitlines()]
        host=[s['probes']['cgroup'] for s in release['samples']]
        out.update(result=result,release_pass=release['pass'],original140_results=[h['eligible'] for h in host],original140_bytes=[h['computed_bytes'] for h in host if 'computed_bytes' in h],raw_references_checked=checked,
            candidate_spawns=[e for e in events if e['event']=='spawn' and e.get('phase')=='candidate.server'],boundaries=[json.loads(s) for s in (root/'host-boundaries.jsonl').read_text().splitlines()],inventory=inventory,archive_sha256=receipt['archive_sha256'])
        out.update(collection_reporting(readback))
        if result.get('pass_'):
            fresh=json.loads((root/'candidate-admission.json').read_text());series=json.loads((root/'host-series.result.json').read_text());rows=[json.loads(s) for s in (root/'host-series.jsonl').read_text().splitlines()]
            release_evidence(release,fresh,events)
            oom_evidence(out['boundaries'],rows,series,events,root)
            if result['qualification']!='POST_CONTROL_CANDIDATE_LOAD_TO_HEALTH_ONLY' or not result['candidate_health200'] or result['surviving_native_group'] or result['surviving_phase_groups'] or result['retained_queries'] or len(out['candidate_spawns'])!=1:raise ValueError('terminal fit/cleanup gate')
            for arm in ('control','candidate'):
                ready=[e for e in events if e['event']=='ready' and e.get('arm')==arm]
                if len(ready)!=1:raise ValueError('one actual readiness required '+arm)
                exits=[e for e in events if e['event']=='exit' and e.get('phase','').startswith(arm+'.health-') and e['returncode']==0 and e['t']<=ready[0]['t']]
                if not exits:raise ValueError('successful raw health missing '+arm)
                data=(root/(exits[-1]['phase']+'.stdout')).read_bytes();header,body=data.split(b'\n',1)
                if json.loads(header)['status']!=200 or len(body)>65536:raise ValueError('actual raw HTTP200 gate '+arm)
            ready_control=next(e['t'] for e in events if e['event']=='ready' and e.get('arm')=='control')
            release_event=next(e['t'] for e in events if e['event']=='release_result')
            prepare=[e for e in events if e['event']=='spawn' and e.get('phase')=='candidate.prepare']
            if len(prepare)!=1 or not ready_control<release_event<=fresh['sample']['start']<=prepare[0]['t'] or prepare[0]['t']-fresh['sample']['end']>2:raise ValueError('raw admission/dispatch chronology')
            if any('replay' in e.get('phase','') or 'functional' in e.get('phase','') for e in events):raise ValueError('extra work')
            if out['issues']:raise ValueError('required artifact gaps')
            out['status']='POST_CONTROL_CANDIDATE_LOAD_TO_HEALTH_ONLY'
        out['original140_not_admission']=True
    except BaseException as error:out['issues'].append(repr(error))
    with (a.destination/'REDUCTION.json').open('x') as stream:json.dump(out,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=out['status'],ratio=None,issues=out['issues'])))
if __name__=='__main__':main()
