"""Offline exact archive/binding/raw feasibility readout. No SDK/network/GPU."""
import argparse,hashlib,json,sys,time,zipfile
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

def main():
    p=argparse.ArgumentParser();p.add_argument('archive_directory',type=Path);p.add_argument('readback_directory',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();a.destination.mkdir(exist_ok=False)
    spec=json.loads((HERE/'packet-spec.json').read_text());out=dict(status='NOT_QUALIFIED',ratio=None,rate_claim=False,adoption=False,issues=[])
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
            candidate_spawns=[e for e in events if e['event']=='spawn' and e.get('phase')=='candidate.server'],boundaries=[json.loads(s) for s in (root/'host-boundaries.jsonl').read_text().splitlines()],inventory=inventory,archive_sha256=receipt['archive_sha256'],terminal_typed_quota=readback.get('typed_quota'))
        if result.get('pass_'):
            fresh=json.loads((root/'candidate-admission.json').read_text());series=json.loads((root/'host-series.result.json').read_text());rows=[json.loads(s) for s in (root/'host-series.jsonl').read_text().splitlines()]
            if not admission.nonhost(fresh['sample'],fresh['admitted_at']) or fresh['admitted_at']-fresh['sample']['end']>2:raise ValueError('raw fresh admission gate')
            if not series['pass_'] or sha(root/'host-series.jsonl')!=series['series_sha256'] or not rows:raise ValueError('observer series gap')
            if any(row['oom_events'][k]!=rows[0]['oom_events'][k] for row in rows for k in ('oom','oom_kill')):raise ValueError('OOM counter increase')
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
