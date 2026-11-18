"""Offline exact archive/binding/raw feasibility readout. No SDK/network/GPU."""
import argparse,hashlib,json,math,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import artifacts,transport,admission
import space

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(value):return type(value) in (int,float) and math.isfinite(value)

def collection_reporting(readback):
    state=str(readback.get('status',{}).get('status','')).upper().split('.')[-1]
    quota=readback.get('typed_quota');terminal=state in {'COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'}
    typed=isinstance(quota,dict) and all(finite(quota.get(k)) for k in ('used_s','reserved_s'))
    complete=bool(terminal and typed and quota['used_s']>=71670.895 and quota['reserved_s']==0)
    return dict(terminal_collection_complete=complete,terminal_typed_quota=quota if complete else None,
        observed_typed_quota=quota,observed_platform_status=state,slot_release_claim=False,
        collection_gaps=[] if complete else ['platform nonterminal or complete terminal typed quota unavailable; no closure/release claim'])

def main():
    p=argparse.ArgumentParser();p.add_argument('archive_directory',type=Path);p.add_argument('readback_directory',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();a.destination.mkdir(exist_ok=False)
    local_space=space.check(a.destination,'derived')
    (a.destination/'space-preflight.json').write_text(json.dumps(local_space,indent=2)+'\n')
    if not local_space['eligible']:raise ValueError('derived2GiB plus1GiB reserve unavailable')
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
        metadata=json.loads((a.readback_directory/'metadata.json').read_text())
        if metadata['id']!=receipt['kernel_id'] or metadata['currentVersionNumber']!=receipt['version'] or metadata['ref']!=receipt['kernel_ref']:raise ValueError('terminal current metadata drift')
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
        import base64,reduce_api
        parent=root.parent
        if values['ASSIGNMENT']!=json.loads((parent/'assignment.json').read_text()) or json.loads((parent/'config.json').read_text())!=spec['config']:raise ValueError('actual config/assignment gap')
        for name,record in values['PAYLOAD'].items():
            data=base64.b64decode(record['base64'],validate=True)
            if (parent/name).read_bytes()!=data or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('actual extracted payload drift '+name)
        result=json.loads((root/'result.json').read_text());events=[json.loads(s) for s in (root/'supervisor.jsonl').read_text().splitlines()]
        terminal=[e for e in events if e['event']=='terminal']
        if len(terminal)!=1 or terminal[0]!=dict(seq=events[-1]['seq'],t=events[-1]['t'],event='terminal',**result):raise ValueError('actual terminal/result join')
        out.update(collection_reporting(readback),inventory=inventory,archive_sha256=receipt['archive_sha256'],result=result,exact_payloads=len(values['PAYLOAD']))
        pair=reduce_api.compare(root);out['independent_pair']=pair
        saved=json.loads((root/'paired-result.json').read_text())
        if pair!=saved:raise ValueError('independent pair reduction contradicts saved result')
        bad=not result.get('pass_') or result.get('qualification')!=pair['verdict'] or result.get('ratio')!=pair['ratio']
        bad=bad or any(result.get(k) for k in ('surviving_phase_groups','surviving_native_group','retained_queries','query_cleanup','cleanup_errors','phase_cleanup_errors','error'))
        if bad:raise ValueError('actual runtime/cleanup failure; saved ratio not qualified')
        if pair['verdict']=='FEASIBILITY_ONLY' or not out['terminal_collection_complete'] or readback['status']['status']!='COMPLETE':raise ValueError('pair or terminal collection not qualified')
        out.update(status=pair['verdict'],ratio=pair['ratio'],rate_claim=True,adoption=False,issues=[])
    except BaseException as error:
        out.update(status='FEASIBILITY_ONLY',ratio=None,rate_claim=False);out['issues'].append(repr(error))
        if isinstance(out.get('independent_pair'),dict):out['independent_pair']['ratio']=None
    encoded=json.dumps(out,indent=2).encode()+b'\n'
    if len(encoded)>space.DERIVED:raise ValueError('derived2GiB bound exceeded; no qualified collection')
    with (a.destination/'REDUCTION.json').open('xb') as stream:stream.write(encoded)
    print(json.dumps(dict(status=out['status'],ratio=out['ratio'],issues=out['issues'])))
if __name__=='__main__':main()
