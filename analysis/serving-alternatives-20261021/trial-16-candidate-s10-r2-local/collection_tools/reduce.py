"""Candidate-alone immutable archive + primary + terminal raw audit. No network."""
import argparse,base64,hashlib,json,math,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import candidate_evidence,primary_bundle,transport
import space
import primary

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def reporting(readback):
    state=readback.get('status',{}).get('status');quota=readback.get('typed_quota')
    valid=state in {'COMPLETE','ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'} and isinstance(quota,dict)
    valid=valid and all(type(quota.get(key)) in (int,float) and math.isfinite(quota[key]) for key in ('used_s','reserved_s'))
    valid=bool(valid and quota['used_s']>=77433.978 and quota['reserved_s']==0)
    return dict(terminal_collection_complete=valid,terminal_typed_quota=quota if valid else None,observed_typed_quota=quota,
        collection_gaps=[] if valid else ['terminal typed quota/status missing or invalid; no closure claim'],slot_release_claim=False)


def reduce(archive,readback,primary_dir):
    archive,readback,primary_dir=map(Path,(archive,readback,primary_dir))
    out=dict(status='INVALID_INCONCLUSIVE',U_resp=None,adoption=False,issues=[],slot_release_claim=False,terminal_collection_complete=False)
    try:
        rb=json.loads((readback/'READBACK.json').read_text());out.update(reporting(rb))
        freeze=json.loads((HERE/'freeze.json').read_text())
        for name,expected in freeze['files'].items():
            if sha(HERE/name)!=expected:raise ValueError('frozen local drift '+name)
        for name,expected in freeze['references'].items():
            if sha(name)!=expected:raise ValueError('frozen reference drift '+name)
        record=json.loads((archive/'ARCHIVE.json').read_text())
        response=json.loads((HERE/'launch-response.json').read_text());spec=json.loads((HERE/'packet-spec.json').read_text())
        if not record.get('complete') or not record.get('CRC_verified') or sha(archive/'output.zip')!=record['archive_sha256']:raise ValueError('archive incomplete/hash')
        if (record['kernel_ref'],record['kernel_id'],record['version'])!=(spec['metadata']['id'],response['kernelId'],1):raise ValueError('archive version identity')
        primary_raw=(primary_dir/'PRIMARY.json').read_bytes();operational=primary.verify(primary_raw,readback)
        bundle=primary_bundle.verify(primary_raw)
        if bundle['missing']:raise ValueError('primary evidence missing '+str(bundle['missing']))
        inventory=[];total=0
        with zipfile.ZipFile(archive/'output.zip') as zipped:
            if len(zipped.infolist())>30000:raise ValueError('archive inventory count')
            for info in zipped.infolist():
                name=Path(info.filename);total+=info.file_size
                if name.is_absolute() or '..' in name.parts or total>space.EXTRACTED or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('unsafe archive')
                if info.is_dir():continue
                data=zipped.read(info);file=archive/'members'/name
                if file.is_symlink() or file.read_bytes()!=data:raise ValueError('archive/extracted bytes mismatch')
                inventory.append(dict(name=str(name),bytes=len(data),CRC=info.CRC,sha256=hashlib.sha256(data).hexdigest()))
        if inventory!=record['members']:raise ValueError('archive inventory receipt mismatch')
        parent=archive/'members'/Path(spec['config']['cwd']).name;root=parent/'output'
        if (archive/'members'/primary_bundle.NAME).read_bytes()!=primary_raw:raise ValueError('primary/version archive equality')
        events=[json.loads(line) for line in (root/'supervisor.jsonl').read_bytes().splitlines()]
        primary_bundle.verify_publication(root,primary_raw,events[0]['work_deadline'],spec['config']['primary_seconds'])
        inner=transport.decode((readback/'source.ipynb').read_bytes());values=transport.literals(transport.cell_source(json.loads(inner)))
        manifest=json.loads((HERE/'review-packet.json').read_text())
        if {name:item['sha256'] for name,item in values['PAYLOAD'].items()}!=manifest['files']:raise ValueError('actual payload map')
        for name,item in values['PAYLOAD'].items():
            data=base64.b64decode(item['base64'],validate=True)
            if (parent/name).read_bytes()!=data or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('actual source payload '+name)
        if json.loads((parent/'assignment.json').read_text())!=values['ASSIGNMENT'] or json.loads((parent/'config.json').read_text())!=spec['config']:raise ValueError('archive config/authority')
        result=json.loads((root/'result.json').read_text())
        if result!=operational['result']:raise ValueError('primary/version actual result equality')
        if any(result.get(key) for key in ('cleanup_errors','phase_cleanup_errors','retained_queries','surviving_phase_groups','primary_bundle_error','late_work')):raise ValueError('actual cleanup/primary/work failure')
        proof=candidate_evidence.check(root,spec['config'],result.get('error'),result.get('failed_stage'))
        if any(result.get(key)!=value for key,value in proof.items()):raise ValueError('independent raw outcome contradicts runtime')
        if not out['terminal_collection_complete']:raise ValueError('terminal collection gap')
        if proof['status']=='CANDIDATE_ALONE_WORKLOAD_FEASIBLE' and rb['status']['status']!='COMPLETE':raise ValueError('feasible screen needs terminal COMPLETE')
        out.update(proof,inventory=inventory,archive_sha256=record['archive_sha256'],payloads=len(values['PAYLOAD']),result=result)
    except BaseException as error:out.update(status='INVALID_INCONCLUSIVE',U_resp=None);out['issues'].append(repr(error))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('readback',type=Path);p.add_argument('primary',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();a.destination.mkdir(exist_ok=False)
    local=space.check(a.destination,'derived');(a.destination/'space-preflight.json').write_text(json.dumps(local))
    if not local['eligible']:raise ValueError('derived2GiB+1GiB reserve required')
    out=reduce(a.archive,a.readback,a.primary)
    raw=json.dumps(out,indent=2).encode()+b'\n'
    if len(raw)>space.DERIVED:raise ValueError('derived byte bound')
    with (a.destination/'REDUCTION.json').open('xb') as stream:stream.write(raw)
    print(json.dumps(dict(status=out['status'],U_resp=out['U_resp'],issues=out['issues'])))
if __name__=='__main__':main()
