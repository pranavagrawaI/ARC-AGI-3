"""Offline reproducible joins for this single attempt; no API or source mutation."""
import base64, datetime as dt, hashlib, json, sys, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OVERLAY=ROOT.parent/'trial16-collection-decimal-overlay-local'
sys.path[:0]=[str(OVERLAY),str(ROOT),str(ROOT/'collection_tools')]
import candidate_evidence, decimal_quota, primary, primary_bundle, reconciliation, transport

def read(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def prove():
    directory=ROOT/'remote-readback'
    archive=directory/'archive-v1'
    readback=directory/'current-primary-fallback-local/fresh-readback'
    primary_dir=directory/'current-primary-fallback-local/primary-current'
    reduced=read(directory/'derived-v1/REDUCTION.json')
    assert reduced['issues']==[] and reduced['status']=='CANDIDATE_ALONE_WORKLOAD_FEASIBLE'
    freeze=read(ROOT/'freeze.json')
    assert sha(ROOT/'freeze.json')=='b51120fedef231e6b5fdecd8882f03bb78653e4e221fa9b256cdd7d908c2ed5b'
    for name,expected in freeze['files'].items():assert sha(ROOT/name)==expected,name
    for name,expected in freeze['references'].items():assert sha(name)==expected,name
    assert len(freeze['files'])+len(freeze['references'])==freeze['bindings']==13118
    pins=read(OVERLAY/'SOURCE-PINS.json')
    assert sha(OVERLAY/'SOURCE-PINS.json')=='c609808e1a09042bb018a786eeda292c9eeb8e495962d65fb1746714613c2568'
    for name,expected in pins['overlay_files'].items():assert sha(OVERLAY/name)==expected,name
    for name,expected in pins['references'].items():assert sha(name)==expected,name
    rb=read(readback/'READBACK.json');meta=read(readback/'metadata.json')
    assert not rb.get('error') and rb['source_exact'] is True and rb['status']['status']=='COMPLETE'
    assert (rb['kernel_id'],rb['version'],meta['id'],meta['currentVersionNumber'])==(140493086,1,140493086,1)
    spec=read(ROOT/'packet-spec.json');config=spec['config']
    local=(ROOT/'kernel'/spec['metadata']['code_file']).read_bytes()
    remote=(readback/'source.ipynb').read_bytes()
    assert transport.cell_source(json.loads(remote))==transport.cell_source(json.loads(local))
    inner=transport.decode(remote);values=transport.literals(transport.cell_source(json.loads(inner)))
    assert transport.sha(inner)==read(ROOT/'launch-attempt.json')['inner_notebook_sha']==rb['inner_sha256']
    record=read(archive/'ARCHIVE.json')
    assert record['complete'] is True and record['CRC_verified'] is True
    assert (record['kernel_id'],record['version'])==(140493086,1)
    assert sha(archive/'output.zip')==record['archive_sha256']==reduced['archive_sha256']
    inventory=[]
    with zipfile.ZipFile(archive/'output.zip') as zipped:
        assert len(zipped.infolist())<=30000
        for info in zipped.infolist():
            name=Path(info.filename)
            assert not name.is_absolute() and '..' not in name.parts
            if info.is_dir():continue
            raw=zipped.read(info) # Includes CRC validation.
            target=archive/'members'/name
            assert not target.is_symlink() and target.read_bytes()==raw
            inventory.append(dict(name=info.filename,bytes=len(raw),CRC=info.CRC,sha256=hashlib.sha256(raw).hexdigest()))
    assert inventory==record['members']==reduced['inventory']
    assert (archive/'output.zip').stat().st_size<=2*1024**3
    assert sum(row['bytes'] for row in inventory)<=8*1024**3
    parent=archive/'members'/Path(config['cwd']).name;out=parent/'output'
    primary_raw=(primary_dir/'PRIMARY.json').read_bytes()
    assert (archive/'members'/primary_bundle.NAME).read_bytes()==primary_raw
    current=primary.verify(primary_raw,readback);bundle=primary_bundle.verify(primary_raw)
    assert bundle['missing']==bundle['truncated']==[]
    for name,item in bundle['files'].items():
        target=parent/name if name in ('config.json','assignment.json') else out/name
        assert target.read_bytes()==base64.b64decode(item['base64'],validate=True),name
    for name,item in values['PAYLOAD'].items():
        data=base64.b64decode(item['base64'],validate=True)
        assert transport.sha(data)==item['sha256'] and (parent/name).read_bytes()==data,name
    assert len(values['PAYLOAD'])==42
    assert read(parent/'config.json')==config and read(parent/'assignment.json')==values['ASSIGNMENT']
    result=read(out/'result.json');assert result==current['result']==reduced['result']
    for key in ('cleanup_errors','phase_cleanup_errors','surviving_phase_groups','retained_queries','query_cleanup'):
        assert result[key]==[]
    events=[json.loads(line) for line in (out/'supervisor.jsonl').read_bytes().splitlines()]
    publication=primary_bundle.verify_publication(out,primary_raw,events[0]['work_deadline'],config['primary_seconds'])
    cleanup=[row for row in events if row['event']=='candidate_cleanup_start'];assert len(cleanup)==1
    begin=cleanup[0];release=read(out/'candidate-release.json')
    assert config['final_resource_seconds']==180
    deadline=min(events[0]['work_deadline'],begin['cleanup_start']+180)
    assert deadline==begin['phase_deadline']==release['absolute_deadline']
    assert release['pass_'] is True and release['error'] is None and release['end']<=deadline
    scope=read(out/'candidate-scope.json');binding=scope['binding'];identity=binding['members'][str(binding['pgid'])]
    replay=[json.loads(line) for line in (out/'candidate/events.jsonl').read_bytes().splitlines()]
    assert str(identity['pgid'])==replay[0]['server_pgid'] and str(identity['starttime'])==replay[0]['server_start']
    assert identity['boot_id']==replay[0]['boot_id']==events[0]['boot_id']
    resource=candidate_evidence.final_resources(out,config,events,reduced['within_arm']['capture'])
    host=candidate_evidence.host_evidence(out,config,events)
    assert resource==reduced['resource_evidence'] and host==reduced['host_evidence'] and host['oom_increased'] is False
    raw_refs={}
    for sample in release['samples']:
        for name in ('gpu','apps'):
            command=sample['probes'][name]['command']
            for ref in list(command['artifacts'].values())+[command['intent'],command['containment']['artifact']]:raw_refs[ref['path']]=ref
        for name in ('disk','port'):ref=sample['probes'][name]['artifact'];raw_refs[ref['path']]=ref
        for ref in sample['probes']['cgroup']['artifacts'].values():raw_refs[ref['path']]=ref
    quota=decimal_quota.validate(rb['typed_quota'],reconciliation.load_policy(),dt.datetime.fromisoformat(rb['typed_quota']['utc']))
    assert quota['reserved_s']==0
    disabled=read(directory/'schedule-disabled-primary404.json');assert disabled['enabled'] is False and disabled['nextRunAt'] is None
    return dict(pass_=True,network_calls=0,original_bindings=13118,overlay_source_pins_exact=True,
        archive_sha256=record['archive_sha256'],CRC_members=len(inventory),extracted_bytes=sum(x['bytes'] for x in inventory),
        payloads=42,primary_sha256=hashlib.sha256(primary_raw).hexdigest(),current_primary_numeric_version_equality=True,
        all_primary_members_exact=True,terminal_status='COMPLETE',terminal_wire_seconds={k:str(v) for k,v in quota.items()},
        status=reduced['status'],descriptive_U_resp=reduced['U_resp'],ratio=None,historical_comparison=False,adoption=False,
        final180=dict(start=begin['cleanup_start'],work_end=events[0]['work_deadline'],deadline=deadline,end=release['end'],samples=len(release['samples']),raw_refs=len(raw_refs)),
        actual_candidate=dict(pgid=identity['pgid'],starttime=identity['starttime'],boot_id=identity['boot_id']),
        replay_pid=replay[0]['pid'],host_evidence=host,publication=publication,followup_disabled=True,
        limitations=reduced['limitations'],gaps=[],slot_release_claim=False)

if __name__=='__main__':
    report=prove()
    target=Path(sys.argv[1])
    with target.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({key:report[key] for key in ('pass_','CRC_members','payloads','status','descriptive_U_resp','gaps')}))
