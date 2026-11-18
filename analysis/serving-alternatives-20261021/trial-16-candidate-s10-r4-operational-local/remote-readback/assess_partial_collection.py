"""CPU-only required-input map. Never reconstruct missing scientific receipts."""
import base64, collections, datetime as dt, hashlib, json, os, shutil, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OVERLAY=ROOT.parent/'trial16-collection-decimal-overlay-local'
sys.path[:0]=[str(OVERLAY),str(ROOT),str(ROOT/'collection_tools')]
import candidate_evidence, decimal_quota, primary, primary_bundle, reconciliation, transport

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def assess(destination):
    destination=Path(destination);destination.mkdir(exist_ok=False)
    d=ROOT/'remote-readback';recovery=read(d/'partial-recovery-v1/RECOVERY.json')
    assert recovery['archive_sha256']==sha(d/'archive-v1/output.zip')=='de1677dd95cd77656ece739e071ba99fb89e475a1c65f5eba5cb2f05cb6fe550'
    freeze=read(ROOT/'freeze.json')
    for name,expected in freeze['files'].items():assert sha(ROOT/name)==expected,name
    for name,expected in freeze['references'].items():assert sha(name)==expected,name
    assert len(freeze['files'])+len(freeze['references'])==freeze['bindings']==13118
    pins=read(OVERLAY/'SOURCE-PINS.json')
    assert sha(OVERLAY/'SOURCE-PINS.json')=='c609808e1a09042bb018a786eeda292c9eeb8e495962d65fb1746714613c2568'
    for name,expected in pins['overlay_files'].items():assert sha(OVERLAY/name)==expected,name
    for name,expected in pins['references'].items():assert sha(name)==expected,name
    fresh=d/'current-primary-fallback-local/fresh-readback';rb=read(fresh/'READBACK.json');meta=read(fresh/'metadata.json')
    assert not rb.get('error') and rb['source_exact'] is True and rb['status']['status']=='COMPLETE'
    assert (meta['id'],meta['currentVersionNumber'],rb['kernel_id'],rb['version'])==(140493086,1,140493086,1)
    spec=read(ROOT/'packet-spec.json');prefix=Path(spec['config']['cwd']).name
    remote=(fresh/'source.ipynb').read_bytes();local=(ROOT/'kernel'/spec['metadata']['code_file']).read_bytes()
    assert transport.cell_source(json.loads(remote))==transport.cell_source(json.loads(local))
    inner=transport.decode(remote);values=transport.literals(transport.cell_source(json.loads(inner)))
    assert transport.sha(inner)==rb['inner_sha256']==read(ROOT/'launch-attempt.json')['inner_notebook_sha']
    quota=decimal_quota.validate(rb['typed_quota'],reconciliation.load_policy(),dt.datetime.fromisoformat(rb['typed_quota']['utc']))
    assert quota['reserved_s']==0
    current=d/'current-primary-fallback-local/primary-current/PRIMARY.json';primary_raw=current.read_bytes()
    primary.verify(primary_raw,fresh);bundle=primary_bundle.verify(primary_raw)
    assert not bundle['missing'] and not bundle['truncated']
    origins={};required={};data_objects=[]
    def need(name,reason,expected_sha=None,expected_bytes=None,conditional=False):
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts
        row=required.setdefault(str(path),dict(name=str(path),reasons=[],expected_sha256=None,expected_bytes=None,conditional=conditional))
        row['conditional']=row['conditional'] and conditional
        if reason not in row['reasons']:row['reasons'].append(reason)
        for key,value in [('expected_sha256',expected_sha),('expected_bytes',expected_bytes)]:
            if value is not None:
                assert row[key] in (None,value),(name,key,row[key],value)
                row[key]=value
    for item in recovery['entries']:
        if not item['retained']:continue
        source=d/'partial-recovery-v1/members'/item['name'];raw=source.read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        target=destination/'evidence'/item['name'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        origins[item['name']]=[dict(kind='CRC_VERIFIED_SEQUENTIAL_LOCAL_MEMBER',source=str(source),sha256=item['sha256'],bytes=item['bytes'],crc32=item['crc32'],offset=item['offset'],next_offset=item['next_offset'])]
    for name,item in bundle['files'].items():
        archive_name=f'{prefix}/{name}' if name in ('config.json','assignment.json') else f'{prefix}/output/{name}'
        raw=base64.b64decode(item['base64'],validate=True)
        target=destination/'evidence'/archive_name
        if target.exists():assert target.read_bytes()==raw,archive_name
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        origins.setdefault(archive_name,[]).append(dict(kind='VERIFIED_CURRENT_PRIMARY_EMBEDDED',source=str(current),bundle_member=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        need(archive_name,'reviewed primary evidence',hashlib.sha256(raw).hexdigest(),len(raw))
        if name.endswith('.json'):data_objects.append(json.loads(raw))
        if name.endswith('.jsonl'):data_objects.extend(json.loads(line) for line in raw.splitlines())
    recovered_primary=destination/'evidence'/primary_bundle.NAME
    assert recovered_primary.exists() and recovered_primary.read_bytes()==primary_raw
    need(primary_bundle.NAME,'numeric-version-request local primary equality',hashlib.sha256(primary_raw).hexdigest(),len(primary_raw))
    for name,item in values['PAYLOAD'].items():
        raw=base64.b64decode(item['base64'],validate=True);assert transport.sha(raw)==item['sha256']
        need(f'{prefix}/{name}','archive payload/source binding',item['sha256'],len(raw))
    parent=destination/'evidence'/prefix;out=parent/'output'
    def output_need(name,reason,expected_sha=None,expected_bytes=None,conditional=False):need(f'{prefix}/output/{name}',reason,expected_sha,expected_bytes,conditional)
    events_path=out/'candidate/events.jsonl';assert events_path.exists()
    events=[json.loads(line) for line in events_path.read_bytes().splitlines()]
    output_need('candidate/events.jsonl','independent S10/window/source/API/accounting/replay identity')
    sends=[row for row in events if row['kind']=='send'];responses={row['rid']:row for row in events if row['kind']=='response'}
    for row in sends:
        rid=row['rid'];output_need('candidate/'+row['transport_ref'],'exact outbound HTTP bytes')
        output_need('candidate/'+row['wire_ref'],'outbound body/source/rendering joins')
        response=responses.get(rid)
        http_ref=response['http_ref'] if response else f'{rid}/http.json'
        output_need('candidate/'+http_ref,'actual socket-write/body/cutoff/concurrency clocks')
        body_ref=response['response_ref'] if response else f'{rid}/body.bin'
        known=out/'candidate'/http_ref
        h=read(known) if known.exists() else None
        if response or h is None or h['status']==200:
            output_need('candidate/'+body_ref,'raw API response/accounting or pending200 persistence',expected_bytes=h['persisted_bytes'] if h else None,conditional=response is None and h is None)
    for name in ('candidate-scope.json','observer-scope.json','host-boundaries.jsonl','host-series.result.json','candidate-release.json','result.json','supervisor.jsonl'):
        output_need(name,'independent candidate/raw resource/host/OOM/identity/lifecycle checks')
    output_need('host-series.jsonl','full sampled raw cgroup/stat/OOM/process evidence',bundle['series_ref']['sha256'],bundle['series_ref']['bytes'])
    for name in ('primary-bundle-receipt.json','primary-publication-completion.json'):output_need(name,'post-persistence and caller-return publication deadline proof')
    plan=read(ROOT/'candidate-plan.json')
    for name,expected in plan['candidate_frontend_source_sha'].items():output_need('candidate-package/installed-source/'+name,'actual installed source byte identity',expected)
    for name in ('installed-source-identity.json','model-source-identity.json','prepared.json','installed-package-identity.json','candidate-launch.json'):
        output_need('candidate-package/'+name,'candidate package/model/assets/flags/rendering identity')
    for item in bundle['health_inventory']:output_need(item['path'],'actual health-probe log inventory',item['sha256'],item['bytes'])
    def references(value):
        if isinstance(value,dict):
            if {'path','sha256','bytes'}<=set(value) and isinstance(value['path'],str) and value['path'].endswith('.raw'):
                output_need('query-artifacts/'+value['path'],'raw query/resource intent/containment/data evidence',value['sha256'],value['bytes'])
            for child in value.values():references(child)
        elif isinstance(value,list):
            for child in value:references(child)
    for value in data_objects:references(value)
    inventory=[];missing=[]
    for name,row in sorted(required.items()):
        target=destination/'evidence'/name
        entry=dict(row,available=target.exists(),origins=origins.get(name,[]))
        if target.exists():
            entry.update(actual_bytes=target.stat().st_size,actual_sha256=sha(target))
            if row['expected_bytes'] is not None:assert row['expected_bytes']==entry['actual_bytes'],name
            if row['expected_sha256'] is not None:assert row['expected_sha256']==entry['actual_sha256'],name
        else:missing.append(entry)
        inventory.append(entry)
    current_reduction=read(d/'derived-v1/REDUCTION.json')
    assert current_reduction['status']=='INVALID_INCONCLUSIVE' and current_reduction['U_resp'] is None
    # Observe the unchanged independent checker on available data; no fake receipts.
    try:
        partial=candidate_evidence.check(out,spec['config'])
        check_attempt=dict(returned_status=partial['status'],within_arm_issues=partial['within_arm']['issues'])
    except Exception as error:check_attempt=dict(error_type=type(error).__name__,message=str(error))
    report=dict(scope='REQUIRED_INPUT_SCOPE_ASSESSMENT_ONLY_NOT_RATE_QUALIFICATION',archive_complete=False,central_directory_verified=False,
        full_archive_gate_blocks=True,scientifically_required_missing_also_blocks=bool(missing),
        required_files=len(inventory),available_required_files=len(inventory)-len(missing),missing_files=len(missing),
        unconditional_missing_files=sum(not x['conditional'] for x in missing),conditional_missing_files=sum(x['conditional'] for x in missing),
        missing_known_bytes=sum(x['expected_bytes'] for x in missing if x['expected_bytes'] is not None),missing_unknown_size_files=sum(x['expected_bytes'] is None for x in missing),
        missing=missing,checker_on_available_data=check_attempt,scope_limitations=[
            'No central directory or complete remote file inventory; unknown additional files cannot be enumerated.',
            'Missing-byte total is a lower bound; many receipt sizes are unknowable without their original bytes.',
            'Pending response body required when original HTTP status200; unknown HTTP receipts remain conditional.',
            'Dependency list derives from pinned reducer source, replay events, primary inventories and raw references; missing files may expose further raw dependencies.',
            'Recovered numeric-version-request primary equality does not establish whole-archive completeness.'])
    (destination/'ORIGINS.json').write_text(json.dumps(origins,indent=2)+'\n')
    (destination/'REQUIRED-INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (destination/'REQUIRED-GAPS.json').write_text(json.dumps(report,indent=2)+'\n')
    proof=dict(pass_=True,scope='PARTIAL_COLLECTION_SOURCE_TERMINAL_PROOF',network_calls=0,
        original_bindings=13118,overlay_source_pins_exact=True,payload_map=42,
        archive_sha256=sha(d/'archive-v1/output.zip'),archive_bytes=(d/'archive-v1/output.zip').stat().st_size,
        recovered_local_members=recovery['verified_members'],recovered_decoded_bytes=recovery['decoded_bytes'],
        complete_archive=False,central_directory_verified=False,recovered_primary_current_bytes_equal=True,
        primary_sha256=sha(current),primary_full_supervisor_result_join=True,
        terminal_status='COMPLETE',terminal_wire_seconds={k:str(v) for k,v in quota.items()},
        frozen_overlay_reducer_status=current_reduction['status'],accepted_rate=None,ratio=None,adoption=False,
        required_missing=len(missing),followup_disabled=True,slot_release_claim=False,
        root_closure_sha256=sha(ROOT.parent/'coordination-20261024/trial16-attempt-closed.json'),
        conclusion='Collection incomplete/inconclusive; primary producer workload-feasible report is not a qualified descriptive rate or candidate-negative.')
    disabled=read(d/'schedule-disabled-primary404.json');assert disabled['enabled'] is False and disabled['nextRunAt'] is None
    (destination/'COLLECTION-PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(proof=proof,gaps={k:v for k,v in report.items() if k not in ('missing','scope_limitations')}),indent=2))

if __name__=='__main__':assess(Path(sys.argv[1]))
