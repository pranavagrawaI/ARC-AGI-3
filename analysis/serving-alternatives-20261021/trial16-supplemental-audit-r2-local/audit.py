import sys
from pathlib import Path
APPROVED=Path(__file__).resolve().parent.parent/'trial16-missing-raw-local'
sys.path.insert(0,str(APPROVED))
import publication as output_publication
"""Independent supplemental audit; original ZIP and reducers stay unchanged."""
import base64,hashlib,json,shutil,sys
from pathlib import Path
from binding import HERE,ROOT,OVERLAY,ASSESSMENT,CURRENT_PRIMARY

ACTUAL_RESULT='serving-candidate-s10-16/output/result.json'

def independent_actual_result(download_dir,bundle):
    download_dir=Path(download_dir)
    receipt=json.loads((download_dir/'COLLECTION.json').read_bytes())
    rows=[row for row in receipt['files'] if row['name']==ACTUAL_RESULT]
    if len(rows)!=1 or rows[0]['origin']!='INDEPENDENT_CURRENT_SESSION_DOWNLOAD' or not rows[0]['complete']:
        raise ValueError('independent fetched actual-result origin required; primary alias forbidden')
    target=download_dir/'members'/ACTUAL_RESULT
    if target.is_symlink():raise ValueError('actual result cannot be an alias')
    embedded=ASSESSMENT/'evidence'/ACTUAL_RESULT
    if embedded.exists() and target.samefile(embedded):raise ValueError('actual result cannot alias primary-embedded inode')
    raw=target.read_bytes()
    if len(raw)!=rows[0]['bytes'] or hashlib.sha256(raw).hexdigest()!=rows[0]['sha256']:
        raise ValueError('actual-result origin/hash mismatch')
    actual=json.loads(raw);published=json.loads(base64.b64decode(bundle['files']['result.json']['base64'],validate=True))
    if actual!=published:raise ValueError('actual result differs from bundled result; original bytes retained')
    return actual

def verify_bound_readbacks(collection,download_dir):
    from remote import validate_readback
    import decimal_quota,reconciliation
    from collector import IDENTITY
    driver=json.loads((download_dir/'DRIVER-COMPLETION.json').read_bytes())
    if driver['schema']!='COLLECTION_DRIVER_COMPLETION_V1' or driver['deadline']!=driver['started']+1800 or not driver['started']<=driver['receipt_persisted_end']<driver['deadline'] or driver['receipt_sha256']!=hashlib.sha256((download_dir/'COLLECTION.json').read_bytes()).hexdigest():raise ValueError('driver completion/persistence deadline mismatch')
    from remote import validate_readback
    # Validate historical receipt chronology at the original receipt instant;
    # fresh checks already ran in the collector before and after every download.
    for label in ('before','after'):
        rb=json.loads((download_dir/label/'READBACK.json').read_bytes())
        checked=validate_readback(download_dir/label,__import__('datetime').datetime.fromisoformat(rb['typed_quota']['utc']))
        if any(collection[label].get(k)!=v for k,v in IDENTITY.items()):raise ValueError('collection exact identity mismatch')
        if hashlib.sha256((download_dir/label/'READBACK.json').read_bytes()).hexdigest()!=collection[label]['readback_sha256']:raise ValueError('collection readback hash drift')
        if collection[label]!=checked:raise ValueError('actual before/after readback contradicts collection receipt')
        quota=decimal_quota.validate(rb['typed_quota'],reconciliation.load_policy(),__import__('datetime').datetime.fromisoformat(rb['typed_quota']['utc']))
        if quota['used_s']!=decimal_quota.Decimal('2714.079') or quota['reserved_s']!=0:raise ValueError('terminal quota drift')
    before=json.loads((download_dir/'before/READBACK.json').read_bytes());after=json.loads((download_dir/'after/READBACK.json').read_bytes())
    if before['utc']>after['utc']:raise ValueError('before/after chronology')
    if before['inner_sha256']!=after['inner_sha256']:raise ValueError('before/after source identity')
    return after

def verify_download_ledger(collection,download_dir,allowed):
    from collector import pending_body_required
    files={row['name']:row for row in collection['files']}
    if len(files)!=len(collection['files']) or set(files)-set(allowed):raise ValueError('unexpected or duplicate supplemental member')
    if not 0<collection['list_pages']==len(collection['pages'])<=100 or any(row['items']>100 for row in collection['pages']):raise ValueError('listing budget/inventory mismatch')
    attempts=[json.loads(line) for line in (download_dir/'attempts.jsonl').read_bytes().splitlines()]
    if not len(attempts)==collection['download_attempts']==len(files)<=1596 or {row['name'] for row in attempts}!=set(files) or any(row['attempt']!=1 for row in attempts):raise ValueError('one-attempt download ledger mismatch')
    if collection['downloaded_bytes']!=sum(row['bytes'] for row in files.values()) or collection['downloaded_bytes']>2*1024**3 or collection['elapsed_seconds']>=1800 or collection['driver_elapsed_seconds']>=1800:raise ValueError('download byte/time cap accounting')
    waived={row['name'] for row in collection['skipped_conditionals']}
    for name in waived:
        if name not in allowed or not allowed[name]['conditional']:raise ValueError('unreviewed required-file waiver')
        h=download_dir/'members'/Path(name).with_name('http.json')
        if pending_body_required(json.loads(h.read_bytes()),name):raise ValueError('required pending body was waived')
    if set(files)|waived!=set(allowed) or set(files)&waived:raise ValueError('required supplemental dependency gap')
    return files,waived

def merge_inputs(download_dir,destination,files,allowed,base_directory=ASSESSMENT):
    base_directory=Path(base_directory)
    origins=json.loads((base_directory/'ORIGINS.json').read_bytes())
    for name,records in origins.items():
        source=base_directory/'evidence'/name;raw=source.read_bytes()
        if source.is_symlink() or any(len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256'] for row in records):raise ValueError('retained evidence origin/hash drift')
    evidence=destination/'evidence';shutil.copytree(base_directory/'evidence',evidence)
    for name,row in files.items():
        source=download_dir/'members'/name;raw=source.read_bytes()
        if source.is_symlink() or row['origin']!='INDEPENDENT_CURRENT_SESSION_DOWNLOAD' or not row['complete'] or len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('supplemental source/provenance mismatch')
        expected=allowed[name]
        if expected.get('expected_bytes') not in (None,len(raw)) or expected.get('expected_sha256') not in (None,row['sha256']):raise ValueError('reviewed missing input hash/bytes drift')
        target=evidence/name
        if target.exists() and name!=ACTUAL_RESULT:raise ValueError('already recovered artifact unexpectedly downloaded')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        origins[name]=[dict(row,source=str(source))]
    return evidence,origins

def supplemental(download_dir,destination):
    """A distinct required-files amendment; never claim whole ZIP/CD completeness."""
    from run import verify_local
    verify_local()
    sys.path[:0]=[str(OVERLAY),str(ROOT),str(ROOT/'collection_tools')]
    import candidate_evidence,primary,primary_bundle,transport,decimal_quota,reconciliation
    from collector import IDENTITY,pending_body_required
    download_dir=Path(download_dir);destination=Path(destination);destination.mkdir(exist_ok=False)
    report=dict(original_collection_status='INVALID_INCONCLUSIVE',original_archive_complete=False,
        central_directory_verified=False,collection_amendment='SUPPLEMENTAL_REQUIRED_FILES_ONLY',
        supplemental_complete=False,status='INVALID_INCONCLUSIVE',U_resp=None,ratio=None,adoption=False,issues=[],dependency_expansion=[])
    try:
        if shutil.disk_usage(destination).free<4*1024**3:raise ValueError('4GiB supplemental assembly reserve unavailable')
        collection=json.loads((download_dir/'COLLECTION.json').read_bytes())
        if not collection['complete']:raise ValueError('supplemental collection incomplete')
        after=verify_bound_readbacks(collection,download_dir)
        primary_raw=(CURRENT_PRIMARY/'PRIMARY.json').read_bytes();bundle=primary_bundle.verify(primary_raw)
        actual=independent_actual_result(download_dir,bundle)
        manifest=json.loads((HERE/'MISSING-MANIFEST.json').read_bytes());allowed={row['name']:row for row in manifest['files']}
        files,waived=verify_download_ledger(collection,download_dir,allowed)
        evidence,origins=merge_inputs(download_dir,destination,files,allowed)
        prefix='serving-candidate-s10-16';parent=evidence/prefix;output=parent/'output'
        spec=json.loads((ROOT/'packet-spec.json').read_bytes());config=spec['config']
        rbdir=download_dir/'after';remote=(rbdir/'source.ipynb').read_bytes()
        expected=(ROOT/'kernel'/spec['metadata']['code_file']).read_bytes()
        if transport.cell_source(json.loads(remote))!=transport.cell_source(json.loads(expected)):raise ValueError('supplemental exact outer source mismatch')
        values=transport.literals(transport.cell_source(json.loads(transport.decode(remote))))
        if hashlib.sha256(transport.decode(remote)).hexdigest()!=after['inner_sha256']:raise ValueError('supplemental exact inner mismatch')
        for name,item in values['PAYLOAD'].items():
            data=base64.b64decode(item['base64'],validate=True)
            if (parent/name).read_bytes()!=data:raise ValueError('actual payload source bytes mismatch')
        if json.loads((parent/'config.json').read_bytes())!=config or json.loads((parent/'assignment.json').read_bytes())!=values['ASSIGNMENT']:raise ValueError('config/assignment mismatch')
        if (evidence/primary_bundle.NAME).read_bytes()!=primary_raw:raise ValueError('recovered numeric-stream/current-primary mismatch')
        primary.verify(primary_raw,rbdir)
        events=[json.loads(line) for line in (output/'supervisor.jsonl').read_bytes().splitlines()]
        publication=primary_bundle.verify_publication(output,primary_raw,events[0]['work_deadline'],config['primary_seconds'])
        proof=candidate_evidence.check(output,config,actual.get('error'),actual.get('failed_stage'))
        if any(actual.get(key)!=value for key,value in proof.items()):raise ValueError('independent scientific outcome contradicts actual result')
        if any(actual.get(key) for key in ('cleanup_errors','phase_cleanup_errors','retained_queries','surviving_phase_groups','primary_bundle_error','late_work')):raise ValueError('actual lifecycle/publication failure')
        completed_proof=dict(proof,publication=publication,actual_result_origin='INDEPENDENT_CURRENT_SESSION_DOWNLOAD',
            original_bindings=13118,payloads=len(values['PAYLOAD']),required_downloaded=len(files),pending_not_required=len(waived))
        return output_publication.publish(destination,report,completed_proof,origins)
    except FileNotFoundError as error:
        report=output_publication.invalid(report,error)
        report['issues'].append('required dependency unavailable')
        missing=Path(error.filename)
        base=destination/'evidence'
        report['dependency_expansion'].append(dict(path=error.filename,remote_name=str(missing.relative_to(base)) if missing.is_relative_to(base) else None,
            reason='unchanged checker requires missing input; expose before further remote authority'))
    except Exception as error:report=output_publication.invalid(report,error)
    return output_publication.failed(destination,report,ValueError('supplemental audit did not complete'))

if __name__=='__main__':
    result=supplemental(Path(sys.argv[1]),Path(sys.argv[2]))
    print(json.dumps({key:result.get(key) for key in ('supplemental_complete','status','U_resp','issues','dependency_expansion')}))
