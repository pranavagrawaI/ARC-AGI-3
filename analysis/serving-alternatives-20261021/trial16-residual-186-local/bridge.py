"""Explicit two-pass ledger; never synthesize a single complete collection receipt."""
import datetime as dt,json
from collections import Counter
from pathlib import Path
from context import *
from collector import pending_body_required


def partition(allowed,first_files,first_attempts,waived,second_files,second_attempts,repeat):
    first_files,waived,second_files=map(set,(first_files,waived,second_files))
    require(not first_files&second_files and not (first_files|second_files)&waived,'redownload/waiver overlap')
    require(first_files|second_files|waived==set(allowed),'required dependency gap/expansion')
    require(all(allowed[n]['conditional'] for n in waived),'unconditional dependency waived')
    require(len(set(first_attempts))==len(first_attempts) and len(set(second_attempts))==len(second_attempts),'duplicate attempt within pass')
    require(set(first_attempts)==first_files|{repeat} and set(second_attempts)==second_files,'attempt/file ledger mismatch')
    require(set(first_attempts)&set(second_attempts)=={repeat},'unapproved repeat connection')
    counts=Counter(first_attempts+second_attempts)
    require(counts[repeat]==2 and all(v==1 for k,v in counts.items() if k!=repeat),'overall connection-attempt contract')
    return dict(successful_files=len(first_files|second_files),waived_files=len(waived),total_attempts=sum(counts.values()),repeated_names=[repeat])


def ledger(directory,state,allowed,bounds,complete):
    directory=Path(directory);files={}
    require(state['complete'] is complete,'pass completion mismatch')
    require(state['addressing']=='CURRENT_SESSION_BEFORE_AFTER_EXACT_V1_NOT_COMPLETE_ZIP','collection addressing contract')
    require(state['skipped_conditionals']==[],'residual waiver must be derived separately')
    require(0<state['list_pages']==len(state['pages'])<=bounds['list_pages'],'list-page count')
    matched=[]
    for i,page in enumerate(state['pages'],1):
        require(page['page']==i and type(page['items']) is int and 0<=len(page['matched'])<=page['items']<=bounds['page_size'],'page size/order')
        matched+=page['matched']
    require(len(set(matched))==len(matched) and set(matched)<=set(allowed),'listed name filter/duplicates')
    attempts=[read_line(line) for line in (directory/'attempts.jsonl').read_bytes().splitlines()]
    names=[safe_name(r['name']) for r in attempts]
    require(len(names)==state['download_attempts']<=bounds['downloads'] and len(set(names))==len(names),'pass attempt budget/duplicates')
    require(set(names)<=set(matched) and all(r['attempt']==1 for r in attempts),'unlisted or repeated attempt')
    for row in state['files']:
        name=safe_name(row['name']);require(name in allowed and name not in files,'unexpected/duplicate completed file')
        source=directory/'members'/name
        require(all(not p.is_symlink() for p in [source,*source.parents] if p.is_relative_to(directory)),'symlink origin')
        require(type(row['bytes']) is int and row['bytes']>=0,'invalid byte count')
        require(row['origin']=='INDEPENDENT_CURRENT_SESSION_DOWNLOAD' and row['complete'] is True and row['attempt']==1,'actual independent origin required')
        require(source.stat().st_size==row['bytes'] and sha(source)==row['sha256'],'retained file hash/bytes mismatch')
        expected=allowed[name]
        require(expected.get('expected_bytes') in (None,row['bytes']) and expected.get('expected_sha256') in (None,row['sha256']),'manifest comparison mismatch; retain actual bytes')
        files[name]=row
    require(set(files)<=set(names) and state['downloaded_bytes']==sum(r['bytes'] for r in files.values())<=bounds['bytes'],'pass byte/attempt accounting')
    require(0<=state['elapsed_seconds']<bounds['wall_seconds'] and 0<=state['driver_elapsed_seconds']<bounds['wall_seconds'],'pass cooperative duration')
    if complete:require(set(files)==set(names)==set(allowed) and not state.get('error') and not state.get('partial'),'incomplete residual pass')
    return files,names


def read_line(line):return json.loads(line)


def driver(directory,seconds,schema):
    directory=Path(directory);d=read(directory/'DRIVER-COMPLETION.json')
    require(d['schema']==schema and d['deadline']==d['started']+seconds,'separate driver cap/schema')
    state=read(directory/'COLLECTION.json')
    require(0<=state['elapsed_seconds']<=state['driver_elapsed_seconds']<=d['receipt_persisted_end']-d['started'],'driver/collector elapsed mismatch')
    require(d['started']<=d['receipt_persisted_end']<d['deadline'] and d['receipt_sha256']==sha(directory/'COLLECTION.json'),'driver persistence/receipt hash')
    return d


def readback(directory,state,label):
    from remote import validate_readback
    raw=read(directory/label/'READBACK.json')
    checked=validate_readback(directory/label,dt.datetime.fromisoformat(raw['typed_quota']['utc']))
    require(state[label]==checked,'saved exact source/status/Decimal readback drift')
    return raw


def prepare():
    """Offline exact first-pass proof and residual derivation; no remote transport."""
    require(sha(REVIEW)==REVIEW_SHA and sha(FIRST/'COLLECTION.json')==FIRST_SHA,'fixed root/first receipt hash')
    review=read(REVIEW);manifest=read(APPROVED/'MISSING-MANIFEST.json');allowed={r['name']:r for r in manifest['files']}
    require(len(allowed)==len(manifest['files'])==1596,'original dependency universe')
    state=read(FIRST/'COLLECTION.json')
    from collector import CAPS as original_caps
    files,attempts=ledger(FIRST,state,allowed,original_caps,False)
    require(len(files)==1400 and len(attempts)==1401 and state['downloaded_bytes']==248479141,'fixed first-pass counts')
    require('after' not in state and not (FIRST/'after').exists(),'first pass must remain failed without after')
    require(state['error']['type']=='ConnectTimeout' and state['safe_error']['http_response']=='UNAVAILABLE','preserve original connection failure')
    p=state['partial'];require(p['name']==attempts[-1]==REPEAT and p['bytes_persisted']==0 and not p['complete'],'exact first failed boundary')
    require(set(attempts)-set(files)=={REPEAT} and not (FIRST/'members'/REPEAT).exists(),'unexpected first partial file')
    d=driver(FIRST,1800,'COLLECTION_DRIVER_COMPLETION_V1');before=readback(FIRST,state,'before')
    waivers={}
    for waiver in review['conditional_waivers']:
        name=safe_name(waiver['name']);http=safe_name(waiver['http_name'])
        require(name in allowed and allowed[name]['conditional'] and name not in files and name not in waivers,'waiver domain')
        require(http==str(Path(name).with_name('http.json')) and http in files and files[http]['sha256']==waiver['http_sha256'],'waiver independent HTTP origin')
        require(waiver['body_required'] is False and pending_body_required(read(FIRST/'members'/http),name) is False,'body is required')
        waivers[name]=waiver
    require(len(waivers)==10 and {n for n,r in allowed.items() if r['conditional']}==set(waivers),'all ten conditional prerequisites')
    remaining=[r for n,r in sorted(allowed.items()) if n not in files and n not in waivers]
    require(remaining==review['remaining_files'] and len(remaining)==186 and all(r['conditional'] is False for r in remaining),'exact residual set/row contract')
    require(set(attempts)&{r['name'] for r in remaining}=={REPEAT},'only one proposed second connection')
    require(review['proposed_second_pass']==CAPS|{'attempts_per_name':1} and review['one_proposed_repeat_connection_name']==REPEAT,'root proposed cap/exception')
    return dict(first=state,first_files=files,first_attempts=attempts,first_driver=d,first_before=before,waivers=waivers,remaining=remaining,allowed=allowed)


def validate(second):
    """Validate both actual ledgers and identity closure, retaining first.complete=false."""
    plan=prepare();second=Path(second);state=read(second/'COLLECTION.json')
    residual={r['name']:r for r in plan['remaining']};files,attempts=ledger(second,state,residual,CAPS,True)
    driver2=driver(second,600,'RESIDUAL_DRIVER_COMPLETION_V1')
    before=readback(second,state,'before');after=readback(second,state,'after')
    times=[dt.datetime.fromisoformat(r['utc']) for r in (plan['first_before'],before,after)]
    require(times==sorted(times),'two-pass before/after chronology')
    from run_residual import authorize
    auth=read(second/'AUTHORITY.json');require(auth['approval_sha256']==sha(auth['approval_path']),'second external authority drift')
    approval=authorize(Path(auth['approval_path']));require(auth['approval']==approval,'second authority record mismatch')
    require(driver2['started']>=plan['first_driver']['receipt_persisted_end'],'two-pass monotonic clock overlap')
    summary=partition(plan['allowed'],plan['first_files'],plan['first_attempts'],set(plan['waivers']),files,attempts,REPEAT)
    require(summary==dict(successful_files=1586,waived_files=10,total_attempts=1587,repeated_names=[REPEAT]),'exact two-pass totals')
    ledger_report=dict(schema='TRIAL16_TWO_PASS_REQUIRED_FILES_V1',first_complete=False,first_after_available=False,
        second_complete=True,original_archive_complete=False,central_directory_verified=False,
        first=dict(directory=str(FIRST),receipt_sha256=FIRST_SHA,driver=plan['first_driver'],caps=read(APPROVED/'MISSING-MANIFEST.json')['bounds'],attempts=1401,files=1400,bytes=248479141),
        second=dict(directory=str(second),receipt_sha256=sha(second/'COLLECTION.json'),driver=driver2,caps=CAPS,attempts=186,files=186,bytes=state['downloaded_bytes']),
        conditional_waivers=list(plan['waivers'].values()),
        repeat_connection_history=dict(name=REPEAT,first_pass=1,second_pass=1,overall=2),
        monotonic_clock_scope='separate per-pass cooperative clocks; no fabricated single clock',**summary)
    return plan,files,after,ledger_report


def assemble(second,destination,plan,second_files):
    # Reuse approved origin validation/copy semantics, then append pass 2 without a fake receipt.
    import importlib.util
    spec=importlib.util.spec_from_file_location('approved_audit_r2',AUDIT_R2/'audit.py');module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(AUDIT_R2));spec.loader.exec_module(module)
    evidence,origins=module.merge_inputs(FIRST,destination,plan['first_files'],plan['allowed'])
    for name in plan['first_files']:
        for row in origins[name]:row.update(collection_pass=1,collection_receipt_sha256=FIRST_SHA)
    append_second(second,evidence,origins,second_files)
    return evidence,origins


def append_second(second,evidence,origins,second_files):
    for name,row in second_files.items():
        source=Path(second)/'members'/safe_name(name);target=evidence/name
        require(not source.is_symlink(),'second source alias')
        require(not target.exists() or name==ACTUAL_RESULT,'second pass would overwrite retained evidence')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        require(sha(target)==row['sha256'],'assembly copy drift')
        origins[name]=[dict(row,source=str(source),collection_pass=2,collection_receipt_sha256=sha(Path(second)/'COLLECTION.json'))]
    return evidence,origins
