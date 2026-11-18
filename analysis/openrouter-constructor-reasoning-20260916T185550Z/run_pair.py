"""Gated off-then-on pair. No paid preflight or automatic retries."""
import argparse, dataclasses, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'engine'))
sys.path.insert(0,str(HERE))
from transport import Transport, Stop, write, exclusive, initial_ledger, ENDPOINT, digest
from constructor import run_construction, ConstructionLimits, ConstructionResult
from prefix import project_transitions
from host import bound_constructor

LIMITS=ConstructionLimits(max_model_requests=6,wall_seconds=240)

def verify_inventory():
    path=HERE/'INVENTORY.json';raw=path.read_bytes();inventory=json.loads(raw)
    for name,sha in inventory['files'].items():
        p=HERE/name
        if not p.is_file() or p.is_symlink() or digest(p.read_bytes())!=sha:raise Stop('candidate drift: '+name)
    return digest(raw)

def gates(review,release):
    sha=verify_inventory()
    for path,role,decision in [(review,'independent-reviewer','PASS'),(release,'Hermes','RELEASE')]:
        d=json.loads(Path(path).read_text())
        if d.get('role')!=role or d.get('decision')!=decision or d.get('inventory_sha256')!=sha or d.get('endpoint')!=ENDPOINT or d.get('task')!='OR-CONSTRUCTOR-REASONING-PAIR':raise Stop('missing/mismatched independent gate')
    return sha

def stop_ledger(path,reason):
    d=json.loads(path.read_text());d['stopped']=True;d['local_stop']=reason;write(path,d)

def episode(arm,output,ledger,*,key=None,fixture_url=None,wall=240):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'tmp').mkdir()
    import tempfile
    tempfile.tempdir=str(output/'tmp')
    started=time.monotonic();deadline=started+wall
    transport=Transport(arm,output/'raw',ledger,deadline,key=key,fixture_url=fixture_url)
    result=ConstructionResult(status='WORKING')
    def progress(r):write(output/'PROGRESS.json',dataclasses.asdict(r))
    def event(e):
        with (output/'tool-events.jsonl').open('ab') as f:
            f.write((json.dumps(e,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
    try:
        with bound_constructor():
            result=run_construction(project_transitions(HERE/'prefix.jsonl'),transport=transport,limits=LIMITS,event_sink=event,progress_sink=progress,result=result)
        if time.monotonic()>deadline:raise Stop('soft deadline: no publication')
        d=json.loads(Path(ledger).read_text());counts=d['arms'][arm]
        if counts['requests']==0:raise Stop('no arm request admitted')
        treatment='verified-provider-reported-off' if arm=='off' and counts['requests']>0 else 'on-not-demonstrated'
        if arm=='on' and counts['reasoning_tokens']>0:treatment='verified-provider-reported-on'
        if arm=='on' and not counts['reasoning_tokens']:raise Stop('on treatment not demonstrated')
        record={'construction_status':result.status,'frozen':False,'treatment':treatment,'prospective':'not_run','supported_model':False,'elapsed_seconds':time.monotonic()-started,'model_requests':result.model_requests,'inspections':result.inspections,'submissions':result.submissions,'usage':result.usage,'repair':{'failed_submissions':sum(e['call']['function']['name']=='submit_candidate' and (json.loads(e['rendered_result_message']['content']).get('status')=='rejected' or json.loads(e['rendered_result_message']['content']).get('visible_evaluation',{}).get('visible',{}).get('status')=='failed') for e in result.tool_events),'accepted_after_repair':result.status=='CANDIDATE_READY' and result.submissions>1}}
        if result.status=='CANDIDATE_READY':
            if result.evaluation['visible']['status']!='passed' or result.evaluation.get('evidence_claim')!='ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL':raise Stop('evaluation claim mismatch')
            exclusive(output/'FROZEN.json',{'candidate_source':result.candidate_source,'source_digest':digest(result.candidate_source.encode()),'evidence_claim':'ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL','supported_model':False,'prospective':'not_run'})
            record['frozen']=True
        if result.status.startswith('INVALID'):raise Stop(result.status)
        write(output/'RESULT.json',record);progress(result)
        return record
    except Exception as e:
        reason=str(e) if isinstance(e,Stop) else type(e).__name__
        result.status='STOPPED';progress(result);stop_ledger(Path(ledger),reason)
        write(output/'RESULT.json',{'construction_status':'STOPPED','reason':reason,'frozen':False,'elapsed_seconds':time.monotonic()-started,'prospective':'not_run'})
        raise Stop(reason) from None

def supervise(command,output,hard_seconds=245):
    """A child may leave forensic files; only a timely exit promotes authority."""
    output=Path(output);started=time.monotonic()
    with (output/'worker.stdout').open('xb') as out,(output/'worker.stderr').open('xb') as err:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        timed_out=False
        try:
            child.wait(timeout=hard_seconds)
        except subprocess.TimeoutExpired:
            timed_out=True
        finally:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
    return {'returncode':child.returncode,'hard_timeout':timed_out,'elapsed_seconds':time.monotonic()-started,'reaped':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',choices=['off','on']);p.add_argument('--review',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--credential-approval',type=Path,required=True);args=p.parse_args()
    sha=gates(args.review,args.release)
    permission=json.loads(args.credential_approval.read_text())
    if permission.get('task')!='OR-CONSTRUCTOR-REASONING-PAIR' or permission.get('credential_use')!='sole-key-existing-loader' or permission.get('decision')!='APPROVED' or permission.get('role')!='PM':raise Stop('credential-use approval pending')
    live=HERE/'live'
    if args.worker:
        if not (live/'CONSUMED.json').is_file():raise Stop('no parent admission')
        arm=args.worker
        if json.loads((live/'ACTIVE_ARM.json').read_text()).get('arm')!=arm:raise Stop('arm order/admission mismatch')
        if arm=='on' and not (live/'off/RESULT.json').is_file():raise Stop('off not promoted')
        exclusive(live/f'{arm}-CONSUMED.json',{'arm':arm,'inventory_sha256':sha})
        if os.geteuid()!=0 or '65534' not in Path('/proc/self/uid_map').read_text() and not any(int(line.split()[2])>65534 for line in Path('/proc/self/uid_map').read_text().splitlines()):raise Stop('mapped namespace missing')
        from auth import load_key
        key,mechanism=load_key(credential_file_authorized=True)
        if not key:raise Stop('credentials unavailable')
        episode(arm,live/arm/'child',live/'ledger.json',key=key)
        return
    if not args.execute:raise Stop('--execute required')
    from auth import load_key
    key,mechanism=load_key(credential_file_authorized=True)
    if not key:raise Stop('credentials unavailable')
    del key
    live.mkdir(exist_ok=False)
    exclusive(live/'CONSUMED.json',{'inventory_sha256':sha,'endpoint':ENDPOINT,'order':['off','on'],'review_sha256':digest(args.review.read_bytes()),'release_sha256':digest(args.release.read_bytes())})
    initial_ledger(live/'ledger.json')
    for arm in ('off','on'):
        root=live/arm;root.mkdir()
        write(live/'ACTIVE_ARM.json',{'arm':arm,'inventory_sha256':sha})
        command=['unshare','--user','--map-auto','--map-root-user',sys.executable,'-B',str(Path(__file__).resolve()),'--worker',arm,'--review',str(args.review.resolve()),'--release',str(args.release.resolve()),'--credential-approval',str(args.credential_approval.resolve())]
        receipt=supervise(command,root);write(root/'SUPERVISION.json',receipt)
        state=json.loads((live/'ledger.json').read_text())
        if receipt['returncode'] or receipt['hard_timeout'] or state['stopped'] or state['unknown_exposure_usd']!='0':
            stop_ledger(live/'ledger.json','child failure/hard deadline or unresolved attempt')
            write(root/'RESULT.json',{'frozen':False,'status':'HOLD','child_artifacts':'forensic only, unpromoted'})
            raise Stop('pair stopped; never rerun')
        result=json.loads((root/'child/RESULT.json').read_text())
        if result['frozen']:
            raw=(root/'child/FROZEN.json').read_bytes();frozen=json.loads(raw)
            if digest(frozen['candidate_source'].encode())!=frozen['source_digest']:raise Stop('child freeze digest')
            exclusive(root/'FROZEN.json',frozen)
        write(root/'RESULT.json',dict(result,authority='timely root promotion'))
    write(live/'RESULT.json',{'status':'PAIR_COMPLETE_DESCRIPTIVE_ONLY','inventory_sha256':sha,'identity':'hosted OpenRouter, not Kaggle checkpoint equivalence','ledger':json.loads((live/'ledger.json').read_text())})

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('BLOCKED:',str(e) if isinstance(e,Stop) else type(e).__name__)
        sys.exit(2)
