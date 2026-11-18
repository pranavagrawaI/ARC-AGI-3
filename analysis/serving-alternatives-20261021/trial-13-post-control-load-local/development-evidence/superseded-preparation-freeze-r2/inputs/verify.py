"""Bounded CPU packet proof; prior64/65 is disclosed, never rewritten as PASS."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SDK=Path('/home/pranav/.local/share/uv/tools/kaggle/bin/python')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bindings():
    frozen=json.loads((HERE/'freeze.json').read_text())
    for name,h in frozen['files'].items():
        if sha(HERE/name)!=h:raise ValueError('input drift '+name)
    for name,h in frozen['references'].items():
        if sha(name)!=h:raise ValueError('reference drift '+name)
    return dict(freeze_sha256=sha(HERE/'freeze.json'),files=len(frozen['files']),references=len(frozen['references']))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
    signal.alarm(600);started=time.monotonic();before=bindings()
    env=dict(os.environ,LOCAL_ADAPTER_PROOF='1',PYTHONPATH=str(HERE)+os.pathsep+str(HERE/'guard'),PACKET_PROOF_DIR=str(out/'cases'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE/'tests'),'-v'],cwd=HERE,env=env,capture_output=True,text=True,timeout=590)
    (out/'tests.stdout').write_text(tests.stdout);(out/'tests.log').write_text(tests.stderr)
    if tests.returncode:raise ValueError('packet suite failed; preserved logs/cases')
    sdk=subprocess.run([str(SDK),str(HERE/'verify_sdk.py'),str(out/'SDK-PROOF.json')],cwd=HERE,capture_output=True,text=True,timeout=60)
    (out/'sdk.stdout').write_text(sdk.stdout);(out/'sdk.stderr').write_text(sdk.stderr)
    if sdk.returncode:raise ValueError('local SDK proof failed')
    sdkproof=json.loads((out/'SDK-PROOF.json').read_text())
    if sdkproof['network_calls'] or sdkproof['network_attempts']:raise ValueError('unexpected network')
    import packet,launch,tempfile
    with tempfile.TemporaryDirectory() as td:report=packet.prepare(Path(td)/'review')
    if report!=json.loads((HERE/'review-packet.json').read_text()):raise ValueError('review build mismatch')
    try:launch.clearance()
    except ValueError as error:
        if not str(error).startswith('no reviewed serialized assignment'):raise
    else:raise ValueError('unexpected authority during local preparation')
    after=bindings()
    if before!=after:raise ValueError('bindings changed during qualification')
    # This records the SINGLE actual baseline run; never calls it a full pass.
    baseline=(HERE/'development-evidence/runtime-first/tests.log').read_text()
    if 'Ran 65 tests' not in baseline or 'FAILED (failures=1)' not in baseline or 'live process identity incomplete' not in baseline:raise ValueError('baseline disclosure drift')
    proof=dict(status='NEW_PACKET_TESTS_AND_SDK_PASS_BASELINE64OF65_REQUIRES_REVIEW',seconds=time.monotonic()-started,bindings=after,
        source_sha256=report['sources_sha256'],config_sha256=report['config_sha256'],policy_sha256=sha(HERE/'prior-reconciliation-policy.json'),
        baseline_tests=65,baseline_pass=64,baseline_failure='unchanged test_clocked_series_rechecks_transient_resources_without_resetting_end: conservative UnsafeIdentity during disappearance',baseline_log_sha256=sha(HERE/'development-evidence/runtime-first/tests.log'),
        packet_tests_log_sha256=sha(out/'tests.log'),sdk_proof_sha256=sha(out/'SDK-PROOF.json'),network_attempts=0,gpu_work=False,local_only=True,
        inventory={str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.is_file()},
        no_launch_authority=True,remaining=['exact independent packet review/new serialized assignment','actual post-control candidate peak/load-to-health unproved','original140 and matched-capacity admission remain unqualified'])
    with (out/'PROOF.json').open('x') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
    print(json.dumps({k:proof[k] for k in ('status','seconds','bindings','source_sha256','config_sha256')}));signal.alarm(0)
if __name__=='__main__':main()
