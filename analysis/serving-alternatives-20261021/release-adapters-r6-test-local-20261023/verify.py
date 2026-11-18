"""Guarded local integration proof, fresh output and600s invocation cap."""
import argparse
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import unittest

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior():
    accepted=json.loads((HERE/'ACCEPTED-CORE.json').read_text());root=Path(accepted['directory'])
    for name,expected in accepted['all_prior_sources'].items():assert sha(root/name)==expected,name
    assert sha(accepted['review_path'])==accepted['review_sha256']
    trial=HERE.parent/'trial-10-exact-wire-local'
    freeze=json.loads((trial/'freeze.json').read_text())
    for name,expected in freeze['files'].items():assert sha(trial/name)==expected,name
    previous=json.loads((HERE/'PRIOR-SNAPSHOT.json').read_text())
    for name,expected in previous['sources'].items():assert sha(Path(previous['old_directory'])/name)==expected,name
    for path,expected in previous['evidence'].items():assert sha(path)==expected,path
    old_delta=json.loads((Path(previous['old_directory'])/'LOCAL-DELTA.json').read_text())
    for name,expected in old_delta['documents'].items():assert sha(Path(previous['old_directory'])/name)==expected,name
    for path,expected in old_delta['references'].items():assert sha(path)==expected,path
    r2=json.loads((HERE/'PRIOR-R2-SNAPSHOT.json').read_text())
    for name,expected in r2['sources'].items():assert sha(Path(r2['old_directory'])/name)==expected,name
    for path,expected in r2['evidence'].items():assert sha(path)==expected,path
    ownership=json.loads((HERE/'OWNERSHIP-SNAPSHOT.json').read_text())
    for name,expected in ownership['sources'].items():assert sha(Path(ownership['directory'])/name)==expected,name
    assert sha(ownership['proof'])==ownership['proof_sha256']
    r4=json.loads((HERE/'PRIOR-R4-SNAPSHOT.json').read_text())
    for name,expected in r4['sources'].items():assert sha(Path(r4['directory'])/name)==expected,name
    for name,expected in r4['documents'].items():assert sha(Path(r4['directory'])/name)==expected,name
    for path,expected in r4['evidence'].items():assert sha(path)==expected,path
    r5=json.loads((HERE/'PRIOR-R5-SNAPSHOT.json').read_text())
    for name,expected in r5['sources'].items():assert sha(Path(r5['directory'])/name)==expected,name
    for name,expected in r5['documents'].items():assert sha(Path(r5['directory'])/name)==expected,name
    for path,expected in r5['evidence'].items():assert sha(path)==expected,path
    return len(freeze['files'])


def refs(value):
    if isinstance(value,dict):
        if {'path','sha256','bytes','complete'}<=value.keys():yield value
        else:
            for child in value.values():yield from refs(child)
    elif isinstance(value,list):
        for child in value:yield from refs(child)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False);signal.alarm(600)
    os.environ['ADAPTER_PROOF_DIR']=str(out/'cases');os.environ['PYTHONPATH']=str(HERE/'guard');os.environ['LOCAL_ADAPTER_PROOF']='1'
    sys.path.insert(0,str(HERE/'guard'));import sitecustomize
    started=time.monotonic();count=prior()
    source_before={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py') if '__pycache__' not in str(p)}
    for path in HERE.rglob('*.py'):compile(path.read_text(),str(path),'exec')
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    from core import ROOT,shutdown
    os.environ['SHUTDOWN_PROOF_DIR']=str(out/'core-cases')
    # Execute the unchanged accepted regression suite against the new narrow seam.
    for name in ('test_shutdown','test_resources','test_ipc'):
        spec=importlib.util.spec_from_file_location('accepted_'+name,ROOT/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if hasattr(module,'shutdown'):module.shutdown=shutdown
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'tests.log').write_text(stream.getvalue());assert result.wasSuccessful(),stream.getvalue()
    import artifacts
    checked=0
    for case in (out/'cases').iterdir():
        values=json.loads((case/'RESULT.json').read_text())
        for ref in refs(values):
            artifacts.verify(case/'raw',ref,time.monotonic()+2);checked+=1
    prior()
    files={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py') if '__pycache__' not in str(p)}
    assert files==source_before,'local source changed during qualification'
    local_modules=('adapters','artifacts','capture','command_child','no_processes','registration','probe_fixture','fixture_wrapper','core','controller_shutdown','runtime_scope')
    for name in local_modules:
        module=sys.modules.get(name)
        if module is not None:assert Path(module.__file__).resolve()==HERE/(name+'.py'),name
    inventory={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    import no_processes
    proof=dict(no_descendant_contract=dict(name='NO_NEW_PROCESSES_THREADS_ALLOWED',arch='x86_64',filter_sha256=no_processes.SHA),
        preserved_adapter_snapshot_sha256=sha(HERE/'PRIOR-SNAPSHOT.json'),preserved_r2_sha256=sha(HERE/'PRIOR-R2-SNAPSHOT.json'),ownership_snapshot_sha256=sha(HERE/'OWNERSHIP-SNAPSHOT.json'),preserved_r4_sha256=sha(HERE/'PRIOR-R4-SNAPSHOT.json'),status='LOCAL_ADAPTER_PLUMBING_ONLY_NOT_GPU_RELEASE',tests=result.testsRun,seconds=time.monotonic()-started,
        sources=files,case_inventory=inventory,artifact_references_checked=checked,prior_core_and_sources_exact=True,
        trial10_inputs_exact=count,invocation_cap_s=600,gpu_queries=0,external_network=0,loopback_bind_only=True,
        intended_commands_sha256=sha(HERE/'INTENDED-COMMANDS.json'),python=sys.version,python_realpath=str(Path(sys.executable).resolve()),python_sha256=sha(sys.executable),
        source_hashes_exact_before_after=True,accepted_27_regressions_run_on_controller_core=True,
        limits=['GPU execution branch source-only; not exercised/authorized locally','native text has no direct watchdog-stop/context-destruction proof',
                'local disk may fail original100GiB gate','seccomp Linux x86-64 contract denies all new processes; actual GPU query compatibility unproved',
                'intended-cgroup/observer/mount and fresh-stderr bindings exercised with actual CPU frontend; real Kaggle/native integration unproved',
                'kernel syscall/scheduling bounds cooperative','140GiB independent and unresolved'])
    (out/'PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(tests=result.testsRun,seconds=proof['seconds'],artifact_refs=checked,proof_sha256=sha(out/'PROOF.json'))))
    signal.alarm(0)


if __name__=='__main__':main()
