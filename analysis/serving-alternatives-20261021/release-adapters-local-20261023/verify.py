"""Guarded local integration proof, fresh output and600s invocation cap."""
import argparse
import hashlib
import io
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
    for path in HERE.rglob('*.py'):compile(path.read_text(),str(path),'exec')
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
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
    inventory={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    proof=dict(status='LOCAL_ADAPTER_PLUMBING_ONLY_NOT_GPU_RELEASE',tests=result.testsRun,seconds=time.monotonic()-started,
        sources=files,case_inventory=inventory,artifact_references_checked=checked,prior_core_and_sources_exact=True,
        trial10_inputs_exact=count,invocation_cap_s=600,gpu_queries=0,external_network=0,loopback_bind_only=True,
        intended_commands_sha256=sha(HERE/'INTENDED-COMMANDS.json'),python=sys.version,
        limits=['GPU execution branch source-only; not exercised/authorized locally','native text has no direct watchdog-stop/context-destruction proof',
                'local disk may fail original100GiB gate','query child PDEATHSIG covers direct exec, not arbitrary detached descendants',
                'kernel syscall/scheduling bounds cooperative','140GiB independent and unresolved'])
    (out/'PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(tests=result.testsRun,seconds=proof['seconds'],artifact_refs=checked,proof_sha256=sha(out/'PROOF.json'))))
    signal.alarm(0)


if __name__=='__main__':main()
