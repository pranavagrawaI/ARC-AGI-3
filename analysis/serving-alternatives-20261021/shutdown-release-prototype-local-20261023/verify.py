"""Run the local CPU proof under a 600s invocation alarm; fresh output only."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import trace
import unittest

HERE=Path(__file__).resolve().parent
TRIAL=HERE.parent/'trial-10-exact-wire-local'


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):value.update(data)
    return value.hexdigest()


def original_bindings():
    assert sha(TRIAL/'freeze.json')=='19fdb8ed412bdddcc81547d58d7dd97c52c80878395e9d31a960713134911fd0'
    freeze=json.loads((TRIAL/'freeze.json').read_text())
    for name,expected in freeze['files'].items():
        assert sha(TRIAL/name)==expected,name
    pin=HERE.parent/'trial10-release-local-20261023'
    old=json.loads((pin/'run2/PROOF.json').read_text())
    for name in ['sources/pinned-sglang-srt-utils-watchdog.py','sources/pinned-sglang-srt-managers-tokenizer_manager.py']:
        assert sha(pin/name)==old['source_sha256'][name]
    return len(freeze['files'])


def risk_metrics(counts):
    """Explicit small local metric, not an existing repo gate or standard radon CC.

    CC proxy: 1 + if/for/while/except/ifexp + extra boolean operands, including
    nested nodes. Coverage is stdlib trace executable lines, parent process only.
    Fork-worker execution is not credited. New functions have no before baseline.
    """
    values={}
    for name in ['shutdown.py','resources.py']:
        path=HERE/name
        executable=trace._find_executable_linenos(str(path))
        for fn in ast.parse(path.read_text()).body:
            if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            nodes=list(ast.walk(fn))
            cc=1+sum(isinstance(n,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp)) for n in nodes)+sum(len(n.values)-1 for n in nodes if isinstance(n,ast.BoolOp))
            lines={n for n in executable if isinstance(n,int) and fn.lineno<=n<=fn.end_lineno}
            covered={n for n in lines if counts.get((str(path),n),0)>0}
            fraction=len(covered)/len(lines) if lines else 0
            values[name+':'+fn.name]=dict(cc_proxy=cc,covered_parent_lines=len(covered),executable_lines=len(lines),
                                        crap_proxy=cc*cc*(1-fraction)**3+cc)
    return values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False)
    signal.alarm(600)
    os.environ['SHUTDOWN_PROOF_DIR']=str(out/'cases')
    os.environ['PYTHONPATH']=str(HERE/'guard')
    sys.path.insert(0,str(HERE/'guard'));import sitecustomize
    started=time.monotonic()
    original=original_bindings()
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    tracer=trace.Trace(count=True,trace=False)
    result=tracer.runfunc(unittest.TextTestRunner(stream=stream,verbosity=2).run,suite)
    (out/'tests.log').write_text(stream.getvalue())
    assert result.wasSuccessful(),stream.getvalue()
    original_bindings()
    cases={}
    for path in sorted((out/'cases').glob('*.json')):
        case=json.loads(path.read_text());assert case['all_fixture_processes_gone']
        assert all(r['guard_active'] for r in case['native_events'] if r['event']=='frontend_ready')
        cases[path.name]=sha(path)
    sources={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py') if '__pycache__' not in str(p)}
    proof=dict(status='LOCAL_CPU_SOURCE_PATH_ONLY_NOT_GPU_QUALIFICATION',tests=result.testsRun,
               elapsed_seconds=time.monotonic()-started,invocation_cap_s=600,
               gpu_calls=0,network_calls=0,python_audit_guard_active=True,
               sources_sha256=sources,cases_sha256=cases,original_trial10_bindings_exact=original,
               python=sys.version,uname=list(os.uname()),risk_metrics=risk_metrics(tracer.results().counts),
               limits=['native GPU collaborators are CPU fixtures','async five-second native polls time-compressed100x',
                       'PID exec/visibility checks cooperative; pidfds prevent numeric PID reuse targeting',
                       'no actual port or GPU query; resource receipts saved/synthetic',
                       'kernel syscall stalls not a proven hard subsecond deadline','140GiB host blocker unresolved'])
    (out/'PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(tests=result.testsRun,elapsed_seconds=proof['elapsed_seconds'],proof_sha256=sha(out/'PROOF.json'),output=str(out))))
    signal.alarm(0)


if __name__=='__main__':main()
