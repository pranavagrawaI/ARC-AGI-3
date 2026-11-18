"""Bounded CPU qualification. No launch, adapter, credentials or model loads.

Risk diagnostic: AST complexity counts decisions/loops/comprehensions and extra
boolean terms; statement coverage traces statement-start lines in capacity.py.
CRAP = CC**2 * (1-coverage)**3 + CC. No threshold or pre-existing baseline.
This diagnostic is not branch coverage or proof of adapter/runtime correctness.
"""
import ast
import contextlib
import json
import signal
import subprocess
import sys
import time
import unittest
import corpus
import fixtures
import capacity


def risk_rows(hits):
    path=corpus.HERE/'capacity.py';tree=ast.parse(path.read_text());rows=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.FunctionDef):continue
        nodes=list(ast.walk(node))
        cc=1+sum(isinstance(n,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp)) for n in nodes)
        cc+=sum(len(n.values)-1 for n in nodes if isinstance(n,ast.BoolOp))
        cc+=sum(1+len(n.ifs) for n in nodes if isinstance(n,ast.comprehension))
        lines={n.lineno for n in nodes if isinstance(n,ast.stmt) and not isinstance(n,ast.FunctionDef)
               and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))}
        coverage=len(lines&hits)/len(lines) if lines else 1.
        rows.append(dict(function=node.name,line=node.lineno,CC=cc,statement_coverage=coverage,
                         CRAP=cc**2*(1-coverage)**3+cc))
    return rows


def preserved_trials():
    repo=corpus.HERE.parents[2];base='analysis/serving-alternatives-20261021/';checks={}
    for trial in ('trial-1','trial-2-allocation'):
        prefix=base+trial+'/'
        files=subprocess.check_output(['git','ls-tree','-r','--name-only','a6ba01398','--',prefix],cwd=repo,text=True).splitlines()
        if not files:raise RuntimeError('missing preserved trial '+trial)
        for name in files:
            old=subprocess.check_output(['git','show','a6ba01398:'+name],cwd=repo)
            if old!=(repo/name).read_bytes():raise RuntimeError('preserved trial modified: '+name)
        checks[trial]=dict(files=len(files),equal_to_local_commit='a6ba01398')
    prior=corpus.HERE.parent/'trial-3-matched-preparation'
    inventory=json.loads((prior/'inventory.json').read_text())
    for name,row in inventory['files'].items():
        if corpus.digest((prior/name).read_bytes())!=row['sha256']:raise RuntimeError('trial-3 inventory changed: '+name)
    checks['trial-3']=dict(files=len(inventory['files']),status='DESIGN_BLOCKED inventory unchanged')
    return checks


def main():
    signal.alarm(1800);began=time.monotonic()
    names=('capacity.py','corpus.py','fixtures.py','test_capacity.py','qualify_local.py','sensitivity.py','EXPERIMENT-CARD.md')
    sources={name:corpus.digest((corpus.HERE/name).read_bytes()) for name in names}
    corpus.verify_corpus();catalog=corpus.load();preservation=preserved_trials();hits=set()
    def trace(frame,event,arg):
        if frame.f_code.co_filename!=str(corpus.HERE/'capacity.py'):return None
        if event=='line':hits.add(frame.f_lineno)
        return trace
    suite=unittest.defaultTestLoader.loadTestsFromName('test_capacity')
    with (corpus.HERE/'checks-final.log').open('w') as log:
        with contextlib.redirect_stdout(log):
            sys.settrace(trace)
            try:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
            finally:sys.settrace(None)
    if not result.wasSuccessful():raise RuntimeError('local contract tests failed; see checks-final.log')
    reader=fixtures.Reader(catalog)
    control=fixtures.build(catalog,reader,label='archived-control')
    candidate=fixtures.build(catalog,reader,label='archived-candidate',decode=1.5,origin=2000.)
    proof=corpus.HERE/'cpu-proof';proof.mkdir(exist_ok=True)
    for name,value in [('control-events.json',control),('candidate-events.json',candidate),
                       ('responses.json',{ref:json.loads(raw) for ref,raw in reader.blobs.items()}),
                       ('request-keys.json',reader.requests),('verdict.json',capacity.compare(control,candidate,catalog,reader))]:
        (proof/name).write_bytes(corpus.canonical(value))
    if sources!={name:corpus.digest((corpus.HERE/name).read_bytes()) for name in names}:
        raise RuntimeError('source changed during qualification')
    report=dict(status='LOCAL_REDUCER_QUALIFIED_ONLY; NO_GPU_AUTHORITY',elapsed_s=time.monotonic()-began,
                tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),source_hashes=sources,
                corpus_manifest_sha=corpus.MANIFEST_SHA,catalog_sha=corpus.CATALOG_SHA,
                corpus_records=len(catalog['records']),games=len(catalog['streams']),preservation=preservation,
                evidence_scope='CPU_FIXTURE_ONLY: real wire/render bytes, synthetic clocks/completions; no engine',
                cap=dict(local_seconds=1800,GPU_seconds=0,paid_spend=0,launches=0,slot_requests=0),
                risk_contract=__doc__,risk=risk_rows(hits),
                cpu_proof_sha={p.name:corpus.digest(p.read_bytes()) for p in sorted(proof.iterdir())})
    (corpus.HERE/'QUALIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','tests','failures','errors','elapsed_s','preservation')}))


if __name__=='__main__':main()
