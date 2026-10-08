"""Report scoped CC/statement coverage/CRAP with the existing stdlib AST contract.
No threshold or quality-improvement claim. Tracing covers the main test thread;
worker-thread line coverage is not represented. GPU-only paths remain unqualified.
"""
import ast
import contextlib
import json
import sys
import trace
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCOPE=['probe.py','build.py','launch.py','audit.py','watch.py','package_contract.py']


def main():
    sys.path[:0]=[str(HERE),str(HERE.parent/'mtp-depth-20261006')]
    tracer=trace.Trace(count=True,trace=False)
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    with contextlib.redirect_stdout(sys.stderr):
        result=tracer.runfunc(unittest.TextTestRunner(stream=sys.stderr).run,suite)
    counts=tracer.results().counts; functions=[]
    for filename in SCOPE:
        path=HERE/filename
        for fn in ast.parse(path.read_text()).body:
            if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            nodes=list(ast.walk(fn))
            cc=1+sum(isinstance(n,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp)) for n in nodes)
            cc+=sum(len(n.values)-1 for n in nodes if isinstance(n,ast.BoolOp))
            cc+=sum(1+len(n.ifs) for n in nodes if isinstance(n,ast.comprehension))
            lines={n.lineno for n in nodes if isinstance(n,ast.stmt) and not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
                   and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))}
            cov=sum(bool(counts.get((str(path),line),0)) for line in lines)/len(lines) if lines else 1
            functions.append(dict(function=f'{filename}:{fn.name}',CC=cc,statement_coverage=cov,CRAP=round(cc**2*(1-cov)**3+cc,3)))
    report=dict(contract=__doc__,source_scope=SCOPE,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                threshold=None,functions=functions,
                baseline='evidence/risk-before.json preserves the 26-test v2-copy baseline. Final scope adds package verification and package regressions; no comparable score reduction claimed.')
    print(json.dumps(report,indent=2));return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
