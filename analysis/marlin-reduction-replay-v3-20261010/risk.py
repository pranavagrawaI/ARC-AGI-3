"""Scoped stdlib AST CC/statement coverage/CRAP; no threshold or improvement claim.
Main-thread CPU behavioral tests only. Unexecuted GPU paths remain unqualified.
All task modules are new: no comparable pre-existing task-function baseline.
"""
import ast
import contextlib
import json
import sys
import trace
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCOPE=['prepare.py','build_notebook.py','marlin_capture.py','reference.py','benchmark.py','audit_capture.py']


def main():
    sys.path.insert(0,str(HERE));tracer=trace.Trace(count=True,trace=False)
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    with contextlib.redirect_stdout(sys.stderr):result=tracer.runfunc(unittest.TextTestRunner(stream=sys.stderr).run,suite)
    counts=tracer.results().counts;functions=[]
    for name in SCOPE:
        path=HERE/name
        for fn in ast.walk(ast.parse(path.read_text())):
            if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            nodes=list(ast.walk(fn))
            cc=1+sum(isinstance(n,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp)) for n in nodes)
            cc+=sum(len(n.values)-1 for n in nodes if isinstance(n,ast.BoolOp))
            cc+=sum(1+len(n.ifs) for n in nodes if isinstance(n,ast.comprehension))
            lines={n.lineno for n in nodes if isinstance(n,ast.stmt) and not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))}
            coverage=sum(bool(counts.get((str(path),line),0)) for line in lines)/len(lines) if lines else 1
            functions.append({'function':f'{name}:{fn.name}:{fn.lineno}','CC':cc,'statement_coverage':coverage,'CRAP':round(cc**2*(1-coverage)**3+cc,3)})
    print(json.dumps({'contract':__doc__,'threshold':None,'scope':SCOPE,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'functions':functions},indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
