"""Scoped CC/statement coverage/CRAP report using stdlib AST and trace. No threshold.

One complexity point per function, if/loop/except/conditional/comprehension/filter,
and extra boolean term. Coverage is traced executable statement starts within the
function, excluding its docstring and nested definitions. Parent coverage includes
nested statements because they are part of that responsibility. Same test universe
is used for before/final; failing contract tests are reported, never called a pass.
"""
import ast
import contextlib
import importlib.util
import json
import sys
import trace
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCOPE = {'analyze.py': ['_summary', 'decide'], 'sweep.py': ['plan_for_time', 'run_launch'],
         'build_notebook.py': ['build_notebook'], 'dryrun_notebook.py': ['main'],
         'launch_watch.py': ['quota_used_s', 'identity', 'priority_resolved', 'main'],
         'verify_prelaunch.py': ['verify']}


def main():
    source = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(HERE))
    for name in ('analyze', 'sweep', 'build_notebook', 'dryrun_notebook', 'launch_watch'):
        path = source / f'{name}.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    tracer = trace.Trace(count=True, trace=False)
    suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'), pattern='test_*.py')
    with contextlib.redirect_stdout(sys.stderr):
        result = tracer.runfunc(unittest.TextTestRunner(stream=sys.stderr).run, suite)
    counts = tracer.results().counts
    functions = []
    for filename, names in SCOPE.items():
        path = source / filename
        if not path.exists():
            continue
        for fn in ast.parse(path.read_text()).body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name not in names:
                continue
            nodes = list(ast.walk(fn))
            cc = 1 + sum(isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp)) for n in nodes)
            cc += sum(len(n.values) - 1 for n in nodes if isinstance(n, ast.BoolOp))
            cc += sum(1 + len(n.ifs) for n in nodes if isinstance(n, ast.comprehension))
            lines = {n.lineno for n in nodes if isinstance(n, ast.stmt)
                     and not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                              and isinstance(n.value.value, str))}
            hit = {line for line in lines if counts.get((str(path), line), 0)}
            cov = len(hit) / len(lines) if lines else 1
            functions.append({'function': f'{filename}:{fn.name}', 'CC': cc,
                              'statement_coverage': cov, 'CRAP': round(cc**2*(1-cov)**3+cc, 3)})
    print(json.dumps({'source': str(source), 'contract': __doc__, 'threshold': None,
                      'tests': result.testsRun, 'failures': len(result.failures),
                      'errors': len(result.errors), 'functions': functions}, indent=2))


if __name__ == '__main__':
    main()
