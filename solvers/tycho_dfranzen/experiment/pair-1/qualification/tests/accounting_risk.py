"""Bounded diagnostic: AST statement coverage and explicit decision-count CRAP.

Uses stdlib tracing because Coverage.py/Radon are absent in the solver venv.
This is not comparable to the canonical solver's Radon/Coverage.py gate.
No threshold is introduced. Run baseline and final with the same test suite.
"""
import ast
import importlib
import json
from pathlib import Path
import sys
import trace
import unittest

suite_name, output = sys.argv[1:]
module = importlib.import_module(suite_name)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
tracer = trace.Trace(count=True, trace=False)
result = tracer.runfunc(unittest.TextTestRunner().run, suite)
if not result.wasSuccessful():
    raise SystemExit('Tests failed: risk measurement invalid')
root = Path(__file__).resolve().parents[1] / 'runtime/hybrid'
rows = []
for name, targets in [('solver.py', {'_play'}), ('accounting.py', {'final_usage'})]:
    path = root / name
    if not path.exists():
        continue
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in targets:
            continue
        statements = {n.lineno for body in node.body for n in ast.walk(body) if isinstance(n, ast.stmt)}
        seen = {line for (file, line), count in tracer.results().counts.items() if count and Path(file).resolve() == path.resolve()}
        coverage = len(statements & seen) / len(statements)
        cc = 1
        for n in ast.walk(node):
            if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)):
                cc += 1
            elif isinstance(n, ast.BoolOp):
                cc += len(n.values) - 1
            elif isinstance(n, ast.comprehension):
                cc += 1 + len(n.ifs)
        rows.append(dict(function=node.name, decisions=cc, covered_statements=len(statements & seen),
                         statements=len(statements), coverage=coverage, crap=cc**2*(1-coverage)**3+cc))
Path(output).write_text(json.dumps(dict(test_suite=suite_name, tests=result.testsRun, rows=rows,
    measurement='stdlib executed AST statement lines; explicit AST decision count; no threshold'), indent=2)+'\n')
