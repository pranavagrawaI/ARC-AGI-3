"""Lean treatments: same bounded diagnostic as lifecycle_risk.py (AST statement coverage, decision-count CRAP).

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
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)) for name in suite_name.split(','))
tracer = trace.Trace(count=True, trace=False)
result = tracer.runfunc(unittest.TextTestRunner().run, suite)
if not result.wasSuccessful():
    raise SystemExit('Tests failed: risk measurement invalid')
root = Path(__file__).resolve().parents[1] / 'runtime/hybrid'
rows = []
for name, targets in [('lean.py', {'lean','treatment_policies','guidance','closed_fields','prune_closed_decisions','prune_for_decision','capped'}),
                      ('agent.py', {'__init__','_resolve_system_prompt'}), ('policy.py', {'system_guidance','attach'}),
                      ('../../experiment/freeze_wave.py', {'arm_from_port','slug','verify_hybrid_arm','main'}),
                      ('../../experiment/wave_run.py', {'verify_frozen','collect_arm'})]:
    path = (root / name).resolve()
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
    measurement='In-process stdlib AST statement coverage only; real worker subprocess execution not counted. Explicit AST decisions; no threshold. New strategy function scores, not comparable to canonical Radon gate.'), indent=2)+'\n')
