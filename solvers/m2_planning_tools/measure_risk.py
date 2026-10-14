"""Local CRAP estimate; no installed CRAP tool or repository gate is available.

Utility coverage uses stdlib trace and behavioral tests. Embedded subprocess
functions conservatively have zero measured coverage, because stdlib trace does
not cross the process boundary. No threshold, exclusion, or CI gate is introduced.
Statement starts and if/for/while/except/conditional/comprehension/boolean branches
follow the earlier B1 estimate. Nested function bodies are measured separately.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import trace
import unittest

HERE = Path(__file__).resolve().parent


def nodes_without_definitions(node):
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from nodes_without_definitions(child)


def score(source, counts, filename, label, selected=None):
    rows = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, ast.FunctionDef) or (selected and function.name not in selected):
            continue
        nodes = list(nodes_without_definitions(function))
        lines = {n.lineno for n in nodes if isinstance(n, ast.stmt) and n is not function
                 and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                          and isinstance(n.value.value, str))}
        branches = sum(isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp)) for n in nodes)
        branches += sum(len(n.values)-1 for n in nodes if isinstance(n, ast.BoolOp))
        branches += sum(1+len(n.ifs) for n in nodes if isinstance(n, ast.comprehension))
        cc = 1 + branches
        covered = sum(bool(counts.get((filename, line))) for line in lines)
        coverage = covered / len(lines) if lines else 1.0
        rows.append({'source': label, 'function': function.name, 'line': function.lineno,
                     'cc': cc, 'coverage': coverage, 'crap': cc**2*(1-coverage)**3+cc})
    return rows


def measure(output, source_root):
    sys.path[:0] = [str(source_root), str(HERE / 'tests'), str(HERE.parent / 'm2_b1_fixed/tests')]
    suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'), pattern='test_bounded_search.py')
    tracer = trace.Trace(count=True, trace=False)
    result = tracer.runfunc(unittest.TextTestRunner(verbosity=0).run, suite)
    if not result.wasSuccessful():
        raise RuntimeError('Risk test universe failed')
    counts = tracer.results().counts
    rows = []
    utility = HERE / 'overlay/inference/utils/bounded_search.py'
    rows.extend(score(utility.read_text(), counts, str(utility), str(utility.relative_to(HERE))))
    # Loading the actual generated bootstrap includes every embedded utility.
    from inference.agent import python_tool_sandbox as sandbox
    rows.extend(score(sandbox._SANDBOX_BOOTSTRAP, {}, '<string>', 'embedded sandbox',
                      {'main', 'action', 'bounded_search', 'probe_transition'}))
    probe = HERE / 'overlay/inference/utils/prospective_probe.py'
    if probe.exists():
        probe_suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'), pattern='test_prospective_probe.py')
        probe_result = tracer.runfunc(unittest.TextTestRunner(verbosity=0).run, probe_suite)
        if not probe_result.wasSuccessful():
            raise RuntimeError('New-function risk tests failed')
        rows.extend(score(probe.read_text(), tracer.results().counts, str(probe), str(probe.relative_to(HERE))))
    report = {'contract': __doc__, 'existing_test_universe': 'tests/test_bounded_search.py (unchanged)',
              'new_function_tests': 'tests/test_prospective_probe.py', 'threshold': None,
              'embedded_coverage_measured': False, 'functions': rows,
              'maximum': max(r['crap'] for r in rows)}
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print('Risk estimate', output, 'maximum', report['maximum'])


if __name__ == '__main__':
    measure(sys.argv[1], Path(sys.argv[2]).resolve())
