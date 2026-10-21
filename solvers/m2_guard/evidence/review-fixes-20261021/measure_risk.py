"""Bounded local CRAP estimate: modified build.main and embedded end guard only.

Reuse existing AST/stdlib trace rules; no threshold or dependencies added.
The unchanged original 13-test universe is frozen by test IDs in risk-before.json.
Expanded tests are a separate final measurement, not a comparable improvement.
The embedded snippet is measured as one function; snippet traced lines are offset
by one for that wrapper. This does not measure GPU behavior or the supervisor.
"""
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import trace
import unittest

HERE = Path(__file__).resolve().parent
GUARD = HERE.parents[1]
ROOT = GUARD.parents[1]
spec = importlib.util.spec_from_file_location('risk_estimator', ROOT / 'solvers/m2_planning_tools/measure_risk.py')
risk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk)
sys.path.insert(0, str(GUARD / 'tests'))
import test_guard


def cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from cases(test)
        else:
            yield test


mode = sys.argv[1]
all_tests = list(cases(unittest.defaultTestLoader.loadTestsFromModule(test_guard)))
if mode == 'before' or mode == 'expanded':
    tests = all_tests
else:
    ids = json.loads((HERE / 'risk-before.json').read_text())['tests']
    tests = [test for test in all_tests if test.id() in ids]
tracer = trace.Trace(count=True, trace=False)
result = tracer.runfunc(unittest.TextTestRunner(verbosity=1).run, unittest.TestSuite(tests))
if not result.wasSuccessful():
    raise SystemExit('Risk test universe failed')
counts = tracer.results().counts
rows = risk.score(test_guard.build.__file__ and Path(test_guard.build.__file__).read_text(), counts,
                  test_guard.build.__file__, 'build.py', {'main'})
snippet = test_guard.end_snippet(test_guard.built('f1x')[1])
wrapped = 'def end_guard(bm, json):\n' + textwrap.indent(snippet, '    ')
shifted = {('end_guard', line + 1): hits for (filename, line), hits in counts.items() if filename == 'm2-guard-end'}
rows.extend(risk.score(wrapped, shifted, 'end_guard', 'embedded END'))
report = {'contract': __doc__, 'mode': mode, 'tests': [test.id() for test in tests],
          'threshold': None, 'functions': rows, 'maximum': max(row['crap'] for row in rows)}
(HERE / f'risk-{mode}.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
