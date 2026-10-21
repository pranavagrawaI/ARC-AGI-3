"""Bounded statement-line CRAP estimate; reuse existing AST/stdlib trace rules.

No threshold, exclusion or dependency is added. Before/after main is measured
with the identical test_requires.py universe. Expanded coverage is separately
reported using test_requires.py plus test_quarantine.py and is not comparable
to the existing-suite baseline. This is a local estimate, not a repository gate.
"""
import importlib.util
import json
from pathlib import Path
import sys
import trace
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'submit_draw.py'
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('existing_risk', ROOT / 'solvers/m2_planning_tools/measure_risk.py')
risk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk)

mode = sys.argv[1]
suite = unittest.defaultTestLoader.discover(str(HERE.parent), pattern='test_requires.py')
if mode == 'expanded':
    suite.addTests(unittest.defaultTestLoader.discover(str(HERE.parent), pattern='test_quarantine.py'))
tracer = trace.Trace(count=True, trace=False)
result = tracer.runfunc(unittest.TextTestRunner(verbosity=1).run, suite)
if not result.wasSuccessful():
    raise SystemExit('Risk test universe failed')
rows = risk.score(SOURCE.read_text(), tracer.results().counts, str(SOURCE), 'submit_draw.py')
report = {'contract': __doc__, 'mode': mode, 'threshold': None, 'functions': rows,
          'maximum': max(row['crap'] for row in rows)}
(HERE / f'risk-{mode}.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
