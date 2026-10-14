"""Final estimates for this task's new orchestration functions.

Same CC/statement rules as the helper report. In-process coverage is measured by
the complete five-test analysis suite; subprocess-only bodies have conservative
zero measured coverage. No threshold or gate is added. These new functions have
no before-score; existing helper comparison is recorded separately.
"""
import json
from pathlib import Path
import sys
import trace
import unittest

HERE=Path(__file__).resolve().parent
TOOLS=HERE.parents[1]/'solvers/m2_planning_tools'
sys.path.insert(0,str(TOOLS))
from measure_risk import score


def main():
    tracer=trace.Trace(count=True,trace=False)
    tests=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    result=tracer.runfunc(unittest.TextTestRunner(verbosity=1).run,tests)
    if not result.wasSuccessful():raise RuntimeError('Risk test universe failed')
    counts=tracer.results().counts
    rows=[]
    for path in [HERE/name for name in ['audit.py','continuation.py','run_matrix.py','qualify_models.py','qualify_probes.py']]+[TOOLS/name for name in ['build.py','variants.py','qualify.py']]:
        rows.extend(score(path.read_text(),counts,str(path),str(path.relative_to(HERE.parents[1]))))
    report={'contract':__doc__,'threshold':None,'function_count':len(rows),'functions':rows,
            'maximum':max(row['crap'] for row in rows)}
    (HERE/'evidence/risk-final.json').write_text(json.dumps(report,indent=2)+'\n')
    print('New-function conservative maximum',report['maximum'])


if __name__=='__main__':main()
