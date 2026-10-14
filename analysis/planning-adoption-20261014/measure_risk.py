"""Same statement-line CRAP rules as the parent; no new threshold or CI gate.

The parent artifact supplies the baseline for existing functions. New boundary
coverage is traced with the fixed test_boundary suite. Process-only functions
have conservative zero measured coverage, not an asserted absence of tests.
"""
import json
import importlib.util
import os
from pathlib import Path
import sys
import trace
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'solvers/m2_planning_tools'))
from measure_risk import score


def main():
    parent=json.loads((ROOT/'analysis/planning-transfer-20261014/evidence/risk-final.json').read_text())
    selected={'post','identity_files','run','main','native_level_count','audit_episode','guidance'}
    baseline={**parent,'functions':[row for row in parent['functions'] if row['function'] in selected],
              'scope':'Copied existing runtime functions; parent fixed five-test analysis suite',
              'parent_commit':'6b1db1e60'}
    (HERE/'evidence/risk-before.json').write_text(json.dumps(baseline,indent=2)+'\n')
    package=HERE/'package/analysis/planning-transfer-20261014'
    os.environ['PLANNING_NATIVE_AUDIT_ROOT']=str(ROOT)
    sys.path.insert(0,str(package))
    for name in ['audit','continuation']:
        spec=importlib.util.spec_from_file_location(name,package/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    tracer=trace.Trace(count=True,trace=False)
    inherited=unittest.defaultTestLoader.discover(str(ROOT/'analysis/planning-transfer-20261014'),pattern='test_*.py')
    result=tracer.runfunc(unittest.TextTestRunner(verbosity=1).run,inherited)
    if not result.wasSuccessful():raise RuntimeError('Inherited fixed risk test universe failed')
    boundary=unittest.defaultTestLoader.discover(str(HERE),pattern='test_boundary.py')
    result=tracer.runfunc(unittest.TextTestRunner(verbosity=1).run,boundary)
    if not result.wasSuccessful():raise RuntimeError('Boundary risk tests failed')
    rows=[]
    counts=tracer.results().counts
    for path in list((HERE/'runtime').glob('*.py'))+[HERE/name for name in ['build.py','qualify.py','launch.py']]:
        measured=(package/path.name) if path.parent==HERE/'runtime' and path.name!='variants.py' else path
        rows.extend(score(path.read_text(),counts,str(measured),str(path.relative_to(HERE))))
    (HERE/'evidence/risk-after.json').write_text(json.dumps({'contract':__doc__,'threshold':None,'functions':rows,
        'maximum':max(row['crap'] for row in rows),'comparison':'Copied existing runtime uses the same five-test parent universe; added boundary tests cover new functions. Process-only coverage remains conservatively unmeasured.'},indent=2)+'\n')
    print('Conservative maximum',max(row['crap'] for row in rows))
    # Separate scope: new collection/reporting code, not a changed baseline.
    analysis_suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_analysis.py')
    result=tracer.runfunc(unittest.TextTestRunner(verbosity=1).run,analysis_suite)
    if not result.wasSuccessful():raise RuntimeError('New reporting tests failed')
    reporting=[]
    for name in ['analyze.py','collect.py','status.py']:
        path=HERE/name
        reporting.extend(score(path.read_text(),tracer.results().counts,str(path),name))
    (HERE/'evidence/risk-reporting.json').write_text(json.dumps({'threshold':None,'functions':reporting,
        'maximum':max(row['crap'] for row in reporting),
        'scope':'Separate final scores for new reporting/collection functions; no comparison with the runtime baseline.',
        'tests':'test_analysis.py; subprocess/network-only coverage conservatively unmeasured'},indent=2)+'\n')


if __name__=='__main__':main()
