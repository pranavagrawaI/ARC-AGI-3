"""Reproducible no-network/no-paid verification. Creates a fresh evidence directory."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import audit_remote
import package_execution
import probe
import verify_package

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    summary={'scope':'OFFLINE_ONLY_NO_PAID_REQUESTS','checks':{}}
    def execute(name,command,*,env=None,expect=0,timeout=240):
        p=subprocess.run(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True,timeout=timeout)
        (output/(name+'.stdout')).write_text(p.stdout)
        (output/(name+'.stderr')).write_text(p.stderr)
        check={'returncode':p.returncode,'expected_returncode':expect,'passed':p.returncode==expect,'command':command}
        count=re.search(r'Ran (\d+) tests',p.stderr)
        if count: check['tests_run']=int(count.group(1))
        skipped=re.search(r'skipped=(\d+)',p.stderr)
        if skipped: check['skipped']=int(skipped.group(1))
        summary['checks'][name]=check
        if not check['passed']: raise RuntimeError(name+' failed: '+p.stderr[-2000:])
        return p
    try:
        summary['package']=verify_package.verify()
        (output/'PACKAGE-VERIFICATION.json').write_text(json.dumps(summary['package'],indent=2)+'\n')
        # Execute real notebook first so all audit tests consume this revision.
        summary['package_fixtures']={}
        for variant in ('success','cached_modules','loader_missing','malformed'):
            summary['package_fixtures'][variant]=package_execution.run(output/('package-'+variant),variant)
        env={**os.environ,'AUDIT_PACKAGE_EVIDENCE':str(output/'package-success'),'PYTHONDONTWRITEBYTECODE':'1'}
        execute('mapped-root-tests',package_execution.MAPPING+[sys.executable,'-B','-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],env=env)
        execute('unprivileged-tests',[sys.executable,'-B','-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],env=env)
        execute('local-real-probe',package_execution.MAPPING+[sys.executable,'-B',str(HERE/'probe.py'),str(output/'local-real'),'--local'])
        sources=json.loads((verify_package.PACKAGE/'BUILD.json').read_text())['assets']
        summary['local_real_audit']=audit_remote.audit_directory(output/'local-real',identity=HERE.name,sources=sources,local_fixture=True)
        assert summary['local_real_audit']['passed']==16
        execute('baseline-reproductions',package_execution.MAPPING+[sys.executable,'-B',str(HERE/'reproduce_baseline.py'),str(output/'BASELINE-REPRODUCTIONS.json')])
        old=HERE.parent/'kaggle-containment-qualification-20260913T160618Z'
        execute('baseline-mapped-tests',package_execution.MAPPING+[sys.executable,'-B','-m','unittest','discover','-s',str(old),'-p','test_qualification.py','-v'])
        for stamp in ('20260913T150421Z','20260913T160618Z'):
            execute('frozen-'+stamp,[sys.executable,'-B',str(HERE.parent/('kaggle-containment-qualification-'+stamp)/'verify_package.py')])
        constructor=HERE.parent/'model-construction-kaggle-v1-20260913'
        execute('constructor-tests',[sys.executable,'-B','-m','unittest','discover','-s',str(constructor/'tests'),'-v'],env={**os.environ,'PYTHONPATH':str(constructor),'PYTHONDONTWRITEBYTECODE':'1'})
        guard=execute('constructor-launch-guard',[sys.executable,'-B',str(constructor/'run.py'),'--execute'],expect=1)
        assert 'Kaggle isolation is unqualified' in guard.stderr
        assert verify_package.verify()==summary['package'], 'SOURCE_CHANGED_DURING_VERIFICATION'
        summary['verified']=True
    except Exception as error:
        summary['verified']=False; summary['error']=type(error).__name__+':'+str(error)
        raise
    finally:
        summary['source_sha256']={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}
        (output/'VERIFICATION.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'verified':summary['verified'],'evidence':str(output),'tests':{n:c.get('tests_run') for n,c in summary['checks'].items() if 'tests_run' in c},'local_cases':summary['local_real_audit'],'package_fixtures':summary['package_fixtures']},indent=2))

if __name__=='__main__': main(Path(sys.argv[1]).resolve())
