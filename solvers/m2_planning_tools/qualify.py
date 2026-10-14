"""Local tests and real native continuation fixtures. No provider calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from build import prepare

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ANALYSIS=ROOT/'analysis/planning-transfer-20261014'


def run(command,log,env=None):
    with log.open('w') as output:
        subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=env,check=True,timeout=120)


def fixture_code(variant):
    code="result={'variant':"+repr(variant)+",'complete':True}\n"
    if variant in ('planner','both'):
        code+='''
def successors(state):
    if state<3: yield 'advance',state+1
def key(state): return state
def goal(state): return state==3
plan=bounded_search(0,successors,state_key=key,is_goal=goal,max_depth=4)
assert plan['status']=='FOUND' and plan['depth']==3
result['search_status']=plan['status']
'''
    if variant in ('probing','both'):
        code+='''
def observe(frame): return {'grid':[list(row) for row in frame.grid]}
def predict(before,action): return before
def grid_key(state): return tuple(tuple(row) for row in state['grid'])
probe=probe_transition({'action':'MOUSE','row':57,'col':36},observe=observe,predict=predict,state_key=grid_key,name='fixture_probe')
assert probe['status'] in ('MATCH','MISMATCH','TERMINAL','REFUSED')
result['probe_status']=probe['status']
'''
    return code+"save_data('fixture_result',result)\nprint('NATIVE_FIXTURE_OK')\n"


def main():
    if sys.version_info[:2]!=(3,12):
        raise SystemExit('Use solvers/tycho_nooa/.venv/bin/python (CPython 3.12 and existing harness dependencies)')
    checks=[]
    with tempfile.TemporaryDirectory(prefix='planning_qualification_') as directory:
        temporary=Path(directory);source=prepare(temporary/'src')
        env={**os.environ,'M2_SOURCE_ROOT':str(source),'PYTHONDONTWRITEBYTECODE':'1'}
        for name,folder in [('tests-312',HERE/'tests'),('inherited-tests-312',HERE.parent/'m2_b1_fixed/tests'),('audit-tests-312',ANALYSIS)]:
            run([sys.executable,'-B','-m','unittest','discover','-s',str(folder),'-v'],HERE/'evidence'/(name+'.log'),env)
            checks.append(name)
            print(name,'PASS',flush=True)
        run([sys.executable,'-B',str(HERE/'tests/gate_package.py')],HERE/'evidence/package-gate.log',
            {**env,'M2_SEARCH_SOURCE':str(source),'ARC3_PLANNING_VARIANT':'planner'})
        checks.append('package-gate')
        print('package-gate PASS',flush=True)
        fixtures=[]
        for checkpoint in ('b1','control'):
            for variant in ('control','planner','probing','both'):
                code=temporary/(checkpoint+'-'+variant+'.py');code.write_text(fixture_code(variant))
                output=temporary/(checkpoint+'-'+variant)
                log=HERE/'evidence'/('native-'+checkpoint+'-'+variant+'.log')
                run([sys.executable,'-B',str(ANALYSIS/'continuation.py'),'--checkpoint',checkpoint,'--variant',variant,
                     '--mode','fixture','--fixture',str(code),'--output',str(output)],log)
                result=json.loads((output/'result.json').read_text())
                saved=result['saved_data'].get('fixture_result',{}).get('value',{})
                if result['status']!='fixture_complete' or not saved.get('complete') or saved.get('variant')!=variant:
                    failure=ANALYSIS/'evidence'/('failed-'+checkpoint+'-'+variant)
                    import shutil
                    if not failure.exists():
                        shutil.copytree(output,failure)
                    raise RuntimeError('Native fixture failed: '+checkpoint+' '+variant)
                if not result['qualified_restoration'] or not result['protected_inputs_unchanged'] or result['new_model_calls']!=0:
                    raise RuntimeError('Fixture restoration/source/call identity failed')
                intervention=json.loads((output/'intervention.json').read_text())
                checkpoint_receipt=json.loads((output/'checkpoint.json').read_text())
                fixtures.append({'checkpoint':checkpoint,'variant':variant,'status':result['status'],
                                 'historical_requests_verified':len(checkpoint_receipt['comparisons']),
                                 'operations':saved,'all_other_messages_identical':intervention['all_other_messages_identical'],
                                 'initial_state_restored':checkpoint_receipt['actions_match'],'model_calls':0})
                print('native',checkpoint,variant,'PASS',flush=True)
        from measure_risk import measure
        measure(HERE/'evidence/risk-after.json',source)
    (HERE/'evidence/native-fixtures.json').write_text(json.dumps(fixtures,indent=2)+'\n')
    provenance=json.loads((HERE/'provenance.json').read_text())
    unchanged=all(hashlib.sha256((HERE.parent/'m2_b1_fixed/src'/relative).read_bytes()).hexdigest()==expected
                  for relative,expected in provenance['parent_files'].items())
    if not unchanged:
        raise RuntimeError('Parent changed')
    # The caller may redirect these streams into an evidence file inventoried
    # below. Flush it before hashing so the final buffered risk line is present.
    sys.stdout.flush()
    sys.stderr.flush()
    verification={'checks':checks,'native_fixtures':len(fixtures),'parent_unchanged':True,
                  'model_calls':0,'gpu_launches':0,'python':sys.version,
                  'files':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p)
                           and p!=HERE/'evidence/verification.json'}}
    (HERE/'evidence/verification.json').write_text(json.dumps(verification,indent=2)+'\n')


if __name__=='__main__':main()
