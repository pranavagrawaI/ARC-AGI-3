#!/usr/bin/env python3
import base64,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];PACKAGE=REPO/'deployment/kaggle'/HERE.name;E=HERE/'evidence'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def count(name):
 text=(E/f'{name}.stderr').read_text();match=re.search(r'Ran (\d+) tests?',text)
 if not match or 'OK' not in text:raise RuntimeError(name+' did not pass')
 return int(match.group(1))
def main():
 checks={}
 checks['tests']={'constructor':count('constructor46'),'containment':count('containment53'),'mapped_root':count('mapped-root-suite')}
 if checks['tests']!={'constructor':46,'containment':53,'mapped_root':21}:raise RuntimeError('test count drift')
 notebooks={}
 for mode in ('normal','cache-poison','local-http'):
  result=json.loads((E/f'notebook-{mode}-fs/working/official-launcher-result.json').read_text())
  construction=result['construction_result']
  notebooks[mode]={'status':result['status'],'candidate_classification':result['candidate_classification'],'gameplay_result':result['gameplay_result'],'model_requests':construction['model_requests'],'inspections':construction['inspections'],'suffix_attempts':construction['suffix_attempts'],'solver_requests':result.get('solver_sessions',{}).get('total_actor_calls')}
  if notebooks[mode] != {'status':'passed','candidate_classification':'INTEGRATION_VALID_CANDIDATE_SUPPORTED','gameplay_result':None,'model_requests':3,'inspections':2,'suffix_attempts':1,'solver_requests':3 if mode=='local-http' else None}:raise RuntimeError('notebook result drift:'+mode)
 checks['notebooks']=notebooks
 local=E/'notebook-local-http-fs/working/fixture-workspaces/construction-evidence'
 checks['local_http_receipts']={name:len(list((local/name).glob('*'))) for name in ('local-wire','http-receipts','raw-http')}
 if checks['local_http_receipts']!={'local-wire':6,'http-receipts':3,'raw-http':3}:raise RuntimeError('local HTTP receipt drift')
 pre=json.loads((E/'PRELAUNCH.json').read_text());checks['prelaunch']=pre['passed'];checks['launch_guard_exit']=int((E/'launch-guard.exit').read_text())
 if not pre['passed'] or checks['launch_guard_exit']!=1 or (PACKAGE/'AUTHORIZATION.json').exists() or 'kaggle kernels push' in (PACKAGE/'launch_once.py').read_text():raise RuntimeError('launch guard open')
 build=json.loads((PACKAGE/'BUILD.json').read_text())
 for name,want in build['assets'].items():
  path=PACKAGE/name if name=='remote_entry.py' else HERE/name
  if digest(path)!=want:raise RuntimeError('build asset drift:'+name)
 for name,want in build['files'].items():
  if digest(PACKAGE/name)!=want:raise RuntimeError('build output drift:'+name)
 checks['build_sha256']=digest(PACKAGE/'BUILD.json');checks['notebook_sha256']=digest(PACKAGE/'construction.ipynb')
 donor=json.loads((REPO/'deployment/kaggle/submission-20260912-v15/submission.ipynb').read_text());assets=''.join(donor['cells'][1]['source']);match=re.search(r"Path\('/tmp/official_launcher_runtime\.py'\)\.write_bytes\(base64\.b64decode\('([^']+)'\)\)",assets)
 if match is None:raise RuntimeError('v15 supervisor asset missing')
 source=base64.b64decode(match.group(1));checks['v15_official_launcher_sha256']=hashlib.sha256(source).hexdigest()
 if checks['v15_official_launcher_sha256']!='fa7a1519595ba7f7306b2071931dc70c6784738da1f720398b0db5b1c8348afd':raise RuntimeError('v15 supervisor drift')
 if '/work/prefix.json' in (HERE/'real_trace.py').read_text():raise RuntimeError('obsolete advertised path')
 checks['qualified_runner_sha256']=digest(HERE/'qualified_runner.py')
 if checks['qualified_runner_sha256']!=digest(REPO/'analysis/kaggle-containment-qualification-20260913T211559Z/runner.py'):raise RuntimeError('candidate runner drift')
 manifest=json.loads((HERE/'MANIFEST.json').read_text())
 for group,root in (('analysis',HERE),('package',PACKAGE)):
  for name,want in manifest[group].items():
   if digest(root/name)!=want:raise RuntimeError('manifest drift:'+group+':'+name)
 checks['manifest_sha256']=digest(HERE/'MANIFEST.json')
 checks['status']='passed';(E/'FINAL-VERIFICATION.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n');print(json.dumps(checks,sort_keys=True))
if __name__=='__main__':main()
