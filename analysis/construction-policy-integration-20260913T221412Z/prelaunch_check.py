#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];PACKAGE=REPO/'deployment/kaggle'/HERE.name
Q=REPO/'analysis/kaggle-containment-qualification-20260913T211559Z';DONOR=REPO/'deployment/kaggle/submission-20260912-v15'
EXPECTED_Q={'RESULT.json':'bccb2dddb20b70a8a25ffb6cfccbbf2a63f20468815ba87d6c2207c6b314ffe7','FINAL-MANIFEST.json':'4e674111eb10784ad4086c589ed4f615a7cb067cb27da66f44dcc9b985fd696a'}
IMAGE='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check():
 errors=[];meta=json.loads((PACKAGE/'kernel-metadata.json').read_text());donor=json.loads((DONOR/'kernel-metadata.json').read_text())
 for key in ('dataset_sources','kernel_sources','competition_sources','model_sources'):
  if meta.get(key)!=donor.get(key):errors.append('source drift:'+key)
 if meta.get('competition_sources')!=['arc-prize-2026-arc-agi-3']:errors.append('competition missing')
 for key,value in {'is_private':True,'enable_internet':False,'enable_gpu':True,'machine_shape':'NvidiaRtxPro6000','docker_image':IMAGE}.items():
  if meta.get(key)!=value:errors.append('metadata drift:'+key)
 if (PACKAGE/'AUTHORIZATION.json').exists():errors.append('launch guard open: authorization exists')
 if 'kaggle kernels push' in (PACKAGE/'launch_once.py').read_text():errors.append('launch guard open: push command present')
 qresult=json.loads((Q/'RESULT.json').read_text()) if (Q/'RESULT.json').exists() else {}
 if qresult.get('qualified') is not True or qresult.get('scope')!='REMOTE_EXACT_VERSION':errors.append('target containment not qualified')
 for n,h in EXPECTED_Q.items():
  if not (Q/n).exists() or digest(Q/n)!=h:errors.append('containment evidence mismatch:'+n)
 if digest(HERE/'qualified_runner.py')!=digest(Q/'runner.py'):errors.append('qualified runner bytes drift')
 build=json.loads((PACKAGE/'BUILD.json').read_text())
 for n,h in build['assets'].items():
  p=(PACKAGE/n) if n=='remote_entry.py' else HERE/n
  if digest(p)!=h:errors.append('asset drift:'+n)
 for n,h in build['files'].items():
  if digest(PACKAGE/n)!=h:errors.append('package drift:'+n)
 return {'passed':not errors,'errors':errors,'launch_guard_closed':not (PACKAGE/'AUTHORIZATION.json').exists(),'competition_sources':meta.get('competition_sources'),'dataset_sources':meta.get('dataset_sources'),'containment_prerequisite':EXPECTED_Q,'image':meta.get('docker_image')}
if __name__=='__main__':
 r=check();print(json.dumps(r,indent=2,sort_keys=True));raise SystemExit(0 if r['passed'] else 2)
