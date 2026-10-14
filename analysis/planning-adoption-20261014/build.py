"""Freeze past-only source/data and a private bounded Kaggle notebook."""
import base64
import hashlib
import json
import lzma
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SLUG='planning-adoption-20261014'
ARCHIVE=ROOT/'analysis/counterexample-continuations-v3-20261013'
TRANSFER='analysis/planning-transfer-20261014'


def sha(data):return hashlib.sha256(data).hexdigest()


def package_files():
    files={}
    for name in ['continuation.py','run_matrix.py','audit.py']:
        files[TRANSFER+'/'+name]=(HERE/'runtime'/name).read_text()
    files['solvers/m2_planning_tools/variants.py']=(HERE/'runtime/variants.py').read_text()
    for path in (ROOT/'solvers/m2_planning_tools/overlay').rglob('*.py'):
        files[str(path.relative_to(ROOT))]=path.read_text()
    for name in ['restore.py','worker.py']:
        files['analysis/counterexample-continuations-v3-20261013/'+name]=(ARCHIVE/name).read_text()
    for name in ['b1','control']:
        files['analysis/counterexample-continuations-v3-20261013/inputs/'+name+'.json']=(ARCHIVE/'inputs'/(name+'.json')).read_text()
        config=json.loads(files['analysis/counterexample-continuations-v3-20261013/inputs/'+name+'.json'])
        for path in Path(config['code']).rglob('*.py'):
            if '__pycache__' not in path.parts:
                files['sources/'+name+'/'+str(path.relative_to(config['code']))]=path.read_text()
    taaf=ROOT/'analysis/m2-winners-20261001/audit/dfranzen-repo/tufa-arc-agi-framework/src'
    for path in taaf.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py','.json','.yaml','.yml','.j2','.html','.css','.txt']:
            files['sources/taaf/'+str(path.relative_to(taaf))]=path.read_text()
    receipt=json.loads((ROOT/'analysis/s5i5-inspection-20261012/engine-replay.json').read_text())
    files['analysis/s5i5-inspection-20261012/engine-replay.json']=json.dumps(
        {key:receipt[key] for key in ['engine_source','engine_sha256']},indent=2)+'\n'
    # A fixed balanced order prevents every treatment occupying the same slot.
    orders=[['control','planner','both','probing'],['planner','probing','control','both'],
            ['probing','both','planner','control'],['both','control','probing','planner']]
    matrix=[]
    for index,(checkpoint,seed) in enumerate([('b1',1701),('control',1701),('b1',2903),('control',2903)]):
        for variant in orders[index]:
            matrix.append({'id':f'{checkpoint}-{seed}-{variant}','checkpoint':checkpoint,'seed':seed,'variant':variant})
    files[TRANSFER+'/matrix.json']=json.dumps(matrix,indent=2)+'\n'
    files['EXPERIMENT-CARD.md']=(HERE/'EXPERIMENT-CARD.md').read_text()
    return files


def materialize(destination,files):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for name,text in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    manifest={name:sha(text.encode()) for name,text in files.items()}
    (destination/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    if (HERE/'evidence/launch-attempt.json').exists():raise RuntimeError('Frozen launch exists; no rebuild')
    files=package_files()
    manifest=materialize(HERE/'package',files)
    packed=base64.b64encode(lzma.compress(json.dumps(files).encode(),preset=9)).decode()
    assert json.loads(lzma.decompress(base64.b64decode(packed)))==files
    install=f'''# Frozen planning/probing package; no future routes or rules in feedback.
import base64,hashlib,json,lzma,os
PLANNING_ROOT=WORKING_DIR/'planning-package'
PLANNING_ROOT.mkdir(exist_ok=True)
_files=json.loads(lzma.decompress(base64.b64decode({packed!r})))
for _name,_text in _files.items():
    _path=PLANNING_ROOT/_name;_path.parent.mkdir(parents=True,exist_ok=True);_path.write_text(_text)
_manifest={{name:hashlib.sha256(text.encode()).hexdigest() for name,text in _files.items()}}
(PLANNING_ROOT/'package-manifest.json').write_text(json.dumps(_manifest,indent=2))
for _name,_hash in _manifest.items():assert hashlib.sha256((PLANNING_ROOT/_name).read_bytes()).hexdigest()==_hash
_env_dir=Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
_engine=_env_dir/'s5i5/18d95033/s5i5.py'
_receipt=json.loads((PLANNING_ROOT/'analysis/s5i5-inspection-20261012/engine-replay.json').read_text())
assert hashlib.sha256(_engine.read_bytes()).hexdigest()==_receipt['engine_sha256']
os.environ.update(PLANNING_ENV_DIR=str(_env_dir),PLANNING_ENGINE_SOURCE=str(_engine),
    PLANNING_CODE_B1=str(PLANNING_ROOT/'sources/b1'),PLANNING_CODE_CONTROL=str(PLANNING_ROOT/'sources/control'),
    PLANNING_TAAF_DIR=str(PLANNING_ROOT/'sources/taaf'))
print('PLANNING_PACKAGE_VERIFIED',len(_manifest),flush=True)
# Restore both actual checkpoints before spending time loading model weights.
for _checkpoint in ('b1','control'):
    _out=WORKING_DIR/'cold-restoration'/_checkpoint
    subprocess.run([sys.executable,'-B',str(PLANNING_ROOT/'{TRANSFER}/continuation.py'),
        '--checkpoint',_checkpoint,'--variant','control','--output',str(_out)],check=True,timeout=120)
    _result=json.loads((_out/'result.json').read_text())
    assert _result['status']=='restored' and _result['qualified_restoration'] and _result['protected_inputs_unchanged']
print('COLD_RESTORATION_VERIFIED',flush=True)
'''
    execute=f'''# Exactly one bounded matrix, with a reserved evidence phase.
if TRUE_SUBMISSION:raise RuntimeError('Diagnostic only; no competition submission')
import urllib.request,os,json,signal,time
assert time.time()-NOTEBOOK_START_TIME<=1300, 'Startup exceeded the frozen allowance'
with urllib.request.urlopen(SERVER_BASE_URL.removesuffix('/v1')+'/health',timeout=10) as _response:
    assert _response.status==200
with urllib.request.urlopen(SERVER_BASE_URL.removesuffix('/v1')+'/get_server_info',timeout=15) as _response:
    _server_info=json.load(_response)
ADOPTION_OUT=WORKING_DIR/'planning-adoption'
ADOPTION_OUT.mkdir(exist_ok=True)
(ADOPTION_OUT/'server-info.json').write_text(json.dumps(_server_info,indent=2))
(ADOPTION_OUT/'package-manifest.json').write_text(json.dumps(_manifest,indent=2))
(ADOPTION_OUT/'source-receipts.json').write_text(json.dumps({{'engine_sha256':_receipt['engine_sha256'],
    'notebook_started':NOTEBOOK_START_TIME,'startup_elapsed':time.time()-NOTEBOOK_START_TIME,
    'same_serving_cells_as':'counterexample-continuations-v3-20261013','CFG':CFG}},indent=2))
os.environ['PLANNING_MATRIX_DEADLINE_EPOCH']=str(NOTEBOOK_START_TIME+8700)
_remaining=NOTEBOOK_START_TIME+8700-time.time()
assert _remaining>0
_command=[sys.executable,'-B',str(PLANNING_ROOT/'{TRANSFER}/run_matrix.py'),'--mode','live',
    '--output',str(ADOPTION_OUT/'episodes'),'--url',SERVER_BASE_URL]
_matrix_proc=subprocess.Popen(_command,start_new_session=True)
try:
    _rc=_matrix_proc.wait(timeout=_remaining)
except subprocess.TimeoutExpired:
    os.killpg(_matrix_proc.pid,signal.SIGKILL);_matrix_proc.wait();_rc=-9
finally:
    _results=ADOPTION_OUT/'episodes/results.json'
    if _results.exists():(ADOPTION_OUT/'results.json').write_bytes(_results.read_bytes())
    print('MATRIX_EXIT',_rc,flush=True)
if _rc:raise RuntimeError('Matrix stopped; evidence preserved; no retry')
print((ADOPTION_OUT/'results.json').read_text(),flush=True)
'''
    source=next((ARCHIVE/'kernel').glob('*.ipynb'));notebook=json.loads(source.read_text())
    cells=[]
    for index,cell in enumerate(notebook['cells']):
        if index in (7,8):continue
        text=''.join(cell['source'])
        if index==1:
            assert text.count('SERVER_STARTUP_TIMEOUT = 12 * 60')==1
            text=text.replace('SERVER_STARTUP_TIMEOUT = 12 * 60','SERVER_STARTUP_TIMEOUT = 1300')
        if index==5:cells.append(code_cell(install))
        cell['source']=text.splitlines(True);cell['outputs']=[];cell['execution_count']=None
        cells.append(cell)
    cells.append(code_cell(execute));notebook['cells']=cells
    target=HERE/'kernel'/(SLUG+'.ipynb');target.write_text(json.dumps(notebook,indent=1)+'\n')
    metadata=json.loads((ARCHIVE/'kernel/kernel-metadata.json').read_text())
    metadata.update(id='pranavagrawai/'+SLUG,title='Planning Adoption 20261014',code_file=target.name,is_private=True)
    (HERE/'kernel/kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    identity={'notebook_sha256':sha(target.read_bytes()),'metadata':metadata,'manifest':manifest,
              'hard_cap_seconds':9000,'generated_token_cap':524288,'episodes':16,
              'card_sha256':sha((HERE/'EXPERIMENT-CARD.md').read_bytes()),
              'parent_serving_notebook_sha256':sha(source.read_bytes())}
    (HERE/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print('Built notebook',target.stat().st_size,'bytes;',len(files),'frozen files')


def code_cell(source):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(True)}


if __name__=='__main__':main()
