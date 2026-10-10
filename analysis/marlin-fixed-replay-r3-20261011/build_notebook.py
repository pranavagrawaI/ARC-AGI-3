"""Reuse qualified installation mechanics; no server or recapture phase."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
V3=HERE.parent/'marlin-reduction-replay-v3-20261010'
BASE=HERE.parent/'serving-headroom-v3-20261008'


def build():
    prior=json.loads((V3/'kernel'/f'{V3.name}.ipynb').read_text())
    source=json.loads((HERE/'capture-source.json').read_text())
    contract=json.loads((BASE/'native-contract.json').read_text())
    modules={name:(HERE/name).read_text() for name in ['benchmark.py','reference.py','runtime_views.py']}
    # Hooks import this module even in ordinary replay; it is the unchanged v3 source.
    modules['marlin_capture.py']=(V3/'marlin_capture.py').read_text()
    patch=json.loads((V3/'patch-manifest.json').read_text())
    replacements={name:(V3/'patched'/name).read_text() for name in patch}
    for name,s in replacements.items():
        assert hashlib.sha256(s.encode()).hexdigest()==patch[name]['after']
    setup="""import hashlib,json,os,re,shlex,shutil,subprocess,sys,time,zipfile
from pathlib import Path
SESSION_DEADLINE=time.monotonic()+3300
WORKING_DIR=Path('/kaggle/working')
OUT=WORKING_DIR/'marlin-replay';OUT.mkdir(exist_ok=True)
KIT=WORKING_DIR/'diagnostic-kit';KIT.mkdir(exist_ok=False)
def _sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()
def run(cmd,*,env=None):
    cmd=list(map(str,cmd));print('$',shlex.join(cmd),flush=True)
    limit=min(900,SESSION_DEADLINE-time.monotonic()-120)
    assert limit>0,'no setup budget remaining'
    done=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=limit)
    print(done.stdout[-6000:],flush=True)
    if done.returncode:
        print(done.stderr[-6000:],flush=True)
        raise RuntimeError('setup subprocess failed')
    return done
def find_unique(root,name,*,directory=False):
    paths=[p for p in Path(root).rglob(name) if (p.is_dir() if directory else p.is_file())]
    assert len(paths)==1,(name,len(paths))
    return paths[0]
"""
    setup+='expected_source='+repr(source)+'\n'
    setup+="""candidates=[p.parent.parent for p in Path('/kaggle/input').rglob('cases.jsonl') if p.parent.name=='capture']
assert len(candidates)==1,('expected one mounted capture',len(candidates))
SOURCE=candidates[0]
readback={name:_sha(SOURCE/name) for name in expected_source['files']}
assert readback==expected_source['files'],'mounted capture differs from accepted v3 version'
(OUT/'capture-source.json').write_text(json.dumps({'kernel_id':expected_source['kernel_id'],'version':expected_source['version'],'files':readback},indent=2))
for name in ['server-args.json','target-config.json']:
    shutil.copy2(SOURCE/name,OUT/name)
WHEELHOUSE_DIR=find_unique('/kaggle/input','requirements.lock').parent
PREFIX='/tmp/marlin-fixed';VENV=PREFIX+'/venv';PYTHON=VENV+'/bin/python';SGLANG=VENV+'/bin/sglang'
"""
    setup+=(BASE/'package_contract.py').read_text()
    setup+='\ncontract='+repr(contract)+'\n'
    setup+="""verify_bundle(Path(WHEELHOUSE_DIR)/'wheels',Path(WHEELHOUSE_DIR)/'requirements.lock',contract)
(OUT/'package-identity.json').write_text(json.dumps(contract['wheel'],indent=2))
"""
    original=''.join(prior['cells'][4]['source'])
    install=original[original.index('WHEELS ='):original.index('# ---- validate target')]
    setup+=install
    patch_cell='modules='+repr(modules)+'\nreplacements='+repr(replacements)+'\npatch='+repr(patch)+'\n'
    patch_cell+="""for name,text in modules.items():(KIT/name).write_text(text)
script="import hashlib,json,pathlib,sglang\\nroot=pathlib.Path(sglang.__file__).parent.parent\\n"
script+="sources="+repr(replacements)+"\\npatch="+repr(patch)+"\\n"
script+="for name,text in sources.items():\\n p=root/name\\n assert hashlib.sha256(p.read_bytes()).hexdigest()==patch[name]['before'],name\\n"
script+="for name,text in sources.items():\\n p=root/name\\n p.write_text(text)\\n assert hashlib.sha256(p.read_bytes()).hexdigest()==patch[name]['after'],name\\n"
script+="(pathlib.Path('/kaggle/working/marlin-replay')/'installed-patch.json').write_text(json.dumps(patch,indent=2))\\n"
(KIT/'apply_patch.py').write_text(script)
run([PYTHON,KIT/'apply_patch.py'],env=env)
env['PYTHONPATH']=str(KIT)
"""
    worker="""remaining=SESSION_DEADLINE-time.monotonic()
assert remaining>=2520,'insufficient time for fixed replay and evidence'
try:
    with (OUT/'benchmark.log').open('xb') as log:
        done=subprocess.run([PYTHON,str(KIT/'benchmark.py'),'--capture',str(SOURCE/'capture'),'--out',str(OUT/'benchmark')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    assert done.returncode==0,'fixed replay worker failed; preserve evidence, no retry'
finally:
    hashes={str(p.relative_to(OUT)):_sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='output-hashes.json'}
    (OUT/'output-hashes.json').write_text(json.dumps(hashes,indent=2))
"""
    hardware=''.join(prior['cells'][1]['source']).replace(V3.name,HERE.name)
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Fixed-capture Marlin diagnostic\n','Numerical variation and paired timings; no serving qualification.\n']}]
    for i,text in enumerate([hardware,setup,patch_cell,worker]):
        compile(text,f'cell-{i}','exec')
        cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':text.splitlines(True)})
    notebook={'cells':cells,'metadata':prior['metadata'],'nbformat':4,'nbformat_minor':4}
    meta=json.loads((V3/'kernel/kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/'+HERE.name,title=HERE.name,code_file=HERE.name+'.ipynb',
                dataset_sources=['dfranzen/pennyroyal-v253'],kernel_sources=[source['slug']])
    folder=HERE/'kernel';folder.mkdir(exist_ok=True)
    (folder/meta['code_file']).write_text(json.dumps(notebook,indent=1)+'\n')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return notebook


if __name__=='__main__':build()
