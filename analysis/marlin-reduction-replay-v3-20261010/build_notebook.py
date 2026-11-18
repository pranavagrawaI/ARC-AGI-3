"""Offline notebook preparation. Does not call Kaggle or upload anything."""
import json
from pathlib import Path
from prepare import build, HERE, FILES, replace_once

RUN_ID=HERE.name
BASE=HERE.parent/'serving-headroom-v3-20261008'


def build_notebook():
    manifest=build()
    notebook=json.loads((BASE/'kernel/serving-headroom-v3-20261008.ipynb').read_text())
    for cell in notebook['cells']:
        source=''.join(cell['source']).replace('serving-headroom-v3-20261008',RUN_ID).replace('/kaggle/working/headroom','/kaggle/working/marlin-replay')
        cell['source']=source.splitlines(True)
        if cell['cell_type']=='code':cell.update(outputs=[],execution_count=None)
    setup=''.join(notebook['cells'][2]['source'])
    setup=replace_once(setup,"OUT = WORKING_DIR / 'headroom'","OUT = WORKING_DIR / 'marlin-replay'")
    notebook['cells'][2]['source']=setup.splitlines(True)
    notebook['cells'][0]['source']=['# Marlin scratch reduction paired replay\n','Prepared only. Requires a fresh launch instruction; no serving-speedup or gameplay claim.\n']
    module_sources={name:(HERE/name).read_text() for name in ['marlin_capture.py','benchmark.py','reference.py']}
    patched_sources={name:(HERE/'patched'/name).read_text() for name in FILES.values()}
    config={'mode':'capture','out':'/kaggle/working/marlin-replay/capture','expected_target_layers':48,
            'layer_ordinals':[0,24,47],'selected_tags':['layer-00','layer-24','layer-47'],'replay_ordinals':[8,16,32],'max_capture_bytes':12*1024**3}
    patch_cell='''# Verify original installed bytes before any task patch, then verify every replacement.
KIT = WORKING_DIR / 'marlin-kit'; KIT.mkdir(exist_ok=False)
'''
    patch_cell+='module_sources='+repr(module_sources)+'\n'
    patch_cell+='patched_sources='+repr(patched_sources)+'\n'
    patch_cell+='patch_manifest='+repr(manifest)+'\n'
    patch_cell+='capture_config='+repr(config)+'\n'
    patch_cell+='''for name, source in module_sources.items(): (KIT / name).write_text(source)
(OUT / 'capture').mkdir(exist_ok=False)
(KIT / 'capture-config.json').write_text(json.dumps(capture_config, indent=2))
script = "import hashlib,json,pathlib,sglang\\nroot=pathlib.Path(sglang.__file__).parent.parent\\n"
script += "sources="+repr(patched_sources)+"\\nmanifest="+repr(patch_manifest)+"\\n"
script += "for name,source in sources.items():\\n p=root/name\\n assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[name]['before'],name\\n"
script += "for name,source in sources.items():\\n p=root/name\\n p.write_text(source)\\n assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[name]['after'],name\\n"
# The payload exceeds Linux's 128 KiB single-argument limit (v2 died with E2BIG); run it from a file.
(KIT / 'apply_patch.py').write_text(script)
run([PYTHON, str(KIT / 'apply_patch.py')], env=env)
env['PYTHONPATH'] = str(KIT)
env['MARLIN_EXPERIMENT_CONFIG'] = str(KIT / 'capture-config.json')
(OUT / 'patch-manifest.json').write_text(json.dumps(patch_manifest, indent=2))
(OUT / 'capture-config.json').write_text(json.dumps(capture_config, indent=2))
'''
    notebook['cells'].insert(6,{'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':patch_cell.splitlines(True)})
    driver=''.join(notebook['cells'][7]['source'])
    driver=replace_once(driver,'PLAN = [("PROFILE1", 1), ("PROFILE10", 10), ("N10A", 10), ("N10B", 10)]','PLAN = [("PROFILE10", 10)]')
    driver=replace_once(driver,'        for label, n in PLAN:\n',"        (out / 'capture' / 'ARM-CAPTURE').write_text('prefix warmup complete\\n')\n        for label, n in PLAN:\n")
    # Reserve worker time as well as batch time; no treatment serving block exists.
    driver=replace_once(driver,'if deadline - time.monotonic() < 660:', 'if deadline - time.monotonic() < 1260:')
    notebook['cells'][7]['source']=driver.splitlines(True)
    worker='''# Server has been stopped and reaped by the preceding cell.
state = json.loads((OUT / 'state.json').read_text())
assert not state.get('stop') and len(state['blocks']) == 1
assert state['blocks'][0]['label'] == 'PROFILE10'
assert json.loads((OUT / 'capture' / 'binding-target.json').read_text())['is_draft_worker'] is False
assert json.loads((OUT / 'capture' / 'binding-draft.json').read_text())['modules'] == []
remaining = SESSION_DEADLINE-time.monotonic()
assert remaining >= 660, 'insufficient remaining budget for replay and collection'
worker_env = dict(env); worker_env.pop('MARLIN_EXPERIMENT_CONFIG', None)
with (OUT / 'benchmark.log').open('xb') as log:
    completed = subprocess.run([PYTHON, str(KIT / 'benchmark.py'), '--capture', str(OUT / 'capture'), '--out', str(OUT / 'benchmark')], env=worker_env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    assert completed.returncode == 0, 'benchmark worker failed; preserve log, no retry'
manifest = {str(p.relative_to(OUT)): _sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'output-hashes.json'}
(OUT / 'output-hashes.json').write_text(json.dumps(manifest, indent=1))
'''
    # Even a failed/terminated worker must retain a new final hash manifest.
    worker=worker[:worker.index('with (OUT')]+'try:\n'+''.join('    '+line+'\n' for line in worker[worker.index('with (OUT'):worker.index('manifest =')].splitlines())+'finally:\n'+''.join('    '+line+'\n' for line in worker[worker.index('manifest ='):].splitlines())
    notebook['cells'].append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':worker.splitlines(True)})
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell-{i}','exec')
    meta=json.loads((BASE/'kernel/kernel-metadata.json').read_text());meta.update(id='pranavagrawai/'+RUN_ID,title=RUN_ID,code_file=RUN_ID+'.ipynb')
    dest=HERE/'kernel';dest.mkdir(exist_ok=True)
    (dest/meta['code_file']).write_text(json.dumps(notebook,indent=1)+'\n')
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (HERE/'capture-config.json').write_text(json.dumps(config,indent=2)+'\n')
    return notebook


if __name__=='__main__':build_notebook()
