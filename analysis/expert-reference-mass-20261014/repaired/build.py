"""Build one private paired screen, borrowing the frozen c14 notebook setup."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
from IPython.core.interactiveshell import InteractiveShell

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO/'analysis/model-serving-20261011/wave-1/e2-c14/kernel/wave1-e2-c14-20261011.ipynb'
SLUG = 'expert-reference-mass-screen-r2-20261014'
FILES = ['routing.py','patch_runtime.py','test_routing.py','screen.py','isolated_phase.py']


def build():
    if (HERE/'launch-attempt.json').exists(): raise RuntimeError('frozen run cannot be rebuilt')
    base = json.loads(BASE.read_text())
    nb = copy.deepcopy(base)
    sources = {str(i): ''.join(base['cells'][i]['source']) for i in [12,14,16,20]}
    sources['16'] = sources['16'].replace("    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:",
                                         "    bm.solver.max_runtime_s_per_game = PLAY_S\n    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:")
    sources['20'] = sources['20'].replace('soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))',
                                         'soft_end = datetime.now() + timedelta(seconds=PLAY_S)')
    launcher = sources['12']
    marker = '# ---- bundled CUDA toolkit and linker symlinks ----'
    install = """site_dir = next(Path(VENV).glob('lib/python*/site-packages'))
run([PYTHON, str(KIT/'patch_runtime.py'), str(wheel), str(site_dir), ARM, str(WORKING_DIR/'routing-install.json')], env=install_env)
if ARM == 'control':
    test_env = dict(install_env, REFERENCE_WHEEL=str(wheel), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    test_result = run([PYTHON, '-B', str(KIT/'test_routing.py')], env=test_env, check=False)
    (WORKING_DIR/'routing-tests.stdout.txt').write_text(test_result.stdout)
    (WORKING_DIR/'routing-tests.stderr.txt').write_text(test_result.stderr)
    (WORKING_DIR/'routing-tests.json').write_text(json.dumps({'returncode':test_result.returncode,'gpu_graph_executed':'REFERENCE_GRAPH_TEST_PASSED' in test_result.stdout}))
    if test_result.returncode: raise RuntimeError('Routing qualification failed; see routing-tests.stderr.txt')
    if 'REFERENCE_GRAPH_TEST_PASSED' not in test_result.stdout: raise RuntimeError('GPU graph qualification did not execute')
"""
    assert launcher.count(marker)==1
    launcher = launcher.replace(marker,install+'\n'+marker)
    launcher = launcher.replace('precache_model_thread.start()',
                               "precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR,DRAFT_MODEL_DIR), kwargs={'threads':3})\nprecache_model_thread.start()")
    launcher = launcher.replace('proc = subprocess.Popen(args, env=env,',"env['ARC3_REFERENCE_TOPK'] = '10' if ARM == 'candidate' else '0'\n    proc = subprocess.Popen(args, env=env,")
    launcher = launcher.replace('print(f"pid {proc.pid} -> {LOG}")', 'OWNED["proc"] = proc\nprint(f"pid {proc.pid} -> {LOG}")')
    sources['12'] = launcher
    # Freeze the owned launcher and phase snippets, not another importable mutable builder.
    assets = {name:(HERE/name).read_text() for name in FILES}
    assets['phase-sources.json'] = json.dumps(sources)
    write_assets = "KIT = Path('/kaggle/reference-mass-kit')\nKIT.mkdir(exist_ok=True)\n"
    for name,content in assets.items():write_assets+=f"(KIT/{name!r}).write_text({content!r})\n"
    # Establish absolute containment before installation/model loading. Offline screen only.
    start = ''.join(nb['cells'][4]['source'])
    start += "\nif TRUE_SUBMISSION: raise RuntimeError('Research screen is not a competition submission')\n"+write_assets
    start += "exec(compile((KIT/'screen.py').read_text(), str(KIT/'screen.py'), 'exec'), globals())\nstart_containment()\n"
    nb['cells'][4]['source'] = start.splitlines(True)
    # Keep setup cells through imports; replace the execution tail with the paired orchestrator.
    nb['cells'] = nb['cells'][:12]+[nbformat.v4.new_code_cell('await run_screen(globals())\n')]
    nb['cells'][-1].pop('id',None)  # Parent uses the pre-cell-id notebook schema.
    nb['cells'][0]['source'] = ('# Expert reference-mass screen\n\nOne private research session; control then active5/reference10. '
                                'One GPU-hour cap including startup. Based on dfranzen c14; same W4A16 weights and top10 MTP.\n')
    for c in nb['cells']:
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
    kd=HERE/'kernel';kd.mkdir(exist_ok=True)
    path=kd/(SLUG+'.ipynb');path.write_text(json.dumps(nb,indent=1)+'\n')
    meta=json.loads(BASE.with_name('kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/'+SLUG,title=SLUG.replace('-',' '),code_file=path.name,is_private=True)
    (kd/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    nbformat.validate(nbformat.read(path,as_version=4))
    shell=InteractiveShell.instance()
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(shell.transform_cell(''.join(c['source'])),f'cell{i}','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    for i,s in sources.items():compile(shell.transform_cell(s),f'phase{i}','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    # The checker searches plain code, so also expose the exact decoded snippets
    # executed by run_screen; checking escaped JSON literals misses quoted anchors.
    expanded=copy.deepcopy(nb)
    expanded['cells'].append({'cell_type':'code','source':'\n'.join(sources.values()),'metadata':{},'outputs':[],'execution_count':None})
    expanded_path=HERE/'evidence/effective-notebook.json'
    expanded_path.write_text(json.dumps(expanded)+'\n')
    check=subprocess.run([sys.executable,'-B',str(REPO/'incumbent/check.py'),str(expanded_path),'--without','b1'],capture_output=True,text=True)
    (HERE/'evidence/incumbent-check.log').write_text(check.stdout+check.stderr)
    # Nested frozen launcher strings preserve anchors although they are embedded in the kit.
    if check.returncode:raise RuntimeError(check.stdout+check.stderr)
    receipt={'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
             'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'metadata_sha256':hashlib.sha256((kd/'kernel-metadata.json').read_bytes()).hexdigest(),
             'asset_sha256':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in assets.items()},
             'hard_cap_s':3600,'self_cap_s':3450,'play_s_per_arm':900,
             'model_sources_unchanged':meta['model_sources']==json.loads(BASE.with_name('kernel-metadata.json').read_text())['model_sources'],
             'b1_exception':'Exact c14 comparison; B1 is separate probation bundle, excluded from both arms',
             'competition_submission_disabled':True,'schema_and_compile':True,'incumbent_check':True}
    receipt['incumbent_check_scope']='notebook plus exact decoded, executed frozen phase snippets'
    receipt['effective_notebook_sha256']=hashlib.sha256(expanded_path.read_bytes()).hexdigest()
    (HERE/'LOCAL-QUALIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':build()
