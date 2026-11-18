"""Freeze both arms of a matched pair from one shared budget block; no launch."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import shutil

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
CONTROL=ROOT/'analysis/m2-winners-20261001/runs/control'
PAIR_SOURCE=HERE/'src/hybrid/pair.py'
RUN_ANCHOR='# Play the benchmark; teardown commands run even if the run raises.'
# Hybrid notebook = donor + install cell at 9; these donor cells may differ (title,
# submission guard, solver swap, Duck viewer). Everything else must be byte-identical.
ALLOWED_SOLVER_CELLS={0,4,20,'last'}


def sha(data):return hashlib.sha256(data if isinstance(data,bytes) else data.encode()).hexdigest()
def cell(source):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                         'source':source.splitlines(True)}
def text(c):return ''.join(c['source'])


def shared_cells(arm,pair_id):
    import hybrid.pair as pair
    watchdog=(HERE/'experiment/watchdog.py').read_text()
    preflight=f'''import json,os,sys,time,subprocess
from pathlib import Path
_qualification_root=Path('/kaggle/working');_qualification_root.mkdir(exist_ok=True)
_qualification_started_monotonic=time.monotonic()
_qualification_started_epoch=time.time()
_watchdog_path=Path('/tmp/pair-watchdog.py')
_watchdog_path.write_text({watchdog!r})
_watchdog=subprocess.Popen([sys.executable,str(_watchdog_path),'--parent',str(os.getpid()),
 '--root',str(_qualification_root),'--started',str(_qualification_started_monotonic),
 '--soft','{pair.SOFT_STOP_SECONDS}','--hard','{pair.SESSION_SECONDS}','--force-margin','10'],start_new_session=True)
os.environ['HYBRID_RUN_MODE']='compare'
_gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip().splitlines()
(_qualification_root/'qualification-start.json').write_text(json.dumps({{'run_id':{pair_id+'-'+arm!r},'arm':{arm!r},
 'started_epoch':_qualification_started_epoch,'gpu_inventory':_gpu_inventory}}))
assert len(_gpu_inventory)==1 and 'RTX PRO 6000' in _gpu_inventory[0], 'exactly one donor GPU required'
'''
    # Identical text in both arms; the hybrid runner re-checks it after unpickling.
    budget=f'''# Matched-pair budget: identical block in both arms (hybrid/pair.py, sha256 {sha(PAIR_SOURCE.read_bytes())}).
_pair_namespace={{}}
exec(compile({PAIR_SOURCE.read_text()!r},'pair.py','exec'),_pair_namespace)
_pair_config=_pair_namespace['apply'](bm)
soft_end=datetime.fromtimestamp(_qualification_started_epoch+_pair_namespace['SOFT_STOP_SECONDS'])
(_qualification_root/'pair-config.json').write_text(json.dumps({{**_pair_config,
 'game_ids':[str(g.env_name) for g in bm.games],'soft_end':soft_end.isoformat()}},indent=2))
'''
    complete='''# Preserve a bounded evidence inventory for either solver arm.
import hashlib
_hybrid_index=_qualification_root/'tycho-dfranzen/evidence-index.json'
if _hybrid_index.exists():
    _pair_files=json.loads(_hybrid_index.read_text())['files']
else:
    _pair_files=[]
    for _path in sorted(_qualification_root.rglob('*')):
        if not _path.is_file(): continue
        _relative=_path.relative_to(_qualification_root)
        if len(_relative.parts)>2: continue
        if _relative.parts[0] not in ('artifacts','transcripts') and _path.name!='benchmark.json' and not _path.name.endswith('_requests.jsonl'): continue
        _pair_files.append({'path':_relative.as_posix(),'sha256':hashlib.sha256(_path.read_bytes()).hexdigest(),
                            'critical':not _path.name.endswith('_requests.jsonl')})
(_qualification_root/'pair-evidence-index.json').write_text(json.dumps({'files':_pair_files},indent=2))
(_qualification_root/'qualification-complete.json').write_text(json.dumps({'notebook_completed':True}))
'''
    return preflight,budget,complete


def arm_notebook(base,arm,pair_id):
    preflight,budget,complete=shared_cells(arm,pair_id)
    notebook=json.loads(json.dumps(base))
    for c in notebook['cells']:
        if c['cell_type']=='code': c['outputs']=[];c['execution_count']=None
    runs=[c for c in notebook['cells'] if RUN_ANCHOR in text(c)]
    assert len(runs)==1 and text(runs[0]).count(RUN_ANCHOR)==1
    runs[0]['source']=text(runs[0]).replace(RUN_ANCHOR,budget+RUN_ANCHOR).splitlines(True)
    notebook['cells'].insert(0,cell(preflight))
    notebook['cells'].append(cell(complete))
    return notebook


def solver_only_difference(control,hybrid):
    """Return donor cell indices that differ; install cell 9 (+1 for preflight) is the hybrid's own."""
    hybrid_cells=hybrid['cells'][:10]+hybrid['cells'][11:]
    assert len(hybrid_cells)==len(control['cells']),'cell count drift beyond the install cell'
    last=len(control['cells'])-2  # final cell before the shared completion cell
    differing=set()
    for i,(a,b) in enumerate(zip(control['cells'],hybrid_cells)):
        # The preflight differs only by its arm label.
        ta,tb=(text(a).replace("'control'","'ARM'").replace("-control'","-ARM'"),
               text(b).replace("'hybrid'","'ARM'").replace("-hybrid'","-ARM'")) if i==0 else (text(a),text(b))
        if ta!=tb or a['cell_type']!=b['cell_type']:
            differing.add('last' if i==last else i-1)  # report donor numbering
    return differing


def main():
    import sys
    import nbformat
    sys.path.insert(0,str(HERE/'src'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-id',default='pair-1')
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--seed',type=int,default=None,help='arm-order draw; recorded')
    args=parser.parse_args()
    out=args.out or HERE/'experiment'/args.pair_id
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert sha((HERE/'kernel/tycho-dfranzen.ipynb').read_bytes())==manifest['notebook_sha256'],'rebuild first'
    assert manifest['source_files']['hybrid/pair.py']==sha(PAIR_SOURCE.read_bytes()),'payload lacks current pair.py'
    bases={'control':json.loads((CONTROL/'dfranzen-m2-control.ipynb').read_text()),
           'hybrid':json.loads((HERE/'kernel/tycho-dfranzen.ipynb').read_text())}
    metas={'control':json.loads((CONTROL/'kernel-metadata.json').read_text()),
           'hybrid':json.loads((HERE/'kernel/kernel-metadata.json').read_text())}
    notebooks={arm:arm_notebook(bases[arm],arm,args.pair_id) for arm in bases}
    differing=solver_only_difference(notebooks['control'],notebooks['hybrid'])
    assert differing<=ALLOWED_SOLVER_CELLS,f'non-solver cells differ: {sorted(map(str,differing-ALLOWED_SOLVER_CELLS))}'
    for key in ('enable_gpu','enable_internet','machine_shape','docker_image','competition_sources','model_sources'):
        assert metas['control'].get(key)==metas['hybrid'].get(key),f'metadata {key} differs'
    assert set(metas['control']['dataset_sources'])<=set(metas['hybrid']['dataset_sources'])
    identities={}
    for arm,notebook in notebooks.items():
        directory=out/arm;(directory/'kernel').mkdir(parents=True)
        nbformat.validate(nbformat.from_dict(notebook))
        for i,c in enumerate(notebook['cells']):
            source=text(c)
            if c['cell_type']=='code' and not any(l.startswith(('!','%')) for l in source.splitlines()):
                compile(source,f'{arm}:{i}','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        path=directory/'kernel/notebook.ipynb'
        path.write_text(json.dumps(notebook,indent=1)+'\n')
        assert path.stat().st_size<1_000_000
        meta=dict(metas[arm],id=f'pranavagrawai/tycho-dfranzen-{args.pair_id}-{arm}',
                  title=f'Tycho dfranzen {args.pair_id} {arm}',code_file=path.name)
        (directory/'kernel/kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        if arm=='hybrid':
            for name in ('manifest.json','canonical-candidate.patch'):shutil.copy2(HERE/name,directory/name)
            shutil.copytree(HERE/'source-bundle',directory/'source-bundle')
        identities[arm]={'notebook_sha256':sha(path.read_bytes()),
            'metadata_sha256':sha((directory/'kernel/kernel-metadata.json').read_bytes()),
            'kernel':meta['id'],'gpu_count':1}
    import hybrid.pair as pair
    seed=args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    order=random.Random(seed).sample(['control','hybrid'],2)
    pair_identity={'pair_id':args.pair_id,'arms':identities,'pair_source_sha256':sha(PAIR_SOURCE.read_bytes()),
        'source_payload_sha256':manifest['source_payload_sha256'],
        'differing_donor_cells':sorted(map(str,differing)),'hard_timeout_seconds':pair.SESSION_SECONDS,
        'arm_order_seed':seed,'launch_order':order,
        'reserved_gpu_hours':2*pair.SESSION_SECONDS/3600}
    (out/'pair-identity.json').write_text(json.dumps(pair_identity,indent=2)+'\n')
    print(json.dumps(pair_identity,indent=2))


if __name__=='__main__':main()
