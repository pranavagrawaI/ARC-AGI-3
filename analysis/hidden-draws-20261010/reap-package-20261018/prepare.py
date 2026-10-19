"""Prepare and exercise a submission-compatible REAP successor; no remote writes."""
import ast
import contextlib
import difflib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT / 'analysis/pruning-transfer-20261011/candidate2/kernel/prune-retain-candidate2-20261011.ipynb'
PRIOR_SHA = '16d3956e154108d5d49407dd19fe43d442974cde5585ec6b764308fd3fbb78d7'
SLUG = 'pranavagrawai/reap448-c18-hidden-ready-20261018'
GUARD = 'assert not TRUE_SUBMISSION, "Public qualification only; never submit this notebook"\n'
OVERRIDE = '\nbm.solver.max_runtime_s_per_game = 7254\n'
OLD_END = 'soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))'
SMOKE_END = 'soft_end = datetime.now() + timedelta(seconds=900)  # interactive packaging qualification only'


def source(nb, index):
    return ''.join(nb['cells'][index]['source'])


def competition_guard(nb):
    """Run the original entry guard with a real competition-mode environment."""
    text = source(nb, 4)
    text = '\n'.join('pass' if line.lstrip().startswith('!') else line for line in text.splitlines())
    tree = ast.parse(text)
    nodes = [node for node in tree.body
             if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TRUE_SUBMISSION' for t in node.targets))
             or (isinstance(node, ast.Assert) and 'TRUE_SUBMISSION' in ast.unparse(node.test))]
    namespace = {'os': SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN': 'true'})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<competition-entry>', 'exec'), namespace)
    return namespace['TRUE_SUBMISSION']


def runtime_config(nb, competition):
    solver = SimpleNamespace()
    benchmark = SimpleNamespace(solver=solver)
    namespace = dict(bm=benchmark, TRUE_SUBMISSION=competition, USE_PRIORITY_SCHEDULING=True,
                     target=SimpleNamespace(max_runtime_s=32400))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source(nb, 16), '<actual-runtime-config>', 'exec'), namespace)
    return dict(per_game_s=solver.max_runtime_s_per_game, concurrency=solver.concurrency,
                passes=benchmark.n_passes, save_request_logs=solver.save_request_logs)


def deadline(nb, competition):
    import datetime as dt
    tree = ast.parse(source(nb, 20))
    index = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'soft_end' for t in n.targets))
    nodes = tree.body[index:index+2]
    assert isinstance(nodes[1], ast.If)
    before = dt.datetime.now()
    namespace = dict(TRUE_SUBMISSION=competition, datetime=dt.datetime, timedelta=dt.timedelta,
                     NOTEBOOK_START_EPOCH=before.timestamp(), target=SimpleNamespace(max_runtime_s=32400))
    exec(compile(ast.Module(body=nodes,type_ignores=[]), '<actual-soft-end>', 'exec'), namespace)
    end = namespace['soft_end']
    return None if end is None else (end-before).total_seconds()


def main():
    original_bytes = PRIOR.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == PRIOR_SHA
    prior = json.loads(original_bytes)
    remote = json.loads((HERE/'remote-prior.ipynb').read_text())
    assert [(c['cell_type'],''.join(c['source'])) for c in prior['cells']] == [(c['cell_type'],''.join(c['source'])) for c in remote['cells']]
    try:
        competition_guard(prior)
    except AssertionError as exc:
        guard_failure = str(exc)
    else:
        raise AssertionError('Expected the old competition-entry refusal')
    old_config = runtime_config(prior, True)
    assert old_config['per_game_s'] == 7254  # reproduced short-budget override
    assert runtime_config(prior,False)['per_game_s'] == 7254
    candidate = json.loads(original_bytes)
    edits = {4: (GUARD, ''), 16: (OVERRIDE, '\n'), 20: (OLD_END, SMOKE_END)}
    changes = []
    for index,(old,new) in edits.items():
        text=source(candidate,index)
        assert text.count(old)==1, (index,'anchor drift')
        revised=text.replace(old,new)
        candidate['cells'][index]['source']=revised.splitlines(True)
        changes.extend(difflib.unified_diff(text.splitlines(True),revised.splitlines(True),
                       fromfile=f'prior/cell-{index}',tofile=f'candidate/cell-{index}'))
    for cell in candidate['cells']:
        if cell['cell_type']=='code':
            cell['outputs']=[]
            cell['execution_count']=None
    changed=[i for i in range(len(prior['cells'])) if source(prior,i)!=source(candidate,i)]
    assert changed==[4,16,20]
    # The frozen loader, embedded pruning mask, sampling, context, and serving cell are unchanged.
    assert source(prior,12)==source(candidate,12)
    assert competition_guard(candidate) is True
    fixed_competition = runtime_config(candidate,True)
    assert fixed_competition==dict(per_game_s=31920,concurrency=120,passes=1,save_request_logs=False)
    assert runtime_config(candidate,False)==runtime_config(prior,False)
    assert deadline(candidate,True) is None
    smoke_s=deadline(candidate,False)
    assert 899 <= smoke_s <= 901
    for index,cell in enumerate(candidate['cells']):
        if cell['cell_type']!='code': continue
        text=source(candidate,index)
        if text.startswith('%%writefile '): continue  # this is unchanged literal patch data
        text='\n'.join('pass' if l.lstrip().startswith('!') else l for l in text.splitlines())
        compile(text,f'<candidate-cell-{index}>','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    original_meta=json.loads((PRIOR.parent/'kernel-metadata.json').read_text())
    meta={**original_meta,'id':SLUG,'title':SLUG.split('/')[1],
          'code_file':SLUG.split('/')[1]+'.ipynb'}
    dest=HERE/'kernel'; dest.mkdir(exist_ok=True)
    candidate_path=dest/meta['code_file']
    candidate_path.write_text(json.dumps(candidate,indent=1)+'\n')
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (HERE/'packaging.patch').write_text(''.join(changes))
    report=dict(status='PREPARED_NOT_LAUNCHED_NOT_QUEUED',parent_sha256=PRIOR_SHA,
                candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256((dest/'kernel-metadata.json').read_bytes()).hexdigest(),
                remote_parent_code_cells_identical=True,changed_cells=changed,
                original_competition_guard_failure=guard_failure,original_competition_runtime=old_config,
                candidate_competition_runtime=fixed_competition,interactive_config_unchanged=True,
                candidate_competition_soft_end=None,interactive_play_s=smoke_s,
                frozen_serving_cell_sha256=hashlib.sha256(source(candidate,12).encode()).hexdigest(),
                qualification_gpu_cap_s=2700,planned_hidden_date='2026-10-20',planned_hidden_draw='D11',
                remaining_gate='One COMPLETE remote qualification of the modified code, then exact-version queueing.',
                no_remote_writes=True)
    (HERE/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
