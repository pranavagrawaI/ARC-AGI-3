"""Read-only Kaggle/source and competition-deadline verification; never launches or submits."""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KERNEL = 'pranavagrawai/m2-f1-screen1-f1'


def code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def main():
    api = KaggleApi()
    api.authenticate()
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = KERNEL.split('/')
    with api.build_kaggle_client() as client:
        remote = json.loads(str(client.kernels.kernels_api_client.get_kernel(req)))
    meta = remote['metadata']
    assert meta['ref'] == KERNEL and meta['currentVersionNumber'] == 1
    assert meta['id'] == 139171846 and meta['isPrivate']
    (HERE / 'remote-identity.json').write_text(json.dumps(meta, indent=2) + '\n')
    source = remote['blob']['source']
    remote_nb = json.loads(source)
    (HERE / 'remote/source.ipynb').write_text(source)
    local = ROOT / 'solvers/m2_f1/experiment/screen-1/f1/kernel/m2-f1-screen1-f1.ipynb'
    screened = json.loads(local.read_text())
    local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
    launch = json.loads((local.parents[1] / 'launch-attempt.json').read_text())
    assert local_hash == launch['identity']['notebook_sha256']
    assert code(remote_nb) == code(screened), 'remote executable differs from screened launch'
    production = json.loads((ROOT / 'solvers/m2_f1/kernel/m2-f1.ipynb').read_text())
    screen_end = 'soft_end = datetime.now() + timedelta(seconds=60*60)  # SCREEN: 60 min of play after server startup'
    full_end = 'soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))'
    assert sum(s.count(screen_end) for s in code(remote_nb)) == 1
    assert [s.replace(screen_end, full_end) for s in code(remote_nb)] == code(production)
    run_cell = next(s for s in code(remote_nb) if 'await bm.run(' in s)
    tree = ast.parse(run_cell)
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'soft_end' for t in n.targets))
    deadline = ast.Module(body=tree.body[start:start+2], type_ignores=[])
    assert isinstance(deadline.body[1], ast.If)
    settings = {'datetime': dt.datetime, 'timedelta': dt.timedelta,
                'target': SimpleNamespace(max_runtime_s=32400.0)}
    outcomes = {}
    for competition in (True, False):
        env = dict(settings, TRUE_SUBMISSION=competition)
        before = dt.datetime.now()
        exec(compile(deadline, '<actual-notebook-deadline>', 'exec'), env)
        if competition:
            assert env['soft_end'] is None
            outcomes['competition_soft_end'] = None
        else:
            seconds = (env['soft_end'] - before).total_seconds()
            assert 3599 <= seconds <= 3601
            outcomes['interactive_soft_end_seconds'] = seconds
    result = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'kernel': KERNEL,
              'version': 1, 'kernel_id': meta['id'], 'code_cells_verified': len(code(remote_nb)),
              'screened_launch_sha256': local_hash, 'matches_screened_launch': True,
              'only_production_difference': 'interactive soft deadline; competition branch unaffected',
              'deadline_execution': outcomes,
              'source_readback_note': 'Version-qualified CLI pull returned HTTP 403 twice; unqualified GetKernel succeeded and explicitly returned currentVersionNumber=1.'}
    (HERE / 'VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
