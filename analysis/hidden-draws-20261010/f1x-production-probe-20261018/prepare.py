"""Build an unlaunched diagnostic from the frozen F1x artifact; never upload."""
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
PARENT = ROOT / 'solvers/m2_f1/experiment/screen-2/f1x/kernel/m2-f1-screen2-f1x.ipynb'
PARENT_SHA = '826dfdc2a483a786bdfe888ad9b9422630bc4c615a187b34bdc9762975fb7df9'
SIMULATOR = ROOT / 'analysis/m2-winners-20261001/audit/dfranzen-repo/tufa-arc-agi-framework/src/taaf/competition_arcade.py'
SLUG = 'f1x-production-probe-20261018'


def source(cell):
    return ''.join(cell['source'])


def main():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    if (HERE / 'launch-attempt.json').exists():
        raise RuntimeError('Preserve the launched artifact; create a successor for any change.')
    nb = json.loads(PARENT.read_text())
    diff = []
    for index, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        original = revised = source(cell)
        if 'TRUE_SUBMISSION = os.environ.get(' in original:
            old = 'TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}'
            new = ('if os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}:\n'
                   '    raise RuntimeError("Diagnostic only; never submit this probe")\n'
                   'TRUE_SUBMISSION = True  # Emulate production flags; local public competition server.')
            assert revised.count(old) == 1
            revised = revised.replace(old, new)
        if 'await bm.run(' in original:
            insert = '''# Probe-only local server, exposing 110 unique clones of public games.
import types
_probe_gateway = types.ModuleType('_f1x_probe_gateway')
sys.modules[_probe_gateway.__name__] = _probe_gateway
exec(compile(SIM_SOURCE, '<probe-gateway>', 'exec'), _probe_gateway.__dict__)
_probe_server = _probe_gateway.CompetitionArcadeServer.official_110()
_probe_server.environments_dir = str(Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files'))
_probe_server.start()
os.environ['ARC_BASE_URL'] = _probe_server.base_url + '/'
os.environ['ARC_API_KEY'] = _probe_server.api_key
_probe_config = dict(emulated_competition=True, actual_competition_rerun=False,
    concurrency=bm.solver.concurrency, save_request_logs=bm.solver.save_request_logs,
    minimal_diagnostics=os.environ['TAAF_MINIMAL_DIAGNOSTICS'],
    games=len(_probe_server.exposed_game_ids), play_cap_s=900)
assert _probe_config['concurrency'] == 120 and _probe_config['save_request_logs'] is False
assert _probe_config['minimal_diagnostics'] == '1' and _probe_config['games'] == 110
(WORKING_DIR / 'probe-config.json').write_text(json.dumps(_probe_config, indent=2))
print('PROBE_CONFIG', json.dumps(_probe_config), flush=True)
'''.replace('SIM_SOURCE', repr(SIMULATOR.read_text()))
            anchor = 'if TRUE_SUBMISSION:\n    # Real submission: play the live competition Arcade served by the Kaggle gateway.'
            assert revised.count(anchor) == 1
            revised = revised.replace(anchor, insert + '\n' + anchor)
            old = 'await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)'
            assert revised.count(old) == 1
            revised = revised.replace(old, (HERE / 'runtime.py').read_text() + '\n'
                'soft_end = datetime.now() + timedelta(seconds=900)  # Probe-only cap.\n'
                'await probe_run(bm, target, WORKING_DIR, soft_end, _probe_server)')
        if original != revised:
            diff.extend(difflib.unified_diff(original.splitlines(True), revised.splitlines(True),
                        fromfile=f'parent/cell-{index}', tofile=f'probe/cell-{index}'))
        cell['source'] = revised.splitlines(True)
        cell['outputs'] = []
        cell['execution_count'] = None
        if not revised.startswith('%%writefile'):
            python = '\n'.join('pass' if line.lstrip().startswith('!') else line for line in revised.splitlines())
            compile(python, f'probe-cell-{index}', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    config = next(source(c) for c in nb['cells'] if 'bm.solver.save_request_logs = True' in source(c))
    bm = SimpleNamespace(solver=SimpleNamespace())
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(config, '<actual-config>', 'exec'),
             {'bm': bm, 'TRUE_SUBMISSION': True, 'USE_PRIORITY_SCHEDULING': True})
    assert bm.solver.concurrency == 120 and bm.solver.save_request_logs is False
    assert bm.solver.max_runtime_s_per_game == 31920
    changed = [i for i, (a, b) in enumerate(zip(json.loads(PARENT.read_text())['cells'], nb['cells']))
               if source(a) != source(b)]
    assert changed == [5, 23], changed
    dest = HERE / 'kernel'
    dest.mkdir(exist_ok=True)
    path = dest / (SLUG + '.ipynb')
    path.write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads((PARENT.parent / 'kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/' + SLUG, title='F1x production-path diagnostic', code_file=path.name)
    (dest / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    (HERE / 'probe.patch').write_text(''.join(diff))
    result = {'status': 'PREPARED_NOT_LAUNCHED', 'parent_sha256': PARENT_SHA,
              'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'changed_cells': changed, 'f1x_model_serving_and_sandbox_unchanged': True,
              'production_config_executed_locally': vars(bm.solver),
              'simulator_sha256': hashlib.sha256(SIMULATOR.read_bytes()).hexdigest(),
              'gpu_cap_s': 2700, 'play_cap_s': 900,
              'limitations': ['No real Kaggle gateway or hidden games.',
                              'F1x does not include F1-only prompt and counter changes.',
                              'A pass establishes bounded operation, not nine-hour stability or hidden efficacy.']}
    (HERE / 'PREPARATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
