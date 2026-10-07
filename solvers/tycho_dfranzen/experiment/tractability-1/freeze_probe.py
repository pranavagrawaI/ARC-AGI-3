"""Assemble one new private branch probe; preserve every earlier frozen payload."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zlib

HERE = Path(__file__).resolve().parent
SOLVER = HERE.parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def pack(files):
    return zlib.compress(json.dumps({name: base64.b64encode(raw).decode()
                                  for name, raw in sorted(files.items())}).encode(), 9)


def freeze():
    if (HERE / 'launch-attempt.json').exists():
        raise RuntimeError('Run attempted: preserve this freeze; do not mutate or retry')
    spec = importlib.util.spec_from_file_location('probe_build', SOLVER / 'build.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    files, provenance, patch = build.assemble(HERE / 'runtime')
    for name in ('branch_probe.py', 'evaluate_actions.py'):
        data = (HERE / name).read_bytes()
        compile(data, name, 'exec')
        files[name] = data
        (HERE / 'runtime' / name).write_bytes(data)
    payload = pack(files)
    packets = HERE / 'packets'
    assert sorted(p.name for p in packets.iterdir() if p.is_dir()) == ['ls20-stall', 're86-stall', 're86-success']
    packet_files = {str(p.relative_to(packets)): p.read_bytes()
                    for p in packets.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    packet_payload = pack(packet_files)
    bundle = HERE / 'source-bundle'
    bundle.mkdir(exist_ok=True)
    (bundle / 'hybrid-runtime.zlib').write_bytes(payload)
    (bundle / 'branch-packets.zlib').write_bytes(packet_payload)
    for notice in ('NOTICE.md', 'LICENSE-APACHE-2.0'):
        shutil.copy2(SOLVER / notice, bundle / notice)
    dataset = 'pranavagrawai/tycho-dfranzen-tractability-1'
    write(bundle / 'dataset-metadata.json', {'id': dataset, 'title': 'Tycho dfranzen tractability 1',
                                          'licenses': [{'name': 'other'}]})
    notebook = json.loads((SOLVER / 'experiment/wave-2/lean_prune/kernel/notebook.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell.update(outputs=[], execution_count=None)
    def source(index):
        return ''.join(notebook['cells'][index]['source'])
    def set_source(index, value):
        notebook['cells'][index]['source'] = value.splitlines(True)
    set_source(0, source(0).replace("'4500'", "'3300'").replace("'4800'", "'3600'")
               .replace('wave-2-lean_prune', 'tractability-1').replace("'arm':'lean_prune'", "'arm':'four-branch-conditions'"))
    set_source(1, '# Terminal presentation and time-aware decisions\n\nPrivate saved-request continuation; no competition submission or full-game score claim.\n')
    install = source(10)
    start = install.index('_hybrid_hashes = ')
    end = install.index('\n_payload_candidates=', start)
    old_hash = install.split("hashlib.sha256(p.read_bytes()).hexdigest()==", 1)[1].split(']', 1)[0].strip().strip("'")
    install = install[:start] + '_hybrid_hashes = ' + repr({name: sha(raw) for name, raw in sorted(files.items())}) + install[end:]
    install = install.replace(old_hash, sha(payload)).replace('tycho-dfranzen-source-v9', 'tycho-dfranzen-tractability-1')
    install += f'''\n# Restore exact saved requests and files; every byte is frozen independently.
PROBE_PACKETS = Path('/kaggle/working/probe-input')
_packets = [p for p in Path('/kaggle/input').rglob('branch-packets.zlib')
            if hashlib.sha256(p.read_bytes()).hexdigest() == {sha(packet_payload)!r}]
if len(_packets) != 1: raise RuntimeError('Exact branch packets required')
for _rel, _encoded in json.loads(zlib.decompress(_packets[0].read_bytes())).items():
    _path = PROBE_PACKETS / _rel
    if not _path.resolve().is_relative_to(PROBE_PACKETS.resolve()): raise ValueError('Unsafe packet path')
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_bytes(base64.b64decode(_encoded))
'''
    install += '''\n# Qualify the inherited Kaggle root/UID sandbox before any model request.
_fixture_env = {**os.environ, 'PYTHONPATH': str(HYBRID_ROOT),
                'TYCHO_KAGGLE_GAME_UID': '60900', 'TYCHO_KAGGLE_GAME_GID': '60900'}
_fixture = subprocess.run([str(HYBRID_PYTHON), str(HYBRID_ROOT / 'branch_probe.py'), 'self-test',
    '--runtime', str(HYBRID_ROOT), '--packet', str(PROBE_PACKETS / 're86-stall'),
    '--output', str(WORKING_DIR / 'branch-preflight')], env=_fixture_env,
    capture_output=True, text=True, timeout=90)
(WORKING_DIR / 'branch-preflight.log').write_text(_fixture.stdout + _fixture.stderr)
if _fixture.returncode: raise RuntimeError('Kaggle branch preflight failed; no model requests launched')
'''
    set_source(10, install)
    set_source(16, '# Saved requests replace the benchmark; no full-game solver is constructed.\n')
    set_source(18, '# Four arms, three packets, two replicas; fixed in EXPERIMENT-CARD.md.\n')
    set_source(22, '''# Run bounded continuation cases on the same local serving allocation.
assert not TRUE_SUBMISSION
_ready_deadline = _qualification_started_monotonic + 1800
while True:
    try:
        with urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/health', timeout=2) as _health:
            if _health.status == 200: break
    except OSError:
        pass
    if time.monotonic() >= _ready_deadline: raise RuntimeError('Serving did not become ready within startup allocation')
    await __import__('asyncio').sleep(2)
with urlopen(SERVER_BASE_URL + '/models', timeout=5) as _models_response:
    _models = json.load(_models_response)
if SERVED_MODEL_NAME not in [m['id'] for m in _models['data']]: raise RuntimeError('Unexpected served model identity')
(WORKING_DIR/'serving-readiness.json').write_text(json.dumps({'models':_models,'elapsed_seconds':time.monotonic()-_qualification_started_monotonic}))
_probe_output = WORKING_DIR / 'tractability-output'
_command = [str(HYBRID_PYTHON), str(HYBRID_ROOT / 'branch_probe.py'), 'run',
            '--runtime', str(HYBRID_ROOT), '--packets', str(PROBE_PACKETS),
            '--output', str(_probe_output), '--endpoint', SERVER_BASE_URL,
            '--seconds', '180', '--tokens', '25000', '--replicas', '2']
_probe_process = subprocess.Popen(_command, env={**os.environ, 'PYTHONPATH': str(HYBRID_ROOT)})
_fields = Path(f'/proc/{_probe_process.pid}/stat').read_text().rsplit(') ',1)[1].split()
(_qualification_root/'qualification-runner.json').write_text(json.dumps({'pid':_probe_process.pid,'start':int(_fields[19])}))
try:
    while _probe_process.poll() is None:
        await __import__('asyncio').sleep(1)
    if _probe_process.returncode: raise RuntimeError(f'Branch suite exited {_probe_process.returncode}')
finally:
    if _probe_process.poll() is None:
        _probe_process.terminate()
        try: _probe_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _probe_process.kill(); _probe_process.wait(timeout=5)
# Score commitments only after prefix agreement with the actual offline engine.
_env_dir = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files'
for _case in _probe_output.glob('block-*/*/result.json'):
    _result = json.loads(_case.read_text())
    if _result.get('action') is None: continue
    _evaluation = subprocess.run([str(HYBRID_PYTHON), str(HYBRID_ROOT / 'evaluate_actions.py'),
                    str(PROBE_PACKETS / _result['packet']), str(_case), _env_dir,
                    str(_case.with_name('engine-evaluation.json'))],
                   env={**os.environ, 'PYTHONPATH': str(HYBRID_ROOT)}, timeout=90,
                   capture_output=True, text=True)
    _case.with_name('engine-evaluation.log').write_text(_evaluation.stdout + _evaluation.stderr)
''')
    set_source(24, "print('Branch outputs: /kaggle/working/tractability-output')\n")
    set_source(25, '''import hashlib
_probe_files=[{'path':str(p.relative_to(WORKING_DIR)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes':p.stat().st_size}
              for p in sorted(WORKING_DIR.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
              and 'tractability-output' in p.parts]
(WORKING_DIR/'probe-evidence-index.json').write_text(json.dumps({'files':_probe_files},indent=2))
(WORKING_DIR/'qualification-complete.json').write_text(json.dumps({'run_id':'tractability-1','elapsed_seconds':time.monotonic()-_qualification_started_monotonic,
 'cases':len(list(_probe_output.glob('block-*/*/result.json')))}))
''')
    kernel = HERE / 'kernel'
    kernel.mkdir(exist_ok=True)
    write(kernel / 'notebook.ipynb', notebook)
    metadata = json.loads((SOLVER / 'experiment/wave-2/lean_prune/kernel/kernel-metadata.json').read_text())
    metadata.update(id='pranavagrawai/tycho-dfranzen-tractability-1', title='Tycho dfranzen tractability 1')
    metadata['dataset_sources'] = [dataset if item == 'pranavagrawai/tycho-dfranzen-source-v9' else item for item in metadata['dataset_sources']]
    write(kernel / 'kernel-metadata.json', metadata)
    identity = {'run_id': 'tractability-1', 'kernel': metadata['id'], 'hard_timeout_seconds': 3600,
                'source_payload_sha256': sha(payload), 'packets_payload_sha256': sha(packet_payload),
                'notebook_sha256': sha((kernel / 'notebook.ipynb').read_bytes()),
                'metadata_sha256': sha((kernel / 'kernel-metadata.json').read_bytes()),
                'card_sha256': sha((HERE / 'EXPERIMENT-CARD.md').read_bytes()),
                'source_files': {name: sha(raw) for name, raw in sorted(files.items())},
                'packet_files': {name: sha(raw) for name, raw in sorted(packet_files.items())}}
    write(HERE / 'identity.json', identity)
    (HERE / 'canonical-candidate.patch').write_text(patch)
    return identity


if __name__ == '__main__':
    print(json.dumps(freeze(), indent=2))
