"""Prepare a notebook/code packet locally. Never calls a platform API or launches."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import release_runner as supervisor
import bootstrap_patch


IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c'


def build(spec, destination, assignment=None):
    config = supervisor.validate(spec['config'], check_cwd=False)
    supervisor.require(config['mode'] == 'GPU_SCREEN', 'packet requires GPU screen mode')
    metadata = dict(spec['metadata'])
    expected = dict(language='python', kernel_type='notebook', is_private=True, enable_gpu=True,
                    enable_tpu=False, enable_internet=False, docker_image=IMAGE, machine_shape='NvidiaRtxPro6000',
                    competition_sources=['arc-prize-2026-arc-agi-3'])
    for key, value in expected.items():
        supervisor.require(metadata.get(key) == value, f'tested allocation envelope mismatch: {key}')
    supervisor.require(not {'pinning_type', 'docker_image_pinning_type'} & metadata.keys(), 'unreviewed pinning type')
    supervisor.require(metadata.get('code_file', '').endswith('.ipynb') and '/' not in metadata['code_file'], 'notebook filename required')
    files = {}
    for name, path in spec['files'].items():
        supervisor.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe packet filename')
        files[name] = Path(path).read_bytes()
    for name in ('supervisor.py', 'http_once.py'):
        supervisor.require(name not in files, 'cannot override reviewed runner')
        files[name] = Path(__file__).with_name(name).read_bytes()
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    sources_sha = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    if assignment is not None:
        supervisor.require(assignment.get('launch_authorized') is True and assignment.get('serialized_assignment'), 'serialized assignment missing')
        supervisor.require(assignment.get('config_sha256') == config_sha and assignment.get('total_seconds') == config['total_seconds'], 'assignment mismatch')
        supervisor.require(assignment.get('sources_sha256') == sources_sha, 'assignment source freeze mismatch')
    payload = {name: dict(base64=base64.b64encode(data).decode(), sha256=hashlib.sha256(data).hexdigest()) for name, data in files.items()}
    source = '''# Prepared locally; no upload or automatic retry.
import base64, datetime, hashlib, json, pathlib, runpy, sys
PAYLOAD = __PAYLOAD__
CONFIG = __CONFIG__
ASSIGNMENT = __ASSIGNMENT__
if ASSIGNMENT is None:
    raise RuntimeError('PREPARATION ONLY: no reviewed serialized assignment; do not execute GPU work')
if ASSIGNMENT.get('launch_authorized') is not True or not ASSIGNMENT.get('serialized_assignment'):
    raise RuntimeError('false/missing launch authority')
if ASSIGNMENT.get('kernel_ref') != __KERNEL_REF__ or ASSIGNMENT.get('platform_hard_timeout_s') != 5400:
    raise RuntimeError('successor identity/platform cap mismatch')
age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(ASSIGNMENT['confirmed_utc'])).total_seconds()
if not 0 <= age < 600:
    raise RuntimeError('stale/future assignment')
source_hashes = {name: record['sha256'] for name, record in PAYLOAD.items()}
if hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest() != ASSIGNMENT['sources_sha256']:
    raise RuntimeError('assignment source freeze mismatch')
if hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest() != ASSIGNMENT['config_sha256'] or CONFIG['total_seconds'] != ASSIGNMENT['total_seconds']:
    raise RuntimeError('assignment configuration/cap mismatch')
if ASSIGNMENT.get('local_only') is not False:
    raise RuntimeError('LOCAL ONLY fixture: no extraction or GPU execution')
root = pathlib.Path(CONFIG['cwd'])
root.mkdir(parents=True, exist_ok=True)
with (root / '.one-attempt').open('x') as marker: marker.write(ASSIGNMENT['serialized_assignment'])
for name, record in PAYLOAD.items():
    data = base64.b64decode(record['base64'], validate=True)
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise RuntimeError('embedded source bytes mismatch')
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f: f.write(data)
(root / 'config.json').write_text(json.dumps(CONFIG, sort_keys=True))
(root / 'assignment.json').write_text(json.dumps(ASSIGNMENT))
sys.argv = [str(root / 'supervisor.py'), str(root / 'config.json'), str(root / 'output'), '--assignment', str(root / 'assignment.json')]
runpy.run_path(str(root / 'supervisor.py'), run_name='__main__')
'''.replace('__PAYLOAD__', repr(payload)).replace('__CONFIG__', repr(config)).replace('__ASSIGNMENT__', repr(assignment)).replace('__KERNEL_REF__',repr(metadata['id']))
    source = bootstrap_patch.apply(source)
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
                    cells=[dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=source.splitlines(keepends=True))])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / metadata['code_file']).write_text(json.dumps(notebook, indent=1) + '\n')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    manifest = dict(run_id=config['run_id'], purpose=config['purpose'], config_sha256=config_sha,
                    sources_sha256=sources_sha,
                    launch_authorized=assignment is not None, total_seconds=config['total_seconds'],
                    files={name: record['sha256'] for name, record in payload.items()},
                    qualification='PACKET_STRUCTURE_ONLY; runtime argv/assets remain the owner review responsibility')
    manifest['packet_files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}
    (destination / 'packet-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    parser.add_argument('destination')
    parser.add_argument('--assignment')
    args = parser.parse_args()
    assignment = json.loads(Path(args.assignment).read_text()) if args.assignment else None
    print(json.dumps(build(json.loads(Path(args.spec).read_text()), args.destination, assignment), indent=2))
