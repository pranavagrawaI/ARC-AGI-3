import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def require_regular(path, label):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f'{label} is not one mounted regular file: {path}')
    return path

input_root = Path('/kaggle/input')

def unique_exact_file(name, expected_sha256, label):
    matches = [
        path
        for path in input_root.rglob(name)
        if path.is_file()
        and not path.is_symlink()
        and sha256_file(path) == expected_sha256
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f'expected one exact {label} under {input_root}, found {matches}'
        )
    return matches[0]

mounted_wheel_path = unique_exact_file(
    'sglang-0.5.19.dev591g8e04f66a7-cp312-cp312-linux_x86_64.whl',
    '06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0',
    'provider-normalized patched wheel',
)
dataset_manifest_path = unique_exact_file(
    'runtime-dataset-manifest.json',
    'f1a50bb224dc26ac48c0fd219c28e4fadc92ede6d2a80a26fd18d9a004a79b94',
    'patch dataset manifest',
)
if mounted_wheel_path.parent != dataset_manifest_path.parent:
    raise RuntimeError(
        'exact patched wheel and dataset manifest are not co-located: '
        f'wheel={mounted_wheel_path}, manifest={dataset_manifest_path}'
    )
patch_root = mounted_wheel_path.parent
lock_path = require_regular(patch_root / 'requirements.lock', 'patch lock')
if sha256_file(lock_path) != 'd473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c':
    raise RuntimeError('mounted patch lock does not match the inherited bytes')
dataset_manifest = json.loads(dataset_manifest_path.read_text())
if (
    dataset_manifest.get('schema') != 'sglang-r8l3b-thin-runtime-dataset-v1'
    or dataset_manifest.get('dataset', {}).get('source')
    != 'pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime'
):
    raise RuntimeError('mounted patch dataset manifest has the wrong semantic identity')
manifest_files = {record['path']: record for record in dataset_manifest.get('files', [])}
local_wheel_record = manifest_files.get('sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl')
if (
    local_wheel_record is None
    or local_wheel_record.get('sha256') != '06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0'
):
    raise RuntimeError('patch dataset manifest does not bind the canonical wheel bytes')

base_wheel_path = unique_exact_file(
    'sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl',
    '9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d',
    'base SGLang wheel',
)
base_lock_paths = [
    path
    for path in input_root.rglob('requirements.lock')
    if path.is_file()
    and not path.is_symlink()
    and path != lock_path
    and sha256_file(path) == 'd473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c'
]
if len(base_lock_paths) != 1:
    raise RuntimeError(
        'expected one exact non-patch base lock under the input root, '
        f'found locks={base_lock_paths}'
    )
base_lock_path = base_lock_paths[0]
if lock_path.read_bytes() != base_lock_path.read_bytes():
    raise RuntimeError('patch lock bytes differ from the inherited base lock')
wheelhouse = base_wheel_path.parent
print(json.dumps({
    'resolved_patch_root': str(patch_root),
    'resolved_patch_wheel': str(mounted_wheel_path),
    'resolved_base_wheel': str(base_wheel_path),
    'resolved_base_lock': str(base_lock_path),
    'resolved_wheelhouse': str(wheelhouse),
}, sort_keys=True))
bootstrap_dir = Path('/kaggle/working/qwen38-fp8-r8l3b-bootstrap')
bootstrap_dir.mkdir(parents=True, exist_ok=False)
canonical_wheel_path = bootstrap_dir / 'sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl'
shutil.copyfile(mounted_wheel_path, canonical_wheel_path)
require_regular(canonical_wheel_path, 'canonical copied patched wheel')
if sha256_file(canonical_wheel_path) != '06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0':
    raise RuntimeError('canonical copied SGLang wheel does not match the pinned bytes')
runtime_provenance = Path('/tmp/sglang-runtime-provenance.json')
runtime_provenance.write_text(json.dumps({
    'dataset_source': 'pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime',
    'dataset_version': 1,
    'base_dataset_source': 'mahdiseddigh/kaggle-t4-stack-2026-08-28',
    'base_dataset_version': 1,
    'repository': 'https://github.com/sgl-project/sglang.git',
    'commit': '8e04f66a7046fa083cda3f5b8e8325632a5af8fd',
    'artifact_path': str(canonical_wheel_path),
    'artifact_sha256': '06cc48d1e067c0eb08a1548c5c244bd0f541b114947c29b87042ce3156bb5cc0',
    'mounted_artifact_path': str(mounted_wheel_path),
    'mounted_artifact_name': 'sglang-0.5.19.dev591g8e04f66a7-cp312-cp312-linux_x86_64.whl',
    'canonical_artifact_name': 'sglang-0.5.19.dev591+g8e04f66a7-cp312-cp312-linux_x86_64.whl',
    'remote_dataset_readback_sha256': 'cee470f08217da591b8a89d3a80f1f9b4f03aecffced29f69d1b2a5c510fda9d',
    'lock_path': str(lock_path),
    'lock_sha256': 'd473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c',
    'dataset_manifest_path': str(dataset_manifest_path),
    'dataset_manifest_sha256': 'f1a50bb224dc26ac48c0fd219c28e4fadc92ede6d2a80a26fd18d9a004a79b94',
    'base_artifact_path': str(base_wheel_path),
    'base_artifact_sha256': '9d07f8bf3b718dc4cd970e72a03415b0b0597da89625acbee7a655d6387e7a8d',
    'base_lock_path': str(base_lock_path),
    'base_lock_sha256': 'd473b6e092e01f66313e12069ac8873f2ff1c6c3b84b51c1b37312229de5475c',
}, indent=2) + '\n')

overrides_path = Path('/tmp/sglang-runtime-overrides.txt')
overrides_path.write_text(
    '\n'.join([
        'flashinfer-python==0.6.17',
        'numba==0.65.1',
        'compressed-tensors==0.18.0',
        'outlines==1.3.3',
        'opencv-python-headless==5.0.0.93',
        'openai==3.5.0',
        '',
    ])
)
uv_install_command = [
    sys.executable,
    '-m',
    'pip',
    'install',
    '--no-index',
    '--find-links',
    str(wheelhouse),
    'uv==0.12.7',
]
with (bootstrap_dir / 'runtime-install.log').open('w') as install_log:
    subprocess.run(
        uv_install_command,
        check=True,
        stdout=install_log,
        stderr=subprocess.STDOUT,
    )
uv_executable = shutil.which('uv')
if not uv_executable:
    raise RuntimeError('uv executable missing after offline installation')
install_command = [
    uv_executable,
    'pip',
    'install',
    '--system',
    '--no-index',
    '--find-links',
    str(wheelhouse),
    '--override',
    str(overrides_path),
    '--upgrade',
    str(canonical_wheel_path),
    'torch==2.13.0+cu130',
    'transformers==5.12.1',
    'flashinfer-python==0.6.17',
    'sglang-kernel==0.4.6.post1',
    'triton==3.7.1',
    'cuda-tile==1.6.0rc5',
    'flash-attn-4==4.0.0b19',
    'opentelemetry-semantic-conventions==0.65b0',
]
(bootstrap_dir / 'runtime-install-command.json').write_text(
    json.dumps({
        'uv_install': uv_install_command,
        'runtime_install': install_command,
        'overrides': overrides_path.read_text().splitlines(),
    }, indent=2) + '\n'
)
with (bootstrap_dir / 'runtime-install.log').open('a') as install_log:
    subprocess.run(
        install_command,
        check=True,
        stdout=install_log,
        stderr=subprocess.STDOUT,
    )
(bootstrap_dir / 'runtime-provenance.json').write_text(
    runtime_provenance.read_text()
)
with (bootstrap_dir / 'runtime-packages.json').open('w') as package_log:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--format', 'json'],
        check=True,
        stdout=package_log,
    )


solver_manifest_path = unique_exact_file(
    'wheelhouse-manifest.json',
    '518f8407b0ab1234430fc0c167e2ecd62ce70318a62292f8397e1a7ea4ed3862',
    'Solver closure manifest',
)
solver_manifest = json.loads(solver_manifest_path.read_text())
if (
    solver_manifest.get('schema') != 'tycho-nooa-full-wheelhouse-v1'
    or solver_manifest.get('artifact_revision') != 11
):
    raise RuntimeError('Solver closure manifest has the wrong identity')
solver_wheelhouse = solver_manifest_path.parent
subprocess.run(
    [sys.executable, str(solver_wheelhouse / 'verify_wheelhouse.py')],
    cwd=solver_wheelhouse,
    check=True,
    timeout=60,
)
solver_wheel_records = [
    record
    for record in solver_manifest['wheels']
    if record['name'] == 'arc-agi-3-tycho-nooa'
]
if len(solver_wheel_records) != 1:
    raise RuntimeError('Solver closure does not declare exactly one Solver wheel')
solver_wheel_record = solver_wheel_records[0]
solver_wheel_path = require_regular(
    solver_wheelhouse / solver_wheel_record['filename'],
    'Solver wheel',
)
if (
    solver_wheel_record['sha256'] != '9ffadfc88ed40667fa14b876b5a17b94ac4992ca277377c887de0db5b0df7faf'
    or sha256_file(solver_wheel_path) != '9ffadfc88ed40667fa14b876b5a17b94ac4992ca277377c887de0db5b0df7faf'
):
    raise RuntimeError('Solver wheel differs from the frozen successor bytes')
uv = shutil.which('uv')
if uv is None:
    raise RuntimeError('Kaggle uv executable is unavailable')
solver_venv = Path('/tmp/solver-venv')
subprocess.run(
    [uv, 'venv', '--python', sys.executable, str(solver_venv)],
    check=True,
    timeout=60,
)
solver_python = solver_venv / 'bin/python'
subprocess.run(
    [
        uv, 'pip', 'install', '--python', str(solver_python), '--offline',
        '--no-cache', '--no-index', '--find-links', str(solver_wheelhouse),
        str(solver_wheel_path),
    ],
    check=True,
    timeout=180,
)
installed_raw = subprocess.check_output(
    [
        str(solver_python), '-c',
        "import importlib.metadata as m,json,re; "
        "c=lambda s:re.sub(r'[-_.]+','-',s).lower(); "
        "print(json.dumps(dict(sorted((c(d.metadata['Name']),d.version) for d in m.distributions()))))",
    ],
    text=True,
    timeout=30,
)
installed = json.loads(installed_raw)
def canonical(value):
    return __import__('re').sub(r'[-_.]+', '-', value).lower()
expected = {canonical(record['name']): record['version'] for record in solver_manifest['wheels']}
if installed != expected:
    raise RuntimeError('installed Solver distributions differ from the frozen manifest')
if 'nooa' not in installed:
    raise RuntimeError('Solver closure is missing the nooa wheel')
join_root = Path('/kaggle/working/nooa-join')
if join_root.exists():
    raise RuntimeError('Join destination exists')
join_root.mkdir(parents=True)
shutil.unpack_archive('/tmp/nooa-join.zip', str(join_root), 'zip')
__import__('os').environ['NOOA_JOIN_ROOT'] = str(join_root)
