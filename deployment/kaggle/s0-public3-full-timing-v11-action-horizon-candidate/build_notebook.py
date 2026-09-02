"""Build the frozen V11 action-horizon full-timing qualification candidate."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V14 = (
    ROOT
    / "analysis/deployment-probes/20260831T160520Z-qwen38-r8l3b-kaggle-v14"
    / "remote-source-v14/prototype-qwen3-8-fp8-r8-b-request-path.ipynb"
)
OUTPUT = HERE / "s0_public3_full_timing.ipynb"
RUNTIME = HERE / "exact_request_outer_runtime.py"
PROBE = HERE / "public25_launcher_probe.py"
ADAPTER = HERE.parent / "agent/my_agent.py"
LOCAL_SOLVER = HERE.parent / "runtime/local_solver.py"
RUNNER = HERE.parent / "runtime/run_official_agent.py"
PUBLIC_ENV_MANIFEST = (
    ROOT
    / "analysis/deployment-probes/20260901T054322Z-public25-kaggle-v1"
    / "public-environment-manifest.json"
)

WHEELHOUSE_MANIFEST_SHA256 = (
    "e95b8176e9b445059f858d511559017bcc6452eba96c1cfd4805e5c1d1d2c187"
)
SOLVER_WHEEL_SHA256 = (
    "0ba1282ea11b679524fce1f813da21c12132f3a9b5a48e66eba4034fbc8f1322"
)
PROBE_SHA256 = "56e627ca9aa4281f36ff5b8f9ea8894a833366679a8e4fecddaf836af47da0bf"
RUNTIME_SHA256 = "c7f5f2845a777ada6c928dc54a3934ce26b34a28914124b849a1d5c4dd41a3ae"
OFFICIAL_REVISION = "135f20aaf44f13341ebc425666bf03b5cac58d3c"
OFFICIAL_SOURCE_HASHES = {
    ".env.example": "ffa415e8a6b5e8e29b69fc351faba6161e588f550daaca950a7fee1687a0c3af",
    ".git/logs/HEAD": "edf83ad6d2902dc27ab28542e7fba3c3af3d8506f3f3b54524f1ea42386d513e",
    "agents/agent.py": "85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087",
    "agents/recorder.py": "0a08d89f4067a760012767c05d4406bd2bf409f426e29a1193106abfcbb696c8",
    "agents/swarm.py": "d9dc48f710f1b90a6552db0921293c7e89c8a925ed00a3faefa07ae19998ad39",
    "agents/templates/random_agent.py": "e5e3610fd56a4819572cc0f0d768cff426ac8b935b549735bb396aefc810705e",
    "agents/tracing.py": "951ca56508c524504e116303f7c64f4eb5cf723c72cab892d4d1a3292b1cc51f",
    "main.py": "864254c750bbbd12a211f2d8aa1b1025d0609283f07dea4ede83722f2435301b",
}


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    if sha256(PROBE) != PROBE_SHA256:
        raise RuntimeError("full-timing launcher probe drifted from its inherited bytes")
    if sha256(RUNTIME) != RUNTIME_SHA256:
        raise RuntimeError("exact-request outer runtime drifted from frozen V10 bytes")
    notebook = json.loads(V14.read_text())
    notebook["cells"][0]["source"] = (
        "# FULL SUCCESSOR-TIMING QUALIFIER — up to 32 decisions on three public games\n\n"
        "This runs the accepted R8-L3B v14 model-server route, exact private "
        "Solver wheelhouse v8 with the action-horizon contract, current MyAgent "
        "adapter, and hash-pinned official ARC-AGI-3-Agents launcher over s5i5, "
        "ls20, and sb26. Sequence 0 may use six Actor calls to form an initial model; "
        "every realized successor receives exactly one required action call at the same 16,000-token ceiling. "
        "Every model request states the exact 32-action used/remaining horizon; sequence 31 says its result is recorded but no next model decision is guaranteed. "
        "The outer runtime reconciles persisted Actor calls against exact SGLang accesses. "
        "This measures continued public gameplay, latency, and level progress, not hidden-game performance. It refuses `KAGGLE_IS_COMPETITION_RERUN` and creates "
        "no competition submission."
    )

    assets = "".join(notebook["cells"][1]["source"])
    for destination, source in (
        ("/tmp/official_launcher_runtime.py", RUNTIME),
        ("/tmp/public25_launcher_probe.py", PROBE),
        ("/tmp/my_agent.py", ADAPTER),
        ("/tmp/local_solver.py", LOCAL_SOLVER),
        ("/tmp/run_official_agent.py", RUNNER),
        ("/tmp/public-environment-manifest.json", PUBLIC_ENV_MANIFEST),
    ):
        assets += (
            f"\nPath({destination!r}).write_bytes("
            f"base64.b64decode({encoded(source)!r}))\n"
        )
    notebook["cells"][1]["source"] = assets

    install = "".join(notebook["cells"][2]["source"])
    install += f"""

# Verify and install the independently frozen Solver closure. The model server
# keeps the accepted R8-L3B environment; the official launcher and worker use
# this separate, non-system-site CPython 3.12 environment.
solver_manifest_path = unique_exact_file(
    'wheelhouse-manifest.json',
    {WHEELHOUSE_MANIFEST_SHA256!r},
    'Solver closure manifest',
)
solver_manifest = json.loads(solver_manifest_path.read_text())
if (
    solver_manifest.get('schema') != 'tycho-nooa-full-wheelhouse-v1'
    or solver_manifest.get('artifact_revision') != 8
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
    solver_wheel_record['sha256'] != {SOLVER_WHEEL_SHA256!r}
    or sha256_file(solver_wheel_path) != {SOLVER_WHEEL_SHA256!r}
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

expected = {{canonical(record['name']): record['version'] for record in solver_manifest['wheels']}}
if installed != expected:
    raise RuntimeError('installed Solver distributions differ from the frozen manifest')
if 'include-system-site-packages = false' not in (solver_venv / 'pyvenv.cfg').read_text():
    raise RuntimeError('Solver venv admits ambient system packages')
source_manifest_path = require_regular(
    solver_wheelhouse / 'solver-source-manifest.json',
    'Solver source manifest',
)
source_manifest = json.loads(source_manifest_path.read_text())
site_packages = next((solver_venv / 'lib').glob('python*/site-packages'))
source_mismatches = []
for record in source_manifest['packaged_sources']:
    installed_source = site_packages / record['wheel_path']
    actual_hash = (
        sha256_file(installed_source) if installed_source.is_file() else None
    )
    if actual_hash != record['sha256']:
        source_mismatches.append({{
            'wheel_path': record['wheel_path'],
            'expected_sha256': record['sha256'],
            'actual_sha256': actual_hash,
        }})
if source_mismatches:
    raise RuntimeError(
        f'installed Solver source closure differs: {{source_mismatches}}'
    )
Path('/tmp/solver-closure-provenance.json').write_text(json.dumps({{
    'dataset': 'pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v8',
    'dataset_version': 1,
    'manifest_sha256': {WHEELHOUSE_MANIFEST_SHA256!r},
    'wheel_count': len(solver_manifest['wheels']),
    'installed_count': len(installed),
    'installed_source_count': len(source_manifest['packaged_sources']),
    'solver_wheel_sha256': {SOLVER_WHEEL_SHA256!r},
    'include_system_site_packages': False,
}}, indent=2) + '\\n')
"""
    notebook["cells"][2]["source"] = install

    notebook["cells"][3]["source"] = f"""import hashlib as official_hashlib
import json
import os
import pandas as pd
import shutil
import subprocess
import sys
from pathlib import Path

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    raise RuntimeError('public-25 qualifier refuses competition rerun mode')

# Full-timing qualification: initial model acquisition followed by bounded
# one-call successor decisions. Single mode deliberately omits automatic Builder.
os.environ['TYCHO_PUBLIC_GAME_IDS'] = 'ls20,s5i5,sb26'
os.environ['TYCHO_KAGGLE_MAX_ACTIONS'] = '32'
os.environ['ARC_LLM_CONTEXT_WINDOW'] = '32768'
os.environ['ARC_LLM_FORCE_ACTION_TOOL'] = '1'
os.environ['ARC_LLM_MAX_ACTOR_CALLS'] = '6'
os.environ['ARC_LLM_MAX_ACTOR_CALLS_AFTER_INITIAL'] = '1'
os.environ['ARC_LLM_MAX_TOKENS'] = '16000'
os.environ['ARC_LLM_MAX_TOKENS_AFTER_INITIAL'] = '16000'
os.environ['ARC_LLM_RETRIES'] = '0'
os.environ['ARC_LLM_TIMEOUT_SECONDS'] = '600'
os.environ['TYCHO_MODE'] = 'single'
os.environ['TYCHO_NOOA_TREATMENT'] = 'single_decision_discipline'
os.environ['TYCHO_NOOA_WORKER_TIMEOUT_SECONDS'] = '7200'

competition_root = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3')
public_root = competition_root / 'environment_files'
public_manifest = json.loads(Path('/tmp/public-environment-manifest.json').read_text())
if public_manifest.get('game_count') != 25 or public_manifest.get('file_count') != 50:
    raise RuntimeError('public environment manifest has the wrong inventory')
declared_public_paths = set()
for record in public_manifest['files']:
    relative = Path(record['path']).relative_to('environment_files')
    path = public_root / relative
    declared_public_paths.add(relative.as_posix())
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record['bytes']
        or official_hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']
    ):
        raise RuntimeError(f'public environment drift: {{relative}}')
actual_public_paths = {{
    path.relative_to(public_root).as_posix()
    for path in public_root.rglob('*')
    if path.is_file()
}}
if actual_public_paths != declared_public_paths:
    raise RuntimeError('mounted public environment inventory differs from manifest')

source_root = competition_root / 'ARC-AGI-3-Agents'
official_root = Path('/kaggle/working/ARC-AGI-3-Agents')
if official_root.exists():
    shutil.rmtree(official_root)
shutil.copytree(source_root, official_root, symlinks=True)
expected_source_hashes = {OFFICIAL_SOURCE_HASHES!r}
for relative, expected_hash in expected_source_hashes.items():
    path = official_root / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f'official source file is absent or a symlink: {{relative}}')
    actual_hash = official_hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f'official source drift: {{relative}} {{actual_hash}}')
head_log = (official_root / '.git/logs/HEAD').read_text()
if {OFFICIAL_REVISION!r} not in head_log:
    raise RuntimeError('official source revision is not the pinned competition revision')

templates = official_root / 'agents/templates'
templates.mkdir(parents=True, exist_ok=True)
shutil.copy2('/tmp/my_agent.py', templates / 'my_agent.py')
shutil.copy2('/tmp/local_solver.py', official_root / 'local_solver.py')
shutil.copy2('/tmp/run_official_agent.py', '/kaggle/working/run_official_agent.py')
(official_root / 'agents/__init__.py').write_text('''from typing import Type
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.my_agent import MyAgent

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{
    "myagent": MyAgent,
}}
''')
(official_root / '.env').write_text(f'''DEBUG=False
SCHEME=http
HOST=127.0.0.1
PORT=18001
ARC_API_KEY=
ARC_BASE_URL=http://127.0.0.1:18001/
OPERATION_MODE=offline
ENVIRONMENTS_DIR={{public_root}}
RECORDINGS_DIR=/kaggle/working/server_recording
''')

output_dir = Path('/kaggle/working/tycho-nooa-public25')
command = [
    sys.executable,
    '/tmp/official_launcher_runtime.py',
    '--output-dir', str(output_dir),
    '--manifest', '/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json',
    '--input-root', '/kaggle/input',
    '--model-source', 'jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
    '--runtime-provenance', '/tmp/sglang-runtime-provenance.json',
    '--startup-timeout-s', '1200',
    '--request-timeout-s', '180',
    '--launcher-timeout-s', '21600',
    '--attention-backend', 'triton',
    '--sampling-backend', 'pytorch',
    '--profile-imports',
    '--r8-l3b',
    '--official-launcher-probe', '/tmp/public25_launcher_probe.py',
    '--launcher-python', '/tmp/solver-venv/bin/python',
]
subprocess.run(command, check=True)
print((output_dir / 'result.json').read_text())

submission = pd.DataFrame(
    data=[['1_0', '1', True, 1]],
    columns=['row_id', 'game_id', 'end_of_game', 'score'],
)
submission.to_parquet('/kaggle/working/submission.parquet', index=False)
"""
    notebook["metadata"]["kaggle"]["isInternetEnabled"] = False
    notebook["metadata"]["kaggle"]["accelerator"] = "nvidiaRtx6000"
    return notebook


def validate(notebook: dict[str, Any]) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    if sha256(LOCAL_SOLVER) != "517572979803525af52e7a6f2066fdee2eb3ce5923fd2780c5b185cb52b06795":
        raise RuntimeError("local Solver adapter drifted from its manifest")
    if sha256(ADAPTER) != "b0e4e2c94cd895eb7cdb5ebde78df6ff00220ed92ee88d510f5d2692da2d5a8f":
        raise RuntimeError("MyAgent adapter drifted from the frozen candidate")
    if sha256(PROBE) != PROBE_SHA256:
        raise RuntimeError("full-timing launcher probe drifted from its inherited bytes")
    if sha256(RUNTIME) != RUNTIME_SHA256:
        raise RuntimeError("exact-request outer runtime drifted from frozen V11 bytes")


if __name__ == "__main__":
    candidate = build()
    validate(candidate)
    OUTPUT.write_text(json.dumps(candidate, indent=1) + "\n")
    print(OUTPUT)
