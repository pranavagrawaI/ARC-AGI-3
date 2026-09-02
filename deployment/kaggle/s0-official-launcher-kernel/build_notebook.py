"""Build the official-launcher candidate from accepted, hash-bound inputs."""

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
OUTPUT = HERE / "s0_official_launcher.ipynb"
RUNTIME = HERE / "official_launcher_runtime.py"
PROBE = HERE / "official_launcher_probe.py"
ADAPTER = HERE.parent / "agent/my_agent.py"
LOCAL_SOLVER = HERE.parent / "runtime/local_solver.py"
RUNNER = HERE.parent / "runtime/run_official_agent.py"

WHEELHOUSE_MANIFEST_SHA256 = (
    "08275ddc46126134a68cde4cdce8dc81196416e9837bf117418efad675e625fe"
)
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
    notebook = json.loads(V14.read_text())
    notebook["cells"][0]["source"] = (
        "# CANDIDATE — Tycho-NOOA official competition launcher\n\n"
        "This joins the accepted R8-L3B v14 model-server route, exact private "
        "Solver wheelhouse v2, current MyAgent adapter, and hash-pinned official "
        "ARC-AGI-3-Agents launcher. A normal commit emits only a dummy parquet. "
        "The official launcher runs only when Kaggle sets "
        "`KAGGLE_IS_COMPETITION_RERUN`."
    )

    assets = "".join(notebook["cells"][1]["source"])
    for destination, source in (
        ("/tmp/official_launcher_runtime.py", RUNTIME),
        ("/tmp/official_launcher_probe.py", PROBE),
        ("/tmp/my_agent.py", ADAPTER),
        ("/tmp/local_solver.py", LOCAL_SOLVER),
        ("/tmp/run_official_agent.py", RUNNER),
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
    or solver_manifest.get('artifact_revision') != 2
):
    raise RuntimeError('Solver closure manifest has the wrong identity')
solver_wheelhouse = solver_manifest_path.parent
subprocess.run(
    [sys.executable, str(solver_wheelhouse / 'verify_wheelhouse.py')],
    cwd=solver_wheelhouse,
    check=True,
    timeout=60,
)
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
        '--no-index', '--find-links', str(solver_wheelhouse),
        'arc-agi-3-tycho-nooa==0.1.0',
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
Path('/tmp/solver-closure-provenance.json').write_text(json.dumps({{
    'dataset': 'pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v2',
    'dataset_version': 2,
    'manifest_sha256': {WHEELHOUSE_MANIFEST_SHA256!r},
    'wheel_count': len(solver_manifest['wheels']),
    'installed_count': len(installed),
    'include_system_site_packages': False,
}}, indent=2) + '\\n')
"""
    notebook["cells"][2]["source"] = install

    notebook["cells"][3]["source"] = f"""import hashlib as official_hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    deadline = time.monotonic() + 600
    while True:
        try:
            with urllib.request.urlopen('http://gateway:8001/api/games', timeout=10) as response:
                if response.status == 200:
                    break
        except Exception:
            if time.monotonic() >= deadline:
                raise TimeoutError('competition gateway was not ready within 600 seconds')
            time.sleep(5)

    source_root = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents')
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
    (official_root / '.env').write_text('''DEBUG=False
SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=competition
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/kaggle/working/server_recording
''')

    output_dir = Path('/kaggle/working/tycho-nooa-official-launcher')
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
        '--launcher-timeout-s', '32400',
        '--attention-backend', 'triton',
        '--sampling-backend', 'pytorch',
        '--profile-imports',
        '--r8-l3b',
        '--official-launcher-probe', '/tmp/official_launcher_probe.py',
        '--launcher-python', '/tmp/solver-venv/bin/python',
    ]
    subprocess.run(command, check=True)
    print((output_dir / 'result.json').read_text())
else:
    import pandas as pd
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
    if sha256(LOCAL_SOLVER) != "2cf9e6705b6a10cc27f2aa57913675f8e60c0e379e449f854d6b89b2b7d0669e":
        raise RuntimeError("local Solver adapter drifted from its manifest")


if __name__ == "__main__":
    candidate = build()
    validate(candidate)
    OUTPUT.write_text(json.dumps(candidate, indent=1) + "\n")
    print(OUTPUT)
