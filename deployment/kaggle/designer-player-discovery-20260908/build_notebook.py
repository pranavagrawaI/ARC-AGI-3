"""Build the NOOA-join public-3 Phase A notebook from the accepted R8-L3B v14 route."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V14 = (
    ROOT
    / "analysis/deployment-probes/20260831T160520Z-qwen38-r8l3b-kaggle-v14"
    / "remote-source-v14/prototype-qwen3-8-fp8-r8-b-request-path.ipynb"
)
OUTPUT = HERE / "s0_nooa_join_public3.ipynb"
# q6 owns this copy rather than borrowing the v15 candidate's, which is frozen
# against V14 bytes: the join needed a reconciler that knows its session schema,
# and every other candidate already carries its own runtime.
RUNTIME = HERE / "exact_request_outer_runtime.py"
PROBE = HERE / "nooa_public_probe.py"
ADAPTER = HERE / "adapter.py"
SHARED_LOCAL_SOLVER = HERE / "shared_local_solver.py"
WRAPPER_LOCAL_SOLVER = HERE / "local_solver.py"
RUNNER = HERE / "run_official_agent.py"
PUBLIC_ENV_MANIFEST = (
    ROOT
    / "analysis/deployment-probes/20260901T054322Z-public25-kaggle-v1"
    / "public-environment-manifest.json"
)

WHEELHOUSE_MANIFEST_SHA256 = (
    "518f8407b0ab1234430fc0c167e2ecd62ce70318a62292f8397e1a7ea4ed3862"
)
SOLVER_WHEEL_SHA256 = "9ffadfc88ed40667fa14b876b5a17b94ac4992ca277377c887de0db5b0df7faf"
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
JOIN_FILES = (
    "assess_candidate.py",
    "chassis/decision_packet.py",
    "chassis/designer_player.md",
    "chassis/step_check.py",
    "stub_agent.py",
    "nooa_worker.py",
    "headless.py",
    "state_delivery.py",
    "chassis/prediction_feedback.py",
    "chassis/evidence_renderer.py",
    "chassis/discovery-demo.md",
    "local_solver.py",
    "nooa_public_probe.py",
    "chassis/arc_llm.py",
    "chassis/solver_agent.py",
    "chassis/host_contract.py",
    "chassis/stall_contract.py",
    "chassis/integrity_guard.py",
    "CHASSIS.json",
    "chassis/skills/interactive-game-solver/SKILL.md",
)


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join_zip_b64() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in JOIN_FILES:
            item = zipfile.ZipInfo(relative, date_time=(2026, 9, 5, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            archive.writestr(item, (HERE / relative).read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


DEFAULT_GAMES = "ls20"
DEFAULT_MAX_ACTIONS = "200"
DEFAULT_LAUNCHER_TIMEOUT_S = 7200
DEFAULT_WORKER_TIMEOUT_S = 2400
DEFAULT_MAX_TOKENS = "16384"
DEFAULT_LLM_TIMEOUT_S = 180
DEFAULT_REQUEST_TIMEOUT_S = 180
DEFAULT_EXTRA_BODY = (
    '{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}'
)
THINKING_ON_EXTRA_BODY = '{"chat_template_kwargs":{"enable_thinking":true}}'


def build(
    *,
    games: str = DEFAULT_GAMES,
    max_actions: str = DEFAULT_MAX_ACTIONS,
    launcher_timeout_s: int = DEFAULT_LAUNCHER_TIMEOUT_S,
    worker_timeout_s: int = DEFAULT_WORKER_TIMEOUT_S,
    max_tokens: str = DEFAULT_MAX_TOKENS,
    llm_timeout_s: int = DEFAULT_LLM_TIMEOUT_S,
    request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S,
    extra_body: str = DEFAULT_EXTRA_BODY,
    heading: str | None = None,
    treatment: bool = False,
    discovery_arm: str = "a",
) -> dict[str, Any]:
    from freeze_candidate import verify

    verify()
    if discovery_arm not in {"a", "b", "c", "d"}:
        raise ValueError("discovery_arm must be one of a, b, c, d")
    notebook = json.loads(V14.read_text())
    notebook["cells"][0]["source"] = heading or (
        "# NOOA JOIN PUBLIC-3 — q6 candidate behind MyAgent\n\n"
        f"Phase A only. Official launcher, offline {games.replace(',', '/')}, "
        f"{max_actions} actions each. "
        "Worker is the q3c NOOA file-IPC host, not Tycho NativeTool. "
        "Refuses competition rerun. One push, no retry."
    )
    assets = "".join(notebook["cells"][1]["source"])
    for destination, source in (
        ("/tmp/official_launcher_runtime.py", RUNTIME),
        ("/tmp/nooa_public_probe.py", PROBE),
        ("/tmp/my_agent.py", ADAPTER),
        ("/tmp/shared_local_solver.py", SHARED_LOCAL_SOLVER),
        ("/tmp/wrapper_local_solver.py", WRAPPER_LOCAL_SOLVER),
        ("/tmp/run_official_agent.py", RUNNER),
        ("/tmp/public-environment-manifest.json", PUBLIC_ENV_MANIFEST),
    ):
        assets += (
            f"\nPath({destination!r}).write_bytes("
            f"base64.b64decode({encoded(source)!r}))\n"
        )
    assets += (
        "\nPath('/tmp/nooa-join.zip').write_bytes("
        f"base64.b64decode({join_zip_b64()!r}))\n"
    )
    notebook["cells"][1]["source"] = assets

    install = "".join(notebook["cells"][2]["source"])
    install += f"""

solver_manifest_path = unique_exact_file(
    'wheelhouse-manifest.json',
    {WHEELHOUSE_MANIFEST_SHA256!r},
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
if 'nooa' not in installed:
    raise RuntimeError('Solver closure is missing the nooa wheel')
join_root = Path('/kaggle/working/nooa-join')
if join_root.exists():
    shutil.rmtree(join_root)
join_root.mkdir(parents=True)
shutil.unpack_archive('/tmp/nooa-join.zip', str(join_root), 'zip')
__import__('os').environ['NOOA_JOIN_ROOT'] = str(join_root)
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
    raise RuntimeError('NOOA join qualifier refuses competition rerun mode')

os.environ['ARC_STALL_REVIEW'] = {"1" if treatment else "0"!r}
os.environ['ARC_DISCOVERY_ARM'] = {discovery_arm!r}
os.environ['TYCHO_PUBLIC_GAME_IDS'] = {games!r}
os.environ['TYCHO_KAGGLE_MAX_ACTIONS'] = {max_actions!r}
os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '7200'
os.environ['ARC_LLM_CONTEXT_WINDOW'] = '262144'
os.environ['ARC_LLM_MAX_TOKENS'] = {max_tokens!r}
os.environ['ARC_LLM_RETRIES'] = '0'
os.environ['ARC_LLM_TIMEOUT_SECONDS'] = {str(llm_timeout_s)!r}
os.environ['ARC_LLM_USE_RESPONSES'] = '0'
os.environ['ARC_LLM_EXTRA_BODY'] = {extra_body!r}
os.environ['TYCHO_NOOA_WORKER_TIMEOUT_SECONDS'] = {str(worker_timeout_s)!r}
os.environ['NOOA_JOIN_ROOT'] = '/kaggle/working/nooa-join'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.pop('TYCHO_KAGGLE_HARDENED_HOST', None)

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
runtime_dir = official_root / 'runtime'
runtime_dir.mkdir(parents=True, exist_ok=True)
(runtime_dir / '__init__.py').write_text('')
shutil.copy2('/tmp/my_agent.py', templates / 'my_agent.py')
shutil.copy2('/tmp/shared_local_solver.py', runtime_dir / 'local_solver.py')
shutil.copy2('/tmp/wrapper_local_solver.py', official_root / 'local_solver.py')
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

output_dir = Path('/kaggle/working/nooa-join-public3')
command = [
    sys.executable,
    '/tmp/official_launcher_runtime.py',
    '--output-dir', str(output_dir),
    '--manifest', '/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json',
    '--input-root', '/kaggle/input',
    '--model-source', 'jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
    '--runtime-provenance', '/tmp/sglang-runtime-provenance.json',
    '--startup-timeout-s', '1200',
    '--request-timeout-s', {str(request_timeout_s)!r},
    '--launcher-timeout-s', {str(launcher_timeout_s)!r},
    '--attention-backend', 'triton',
    '--sampling-backend', 'pytorch',
    '--profile-imports',
    '--r8-l3b',
    '--official-launcher-probe', '/tmp/nooa_public_probe.py',
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


def validate(
    notebook: dict[str, Any],
    *,
    games: str = DEFAULT_GAMES,
    max_actions: str = DEFAULT_MAX_ACTIONS,
    launcher_timeout_s: int = DEFAULT_LAUNCHER_TIMEOUT_S,
    extra_body: str = DEFAULT_EXTRA_BODY,
) -> None:
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    if f"TYCHO_PUBLIC_GAME_IDS'] = {games!r}" not in source:
        raise RuntimeError("public game subset drifted")
    if f"TYCHO_KAGGLE_MAX_ACTIONS'] = {max_actions!r}" not in source:
        raise RuntimeError("max-actions cap drifted")
    if f"'--launcher-timeout-s', {str(launcher_timeout_s)!r}" not in source:
        raise RuntimeError("launcher timeout drifted")
    if "'--request-timeout-s', '180'" not in source:
        raise RuntimeError("R8-L3B request timeout must stay 180")
    if f"ARC_LLM_EXTRA_BODY'] = {extra_body!r}" not in source:
        raise RuntimeError("LLM extra_body drifted")
    if "NOOA join qualifier refuses competition rerun mode" not in source:
        raise RuntimeError("competition-rerun refusal drifted")
    if "kaggle competitions submit" in source:
        raise RuntimeError("notebook must not submit")
    if "nooa-join.zip" not in source:
        raise RuntimeError("join zip is not embedded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", action="store_true")
    parser.add_argument("--discovery-arm", choices=["a", "b", "c", "d"], default="a")
    parser.add_argument("--games", default=DEFAULT_GAMES)
    parser.add_argument("--max-actions", default=DEFAULT_MAX_ACTIONS)
    parser.add_argument(
        "--launcher-timeout-s", type=int, default=DEFAULT_LAUNCHER_TIMEOUT_S
    )
    parser.add_argument(
        "--worker-timeout-s", type=int, default=DEFAULT_WORKER_TIMEOUT_S
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--heading", default=None)
    parser.add_argument("--thinking-on", action="store_true")
    parser.add_argument("--max-tokens", default=None)
    parser.add_argument("--llm-timeout-s", type=int, default=None)
    parser.add_argument("--request-timeout-s", type=int, default=None)
    parser.add_argument("--extra-body", default=None)
    args = parser.parse_args()
    extra_body = args.extra_body or (
        THINKING_ON_EXTRA_BODY if args.thinking_on else DEFAULT_EXTRA_BODY
    )
    max_tokens = args.max_tokens or ("8192" if args.thinking_on else DEFAULT_MAX_TOKENS)
    llm_timeout_s = args.llm_timeout_s or (
        300 if args.thinking_on else DEFAULT_LLM_TIMEOUT_S
    )
    # R8-L3B argparse rejects any --request-timeout-s other than 180.
    request_timeout_s = args.request_timeout_s or DEFAULT_REQUEST_TIMEOUT_S
    candidate = build(
        games=args.games,
        max_actions=args.max_actions,
        launcher_timeout_s=args.launcher_timeout_s,
        worker_timeout_s=args.worker_timeout_s,
        max_tokens=max_tokens,
        llm_timeout_s=llm_timeout_s,
        request_timeout_s=request_timeout_s,
        extra_body=extra_body,
        heading=args.heading,
        treatment=args.treatment,
        discovery_arm=args.discovery_arm,
    )
    validate(
        candidate,
        games=args.games,
        max_actions=args.max_actions,
        launcher_timeout_s=args.launcher_timeout_s,
        extra_body=extra_body,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate, indent=1) + "\n")
    print(args.output)
