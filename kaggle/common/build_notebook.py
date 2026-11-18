"""Build one reference-arm Phase A notebook on the qualified R8-L3B Kaggle route.

Both arms are built by this one file from one base notebook, so the serving route,
the environment checks, the official-source pinning and the containment are
identical between them and only the arm differs. The base is the V14 remote source
that last booted SGLang with Qwen3.8-27B-FP8 on an RTX PRO 6000 with the network
disabled.

    python common/build_notebook.py --arm tycho-upstream
    python common/build_notebook.py --arm nooa-upstream

Phase A only: the notebook refuses competition-rerun mode and never submits.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent
ROUTE = HERE / "route"
BASE_NOTEBOOK = ROUTE / "base-notebook-v14.ipynb"
PROBE = HERE / "reference_probe.py"

# Frozen identity of the offline wheelhouse the notebook installs from. Revision 11
# carries `arc-agi-3-tycho 0.1.0` and `nooa 0.0.1.dev95`, both verified
# byte-identical to the pristine reference checkouts.
WHEELHOUSE_MANIFEST_SHA256 = "518f8407b0ab1234430fc0c167e2ecd62ce70318a62292f8397e1a7ea4ed3862"
WHEELHOUSE_REVISION = 11
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

PUBLIC_25 = (
    "ar25,bp35,cd82,cn04,dc22,ft09,g50t,ka59,lf52,lp85,ls20,m0r0,r11l,re86,"
    "s5i5,sb26,sc25,sk48,sp80,su15,tn36,tr87,tu93,vc33,wa30"
)

# R8-L3B's argparse rejects any other value; the lifecycle probes pinned it.
REQUEST_TIMEOUT_S = 180
STARTUP_TIMEOUT_S = 1200


class Arm:
    """One reference harness, its join payload, and the environment it reads."""

    def __init__(
        self,
        name: str,
        *,
        slug_stem: str,
        title_stem: str,
        join_files: tuple[str, ...],
        description: str,
        environment: dict[str, str],
    ) -> None:
        self.name = name
        self.slug_stem = slug_stem
        self.title_stem = title_stem
        self.join_files = join_files
        self.description = description
        self.environment = environment

    def slug(self, label: str) -> str:
        return f"{self.slug_stem}-{label}"

    def title(self, label: str) -> str:
        return f"{self.title_stem} {label}"

    @property
    def directory(self) -> Path:
        return KAGGLE / self.name

    @property
    def wrapper_local_solver(self) -> Path:
        return self.directory / "worker" / "local_solver.py"


# Upstream NOOA reads ARC_LLM_*; the ladder and visual mode come from the
# reference's own competition config (`configs/competition.yaml`). The `memory`
# variant is unavailable offline — it needs an embeddings gateway and the
# nooa-memory distribution — so this arm runs the `mdfiles` variant, which is the
# paper's ablation arm (38.4 RHAE on GPT-5.5 against 50.2 for memory).
NOOA = Arm(
    "nooa-upstream",
    slug_stem="prototype-nooa-upstream",
    title_stem="PROTOTYPE NOOA upstream",
    join_files=(
        "worker/nooa_worker.py",
        "worker/nooa_headless.py",
        "worker/local_solver.py",
        "worker/stub_agent.py",
        "upstream/solver_agent.py",
        "upstream/arc_llm.py",
        "upstream/skills/interactive-game-solver/SKILL.md",
    ),
    description=(
        "nooa-pin examples/arc_agi_3 MdArcSolverAgent (mdfiles variant), "
        "pristine solver_agent.py and 51-line SKILL.md"
    ),
    environment={
        "ARC_LLM_CONTEXT_WINDOW": "32768",
        "ARC_LLM_MAX_TOKENS": "8192",
        "ARC_LLM_RETRIES": "0",
        "ARC_LLM_TIMEOUT_SECONDS": "170",
        "ARC_LLM_USE_RESPONSES": "0",
        "ARC_LLM_EXTRA_BODY": '{"chat_template_kwargs":{"enable_thinking":true}}',
    },
)

# Upstream Tycho reads TYCHO_*/LLM_*. Values below track `configs/paper/*.yaml`
# where the local route can honour them and are reduced only where the route or
# the Kaggle session forces it; every reduction is noted in SOURCES.json.
TYCHO = Arm(
    "tycho-upstream",
    slug_stem="prototype-tycho-upstream",
    title_stem="PROTOTYPE Tycho upstream",
    join_files=(
        "worker/tycho_worker.py",
        "worker/local_solver.py",
    ),
    description=(
        "arc-agi-3-tycho 0.1.0 from the offline wheelhouse, byte-identical to the "
        "Initial release; TychoAgent driven by upstream run_env"
    ),
    environment={
        # `single` is the paper's mid policy (85.36 RHAE on Opus 4.8) and the one
        # that needs no builder subprocess.
        "TYCHO_MODE": "single",
        # vision.py resolves Qwen as the one family whose encoder is lossless per
        # ARC cell; this is the reference's own default and the reason to use it.
        "TYCHO_VISION": "1",
        "TYCHO_TEXT_GRID": "full",
        "TYCHO_GRID_KEEP": "3",
        "TYCHO_IMAGE_KEEP": "3",
        "TYCHO_IMAGE_CAP": "4",
        "TYCHO_REASONING_KEEP": "3",
        "TYCHO_HISTORY": "tail_evict",
        # Local SGLang has no prompt cache API; the paper configs enable it on
        # Anthropic only.
        "TYCHO_PROMPT_CACHING": "0",
        "TYCHO_MAX_TOKENS": "8192",
        "TYCHO_MAX_TOOL_STEPS": "25",
        "TYCHO_MAX_LLM_CALLS": "1100",
        # A local model has no public price; the cost ceiling is meaningless here
        # and 0 disables it, leaving the action budget as the only limit.
        "TYCHO_MAX_INFERENCE_COST_PER_GAME": "0",
        "TYCHO_MAX_INFERENCE_COST_PER_LEVEL": "0",
        # Matched to the NOOA arm's 32,768-token context bound so both arms
        # face the same budget on one local SGLang process.
        "TYCHO_CONTEXT_EMERGENCY_SOFT_TOKENS": "28000",
        "LLM_EFFORT": "xhigh",
        "TYCHO_SYNTHETIC_BASELINE": "512",
        "TYCHO_SYNTHETIC_LEVELS": "32",
    },
)

ARMS = {arm.name: arm for arm in (NOOA, TYCHO)}


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join_zip_b64(arm: Arm) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in arm.join_files:
            archive.write(arm.directory / relative, arcname=relative)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build(
    arm: Arm,
    *,
    label: str,
    games: str = PUBLIC_25,
    max_actions: str = "16",
    launcher_timeout_s: int = 25_200,
    worker_timeout_s: int = 5_400,
) -> dict[str, Any]:
    notebook = json.loads(BASE_NOTEBOOK.read_text())
    join_root = f"/kaggle/working/{arm.name}-join"
    notebook["cells"][0]["source"] = (
        f"# {arm.title(label).upper()} — reference harness behind the official seam\n\n"
        f"Phase A only. Official ARC-AGI-3 launcher, offline, "
        f"{len(games.split(','))} public games, {max_actions} actions each.\n\n"
        f"Arm: {arm.description}.\n\n"
        "Refuses competition rerun. Creates no submission. One push, no retry."
    )

    assets = "".join(notebook["cells"][1]["source"])
    for destination, source in (
        ("/tmp/official_launcher_runtime.py", ROUTE / "exact_request_outer_runtime.py"),
        ("/tmp/reference_probe.py", PROBE),
        ("/tmp/my_agent.py", ROUTE / "my_agent.py"),
        ("/tmp/shared_local_solver.py", ROUTE / "shared_local_solver.py"),
        ("/tmp/wrapper_local_solver.py", arm.wrapper_local_solver),
        ("/tmp/run_official_agent.py", ROUTE / "run_official_agent.py"),
        ("/tmp/public-environment-manifest.json", ROUTE / "public-environment-manifest.json"),
    ):
        assets += (
            f"\nPath({destination!r}).write_bytes("
            f"base64.b64decode({encoded(source)!r}))\n"
        )
    assets += (
        f"\nPath('/tmp/{arm.name}-join.zip').write_bytes("
        f"base64.b64decode({join_zip_b64(arm)!r}))\n"
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
    or solver_manifest.get('artifact_revision') != {WHEELHOUSE_REVISION}
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
for required in ('nooa', 'arc-agi-3-tycho'):
    if required not in installed:
        raise RuntimeError(f'Solver closure is missing the {{required}} wheel')
join_root = Path({join_root!r})
if join_root.exists():
    shutil.rmtree(join_root)
join_root.mkdir(parents=True)
shutil.unpack_archive('/tmp/{arm.name}-join.zip', str(join_root), 'zip')
__import__('os').environ['REFERENCE_JOIN_ROOT'] = str(join_root)
"""
    notebook["cells"][2]["source"] = install

    arm_env = "".join(
        f"os.environ[{key!r}] = {value!r}\n"
        for key, value in sorted(arm.environment.items())
    )
    notebook["cells"][3]["source"] = f"""import hashlib as official_hashlib
import json
import os
import pandas as pd
import shutil
import subprocess
import sys
from pathlib import Path

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    raise RuntimeError({arm.name!r} + ' Phase A qualifier refuses competition rerun mode')

os.environ['REFERENCE_ARM'] = {arm.name!r}
os.environ['REFERENCE_JOIN_ROOT'] = {join_root!r}
os.environ['NOOA_JOIN_ROOT'] = {join_root!r}
os.environ['TYCHO_UPSTREAM_JOIN_ROOT'] = {join_root!r}
os.environ['TYCHO_PUBLIC_GAME_IDS'] = {games!r}
os.environ['TYCHO_KAGGLE_MAX_ACTIONS'] = {max_actions!r}
os.environ['TYCHO_NOOA_WORKER_TIMEOUT_SECONDS'] = {str(worker_timeout_s)!r}
os.environ['TYCHO_SANDBOX_RUNTIME'] = 'host'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.pop('TYCHO_KAGGLE_HARDENED_HOST', None)
{arm_env}
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

output_dir = Path('/kaggle/working/' + {arm.name!r})
command = [
    sys.executable,
    '/tmp/official_launcher_runtime.py',
    '--output-dir', str(output_dir),
    '--manifest', '/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json',
    '--input-root', '/kaggle/input',
    '--model-source', 'jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
    '--runtime-provenance', '/tmp/sglang-runtime-provenance.json',
    '--startup-timeout-s', {str(STARTUP_TIMEOUT_S)!r},
    '--request-timeout-s', {str(REQUEST_TIMEOUT_S)!r},
    '--launcher-timeout-s', {str(launcher_timeout_s)!r},
    '--attention-backend', 'triton',
    '--sampling-backend', 'pytorch',
    '--profile-imports',
    '--r8-l3b',
    '--official-launcher-probe', '/tmp/reference_probe.py',
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


def validate(notebook: dict[str, Any], arm: Arm, *, games: str, max_actions: str) -> None:
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    if f"REFERENCE_ARM'] = {arm.name!r}" not in source:
        raise RuntimeError("arm identity drifted")
    if f"TYCHO_PUBLIC_GAME_IDS'] = {games!r}" not in source:
        raise RuntimeError("public game subset drifted")
    if f"TYCHO_KAGGLE_MAX_ACTIONS'] = {max_actions!r}" not in source:
        raise RuntimeError("max-actions cap drifted")
    if f"'--request-timeout-s', '{REQUEST_TIMEOUT_S}'" not in source:
        raise RuntimeError("R8-L3B request timeout must stay 180")
    if "Phase A qualifier refuses competition rerun mode" not in source:
        raise RuntimeError("competition-rerun refusal drifted")
    if "kaggle competitions submit" in source:
        raise RuntimeError("notebook must not submit")
    if f"{arm.name}-join.zip" not in source:
        raise RuntimeError("join zip is not embedded")
    for key, value in arm.environment.items():
        if f"os.environ[{key!r}] = {value!r}" not in source:
            raise RuntimeError(f"arm setting {key} drifted")
    other = "tycho-upstream" if arm.name == "nooa-upstream" else "nooa-upstream"
    if f"REFERENCE_ARM'] = {other!r}" in source:
        raise RuntimeError("the other arm leaked into this notebook")


def kernel_metadata(arm: Arm, code_file: str, label: str) -> dict[str, Any]:
    return {
        "id": f"pranavagrawai/{arm.slug(label)}",
        "title": arm.title(label),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaRtxPro6000",
        "enable_internet": False,
        "enable_tpu": False,
        "dataset_sources": [
            "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot",
            "mahdiseddigh/kaggle-t4-stack-2026-08-28",
            "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime",
            "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--games", default=PUBLIC_25)
    parser.add_argument("--max-actions", default="16")
    parser.add_argument("--launcher-timeout-s", type=int, default=25_200)
    parser.add_argument("--worker-timeout-s", type=int, default=5_400)
    parser.add_argument("--label", default=None, help="kernel slug suffix; default public<N>")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    arm = ARMS[args.arm]
    label = args.label or f"public{len(args.games.split(','))}"
    notebook = build(
        arm,
        label=label,
        games=args.games,
        max_actions=args.max_actions,
        launcher_timeout_s=args.launcher_timeout_s,
        worker_timeout_s=args.worker_timeout_s,
    )
    validate(notebook, arm, games=args.games, max_actions=args.max_actions)

    output_dir = args.output_dir or (KAGGLE / "notebooks" / arm.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{arm.name.replace('-', '_')}_{label}.ipynb"
    notebook_path = output_dir / code_file
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata_path = output_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(kernel_metadata(arm, code_file, label), indent=2) + "\n"
    )
    print(notebook_path)
    print(f"  notebook sha256 {sha256(notebook_path)}")
    print(f"  metadata sha256 {sha256(metadata_path)}")


if __name__ == "__main__":
    main()
