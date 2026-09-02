"""Assemble the resumable public-25 outer callback candidate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import resumable_public25 as harness

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V14 = (
    ROOT
    / "analysis/deployment-probes/20260831T160520Z-qwen38-r8l3b-kaggle-v14"
    / "remote-source-v14/prototype-qwen3-8-fp8-r8-b-request-path.ipynb"
)
OUTPUT = HERE / "s0_public25_resumable_outer.ipynb"
RUNTIME = HERE / "all25_outer_runtime.py"
HARNESS = HERE / "resumable_public25.py"
CALLBACK_CONTRACT = HERE / "outer_callback_contract.py"
CHECKPOINT_TRANSPORT = HERE / "checkpoint_dataset_transport.py"
PROBE = (
    HERE.parent
    / "s0-public3-full-timing-v11-action-horizon-candidate/public25_launcher_probe.py"
)
ADAPTER = HERE.parent / "agent/my_agent.py"
LOCAL_SOLVER = HERE.parent / "runtime/local_solver.py"
RUNNER = HERE.parent / "runtime/run_official_agent.py"
PUBLIC_ENV_MANIFEST = (
    ROOT
    / "analysis/deployment-probes/20260901T054322Z-public25-kaggle-v1"
    / "public-environment-manifest.json"
)
PROJECTION_ESTIMATOR = HERE.parent / "scripts/derive_public25_runtime_projection.py"
PROJECTION_POLICY = HERE.parent / "public25-runtime-projection-policy-v1.json"
AUTHORITATIVE_HARNESS = (
    HERE.parent / "s0-public25-resumable-v2-candidate/resumable_public25.py"
)
AUTHORITATIVE_CALLBACK_CONTRACT = (
    HERE.parent / "s0-public25-resumable-v2-candidate/outer_callback_contract.py"
)
AUTHORITATIVE_CHECKPOINT_TRANSPORT = (
    HERE.parent
    / "s0-public25-resumable-v2-candidate/checkpoint_dataset_transport.py"
)

WHEELHOUSE_MANIFEST_SHA256 = (
    "e95b8176e9b445059f858d511559017bcc6452eba96c1cfd4805e5c1d1d2c187"
)
SOLVER_WHEEL_SHA256 = (
    "0ba1282ea11b679524fce1f813da21c12132f3a9b5a48e66eba4034fbc8f1322"
)
PROBE_SHA256 = "56e627ca9aa4281f36ff5b8f9ea8894a833366679a8e4fecddaf836af47da0bf"
CHECKPOINT_TRANSPORT_SHA256 = (
    "ab015a7b894a3c5633eabcafd7de75bfc450515d91b1953637439101ae3efb62"
)
CHECKPOINT_BINDING_SCHEMA = "tycho-nooa-public25-checkpoint-dataset-binding-v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
DATASET_ID = re.compile(
    r"[a-z0-9][a-z0-9_-]{1,38}/[a-z0-9][a-z0-9-]{5,49}"
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
EXPECTED_CONTRACT_ARTIFACT_HASHES = {
    "adapter": "b0e4e2c94cd895eb7cdb5ebde78df6ff00220ed92ee88d510f5d2692da2d5a8f",
    "checkpoint_transport": CHECKPOINT_TRANSPORT_SHA256,
    "continued_play_probe": PROBE_SHA256,
    "guarded_runner": "e34b6ec37fee36935846b9f081cea9f57101dc22bd9b86469b9bf654e374a689",
    "local_solver": "517572979803525af52e7a6f2066fdee2eb3ce5923fd2780c5b185cb52b06795",
    "model_manifest": "2c2c2f2a4344c33d6f54d9f065a4975b91822cd2836d4300ff50f425e0a322a4",
    "official_source_manifest": "7c9a9b7c9a9880e7f3da38282dcef3bf1ed2b9aa9c559cd1082c5abb80b459ec",
    "outer_callback_contract": "6d1e8ec95d335e0defa3836d9236b819a403092a1587362a08ff1a8fc0d94856",
    "outer_runtime": "e238ab7da771604ab3338c0304c22ffc7ce51883955c55a0c0fbc15cf2ba7143",
    "public_environment_manifest": "842477b6336e4f556be999d6c9b76664238a5c7d991740263abf72fc1f864a35",
    "resumable_harness": "1145c9d5b4d6e58846ac0387dd1dcf76a2c7cadedfc7ab6990fcf96952f6633b",
    "runtime_dataset_manifest": "f1a50bb224dc26ac48c0fd219c28e4fadc92ede6d2a80a26fd18d9a004a79b94",
    "runtime_projection_estimator": "b1b92118059c91dafa35a95fd1986d95d2970a6842514d7abf1bc01070eec86e",
    "runtime_projection_policy": "249d61aecbacadb36f8cbcae5144ad9c2a9c3fe8c21a19756214ac3cbce2b640",
    "solver_wheel": SOLVER_WHEEL_SHA256,
    "solver_wheelhouse_manifest": WHEELHOUSE_MANIFEST_SHA256,
}
BASE_NOTEBOOK_SHA256 = (
    "e73b9bde9474d639d5bd75edb136b5d4149592517e41a08043c267f9b37edadd"
)
LOCAL_ARTIFACT_PATHS = {
    "adapter": ADAPTER,
    "checkpoint_transport": CHECKPOINT_TRANSPORT,
    "continued_play_probe": PROBE,
    "guarded_runner": RUNNER,
    "local_solver": LOCAL_SOLVER,
    "outer_callback_contract": CALLBACK_CONTRACT,
    "outer_runtime": RUNTIME,
    "public_environment_manifest": PUBLIC_ENV_MANIFEST,
    "resumable_harness": HARNESS,
    "runtime_projection_estimator": PROJECTION_ESTIMATOR,
    "runtime_projection_policy": PROJECTION_POLICY,
}


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_closure() -> None:
    """Reject any local source bytes not named by the frozen contract."""
    if sha256(V14) != BASE_NOTEBOOK_SHA256:
        raise RuntimeError("accepted V14 base notebook bytes differ")
    for role, path in LOCAL_ARTIFACT_PATHS.items():
        expected = EXPECTED_CONTRACT_ARTIFACT_HASHES[role]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"local contract artifact {role} differs: {actual} != {expected}"
            )


def load_predecessor_binding(
    path: Path | None,
    *,
    contract_sha256: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("predecessor binding is not one regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"predecessor binding is not readable JSON: {error}") from error
    expected_fields = {
        "schema",
        "dataset_id",
        "dataset_version",
        "transport_manifest_sha256",
        "contract_sha256",
        "predecessor_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise RuntimeError("predecessor binding fields differ")
    if value["schema"] != CHECKPOINT_BINDING_SCHEMA:
        raise RuntimeError("predecessor binding schema differs")
    if value["contract_sha256"] != contract_sha256:
        raise RuntimeError("predecessor binding contract differs")
    if value["dataset_version"] != 1:
        raise RuntimeError("predecessor dataset must be immutable version 1")
    if not isinstance(value["dataset_id"], str) or DATASET_ID.fullmatch(
        value["dataset_id"]
    ) is None:
        raise RuntimeError("predecessor dataset identity is malformed")
    for field in ("transport_manifest_sha256", "predecessor_manifest_sha256"):
        digest = value[field]
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise RuntimeError(f"predecessor binding {field} is malformed")
    return value


def load_runtime_derivation(
    path: Path,
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("runtime derivation is not one regular file")
    try:
        evidence = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"runtime derivation is not readable JSON: {error}") from error
    if not isinstance(evidence, dict):
        raise TypeError("runtime derivation is not a JSON object")
    projection = evidence.get("runtime_projection")
    expected_projection_fields = {
        "initial_decision_seconds",
        "successor_decision_seconds",
        "per_game_overhead_seconds",
        "per_game_scheduler_reserve_seconds",
        "safety_multiplier",
        "next_scratch_bound_bytes",
        "storage_publication_margin_bytes",
        "source_result_sha256",
        "estimator_schema",
        "projection_policy_sha256",
        "source_artifact_manifest_sha256",
    }
    if (
        evidence.get("schema")
        != "tycho-nooa-public25-runtime-projection-evidence-v1"
        or not isinstance(projection, dict)
        or set(projection) != expected_projection_fields
        or projection.get("estimator_schema") != evidence.get("schema")
    ):
        raise RuntimeError("runtime derivation projection schema differs")
    policy = evidence.get("policy")
    source_artifacts = evidence.get("source_artifacts")
    canonical_run = evidence.get("canonical_run")
    if (
        not isinstance(policy, dict)
        or not isinstance(policy.get("file"), dict)
        or policy["file"].get("sha256")
        != projection["projection_policy_sha256"]
        or not isinstance(source_artifacts, dict)
        or source_artifacts.get("manifest_sha256")
        != projection["source_artifact_manifest_sha256"]
        or not isinstance(canonical_run, dict)
    ):
        raise RuntimeError("runtime derivation provenance differs")
    canonical_identity = canonical_run.get(
        "candidate_artifact_manifest_sha256"
    )
    if canonical_identity != harness.CANONICAL_TIMING_RUN_IDENTITY_SHA256:
        raise RuntimeError("runtime derivation canonical identity differs")
    if evidence.get("cohort") != {
        "solver_decision_ceiling": 1_056,
        "whole_model_request_ceiling": 1_181,
    }:
        raise RuntimeError("runtime derivation cohort ceiling differs")
    derivation_harness = evidence.get("harness")
    if (
        not isinstance(derivation_harness, dict)
        or derivation_harness.get("sha256") != sha256(HARNESS)
        or sha256(HARNESS)
        != EXPECTED_CONTRACT_ARTIFACT_HASHES["resumable_harness"]
        or sha256(PROJECTION_ESTIMATOR)
        != EXPECTED_CONTRACT_ARTIFACT_HASHES["runtime_projection_estimator"]
        or sha256(PROJECTION_POLICY)
        != EXPECTED_CONTRACT_ARTIFACT_HASHES["runtime_projection_policy"]
        or projection["projection_policy_sha256"] != sha256(PROJECTION_POLICY)
    ):
        raise RuntimeError("runtime derivation local artifact closure differs")
    expected_contract_projection = {
        **projection,
        "derivation_artifact_sha256": sha256(path),
        "canonical_run_identity_sha256": canonical_identity,
    }
    if contract.get("runtime_projection") != expected_contract_projection:
        raise RuntimeError("contract runtime projection does not match derivation")
    return evidence


def build(
    *,
    contract_path: Path,
    derivation_path: Path,
    predecessor_binding_path: Path | None = None,
) -> dict[str, Any]:
    validate_source_closure()
    if not contract_path.is_file() or contract_path.is_symlink():
        raise RuntimeError("measured public-25 contract is not one regular file")
    contract = json.loads(contract_path.read_text())
    contract_sha256 = harness.validate_contract(contract)
    if contract.get("artifact_hashes") != EXPECTED_CONTRACT_ARTIFACT_HASHES:
        raise RuntimeError("measured public-25 contract artifact hashes differ")
    load_runtime_derivation(derivation_path, contract=contract)
    predecessor_binding = load_predecessor_binding(
        predecessor_binding_path,
        contract_sha256=contract_sha256,
    )
    notebook = json.loads(V14.read_text())
    notebook["cells"][0]["source"] = (
        "# RESUMABLE PUBLIC-25 V8/V2 OUTER CALLBACK CANDIDATE\n\n"
        "This joins the accepted R8-L3B v14 model-server lifecycle to the "
        "manifested sequential public-25 harness and its schema-bound callback "
        "contract. Callback PASS is not cohort completion: only an explicit "
        "complete qualification means all 25 public games completed. This "
        "candidate refuses competition rerun mode and creates no competition "
        "submission."
    )

    assets = "".join(notebook["cells"][1]["source"])
    embedded_assets = [
        ("/tmp/official_launcher_runtime.py", RUNTIME),
        ("/tmp/resumable_public25.py", HARNESS),
        ("/tmp/outer_callback_contract.py", CALLBACK_CONTRACT),
        ("/tmp/public25_launcher_probe.py", PROBE),
        ("/tmp/my_agent.py", ADAPTER),
        ("/tmp/local_solver.py", LOCAL_SOLVER),
        ("/tmp/run_official_agent.py", RUNNER),
        ("/tmp/public-environment-manifest.json", PUBLIC_ENV_MANIFEST),
        ("/tmp/public25-contract.json", contract_path),
        ("/tmp/public25-runtime-projection-evidence.json", derivation_path),
    ]
    if predecessor_binding_path is not None:
        embedded_assets.extend(
            [
                ("/tmp/checkpoint_dataset_transport.py", CHECKPOINT_TRANSPORT),
                ("/tmp/public25-predecessor-binding.json", predecessor_binding_path),
            ]
        )
    for destination, source in embedded_assets:
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

    predecessor_setup = ""
    predecessor_command_extension = ""
    if predecessor_binding is not None:
        predecessor_setup = f"""

predecessor_binding = json.loads(
    Path('/tmp/public25-predecessor-binding.json').read_text()
)
if predecessor_binding != {predecessor_binding!r}:
    raise RuntimeError('embedded predecessor binding differs from frozen values')
predecessor_root = Path('/kaggle/working/public25-predecessor')
import_command = [
    sys.executable,
    '/tmp/checkpoint_dataset_transport.py', 'import',
    '--input-root', '/kaggle/input',
    '--destination', str(predecessor_root),
    '--expected-transport-manifest-sha256',
    {predecessor_binding['transport_manifest_sha256']!r},
    '--expected-contract-sha256', {predecessor_binding['contract_sha256']!r},
    '--expected-predecessor-manifest-sha256',
    {predecessor_binding['predecessor_manifest_sha256']!r},
    '--expected-dataset-id', {predecessor_binding['dataset_id']!r},
]
imported = subprocess.run(
    import_command,
    check=True,
    capture_output=True,
    text=True,
    timeout=120,
)
import_receipt = json.loads(imported.stdout)
expected_import_receipt = {{
    'schema': 'tycho-nooa-public25-checkpoint-import-receipt-v1',
    'checkpoint_root': str(predecessor_root),
    'dataset_id': {predecessor_binding['dataset_id']!r},
    'transport_manifest_sha256':
        {predecessor_binding['transport_manifest_sha256']!r},
    'checkpoint_manifest_sha256':
        {predecessor_binding['predecessor_manifest_sha256']!r},
    'contract_sha256': {predecessor_binding['contract_sha256']!r},
}}
if import_receipt != expected_import_receipt:
    raise RuntimeError('checkpoint import receipt differs from frozen binding')
Path('/kaggle/working/public25-predecessor-import-receipt.json').write_text(
    json.dumps(import_receipt, indent=2, sort_keys=True) + '\\n'
)
""".rstrip()
        predecessor_command_extension = f"""
command.extend([
    '--public25-predecessor', str(predecessor_root),
    '--expected-predecessor-sha256',
    {predecessor_binding['predecessor_manifest_sha256']!r},
])
""".rstrip()

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

# Public-25 cohort: each atomic shard selects one public game and its bound cap.
# Single mode deliberately omits automatic Builder.
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
template_root = Path('/kaggle/working/public25-template')
if template_root.exists():
    shutil.rmtree(template_root)
template_root.mkdir()
official_root = template_root / 'ARC-AGI-3-Agents'
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
shutil.copy2('/tmp/run_official_agent.py', template_root / 'run_official_agent.py')
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
'''){predecessor_setup}

output_dir = Path('/kaggle/working/tycho-nooa-public25-outer')
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
    '--launcher-timeout-s', '25200',
    '--attention-backend', 'triton',
    '--sampling-backend', 'pytorch',
    '--profile-imports',
    '--r8-l3b',
    '--resumable-public25-probe', '/tmp/resumable_public25.py',
    '--public25-game-probe', '/tmp/public25_launcher_probe.py',
    '--public25-contract', '/tmp/public25-contract.json',
    '--public25-template-root', str(template_root),
    '--public25-checkpoint-root', '/kaggle/working/public25-checkpoint',
    '--public25-workspace-root', '/kaggle/working/public25-orchestrator',
    '--launcher-python', '/tmp/solver-venv/bin/python',
]{predecessor_command_extension}
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
    contract_path: Path,
    derivation_path: Path,
    predecessor_binding_path: Path | None = None,
) -> None:
    validate_source_closure()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    contract_sha256 = harness.validate_contract(json.loads(contract_path.read_text()))
    load_runtime_derivation(
        derivation_path,
        contract=json.loads(contract_path.read_text()),
    )
    predecessor_binding = load_predecessor_binding(
        predecessor_binding_path,
        contract_sha256=contract_sha256,
    )
    if HARNESS.read_bytes() != AUTHORITATIVE_HARNESS.read_bytes():
        raise RuntimeError("assembled resumable harness differs from the repaired source")
    if CALLBACK_CONTRACT.read_bytes() != AUTHORITATIVE_CALLBACK_CONTRACT.read_bytes():
        raise RuntimeError("assembled callback contract differs from the repaired source")
    if (
        predecessor_binding is not None
        and CHECKPOINT_TRANSPORT.read_bytes()
        != AUTHORITATIVE_CHECKPOINT_TRANSPORT.read_bytes()
    ):
        raise RuntimeError("assembled checkpoint transport differs from its source")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--derivation", type=Path, required=True)
    parser.add_argument("--predecessor-binding", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    candidate = build(
        contract_path=arguments.contract,
        derivation_path=arguments.derivation,
        predecessor_binding_path=arguments.predecessor_binding,
    )
    validate(
        candidate,
        contract_path=arguments.contract,
        derivation_path=arguments.derivation,
        predecessor_binding_path=arguments.predecessor_binding,
    )
    arguments.output.write_text(json.dumps(candidate, indent=1) + "\n")
    print(arguments.output)
