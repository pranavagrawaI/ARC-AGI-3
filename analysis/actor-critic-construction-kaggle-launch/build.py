#!/usr/bin/env python3
"""Build the Kaggle smoke package from an allowlist. Does not push."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACKAGE = REPO / "deployment/kaggle/actor-critic-construction-kaggle-smoke"
DONOR_NB = REPO / "deployment/kaggle/construction-policy-live-20260914T093351Z/construction.ipynb"
DONOR_META = REPO / "deployment/kaggle/construction-policy-live-20260914T093351Z/kernel-metadata.json"
DONOR_NB_SHA = "50d345ec00f2a5d089e19b4ee74412576f54a3902ba67859bd4bf9258d02dbd6"
V2 = REPO / "analysis/actor-critic-construction-live-launch-v2"
OVERLAY = REPO / "analysis/actor-critic-construction-integrated"
DONOR_SRC = REPO / "analysis/construction-policy-live-20260914T093351Z"
V11 = REPO / "deployment/kaggle/submission-runtime-20260912-v11"
TESTS = REPO / "deployment/kaggle/tests"

ALLOWLIST = [
    (HERE / "local_wire.py", "local_wire.py"),
    (HERE / "kaggle_probe.py", "kaggle_probe.py"),
    (HERE / "EXPERIMENT-CARD.md", "EXPERIMENT-CARD.md"),
    (V2 / "audit.py", "audit.py"),
    (V2 / "live_agent.py", "live_agent.py"),
    (V2 / "headless.py", "headless.py"),
    (V2 / "persist.py", "persist.py"),
    (V2 / "persist_job.py", "persist_job.py"),
    (OVERLAY / "construction_agent.py", "construction_agent.py"),
    (OVERLAY / "construction_service.py", "construction_service.py"),
    (OVERLAY / "construction_job.py", "construction_job.py"),
    (OVERLAY / "host.py", "host.py"),
    (OVERLAY / "observation_abi.py", "observation_abi.py"),
    (OVERLAY / "prefix.py", "prefix.py"),
    (OVERLAY / "evaluate_visible.py", "evaluate_visible.py"),
    (OVERLAY / "frozen_client.py", "frozen_client.py"),
    (DONOR_SRC / "accelerator.py", "accelerator.py"),
    (DONOR_SRC / "constructor.py", "constructor.py"),
    (DONOR_SRC / "request_contract.py", "request_contract.py"),
    (DONOR_SRC / "real_trace.py", "real_trace.py"),
    (DONOR_SRC / "integration_boundary.py", "integration_boundary.py"),
    (DONOR_SRC / "inspection_runner.py", "inspection_runner.py"),
    (DONOR_SRC / "containment_base.py", "containment_base.py"),
    (DONOR_SRC / "qualified_runner.py", "qualified_runner.py"),
    (V11 / "probe_lifecycle.py", "probe_lifecycle.py"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(path, names):
    tree = ast.parse(path.read_text())
    return '\n\n'.join(ast.unparse(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names) + '\n'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('source adaptation anchor drift: ' + old[:80])
    return source.replace(old, new)


def build(package=PACKAGE, *, write_manifest=True):
    PACKAGE = Path(package)
    if sha(DONOR_NB) != DONOR_NB_SHA:
        raise RuntimeError('donor hash drift')
    if PACKAGE.is_symlink():
        raise RuntimeError('symlink package root')
    PACKAGE.mkdir(parents=True, exist_ok=True)
    assets, origins = {}, {}

    def put(name, text, source=None):
        raw = text.encode() if isinstance(text, str) else text
        assets[name] = raw
        if source:
            origins[str(source.relative_to(REPO))] = sha(source)

    for source, name in ALLOWLIST:
        text = source.read_text()
        if name == 'headless.py':
            text = replace_once(text, 'V11 = ROOT / "deployment/kaggle/submission-runtime-20260912-v11"', 'V11 = Path(__file__).resolve().parent / "nooa-join"')
            text = replace_once(text, 'OVERLAY = Path(__file__).resolve().parent.parent / "actor-critic-construction-integrated"', 'OVERLAY = Path(__file__).resolve().parent')
        if name == 'constructor.py':
            text = replace_once(text, "ROUTING = json.loads((Path(__file__).resolve().parent / 'PROVIDER-SELECTION.json').read_text())['routing']", 'ROUTING = {}  # logical relay marker, never a gateway provider claim')
        if name == 'request_contract.py':
            text = '''"""Constructor logical relay contract; no gateway routing or billing."""
from local_wire import project_messages

def validate_request(body):
    if set(body) != {'model', 'messages', 'provider', 'tools', 'tool_choice', 'max_tokens', 'reasoning'}:
        raise ValueError('constructor fields')
    if body['model'] != 'qwen/qwen3.8-27b' or body['provider'] != {} or body['reasoning'] != {'enabled': False}:
        raise ValueError('constructor logical identity/control')
    if type(body['max_tokens']) is not int or not 1 <= body['max_tokens'] <= 4096 or body['tool_choice'] != 'required':
        raise ValueError('constructor bound')
    if project_messages(body['messages']) != body['messages']:
        raise ValueError('unprojected constructor messages')
    names = {t['function']['name'] for t in body['tools']}
    if not names <= {'execute_python', 'submit_candidate', 'request_evidence'} or 'request_evidence' not in names:
        raise ValueError('constructor tools')
    return {'passed': True, 'wire': 'logical-local-relay'}
'''
        if name == 'audit.py':
            start = text.index('def request_response_accounted(')
            end = text.index('\ndef bound_use(', start)
            text = text[:start] + 'def request_response_accounted(run_dir):\n    from local_accounting import accounting\n    return accounting(run_dir)["complete"]\n\n' + text[end:]
        # Every copied overlay now resolves its siblings inside this flat capsule.
        for line in text.splitlines():
            if line.startswith(('LIVE = ', 'OVERLAY = ')) and any(s in line for s in ('construction-policy-live-', 'actor-critic-construction-integrated')):
                text = text.replace(line, line.split('=')[0] + '= __import__("pathlib").Path(__file__).resolve().parent')
        put(name, text, source)
    for name in ('driver_patch.py', 'local_accounting.py', 'containment_preflight.py', 'process_guard.py', 'donor_bootstrap.py', 'runtime_install.py'):
        put(name, (HERE / name).read_bytes(), HERE / name)
    runtime_names = ('headless.py', 'local_solver.py', 'shared_local_solver.py', 'nooa_worker.py', 'state_delivery.py', 'adapter.py', 'probe_lifecycle.py')
    runtime_sources = [V11 / name for name in runtime_names]
    runtime_sources += [p for p in (V11 / 'chassis').rglob('*') if p.is_file() and p.suffix in {'.py', '.md'}]
    for path in runtime_sources:
        put('nooa-join/' + str(path.relative_to(V11)), path.read_bytes(), path)
    chassis = json.loads((V11 / 'CHASSIS.json').read_text())
    put('nooa-join/CHASSIS.json', json.dumps({'schema': 'packaged-v11-closure-v1', 'identity': 'actor-critic-construction-kaggle-smoke', 'contract_sha256': chassis['contract_sha256'], 'files': {name.removeprefix('nooa-join/'): hashlib.sha256(raw).hexdigest() for name, raw in assets.items() if name.startswith('nooa-join/')}}, indent=2), V11 / 'CHASSIS.json')
    # Extract only the real driver functions and its test Agent base; never ship tests/fixtures.
    header = 'from __future__ import annotations\nimport importlib.util, sys\nfrom pathlib import Path\nfrom types import ModuleType\nfrom typing import Any\nfrom arcengine import FrameData, GameAction, GameState\n'
    put('test_my_agent.py', header + 'AGENT_PATH = Path(__file__).resolve().parent / "nooa-join/adapter.py"\n' + selected(TESTS / 'test_my_agent.py', {'StubAgent', 'load_agent_module', 'frame'}), TESTS / 'test_my_agent.py')
    put('test_nooa_join.py', header + 'CANDIDATE = Path(__file__).resolve().parent / "nooa-join"\n' + selected(TESTS / 'test_nooa_join.py', {'load_create_solver'}), TESTS / 'test_nooa_join.py')
    put('test_join_qualification.py', header + 'from test_my_agent import load_agent_module\nfrom test_nooa_join import load_create_solver\n' + selected(TESTS / 'test_join_qualification.py', {'build_agent'}), TESTS / 'test_join_qualification.py')
    original = REPO / 'analysis/deployment-probes/20260904T-nooa-join-public3-depth/kaggle-output/ARC-AGI-3-Agents/agents/agent.py'
    if sha(original) != '85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087':
        raise RuntimeError('frame converter donor drift')
    cls = next(n for n in ast.parse(original.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Agent')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_convert_raw_frame_data')
    put('convert_frame.py', 'from __future__ import annotations\nfrom arcengine import FrameData, FrameDataRaw\n' + ast.unparse(method) + '\ndef convert(raw): return _convert_raw_frame_data(None, raw)\n', original)
    # Decode ONLY the hashed runtime supervisor and model manifest. No donor case/archive/entry.
    tree = ast.parse(''.join(json.loads(DONOR_NB.read_text())['cells'][1]['source']))
    branch = next(n for n in tree.body if isinstance(n, ast.If))
    nested = ast.parse(branch.orelse[0].value.args[0].value)
    for n in nested.body:
        if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Call) or not isinstance(n.value.func, ast.Attribute) or n.value.func.attr != 'write_bytes':
            continue
        name = Path(n.value.func.value.args[0].value).name
        if name not in {'official_launcher_runtime.py', 'PROTOTYPE-qwen3_8_fp8-manifest.json'}:
            continue
        raw = base64.b64decode(n.value.args[0].args[0].value)
        if name.endswith('.json'):
            put('model-metadata.json', raw)
            continue
        text = raw.decode()
        start = text.index('def expected_launcher_solver_requests(')
        end = text.index('\ndef reconcile_launcher_request_count(', start)
        text = text[:start] + 'def expected_launcher_solver_requests(launcher_result):\n    from local_accounting import expected_requests\n    return expected_requests(launcher_result)\n\n' + text[end:]
        claim_start = text.index('    if args.official_launcher_probe:\n        claim_ceiling =')
        claim_end = text.index('    elif args.readiness_only:', claim_start)
        text = text[:claim_start] + '''    if args.official_launcher_probe:
        claim_ceiling = (
            "Local Actor/critic/construction integration through test_join_qualification.build_agent "
            "with a test Agent base, real worker and offline game; NOT Agent.main. "
            "Passing requires frozen model use, critic, local receipt accounting and lifecycle retirement. "
            "It does not establish prediction accuracy, gameplay improvement, submission acceptance or general competence."
        )
''' + text[claim_end:]
        text = replace_once(text, '                    str(launcher_timeout - 60),', '                    str(min(900, launcher_timeout - 60)),\n                    "--startup-argv-json",\n                    json.dumps(command),')
        text = replace_once(text, '            launcher_result = json.loads(launcher_result_path.read_text())', '            launcher_result = json.loads(launcher_result_path.read_text())\n            from local_accounting import accounting\n            if launcher_result.get("local_requests") != accounting(solver_workspace / "raw"):\n                raise RuntimeError("launcher local evidence differs from persisted receipts")')
        text = replace_once(text, 'output_dir / "server.log", process, expected_count=2', 'output_dir / "server.log", process, expected_count=1 + expected_launcher_solver_requests(launcher_result)')
        text = text.replace('"TYCHO_KAGGLE_HARDENED_HOST": "1"', '"TYCHO_KAGGLE_HARDENED_HOST": "0"')
        put(name, text)
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in assets.items()}
    put('ASSET-HASHES.json', json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    for name, raw in assets.items():
        target = PACKAGE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    # Refuse hidden stale deploy assets rather than silently upload them.
    permitted = set(assets) | {'construction.ipynb', 'kernel-metadata.json', 'MANIFEST.json'}
    for path in PACKAGE.rglob('*'):
        if path.is_file() and str(path.relative_to(PACKAGE)) not in permitted and '__pycache__' not in path.parts:
            raise RuntimeError('stale package file: ' + str(path))
    meta = json.loads(DONOR_META.read_text())
    meta.update(id='pranavagrawai/actor-critic-construction-kaggle-smoke', title='actor-critic-construction-kaggle-smoke', code_file='construction.ipynb', is_private=True, enable_gpu=True, enable_internet=False, competition_sources=['arc-prize-2026-arc-agi-3'])
    (PACKAGE / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    embedded = {name: base64.b64encode(raw).decode() for name, raw in assets.items()}
    materialize = '''import base64, hashlib, json, os, subprocess, sys
from pathlib import Path
working = Path('/kaggle/working')
root = working / 'actor-critic-construction'
ASSETS = EMBEDDED_PLACEHOLDER
EXPECTED = HASH_PLACEHOLDER
if root.is_symlink():
    raise RuntimeError('symlink capsule root')
root.mkdir(parents=True, exist_ok=True)
for path in root.rglob('*'):
    if path.is_symlink():
        raise RuntimeError('symlink in capsule')
for name, value in ASSETS.items():
    path = root / name
    raw = base64.b64decode(value, validate=True)
    if hashlib.sha256(raw).hexdigest() != EXPECTED[name]:
        raise RuntimeError('embedded source drift: ' + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and path.read_bytes() != raw):
        raise RuntimeError('existing asset differs: ' + name)
    if not path.exists():
        path.write_bytes(raw)
for path in root.rglob('*'):
    if path.is_file() and str(path.relative_to(root)) not in ASSETS:
        raise RuntimeError('unexpected existing capsule file: ' + str(path))
'''.replace('EMBEDDED_PLACEHOLDER', repr(embedded)).replace('HASH_PLACEHOLDER', repr({name: hashlib.sha256(raw).hexdigest() for name, raw in assets.items()}))
    launch = '''# Fresh isolated guardian and bootstrap ignore cached notebook modules.
env = {'PATH': '/opt/bin:/usr/local/bin:/usr/bin:/bin', 'HOME': str(root), 'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1', 'ARC_DISABLE_DOTENV': '1'}
subprocess.run([sys.executable, '-I', '-S', '-B', str(root / 'process_guard.py'), str(os.getpid()), '3500', str(root / 'PROCESS-GUARD.json'), sys.executable, '-I', '-S', '-B', str(root / 'donor_bootstrap.py')], env=env, check=True)
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'python3', 'language': 'python', 'name': 'python3'}}, 'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# Actor–critic construction integration\nPrivate local Qwen smoke; test_join_qualification driver, not Agent.main. No submission or launch authorization.\n']},
        *[{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)} for code in (materialize, launch)]]}
    (PACKAGE / 'construction.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    manifest = {'identity': 'actor-critic-construction-kaggle-smoke', 'complete': False, 'push': False,
        'note': 'Packaging only; local checks and remote prerequisites tracked separately. Requires coordinator exact-hash review/launch gate.',
        'donor_notebook_sha256': DONOR_NB_SHA, 'origins': origins,
        'assets': {name: sha(PACKAGE / name) for name in assets},
        'files': {name: sha(PACKAGE / name) for name in ('construction.ipynb', 'kernel-metadata.json')},
        'builder_sha256': sha(HERE / 'build.py'),
        'openrouter_manifest_sha256': '6fb19b860a7bce1a104e5e2dc2f5f9210aca201d929ab545e5480018ef9e0661',
        'caps': {'boot_seconds': 1200, 'model_admission_seconds': 900, 'calls': 24, 'actor_summarizer': 20, 'actions': 8, 'constructor_wall_seconds': 45, 'constructor_hard_seconds': 50, 'http_post_seconds': 30, 'reservation_seconds': 3600, 'budget': 'gpu-wall'},
        'driver': 'test_join_qualification.build_agent: real local worker/game, test Agent base, not Agent.main'}
    for target in ((HERE, PACKAGE) if write_manifest else (PACKAGE,)):
        (target / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    result = build()
    print(json.dumps({'assets': len(result['assets']), 'complete': result['complete'], 'push': result['push']}))


