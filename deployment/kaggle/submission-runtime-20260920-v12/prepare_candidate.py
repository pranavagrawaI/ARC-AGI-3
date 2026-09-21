"""Build/check the v12 sb26 critic-off sequential notebook."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import build_notebook as builder
from freeze_candidate import verify

HERE = Path(__file__).resolve().parent
GAMES = 'sb26'
PASSES = 2
EPISODES = 2
ARMS = ('off',)
ACTIONS_PER_ARM = '200'
ADMISSION_SECONDS = '1800'
LAUNCHER_TIMEOUT_SECONDS = 4200
WORKER_TIMEOUT_SECONDS = 2400
KEEP_TURNS = '4'
OUT = HERE.parents[2] / 'analysis/deployment-probes/20260921-v12-sb26-critic-off'


def build():
    extra = '{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}'
    nb = builder.build(
        games=GAMES,
        max_actions=ACTIONS_PER_ARM,
        worker_timeout_s=WORKER_TIMEOUT_SECONDS,
        launcher_timeout_s=LAUNCHER_TIMEOUT_SECONDS,
        max_tokens='16384',
        llm_timeout_s=300,
        request_timeout_s=180,
        extra_body=extra,
        heading=(
            '# v12 sb26 critic-off\n\n'
            'One arm: runtime v12 eviction keep=4, Actor reasoning off, discovery A, '
            'contingent off, epistemic critic off. Public sb26 only, 2 sequential '
            '1800s passes. A later-episode worker abort must not skip assessment. '
            'No submission or automatic promotion.'
        ),
    )
    builder.validate(
        nb,
        games=GAMES,
        max_actions=ACTIONS_PER_ARM,
        launcher_timeout_s=LAUNCHER_TIMEOUT_SECONDS,
        extra_body=extra,
    )
    body = ''.join(nb['cells'][3]['source']).split('submission = pd.DataFrame(')[0]
    body = body.replace(
        "os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '7200'",
        f"os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = {ADMISSION_SECONDS!r}",
    )
    body = body.replace('subprocess.run(command, check=True)', 'launcher = subprocess.run(command, check=False)')
    cells = nb['cells'][:3]
    arm = ARMS[0]
    root = '/kaggle/working/v12-sb26-critic-off'
    code = (
        "import os, shutil\nfrom pathlib import Path\n"
        f"arm_root=Path({root!r})\n"
        "if arm_root.exists(): raise RuntimeError('Arm directory exists; no retry into old evidence')\n"
        "arm_root.mkdir()\nshutil.copytree('/kaggle/working/nooa-join',arm_root/'nooa-join')\n"
        "os.environ['ARC_SOLVER_ARM']='loop'\n"
        "os.environ['ARC_DISCOVERY_ARM']='a'\n"
        f"os.environ['ARC_CONTINGENT_ARM']={arm!r}\n"
        "os.environ['ARC_EPISTEMIC_CRITIC']='0'\n"
        f"os.environ['ARC_TRANSCRIPT_KEEP_TURNS']={KEEP_TURNS!r}\n"
        f"os.environ['TYCHO_PUBLIC_GAME_PASSES']={str(PASSES)!r}\n"
        f"os.environ['TYCHO_PUBLIC25_WORKING_ROOT']={root!r}\n"
    )
    code += body.replace('/kaggle/working/', root + '/')
    code += (
        "print('launcher_returncode', launcher.returncode)\n"
        "print((output_dir / 'result.json').read_text())\n"
        "assess = subprocess.run([sys.executable,str(arm_root/'nooa-join/assess_candidate.py'),str(output_dir),"
        "'--arm',os.environ['ARC_SOLVER_ARM'],'--reasoning','off',"
        "'--contingent-arm',os.environ['ARC_CONTINGENT_ARM'],"
        "'--epistemic-critic',os.environ['ARC_EPISTEMIC_CRITIC'],"
        "'--action-limit',os.environ['TYCHO_KAGGLE_MAX_ACTIONS'],"
        "'--admission-seconds',os.environ['TYCHO_GAMEPLAY_ADMISSION_SECONDS'],"
        "'--output',str(output_dir/'ASSESSMENT.json')],check=False)\n"
        "print('assess_returncode', assess.returncode)\n"
    )
    ast.parse(code)
    cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=code))
    nb['cells'] = cells
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = verify()
    nb = build()
    path = HERE / 'probe.ipynb'
    if args.check:
        assert json.loads(path.read_text()) == nb, 'Notebook source drift'
    else:
        path.write_text(json.dumps(nb, indent=2) + '\n')
    OUT.mkdir(parents=True, exist_ok=True)
    extracted = OUT / 'extracted'
    extracted.mkdir(exist_ok=True)
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    payloads = []
    for node in ast.walk(ast.parse(code)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'b64decode'
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            raw = base64.b64decode(node.args[0].value)
            payloads.append(raw)
            if raw.startswith(b'PK'):
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for name in archive.namelist():
                        assert not Path(name).is_absolute() and '..' not in Path(name).parts
                        p = extracted / name
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(archive.read(name))
    for name in (
        'exact_request_outer_runtime.py',
        'adapter.py',
        'shared_local_solver.py',
        'run_official_agent.py',
        'nooa_public_probe.py',
    ):
        raw = (HERE / name).read_bytes()
        assert raw in payloads, name
        (extracted / name).write_bytes(raw)
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == digest, name
    metadata = json.loads((HERE / 'kernel-metadata.json').read_text())
    assert metadata['competition_sources'] == ['arc-prize-2026-arc-agi-3']
    assert metadata['machine_shape'] == 'NvidiaRtxPro6000'
    launch = dict(
        kernel=metadata['id'],
        session_timeout_seconds=10800,
        arms=list(ARMS),
        factor='v12-critic-off',
        games=GAMES.split(','),
        passes=PASSES,
        episodes=EPISODES,
        discovery_arm='a',
        epistemic_critic='0',
        admission_seconds_per_arm=int(ADMISSION_SECONDS),
        launcher_timeout_seconds_per_arm=LAUNCHER_TIMEOUT_SECONDS,
        actions_per_arm=int(ACTIONS_PER_ARM),
        transcript_keep_turns=int(KEEP_TURNS),
        max_output_tokens_per_request=16384,
        model_context_tokens=262144,
        contract_sha256=manifest['contract_sha256'],
        promotion=False,
        launch_ready=True,
        authorization=(
            'One private RTX PRO 6000 kernel: v12 eviction chassis, contingent off, '
            'epistemic critic off, public sb26, 2 sequential 1800s passes. No submission, retry or automatic promotion.'
        ),
        openrouter_qualified=False,
        automatic_retries=0,
        argv=[
            'kaggle',
            'kernels',
            'push',
            '--path',
            str(HERE),
            '--timeout',
            '10800',
            '--accelerator',
            'NvidiaRtxPro6000',
        ],
        files={
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in (
                'probe.ipynb',
                'kernel-metadata.json',
                'prepare_candidate.py',
                'KAGGLE-EXPERIMENT-CARD.md',
                'assess_candidate.py',
                'verify_candidate.py',
            )
        },
    )
    path = HERE / 'LAUNCH.json'
    if args.check:
        assert json.loads(path.read_text()) == launch, 'Launch identity drift'
    else:
        path.write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps(launch, indent=2))


if __name__ == '__main__':
    main()
