"""Package the frozen live-v3 NOOA runtime for public and competition modes."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / 'submission-runtime-20260910'


def build():
    sys.path.insert(0, str(DONOR))
    import build_notebook as donor
    notebook = donor.build(games='ls20,s5i5,cd82', max_actions='10', launcher_timeout_s=4800,
                           worker_timeout_s=300, llm_timeout_s=180, max_tokens='16384')
    notebook['cells'] = notebook['cells'][:3]
    notebook['cells'][0]['source'] = ('# First ARC-AGI-3 submission — 10 September 2026\n\n'
        'Contingent-controller-derived package with host-admitted successor publication, reasoning off, FP8 27B, MTP off. '
        'Commit mode qualifies ls20/s5i5/cd82; competition mode plays the gateway-provided games. '
        'The official Agent loop runs sequentially. Public success is not a score claim.')
    play = (HERE / 'play.py').read_bytes()
    assets = ''.join(notebook['cells'][1]['source'])
    assets += f"\nPath('/tmp/submission_play.py').write_bytes(base64.b64decode({base64.b64encode(play).decode()!r}))\n"
    notebook['cells'][1]['source'] = assets
    source = '''import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from urllib.request import urlopen

rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1', 'true')
competition = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3')
framework = Path('/kaggle/working/ARC-AGI-3-Agents')
if framework.exists():
    raise RuntimeError('Framework destination exists; use a fresh execution')
shutil.copytree(competition / 'ARC-AGI-3-Agents', framework, symlinks=True)
for name, digest in OFFICIAL_HASHES.items():
    if hashlib.sha256((framework/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'Official framework drift: {name}')
(framework/'runtime').mkdir(exist_ok=True)
(framework/'runtime/__init__.py').write_text('')
shutil.copy2('/tmp/my_agent.py', framework/'agents/templates/my_agent.py')
shutil.copy2('/tmp/shared_local_solver.py', framework/'runtime/local_solver.py')
shutil.copy2('/tmp/wrapper_local_solver.py', framework/'local_solver.py')
(framework/'agents/__init__.py').write_text('from .agent import Agent, Playback\\nfrom .swarm import Swarm\\nfrom .templates.my_agent import MyAgent\\nAVAILABLE_AGENTS = {"myagent": MyAgent}\\n')
(framework/'.env').write_text('')
os.environ.update({
    'ARC_STALL_REVIEW':'0', 'ARC_DISCOVERY_ARM':'a', 'ARC_SOLVER_ARM':'loop', 'ARC_CONTINGENT_ARM':'on',
    'ARC_LLM_CONTEXT_WINDOW':'262144', 'ARC_LLM_MAX_TOKENS':'16384', 'ARC_LLM_RETRIES':'0',
    'ARC_LLM_TIMEOUT_SECONDS':'90', 'ARC_LLM_USE_RESPONSES':'0',
    'ARC_LLM_EXTRA_BODY':'{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}',
    'TYCHO_NOOA_WORKER_TIMEOUT_SECONDS':'120', 'NOOA_JOIN_ROOT':'/kaggle/working/nooa-join',
    'SUBMISSION_FRAMEWORK':str(framework), 'SUBMISSION_ENVIRONMENTS':str(competition/'environment_files'),
    'SUBMISSION_PUBLIC_GAMES':'ls20,s5i5,cd82', 'PYTHONUNBUFFERED':'1',
})
if rerun:
    deadline = time.monotonic()+600
    while True:
        try:
            with urlopen('http://gateway:8001/api/games', timeout=10) as response:
                if response.status == 200:
                    break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
else:
    manifest = json.loads(Path('/tmp/public-environment-manifest.json').read_text())
    for item in manifest['files']:
        path = competition/item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'Public environment drift: {path}')
output = Path('/kaggle/working/qualification')
if output.exists():
    raise RuntimeError('Evidence destination exists; no overwrite')
command = [sys.executable, '/tmp/official_launcher_runtime.py', '--output-dir', str(output),
    '--manifest','/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json','--input-root','/kaggle/input',
    '--model-source','jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
    '--runtime-provenance','/tmp/sglang-runtime-provenance.json', '--startup-timeout-s','1200',
    '--request-timeout-s','180','--launcher-timeout-s', '25200' if rerun else '4800',
    '--attention-backend','triton','--sampling-backend','pytorch','--profile-imports','--r8-l3b',
    '--official-launcher-probe','/tmp/submission_play.py','--launcher-python','/tmp/solver-venv/bin/python']
subprocess.run(command, check=True, timeout=27600 if rerun else 6300)
print((output/'result.json').read_text())
if not rerun:
    import pandas as pd
    pd.DataFrame([['1_0','1',True,1]], columns=['row_id','game_id','end_of_game','score']).to_parquet('/kaggle/working/submission.parquet', index=False)
'''
    source = 'OFFICIAL_HASHES = ' + repr(donor.OFFICIAL_SOURCE_HASHES) + '\n' + source
    ast.parse(source)
    notebook['cells'].append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=source))
    notebook.setdefault('metadata', {})['kernelspec'] = dict(name='python3', language='python', display_name='Python 3')
    for i, cell in enumerate(notebook['cells']):
        cell['id'] = f'first-submission-{i}'
    metadata = json.loads((DONOR / 'kernel-metadata.json').read_text())
    metadata.update(id='pranavagrawai/arc3-first-submission-20260910',
                    title='ARC3 First Submission 20260910', code_file='submission.ipynb')
    (HERE / 'submission.ipynb').write_text(json.dumps(notebook, indent=1)+'\n')
    (HERE / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    manifest = dict(donor=str(DONOR), contract_sha256=json.loads((DONOR/'CHASSIS.json').read_text())['contract_sha256'],
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [HERE/'build.py', HERE/'play.py', HERE/'submission.ipynb', HERE/'kernel-metadata.json']})
    (HERE/'BUILD.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    build()
