"""Build one private, supervised vLLM trial from the audited donor setup."""
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT / 'analysis/m2-winners-20261001/audit'
SLUG = 'serving-vllm-smoke1-20261021'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    donor = json.loads((AUDIT / 'lord-bundle/setup_commands.json').read_text())[0]
    python = '\n'.join(donor.splitlines()[1:-1]) + '\n'
    tree = ast.parse(python)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
            break  # do not execute donor server/watchdog/benchmark entrypoint
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WATCHDOG_SCRIPT_TEXT' for t in node.targets):
            continue
        nodes.append(node)
    library = ast.unparse(ast.Module(body=nodes, type_ignores=[])) + '\n'
    vocab = (AUDIT / 'lord-bundle/src/ARC3-Inference/configs/draft_vocab_32k.json').read_bytes()
    runtime_manifest = (HERE / 'input-meta/runtime/runtime-manifest.json').read_bytes()
    runtime = json.loads(runtime_manifest)
    # The original donor guards must accept the selected published manifest before any GPU work.
    identity = next(n.value for n in nodes if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FINE_PREFIX_CACHE_PATCH_IDENTITY' for t in n.targets))
    expected_patch = ast.literal_eval(identity)
    assert all(runtime['patch'][k] == v for k, v in expected_patch.items()), 'donor setup and current runtime differ'
    bootstrap = '''import os, json, base64, zlib, hashlib
from pathlib import Path
OUT = Path('/kaggle/working/trial'); OUT.mkdir(parents=True, exist_ok=True)
os.environ['TAAF_KAGGLE_WORKING_DIR'] = str(OUT)
os.environ['TAAF_KAGGLE_BUNDLE_DIR'] = '/kaggle/working/donor-bundle'
os.environ['TAAF_RUN_AS_SUBMISSION'] = '0'
vocab = Path(os.environ['TAAF_KAGGLE_BUNDLE_DIR']) / 'src/ARC3-Inference/configs/draft_vocab_32k.json'
vocab.parent.mkdir(parents=True, exist_ok=True)
vocab.write_bytes(zlib.decompress(base64.b64decode(__VOCAB__)))
'''.replace('__VOCAB__', repr(base64.b64encode(zlib.compress(vocab, 9)).decode()))
    overrides = '''
# Predeclared differences from the donor's gaming harness profile.
VLLM_MAX_NUM_SEQS = 10
VLLM_MAX_MODEL_LEN = 139264
VLLM_GPU_MEMORY_UTILIZATION = '0.96'
VLLM_IMAGE_MAX_PIXELS = 640 * 640
'''
    runner = bootstrap + library + overrides + (HERE / 'run.py').read_text()
    runner = runner.replace('__RUNTIME_MANIFEST_SHA__', sha(runtime_manifest))
    compile(runner, 'trial.py', 'exec')
    (HERE / 'trial.py').write_text(runner)
    code = '''import base64, zlib, subprocess, sys, os, signal, json, time
from pathlib import Path
p = Path('/kaggle/working/trial.py')
p.write_bytes(zlib.decompress(base64.b64decode(__RUNNER__)))
proc = subprocess.Popen([sys.executable, '-u', str(p)], start_new_session=True)
try:
    rc = proc.wait(timeout=5100)
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGTERM)
    try: proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
    Path('/kaggle/working/supervisor-timeout.json').write_text(json.dumps({'timeout_s': 5100, 'time': time.time()}))
    raise RuntimeError('trial supervisor cutoff; no retry')
finally:
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: pass
assert rc == 0, f'trial failed with exit {rc}; raw outputs preserved'
'''.replace('__RUNNER__', repr(base64.b64encode(zlib.compress(runner.encode(), 9)).decode()))
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
          'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# One capped vLLM full-image ARC request replay\n\nNo competition submission. No retries. Card and source hashes embedded in the local trial directory. Runtime and model are the lordhansolo published artifacts; replay is the frozen 2026-10-06 corpus.']},
                    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
    meta = json.loads((ROOT / 'analysis/m2-winners-20261001/lordhansolo/kernel-metadata.json').read_text())
    meta.pop('id_no', None)
    meta.update(id='pranavagrawai/' + SLUG, title=SLUG, code_file=SLUG + '.ipynb', is_private=True,
                dataset_sources=['lordhansolo/vllm-main-e975732-arc3', 'pranavagrawai/mtp-depth-replay-data-20261006/2'],
                competition_sources=[], kernel_sources=[], docker_image_pinning_type='original')
    out = HERE / 'kernel'; out.mkdir(exist_ok=True)
    (out / (SLUG + '.ipynb')).write_text(json.dumps(nb, indent=1) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    import nbformat
    nbformat.validate(nbformat.from_dict(nb))
    manifest = {'donor_setup_sha256': sha(donor.encode()), 'runtime_manifest_sha256': sha(runtime_manifest),
                'runtime_patch': runtime['patch']['overlay_sha256'], 'draft_vocab_sha256': sha(vocab),
                'files': {str(p.relative_to(HERE)): sha(p.read_bytes()) for p in [HERE / 'build.py', HERE / 'run.py', HERE / 'trial.py', HERE / 'EXPERIMENT-CARD.md', *out.iterdir()]}}
    (HERE / 'freeze.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'kernel': meta['id'], 'runner_bytes': len(runner), 'runtime_patch': runtime['patch']['overlay_sha256']}))


if __name__ == '__main__':
    main()
