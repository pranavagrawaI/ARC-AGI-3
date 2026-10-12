"""Build isolated retained-pruning experiments from the frozen c14 notebook."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'model-serving-20261011/wave-1/e2-c14/kernel'
BASE_FILE = BASE / 'wave1-e2-c14-20261011.ipynb'
BASE_SHA = '543cae3c0f880476746bf0af0d32cc54cb3b76690edd7ef7846508c973cad0ad'
WHEEL_SHA = 'd0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225'
LATE_READY_LIMIT = 1300


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Notebook anchor changed: ' + old[:100])
    return source.replace(old, new)


def build(run, mode, streams, probe=False):
    root = HERE / run
    if (root / 'launch-attempt.json').exists():
        raise ValueError('Cannot rebuild launched experiment')
    if hashlib.sha256(BASE_FILE.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Retained c14 notebook changed')
    notebook = json.loads(BASE_FILE.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []; cell['execution_count'] = None
    setup = ''.join(notebook['cells'][4]['source'])
    if streams != 14:
        setup = replace_once(setup, "'ARC3_MAX_ACTIVE_STREAMS': 14,", f"'ARC3_MAX_ACTIVE_STREAMS': {streams},")
    setup += '\nassert not TRUE_SUBMISSION, "Public qualification only; never submit this notebook"\n'
    notebook['cells'][4]['source'] = setup.splitlines(True)
    launcher = ''.join(notebook['cells'][12]['source'])
    if streams != 14:
        for old, new in [('MAXREQ=14,', f'MAXREQ={streams},'), ('CUDAGRAPH_MAXBS=14,', f'CUDAGRAPH_MAXBS={streams},'),
                         ('MAMBA_CACHE=84,', f'MAMBA_CACHE={streams*6},')]:
            launcher = replace_once(launcher, old, new)
        launcher = replace_once(launcher, '{1, 2, 4, 7, 8, 9, 10, 12, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]}',
                                 '{1, 2, 4, 7, 8, 9, 10, 12, 14, 16, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]}')
    guard = f"\nassert hashlib.sha256(wheel.read_bytes()).hexdigest() == {WHEEL_SHA!r}, 'Packaged wheel identity differs'\n"
    launcher = replace_once(launcher, 'wheel = sglang_wheels[0]\n', 'wheel = sglang_wheels[0]\n' + guard)
    payload_names = ['pruning.py', 'install.py', 'smoke.py', 'mask.json', 'router-identity.json', 'contract.json']
    payload = {name: (HERE / name).read_text() for name in payload_names}
    install = '''
# Retained pruning: original dataset and draft config stay immutable.
PRUNE_OUT = Path('/kaggle/working/pruning'); PRUNE_OUT.mkdir(exist_ok=True)
KIT = Path('/tmp/retained-pruning-kit'); KIT.mkdir(exist_ok=True)
'''
    install += 'for name, content in ' + repr(payload) + '.items():\n    (KIT / name).write_text(content)\n'
    install += f"PRUNE_MODE = {mode!r}\n"
    install += '''ORIGINAL_TARGET = MODEL_DIR
PRUNED_VIEW = Path('/tmp/retained-pruning-target')
run([PYTHON, str(KIT/'install.py'), '--source', MODEL_DIR, '--view', str(PRUNED_VIEW),
     '--receipt', str(PRUNE_OUT/'install.json'), '--mode', PRUNE_MODE], env=env)
env['RETAINED_PRUNING_RECEIPT_DIR'] = str(PRUNE_OUT)
if PRUNE_MODE == 'pruned': MODEL_DIR = str(PRUNED_VIEW)
if DRAFT_VIEW is None: raise RuntimeError('Expected unchanged MTP drafter')
draft_config = json.loads((Path(DRAFT_VIEW)/'config.json').read_text())
assert draft_config.get('text_config', draft_config)['num_experts'] == 512
(PRUNE_OUT/'draft-config.json').write_text(json.dumps(draft_config))
(PRUNE_OUT/'target-config.json').write_text(Path(MODEL_DIR,'config.json').read_text())
'''
    launcher = replace_once(launcher, 'args = [SGLANG, "serve",', install + '\nargs = [SGLANG, "serve",')
    # Both arms must reach health; the original fail-fast did not reject a live hung server.
    # A slow Kaggle disk can push READY past dfranzen's 722 s deadline (6 of 32 kernels; candidate
    # 138302732 died here). Keep waiting while the server lives, so both arms start play READY.
    launcher += f'''
LATE_READY_LIMIT = {LATE_READY_LIMIT}  # s from notebook start; + 7254 s play stays inside the 9000 s cap
while True:
    try:
        with urllib.request.urlopen(url, timeout=5) as health:
            if health.status == 200:
                break
    except Exception:
        pass
    if proc.poll() is not None:
        show_log_tail(); raise RuntimeError(f"SGLang server exited during startup (code {{proc.returncode}})")
    if time.time() - NOTEBOOK_START_TIME > LATE_READY_LIMIT:
        show_log_tail(); raise RuntimeError("Server not READY by LATE_READY_LIMIT")
    time.sleep(3)
print(f"SERVER READY at {{int(time.time()-NOTEBOOK_START_TIME)}}s from notebook start", flush=True)
'''
    launcher += '''
with urllib.request.urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/get_server_info', timeout=10) as r:
    (PRUNE_OUT/'server-info-start.json').write_bytes(r.read())
target_receipt = json.loads((PRUNE_OUT/'target.json').read_text())
draft_receipt = json.loads((PRUNE_OUT/'draft.json').read_text())
assert target_receipt['complete'] and target_receipt['target_experts'] == (448 if PRUNE_MODE == 'pruned' else 512)
assert draft_receipt['draft_experts'] == 512
'''
    notebook['cells'][12]['source'] = launcher.splitlines(True)
    if probe:
        notebook['cells'] = notebook['cells'][:13]
        ending = '''import signal
try:
    subprocess.run([PYTHON, str(KIT/'smoke.py')], env=env, check=True, timeout=600)
finally:
    os.killpg(proc.pid, signal.SIGTERM)
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=15)
'''
        notebook['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': ending.splitlines(True)})
    else:
        # Explicit 121-minute gameplay horizon in both arms; platform hard cap 150 minutes.
        source = ''.join(notebook['cells'][16]['source'])
        source += '\nbm.solver.max_runtime_s_per_game = 7254\n'
        notebook['cells'][16]['source'] = source.splitlines(True)
    slug = f'prune-retain-{run}-20261011'
    meta = json.loads((BASE / 'kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/' + slug, title=slug, code_file=slug + '.ipynb')
    dest = root / 'kernel'; dest.mkdir(parents=True, exist_ok=True)
    nbpath = dest / meta['code_file']; nbpath.write_text(json.dumps(notebook, indent=1) + '\n')
    (dest / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return {'run': run, 'slug': meta['id'], 'mode': mode, 'streams': streams, 'hard_cap_s': 3600 if probe else 9000,
            'notebook_sha256': hashlib.sha256(nbpath.read_bytes()).hexdigest(), 'metadata': meta}


def main():
    mask = Path('/home/pranav/projects/arc-agi-3-reference/duck-qwen38/runs/flash-next/prune/k448_mask_layers.json')
    (HERE / 'mask.json').write_bytes(mask.read_bytes())
    config = HERE / 'checkpoints/intel-config.json'
    index = json.loads((HERE / 'checkpoints/intel-model.safetensors.index.json').read_text())
    contract = {'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                'index_tensor_names_sha256': hashlib.sha256(json.dumps(sorted(index['weight_map'])).encode()).hexdigest(),
                'note': 'Kaggle splits large HF shards; tensor names, config and router bytes are verified independently of sharding.'}
    (HERE / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    # Probe 1 died loading the drafter (probe/RESULT.json); its cap stays committed against the ceiling.
    retired = [r for r in json.loads((HERE / 'runs.json').read_text()).get('retired', []) +
               json.loads((HERE / 'runs.json').read_text())['runs'] if (HERE / r['run'] / 'launch-attempt.json').exists()]
    if (HERE / 'candidate/launch-attempt.json').exists():
        # 2026-10-12 rerun: candidate ERRORed at startup (readout/README.md); only candidate2 is new.
        plan = json.loads((HERE / 'runs.json').read_text())
        if any(r['run'] == 'candidate2' for r in plan['runs']):
            raise ValueError('candidate2 already planned')
        plan['runs'].append(build('candidate2', 'pruned', 18))
        (HERE / 'runs.json').write_text(json.dumps(plan, indent=2) + '\n')
        print(json.dumps({k: v for k, v in plan['runs'][-1].items() if k != 'metadata'}, indent=2))
        return
    runs = [build('probe2', 'pruned', 18, True), build('control', 'control', 14), build('candidate', 'pruned', 18)]
    (HERE / 'runs.json').write_text(json.dumps({'baseline_notebook_sha256': BASE_SHA, 'retired': retired, 'runs': runs}, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'metadata'} for r in runs], indent=2))


if __name__ == '__main__':
    main()
