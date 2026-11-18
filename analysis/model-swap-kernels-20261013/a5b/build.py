"""Build the A5B (top-5 routing) smoke notebook from the c14 ledger notebook.

Variant (B): the incumbent Intel W4A16 checkpoint, unchanged weights, text_config.num_experts_per_tok 10 -> 5
via a config.json overlay in a symlink view. MTP drafter unchanged (its own config keeps top-10).
Run with the model-swap .venv python and -B.
"""
import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / 'analysis/model-serving-20261011/wave-1/e2-c14/kernel/wave1-e2-c14-20261011.ipynb'
SLUG = 'model-swap-a5b-smoke-20261013'
VIEW = '/kaggle/a5b-model-view'


def replace(nb, old, new):
    hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and old in ''.join(c['source'])]
    assert len(hits) == 1, (old, len(hits))
    s = ''.join(hits[0]['source'])
    assert s.count(old) == 1, old
    hits[0]['source'] = s.replace(old, new).splitlines(True)


ROUTING_VIEW = '''INCUMBENT_MODEL_DIR = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
A5B_TOP_K = 5


def routing_view(source, view, top_k):
    """Incumbent checkpoint with one change: text_config.num_experts_per_tok 10 -> top_k.

    Every entry except config.json is a symlink into the read-only mount; no weight is copied,
    edited or requantized. The MTP drafter keeps its own config (top-10)."""
    import hashlib as _hashlib
    source, view = Path(source), Path(view)
    raw = (source / 'config.json').read_bytes()
    cfg = json.loads(raw)
    text = cfg['text_config']
    assert text['num_experts'] == 512 and text['num_experts_per_tok'] == 10, (text['num_experts'], text['num_experts_per_tok'])
    view.mkdir(parents=True, exist_ok=False)
    linked = []
    for entry in sorted(source.iterdir()):
        if entry.name == 'config.json':
            continue
        (view / entry.name).symlink_to(entry.resolve())
        linked.append(entry.name)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    missing = [s for s in shards if not (view / s).is_file()]
    assert not missing, missing
    text['num_experts_per_tok'] = top_k
    (view / 'config.json').write_text(json.dumps(cfg, indent=2) + '\\n')
    check = json.loads((view / 'config.json').read_text())
    diff = sorted(k for k in set(check['text_config']) | set(json.loads(raw)['text_config'])
                  if check['text_config'].get(k) != json.loads(raw)['text_config'].get(k))
    assert diff == ['num_experts_per_tok'] and {k: v for k, v in check.items() if k != 'text_config'} == {k: v for k, v in json.loads(raw).items() if k != 'text_config'}, diff
    receipt = {'source': str(source), 'view': str(view), 'top_k': top_k, 'source_top_k': 10, 'num_experts': 512,
               'config_keys_changed': ['text_config.num_experts_per_tok'], 'linked_entries': len(linked),
               'indexed_shards': len(shards), 'indexed_tensors': len(index['weight_map']),
               'source_config_sha256': _hashlib.sha256(raw).hexdigest(),
               'view_config_sha256': _hashlib.sha256((view / 'config.json').read_bytes()).hexdigest()}
    Path('/kaggle/working').mkdir(parents=True, exist_ok=True)
    Path('/kaggle/working/model-swap-assembly.json').write_text(json.dumps(receipt, indent=2) + '\\n')
    print('A5B_ROUTING_VIEW', json.dumps(receipt), flush=True)
    return str(view)


MODEL_DIR = routing_view(INCUMBENT_MODEL_DIR, '%s', A5B_TOP_K)''' % VIEW

PROBE = '''
if not TRUE_SUBMISSION:  # interactive smoke evidence only; the competition rerun never enters this block
    import math as _math
    from urllib.request import Request as _Request
    # The c14 deadline releases the benchmark with the server still loading; the smoke waits for READY instead.
    while True:
        if proc.poll() is not None:
            show_log_tail()
            raise RuntimeError(f"SGLang server exited during startup (code {proc.returncode}); see log tail above")
        try:
            with urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/health', timeout=5) as _r:
                if _r.status == 200:
                    print(f'SMOKE_READY at {int(time.time()-NOTEBOOK_START_TIME)}s from notebook start', flush=True)
                    break
        except Exception:
            pass
        if time.time() - NOTEBOOK_START_TIME > 3600:
            raise RuntimeError('smoke: server not healthy within 3600 s of notebook start')
        time.sleep(5)
    with urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/get_server_info', timeout=30) as _r:
        _info = json.loads(_r.read())
    (WORKING_DIR / 'model-swap-server-info.json').write_text(json.dumps(_info, indent=2))
    # Routing as SGLang itself parses it (same get_config the server's ModelConfig calls), target and draft.
    _code = ("import json,sys\\n"
             "from sglang.srt.utils.hf_transformers_utils import get_config\\n"
             "out={}\\n"
             "for k,p in (('target',sys.argv[1]),('draft',sys.argv[2])):\\n"
             "    c=get_config(p,trust_remote_code=True);t=getattr(c,'text_config',c)\\n"
             "    out[k]={'path':p,'num_experts':getattr(t,'num_experts',None),'num_experts_per_tok':getattr(t,'num_experts_per_tok',None),'norm_topk_prob':getattr(t,'norm_topk_prob',None)}\\n"
             "print('A5B_ROUTING_PARSED '+json.dumps(out))\\n")
    try:  # CPU-only parse (the server holds the GPU); evidence, never fatal
        _routing = subprocess.run([PYTHON, "-c", _code, MODEL_DIR, str(DRAFT_VIEW)], text=True, capture_output=True,
                                  env=dict(env, CUDA_VISIBLE_DEVICES=''), timeout=300)
        _rout = {'returncode': _routing.returncode, 'stdout': _routing.stdout[-4000:], 'stderr': _routing.stderr[-4000:]}
    except Exception as _e:
        _rout = {'returncode': None, 'stdout': '', 'stderr': repr(_e)}
    (WORKING_DIR / 'model-swap-routing.json').write_text(json.dumps(_rout, indent=2))
    print([l for l in _rout['stdout'].splitlines() if l.startswith('A5B_ROUTING_PARSED')] or _rout['stderr'][-800:], flush=True)
    _payload = dict(model=SERVED_MODEL_NAME, prompt='Compute 2+2. Answer with the numeral only: ', max_tokens=32, temperature=0, logprobs=1)
    _req = _Request(SERVER_BASE_URL + '/completions', data=json.dumps(_payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(_req, timeout=180) as _r:
        _answer = json.loads(_r.read())
    (WORKING_DIR / 'model-swap-finite-answer.json').write_text(json.dumps(_answer, indent=2))
    _probs = _answer['choices'][0]['logprobs']['token_logprobs']
    assert _probs and all(_math.isfinite(float(_p)) for _p in _probs if _p is not None), 'nonfinite or empty token logprobs'
    print('MODEL_SWAP_FINITE_ANSWER', len(_probs), repr(_answer['choices'][0]['text']), flush=True)
'''


def submission_bodies(nb, shell):
    out = []
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c['source'])
        if s.startswith('%%'):
            continue
        for node in ast.walk(ast.parse(shell.transform_cell(s))):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'TRUE_SUBMISSION':
                out.append([ast.dump(x, include_attributes=False) for x in node.body])
    return out


def build():
    kd = HERE / 'kernel'
    assert not (HERE / 'launch-attempt.json').exists(), 'already frozen/launched'
    base = json.loads(BASE.read_text())
    nb = copy.deepcopy(base)
    replace(nb, "MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'", ROUTING_VIEW)
    # Interactive-only smoke controls (Tinfield r3 pattern). Every TRUE_SUBMISSION body stays literal.
    replace(nb, 'NOTEBOOK_START_EPOCH = time.time()',
            'NOTEBOOK_START_EPOCH = time.time()\nif not TRUE_SUBMISSION:\n    SERVER_STARTUP_TIMEOUT = 1300\n')
    replace(nb, "    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:",
            "    bm.solver.max_runtime_s_per_game = 40*60  # interactive smoke only\n    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:")
    replace(nb, 'soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))',
            'soft_end = datetime.now() + timedelta(seconds=40*60)  # interactive play cap, after server READY')
    cell12 = [i for i, c in enumerate(nb['cells']) if 'raise RuntimeError(f"SGLang server exited during startup' in ''.join(c['source'])]
    assert cell12 == [12], cell12
    nb['cells'][12]['source'] = (''.join(nb['cells'][12]['source']) + PROBE).splitlines(True)

    kd.mkdir(parents=True, exist_ok=True)
    notebook = kd / (SLUG + '.ipynb')
    notebook.write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads(BASE.with_name('kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/' + SLUG, title=SLUG.replace('-', ' '), code_file=notebook.name, is_private=True)
    (kd / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    assert meta['title'].replace(' ', '-') == meta['id'].split('/')[1]

    nbformat.validate(nbformat.read(notebook, as_version=4))
    shell = InteractiveShell.instance()
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            compile(shell.transform_cell(''.join(c['source'])), f'cell-{i}', 'exec', flags=0x2000)
    assert submission_bodies(nb, shell) == submission_bodies(base, shell)
    check = subprocess.run([sys.executable, '-B', str(REPO / 'incumbent/check.py'), str(notebook), '--without', 'b1'], text=True, capture_output=True)
    (HERE / 'incumbent-check.log').write_text(check.stdout + check.stderr)
    assert check.returncode == 0, check.stdout + check.stderr
    changed = [i for i, (a, b) in enumerate(zip(base['cells'], nb['cells'])) if a['source'] != b['source']]
    assert changed == [4, 12, 16, 20], changed
    assert meta['model_sources'] == json.loads(BASE.with_name('kernel-metadata.json').read_text())['model_sources']
    receipt = {'notebook': str(notebook.relative_to(REPO)), 'base': str(BASE.relative_to(REPO)),
               'base_sha256': hashlib.sha256(BASE.read_bytes()).hexdigest(),
               'notebook_sha256': hashlib.sha256(notebook.read_bytes()).hexdigest(),
               'metadata_sha256': hashlib.sha256((kd / 'kernel-metadata.json').read_bytes()).hexdigest(),
               'changed_cells_vs_base': changed, 'model_sources_unchanged': True,
               'submission_if_bodies_identical': True, 'notebook_schema_valid': True, 'all_code_cells_compile': True,
               'incumbent_check_without_b1': 'OK',
               'b1_exception': 'Named exact c14 base lacks B1 (probation; D3 23.56 below controls, thread 682); model-only change forbids adding it. Same exception as Tinfield (board 684).'}
    (HERE / 'LOCAL-QUALIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    build()
