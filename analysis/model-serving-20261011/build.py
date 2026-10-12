"""Build the wave-1 run directories from the dfranzen control notebook. Never uploads or runs anything.

wave-1/e1-swift : MODEL_DIR -> a merged symlink view of the two Swift-1.5 Flash-Next W4A16 AutoRound parts
                  (phuongncn/arc3-qwen38-swift-w4a16-autoround w4a16-a + w4a16-b). Everything else is control.
wave-1/e2-c14   : control model; 14 concurrent streams (gate, server, cuda graphs), analyzer context 128K -> 92K
                  and drain 58K -> 40K so 14 streams fit the same KV pool; mamba slots 60 -> 84 (6 per stream).
Both            : one fail-fast line: stop the notebook if the server process exited during startup (control would
                  otherwise run every game against a dead server until the cap).
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTROL = REPO / 'analysis/m2-winners-20261001/runs/control'
SWIFT = 'phuongncn/arc3-qwen38-swift-w4a16-autoround'
INTEL = 'dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def edit(nb, old, new, count=1):
    """Replace old with new in exactly one code cell, where it must occur exactly `count` times."""
    hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and old in ''.join(c['source'])]
    assert len(hits) == 1, f'{old[:60]!r}: in {len(hits)} cells'
    src = ''.join(hits[0]['source'])
    assert src.count(old) == count, f'{old[:60]!r}: {src.count(old)} occurrences'
    hits[0]['source'] = src.replace(old, new).splitlines(True)


FAIL_FAST_OLD = '''    print(f"\\nDEADLINE at {int(time.time()-NOTEBOOK_START_TIME)}s from notebook start; "
          f"releasing the benchmark with the server still loading")'''
FAIL_FAST_NEW = FAIL_FAST_OLD + '''
if proc.poll() is not None:  # WAVE-1: a dead server must end the run, not burn the cap
    raise RuntimeError(f"SGLang server exited during startup (code {proc.returncode}); see log tail above")
'''

SWIFT_VIEW = '''MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
'''
SWIFT_VIEW_NEW = '''# WAVE-1 E1: Swift-1.5 Flash-Next W4A16 AutoRound, mounted as two Kaggle instances (50-file limit); link both
# into one read-only view. Parts are disjoint: a = shards 1-48, b = shards 49-81 + config/tokenizer/index.
import glob as _glob
_swift_parts = sorted(_glob.glob('/kaggle/input/models/phuongncn/arc3-qwen38-swift-w4a16-autoround/*/w4a16-[ab]/*'))
assert len(_swift_parts) == 2, f"expected the two Swift parts, found {_swift_parts}"
_swift_view = Path('/tmp/swift-w4a16-view'); _swift_view.mkdir(parents=True, exist_ok=True)
_swift_seen = {}
for _part in _swift_parts:
    for _f in Path(_part).iterdir():
        assert _f.name not in _swift_seen, f"Swift parts overlap on {_f.name}"
        _swift_seen[_f.name] = _f
        _dst = _swift_view / _f.name
        if not _dst.is_symlink():
            _dst.symlink_to(_f.resolve())
_swift_index = json.loads((_swift_view / 'model.safetensors.index.json').read_text())
_swift_missing = sorted(set(_swift_index['weight_map'].values()) - set(_swift_seen))
assert not _swift_missing, f"Swift view lacks indexed shards: {_swift_missing[:5]}"
print(f"Swift view: {len(_swift_seen)} files from {len(_swift_parts)} parts -> {_swift_view}")
MODEL_DIR         = str(_swift_view)
'''


def e1(control):
    nb = json.loads(json.dumps(control))
    edit(nb, SWIFT_VIEW, SWIFT_VIEW_NEW)
    edit(nb, FAIL_FAST_OLD, FAIL_FAST_NEW)
    return nb


def e2(control):
    nb = json.loads(json.dumps(control))
    edit(nb, "    'ARC3_MAX_ACTIVE_STREAMS': 10,\n", "    'ARC3_MAX_ACTIVE_STREAMS': 14,  # WAVE-1 E2 (control 10)\n")
    edit(nb, "    'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+12)*1024,\n",
         "    'LOCAL_ANALYZER_CONTEXT_WINDOW': (80+12)*1024,  # WAVE-1 E2 (control (116+12)*1024)\n")
    edit(nb, "    'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024,\n", "    'ARC3_CONTEXT_DRAIN_TOKENS': 40*1024,  # WAVE-1 E2 (control 58*1024)\n")
    edit(nb, '    CTX=(116+12+8)*1024,\n', '    CTX=(80+12+8)*1024,  # WAVE-1 E2 (control (116+12+8)*1024)\n')
    edit(nb, '    MAXREQ=10,\n', '    MAXREQ=14,  # WAVE-1 E2 (control 10)\n')
    edit(nb, '    CUDAGRAPH_MAXBS=10,\n', '    CUDAGRAPH_MAXBS=14,  # WAVE-1 E2 (control 10)\n')
    edit(nb, '    MAMBA_CACHE=60,\n', '    MAMBA_CACHE=84,  # WAVE-1 E2 (control 60; 6 slots per stream)\n')
    edit(nb, 'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})\n',
         'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, 12, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})  # WAVE-1 E2: +12\n')
    edit(nb, FAIL_FAST_OLD, FAIL_FAST_NEW)
    return nb


def write(run, nb, slug, title, models):
    d = HERE / run / 'kernel'; d.mkdir(parents=True, exist_ok=True)
    (d / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads((CONTROL / 'kernel-metadata.json').read_text())
    meta.update(id=f'pranavagrawai/{slug}', title=title, code_file=f'{slug}.ipynb', is_private=True, model_sources=models)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return {'run': run, 'slug': slug, 'notebook_sha256': sha(d / f'{slug}.ipynb'), 'metadata': meta}


def main():
    control = json.loads((CONTROL / 'dfranzen-m2-control.ipynb').read_text())
    base_models = json.loads((CONTROL / 'kernel-metadata.json').read_text())['model_sources']
    assert INTEL in base_models, base_models
    swift_models = [m for m in base_models if m != INTEL] + [f'{SWIFT}/PyTorch/w4a16-a/1', f'{SWIFT}/PyTorch/w4a16-b/1']
    previous = {r['run']: r for r in json.loads((HERE / 'runs.json').read_text())['runs']} if (HERE / 'runs.json').exists() else {}

    def build(run, nb, slug, title, models):
        if (HERE / run / 'launch-attempt.json').exists():  # a launched notebook is evidence: never rebuild it
            return previous[run]
        return write(run, nb, slug, title, models)

    runs = [build('wave-1/e1-swift', e1(control), 'wave1-e1-swift-20261011', 'wave1 e1 swift 20261011', swift_models),
            build('wave-1/e2-c14', e2(control), 'wave1-e2-c14-20261011', 'wave1 e2 c14 20261011', base_models)]
    # each arm differs from control only in the code cells it names
    c_code = [''.join(c['source']) for c in control['cells'] if c['cell_type'] == 'code']
    for r in runs:
        nb = json.loads((HERE / r['run'] / 'kernel' / f"{r['slug']}.ipynb").read_text())
        code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        changed = [i for i, (a, b) in enumerate(zip(code, c_code)) if a != b]
        assert len(code) == len(c_code) and changed, (r['run'], changed)
        r['changed_code_cells'] = changed
    (HERE / 'runs.json').write_text(json.dumps({'control_sha256': sha(CONTROL / 'dfranzen-m2-control.ipynb'), 'runs': runs}, indent=2) + '\n')
    print(json.dumps([(r['run'], r['slug'], r['changed_code_cells']) for r in runs]))


if __name__ == '__main__':
    main()
