"""Build the M2 F1 friction notebook and its 60-minute screen runs from the frozen B1 notebook. Never uploads.

F1 = B1-fixed (solvers/m2_b1_fixed: B1 with its false frame-API instruction repaired) + two friction fixes
(agreed plan 2026-10-16; evidence in README.md):
  preload  : the sandbox preloads allowed stdlib modules, so a snippet that forgets `import collections` still runs.
  out16k   : output cap 12K -> 16K; the analyzer window and server context grow by the same 4K, so the input budget
             (116K) is unchanged.
Screen runs (experiment/screen-1): f1 and b1ctl (B1 as drawn in D5), identical except for the B1-fixed overlay and
F1's changes; both stop 60 min after the
server is up (interactive branch only; every TRUE_SUBMISSION path is untouched by the screen edit).
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
B1 = HERE.parent / 'm2_b1'          # control: the notebook drawn as hidden D5
BASE = HERE.parent / 'm2_b1_fixed'  # F1 builds on this
BASE_NB = 'kernel/m2-b1-fixed.ipynb'
EXP = HERE / 'experiment'
SANDBOX_REL = 'inference/agent/python_tool_sandbox.py'
B1_INSTALL = '# M2 B1: install verified source before importing the harness.'
ANCHOR = '        runtime_globals["__builtins__"]["__import__"] = _safe_import\n'
PRELOAD_MODULES = ('bisect', 'collections', 'copy', 'functools', 'heapq', 'itertools', 'json', 'math', 'statistics')
PRELOAD = ('        # F1: preload allowed stdlib modules; a forgotten import cost a whole snippet (~8 per 25-game run).\n'
           f'        for _f1_module in {PRELOAD_MODULES!r}:\n'
           '            runtime_globals[_f1_module] = _safe_import(_f1_module)\n')
CONFIG = [  # (cell anchor, B1 text, F1 text): input budget = window - reply reserve (= max output) stays 116K
    ("    'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024,", "    'LOCAL_ANALYZER_MAX_OUTPUT': 16*1024,  # F1: was 12*1024"),
    ("    'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+12)*1024,", "    'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+16)*1024,  # F1: was (116+12)"),
    ("    CTX=(116+12+8)*1024,", "    CTX=(116+16+8)*1024,  # F1: was (116+12+8)"),
]
SOFT_END = 'soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))'
SCREEN_END = 'soft_end = datetime.now() + timedelta(seconds=60*60)  # SCREEN: 60 min of play after server startup'


def sha(data): return hashlib.sha256(data).hexdigest()


def patched_sandbox():
    old = (BASE / 'src' / SANDBOX_REL).read_bytes()
    for d in (B1, BASE):  # B1-fixed leaves the sandbox byte-identical to B1
        want = json.loads((d / 'manifest.json').read_text())['changes'][SANDBOX_REL]['after']
        assert sha(old) == want, f'{d.name} sandbox source drifted from its manifest'
    text = old.decode()
    assert text.count(ANCHOR) == 1
    new = text.replace(ANCHOR, ANCHOR + PRELOAD, 1).encode()
    compile(new, SANDBOX_REL, 'exec')
    return old, new


def install_cell(old, new):
    return f'''# M2 F1: preload allowed stdlib modules in the sandbox. Runs after the B1 install, before harness imports.
import hashlib
_f1_path = BUNDLE_DIR / "src" / "ARC3-Inference" / {SANDBOX_REL!r}
_f1_old = _f1_path.read_bytes()
if hashlib.sha256(_f1_old).hexdigest() != {sha(old)!r}:
    raise RuntimeError("M2 F1: sandbox is not the B1 source")
_f1_new = _f1_old.replace({ANCHOR.encode()!r}, {(ANCHOR + PRELOAD).encode()!r}, 1)
if hashlib.sha256(_f1_new).hexdigest() != {sha(new)!r}:
    raise RuntimeError("M2 F1: patched sandbox hash mismatch")
_f1_path.write_bytes(_f1_new)
print("M2 F1 installed", {sha(new)!r})
'''


def replace_once(cells, old, new):
    hits = [c for c in cells if old in ''.join(c['source'])]
    assert len(hits) == 1 and ''.join(hits[0]['source']).count(old) == 1, old
    hits[0]['source'] = ''.join(hits[0]['source']).replace(old, new).splitlines(True)


def f1_notebook(base_nb=BASE / BASE_NB, header='B1-fixed plus friction fixes'):
    nb = json.loads(base_nb.read_text())
    old, new = patched_sandbox()
    idx = [i for i, c in enumerate(nb['cells']) if ''.join(c['source']).startswith(B1_INSTALL)]
    assert len(idx) == 1
    nb['cells'].insert(idx[0] + 1, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                                    'source': install_cell(old, new).splitlines(True)})
    for a, b in CONFIG:
        replace_once(nb['cells'], a, b)
    nb['cells'][0]['source'] = ['# M2 F1 candidate\n', f'{header}: sandbox stdlib preload and a 16K output cap '
                                '(input budget unchanged). See solvers/m2_f1/README.md.\n\n'] + nb['cells'][0]['source']
    return nb, sha(new)


def screen(nb):
    nb = json.loads(json.dumps(nb))
    replace_once(nb['cells'], SOFT_END, SCREEN_END)
    return nb


def write(d, nb, slug, title):
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads((B1 / 'kernel/kernel-metadata.json').read_text())
    meta.update(id=f'pranavagrawai/{slug}', title=title, code_file=f'{slug}.ipynb', is_private=True)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return {'slug': slug, 'notebook_sha256': sha((d / f'{slug}.ipynb').read_bytes()), 'metadata': meta}


def code_cells(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def screen2(previous):
    """screen-2 (2026-10-17): F1 minus the B1-fixed frame-API rewrite (f1x = B1 + preload + out16k) vs a fresh B1
    control. f1 and the lexicon arm, the two arms that rewrite the frame text, both started ~15% slower."""
    b1_path = B1 / 'kernel/m2-b1.ipynb'
    f1x, _ = f1_notebook(b1_path, 'B1 (frame-API text untouched) plus friction fixes')
    b1 = json.loads(b1_path.read_text())
    runs = []
    for run, nb, slug, title in [('screen-2/f1x', screen(f1x), 'm2-f1-screen2-f1x', 'M2 F1 screen2 f1x'),
                                 ('screen-2/b1ctl', screen(b1), 'm2-f1-screen2-b1ctl', 'M2 F1 screen2 b1ctl')]:
        if (EXP / run / 'launch-attempt.json').exists():
            runs.append(previous[run]); continue
        runs.append({'run': run, **write(EXP / run / 'kernel', nb, slug, title)})
    arms = [json.loads((EXP / r['run'] / 'kernel' / f"{r['slug']}.ipynb").read_text()) for r in runs]
    f1x_code = [s for s in code_cells(arms[0]) if not s.startswith('# M2 F1:')]
    for a, b in CONFIG:
        f1x_code = [s.replace(b, a) for s in f1x_code]
    assert f1x_code == code_cells(arms[1]), 'f1x differs from the B1 control beyond F1'
    screen1_ctl = EXP / 'screen-1/b1ctl/kernel/m2-f1-screen1-b1ctl.ipynb'
    assert code_cells(arms[1]) == code_cells(json.loads(screen1_ctl.read_text())), 'screen-2 control != screen-1 control'
    return runs


def main():
    f1, sandbox_after = f1_notebook()
    kernel = write(HERE / 'kernel', f1, 'm2-f1', 'M2 F1')
    b1 = json.loads((B1 / 'kernel/m2-b1.ipynb').read_text())
    runs_path = EXP / 'runs.json'
    previous = {r['run']: r for r in json.loads(runs_path.read_text())['runs']} if runs_path.exists() else {}
    runs = []
    for run, nb, slug, title in [('screen-1/f1', screen(f1), 'm2-f1-screen1-f1', 'M2 F1 screen1 f1'),
                                 ('screen-1/b1ctl', screen(b1), 'm2-f1-screen1-b1ctl', 'M2 F1 screen1 b1ctl')]:
        if (EXP / run / 'launch-attempt.json').exists():  # a launched run's notebook is evidence: never rebuild
            runs.append(previous[run]); continue
        runs.append({'run': run, **write(EXP / run / 'kernel', nb, slug, title)})
    # The f1 arm minus F1's cell and config lines must be the B1-fixed notebook; B1-fixed may differ from B1
    # only in its overlay cell (checked below), so the arms differ by nothing else.
    arms = [json.loads((EXP / r['run'] / 'kernel' / f"{r['slug']}.ipynb").read_text()) for r in runs]
    code = [[''.join(c['source']) for c in a['cells'] if c['cell_type'] == 'code'] for a in arms]
    f1_code = [s for s in code[0] if not s.startswith('# M2 F1:')]
    for a, b in CONFIG:
        f1_code = [s.replace(b, a) for s in f1_code]
    base = screen(json.loads((BASE / BASE_NB).read_text()))
    assert f1_code == [''.join(c['source']) for c in base['cells'] if c['cell_type'] == 'code'], 'f1 differs from B1-fixed beyond F1'
    diff = [i for i, (x, y) in enumerate(zip(f1_code, code[1])) if x != y]
    assert len(f1_code) == len(code[1]) and len(diff) == 1 and f1_code[diff[0]].startswith('# M2 B1: install'), diff
    runs += screen2(previous)
    (HERE / 'manifest.json').write_text(json.dumps({
        'b1_notebook_sha256': sha((B1 / 'kernel/m2-b1.ipynb').read_bytes()),
        'b1_fixed_notebook_sha256': sha((BASE / BASE_NB).read_bytes()),
        'f1_notebook_sha256': kernel['notebook_sha256'], 'sandbox_after_sha256': sandbox_after,
        'preload_modules': PRELOAD_MODULES, 'config': CONFIG}, indent=2) + '\n')
    EXP.mkdir(exist_ok=True)
    runs_path.write_text(json.dumps({'f1_notebook_sha256': kernel['notebook_sha256'], 'runs': runs}, indent=2) + '\n')
    print(json.dumps({'kernel': kernel['slug'], 'runs': [r['run'] for r in runs]}))


if __name__ == '__main__':
    main()
