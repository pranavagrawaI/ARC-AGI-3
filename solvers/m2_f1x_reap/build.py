"""Build the F1x + REAP448 notebook and its 60-minute screen kernel from two frozen, drawn parents. Never uploads.

F1x  = solvers/m2_f1 screen-2 f1x v1: B1 + stdlib preload + 16K output cap at a 116K input, 10 streams. Hidden D10 30.88.
REAP = analysis/hidden-draws-20261010/reap-package-20261018 v1: REAP448 target (full 512-expert drafter), fail-fast
       startup guard, 18 streams at an 80K input, raw dfranzen harness. Hidden D11 (2026-10-20).
The merge keeps every F1x cell and swaps in REAP's serving cell at F1x's context. REAP's freed KV goes to streams,
not to a shorter context: 14 streams keep the pool's load at F1x's own (README.md). REAP's late-READY limit becomes
screen-only, so a submission on a slow disk keeps waiting for a live server instead of dying with a zero.
The screen kernel is the draw candidate: its 60-minute soft end applies only when not TRUE_SUBMISSION.
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
F1X_NB = ROOT / 'solvers/m2_f1/experiment/screen-2/f1x/kernel/m2-f1-screen2-f1x.ipynb'
F1X_SHA = '826dfdc2a483a786bdfe888ad9b9422630bc4c615a187b34bdc9762975fb7df9'
REAP_NB = ROOT / 'analysis/hidden-draws-20261010/reap-package-20261018/kernel/reap448-c18-hidden-ready-20261018.ipynb'
REAP_SHA = 'b0e3617055aee24555fa94a33ec8e257f19ec31353c2579d135b149f6c39e0e1'
EXP = HERE / 'experiment'
LAUNCHER = '# Paste this entire file into the launcher cell.'
STREAMS = 14
SERVING = [  # (REAP launcher text, merge text)
    ('    CTX=(80+12+8)*1024,  # WAVE-1 E2 (control (116+12+8)*1024)',
     "    CTX=(116+16+8)*1024,  # F1x+REAP: F1x's window (116K input + 16K output) + 8K"),
    ('    MAXREQ=18,  # WAVE-1 E2 (control 10)',
     "    MAXREQ=14,  # F1x+REAP: REAP448's freed KV spent on streams at F1x's context"),
    ('    CUDAGRAPH_MAXBS=18,  # WAVE-1 E2 (control 10)', '    CUDAGRAPH_MAXBS=14,  # F1x+REAP'),
    ('    MAMBA_CACHE=108,  # WAVE-1 E2 (control 60; 6 slots per stream)', '    MAMBA_CACHE=84,  # F1x+REAP: 6 slots per stream'),
    ('    if time.time() - NOTEBOOK_START_TIME > LATE_READY_LIMIT:',
     '    if not TRUE_SUBMISSION and time.time() - NOTEBOOK_START_TIME > LATE_READY_LIMIT:  '
     '# F1x+REAP: screens only; a submission waits while the server lives'),
]
CONFIG = [("    'ARC3_MAX_ACTIVE_STREAMS': 10,", "    'ARC3_MAX_ACTIVE_STREAMS': 14,  # F1x+REAP: was 10")]
SCREEN_END = 'soft_end = datetime.now() + timedelta(seconds=60*60)  # SCREEN: 60 min of play after server startup'
REAP_SMOKE_END = 'soft_end = datetime.now() + timedelta(seconds=900)  # interactive packaging qualification only'
HEADER = ['# M2 F1x + REAP448 candidate\n',
          'F1x (B1 + sandbox stdlib preload + 16K output cap, 116K input) served by REAP448 (448 target experts, '
          'full 512-expert MTP drafter) at 14 streams. See solvers/m2_f1x_reap/README.md.\n\n']


def sha(data): return hashlib.sha256(data).hexdigest()


def load(path, digest):
    data = path.read_bytes()
    assert sha(data) == digest, f'{path.name} drifted from the drawn version'
    return json.loads(data)


def source(c): return ''.join(c['source'])


def swap(text, pairs):
    for old, new in pairs:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def merge_notebook():
    f1x, reap = load(F1X_NB, F1X_SHA), load(REAP_NB, REAP_SHA)
    [launcher] = [c for c in reap['cells'] if source(c).startswith(LAUNCHER)]
    nb = json.loads(json.dumps(f1x))
    hits = [i for i, c in enumerate(nb['cells']) if source(c).startswith(LAUNCHER)]
    assert len(hits) == 1
    nb['cells'][hits[0]]['source'] = swap(source(launcher), SERVING).splitlines(True)
    [cfg] = [c for c in nb['cells'] if CONFIG[0][0] in source(c)]
    cfg['source'] = swap(source(cfg), CONFIG).splitlines(True)
    assert source(nb['cells'][0]).startswith('# M2 F1 candidate')
    nb['cells'][0]['source'] = HEADER + nb['cells'][0]['source']
    return nb


def write(d, nb, slug, title):
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads((F1X_NB.parent / 'kernel-metadata.json').read_text())
    meta.update(id=f'pranavagrawai/{slug}', title=title, code_file=f'{slug}.ipynb', is_private=True)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return {'slug': slug, 'notebook_sha256': sha((d / f'{slug}.ipynb').read_bytes()), 'metadata': meta}


def main():
    run = 'screen-1/merge'
    runs_path = EXP / 'runs.json'
    if (EXP / run / 'launch-attempt.json').exists():  # a launched notebook is evidence: never rebuild it
        print(f'{run} launched; not rebuilding'); return
    entry = {'run': run, **write(EXP / run / 'kernel', merge_notebook(), 'm2-f1x-reap14-screen1', 'M2 F1x REAP14 screen1')}
    runs_path.write_text(json.dumps({'parents': {'f1x': F1X_SHA, 'reap': REAP_SHA}, 'runs': [entry]}, indent=2) + '\n')
    (HERE / 'manifest.json').write_text(json.dumps({
        'f1x_notebook_sha256': F1X_SHA, 'reap_notebook_sha256': REAP_SHA, 'streams': STREAMS,
        'serving': SERVING, 'config': CONFIG, 'merge_notebook_sha256': entry['notebook_sha256']}, indent=2) + '\n')
    print(json.dumps({'run': run, 'sha256': entry['notebook_sha256']}))


if __name__ == '__main__':
    main()
