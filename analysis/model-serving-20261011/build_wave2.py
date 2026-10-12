"""Build wave-2/b1-c14: the frozen B1 notebook (solvers/m2_b1, as run in pair-1) plus exactly E2's edits
(14 streams, 92K context, drain 40K, mamba 84, graphs to 14 + bs 12) and the startup fail-fast. Never uploads or runs.
Both halves already ran cleanly on GPU: B1 in m2-b1 pair-1 (102/117 levels), E2 in wave-1 (107 levels, +19.8% tok/s).
"""
import hashlib, json
from pathlib import Path
import build as W1  # wave-1 edit helpers; the same edit list, applied to B1

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
B1 = REPO / 'solvers/m2_b1/kernel/m2-b1.ipynb'
B1_PAIR = REPO / 'solvers/m2_b1/experiment/pair-1/b1-r1/kernel'
RUN = 'wave-2/b1-c14'
SLUG, TITLE = 'wave2-b1-c14-20261011', 'wave2 b1 c14 20261011'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert sha(B1) == sha(B1_PAIR / 'm2-b1-pair1-r1.ipynb'), 'B1 source drifted from the notebook that ran in pair-1'
    if (HERE / RUN / 'launch-attempt.json').exists():
        raise SystemExit(f'{RUN} already launched; never rebuild it')
    b1 = json.loads(B1.read_text())
    nb = W1.e2(b1)
    d = HERE / RUN / 'kernel'; d.mkdir(parents=True, exist_ok=True)
    (d / f'{SLUG}.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads((B1_PAIR / 'kernel-metadata.json').read_text())
    meta.update(id=f'pranavagrawai/{SLUG}', title=TITLE, code_file=f'{SLUG}.ipynb', is_private=True)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    # the only differences from B1 are E2's edits: compare against E2-vs-control, cell by cell
    control = json.loads(W1.CONTROL.joinpath('dfranzen-m2-control.ipynb').read_text())
    e2 = W1.e2(control)
    code = lambda n: [''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code']
    b1c, nbc, cc, e2c = code(b1), code(nb), code(control), code(e2)
    changed = [i for i, (a, b) in enumerate(zip(b1c, nbc)) if a != b]
    overlay = [i for i, s in enumerate(b1c) if s.startswith('# M2 B1: install')]
    b1_wo = [s for i, s in enumerate(b1c) if i not in overlay]
    nb_wo = [s for i, s in enumerate(nbc) if i not in overlay]
    assert b1_wo == cc and nb_wo == e2c, 'b1-c14 is not exactly B1 overlay + E2 edits'
    runs = {'b1_sha256': sha(B1), 'runs': [{'run': RUN, 'slug': SLUG, 'notebook_sha256': sha(d / f'{SLUG}.ipynb'),
                                           'metadata': meta, 'changed_code_cells_vs_b1': changed}]}
    (HERE / 'runs-wave2.json').write_text(json.dumps(runs, indent=2) + '\n')
    print(json.dumps(runs['runs'][0]['changed_code_cells_vs_b1']), meta['id'], meta['model_sources'])


if __name__ == '__main__':
    main()
