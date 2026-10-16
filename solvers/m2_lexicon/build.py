"""Build the lexicon screen: dfranzen M2 control + the ARCAID3 lexicon cell, against the unchanged control. Never uploads.

Source: habanwer/arcaid3-lexicon-view-dfranzen-m2 version 4 (Shehab Anwer, Apache-2.0; NOTICE.md), hidden 31.90 (n=1).
Its notebook equals our control except for (a) the lexicon install cell, (b) a final counters cell and (c) its own
interactive-only config (10 games, 25 min), which we do not take. In submission mode lex == that notebook's code.
Both screen arms stop 60 min after server startup (interactive branch only).
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTROL = REPO / 'analysis/m2-winners-20261001/runs/control'
DONOR = REPO / 'analysis/notebook-mining-20261016/pulled/habanwer_arcaid3-lexicon-view-dfranzen-m2/arcaid3-lexicon-view-dfranzen-m2.ipynb'
EXP = HERE / 'experiment'
LEX_HEAD = '# ---------------------------------------------------------------------------\n# ARCAID3 lexicon view'
COUNTERS_HEAD = '# ARCAID3 lexicon view: final counters'
BENCH_HEAD = "print('Starting benchmark...')"
SOFT_END = 'soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))'
SCREEN_END = 'soft_end = datetime.now() + timedelta(seconds=60*60)  # SCREEN: 60 min of play after server startup'


def sha(data): return hashlib.sha256(data).hexdigest()
def src(cell): return ''.join(cell['source'])


def donor_cells():
    raw = DONOR.read_bytes()
    pinned = json.loads((HERE / 'manifest.json').read_text())['donor_sha256'] if (HERE / 'manifest.json').exists() else None
    assert pinned in (None, sha(raw)), 'donor notebook changed since it was pinned'
    cells = [c for c in json.loads(raw)['cells'] if c['cell_type'] == 'code']
    lex = [c for c in cells if src(c).startswith(LEX_HEAD)]
    counters = [c for c in cells if src(c).startswith(COUNTERS_HEAD)]
    assert len(lex) == len(counters) == 1
    return sha(raw), lex[0], counters[0], cells


def lexicon(control, lex, counters):
    nb = json.loads(json.dumps(control))
    at = [i for i, c in enumerate(nb['cells']) if src(c).startswith(BENCH_HEAD)]
    assert len(at) == 1
    nb['cells'].insert(at[0], json.loads(json.dumps(lex)))
    nb['cells'].append(json.loads(json.dumps(counters)))
    nb['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# M2 lexicon screen\n', 'dfranzen M2 control plus the ARCAID3 lexicon view cell from '
        'habanwer/arcaid3-lexicon-view-dfranzen-m2 v4 (Shehab Anwer, Apache-2.0). See solvers/m2_lexicon/NOTICE.md.\n']})
    return nb


def screen(nb):
    nb = json.loads(json.dumps(nb))
    hits = [c for c in nb['cells'] if SOFT_END in src(c)]
    assert len(hits) == 1 and src(hits[0]).count(SOFT_END) == 1
    hits[0]['source'] = src(hits[0]).replace(SOFT_END, SCREEN_END).splitlines(True)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['execution_count'] = None; c['outputs'] = []
    return nb


def write(d, nb, slug, title):
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads((CONTROL / 'kernel-metadata.json').read_text())
    meta.update(id=f'pranavagrawai/{slug}', title=title, code_file=f'{slug}.ipynb', is_private=True)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return {'slug': slug, 'notebook_sha256': sha((d / f'{slug}.ipynb').read_bytes()), 'metadata': meta}


def main():
    donor_sha, lex, counters, donor_code = donor_cells()
    control = json.loads((CONTROL / 'dfranzen-m2-control.ipynb').read_text())
    control_code = [src(c) for c in control['cells'] if c['cell_type'] == 'code']
    # Outside its own config cell, the donor must be our control plus exactly the two lexicon cells.
    rest = [s for s in map(src, donor_code) if not s.startswith((LEX_HEAD, COUNTERS_HEAD))]
    assert len(rest) == len(control_code) and sum(a != b for a, b in zip(rest, control_code)) == 1
    assert [a for a, b in zip(rest, control_code) if a != b][0].startswith('# Make one-off changes to `bm`')
    arms = [('screen-1/lex', screen(lexicon(control, lex, counters)), 'm2-lex-screen1-lex', 'M2 lex screen1 lex'),
            ('screen-1/rawctl', screen(control), 'm2-lex-screen1-rawctl', 'M2 lex screen1 rawctl')]
    runs_path = EXP / 'runs.json'
    previous = {r['run']: r for r in json.loads(runs_path.read_text())['runs']} if runs_path.exists() else {}
    runs = []
    for run, nb, slug, title in arms:
        if (EXP / run / 'launch-attempt.json').exists():  # a launched run's notebook is evidence: never rebuild
            runs.append(previous[run]); continue
        runs.append({'run': run, **write(EXP / run / 'kernel', nb, slug, title)})
    (HERE / 'manifest.json').write_text(json.dumps({
        'donor': str(DONOR.relative_to(REPO)), 'donor_sha256': donor_sha,
        'control_sha256': sha((CONTROL / 'dfranzen-m2-control.ipynb').read_bytes()),
        'lexicon_cell_sha256': sha(src(lex).encode()), 'counters_cell_sha256': sha(src(counters).encode())}, indent=2) + '\n')
    runs_path.write_text(json.dumps({'runs': runs}, indent=2) + '\n')
    print(json.dumps([r['run'] for r in runs]))


if __name__ == '__main__':
    main()
