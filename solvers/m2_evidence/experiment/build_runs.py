"""Build the three run directories from frozen inputs. Never uploads or runs anything.
smoke-1   : treatment notebook, games restricted to lf52/s5i5/ft09 (cell 'Customization hook' edit only).
pair-1/control   : the preserved dfranzen control notebook, byte-identical source cells.
pair-1/treatment : kernel/m2-evidence.ipynb as built and qualified.
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVER = HERE.parent
CONTROL = SOLVER.parents[1] / 'analysis/m2-winners-20261001/runs/control'
ALL_GAMES = ['ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59', 'lf52', 'lp85', 'ls20', 'm0r0', 'r11l',
             're86', 's5i5', 'sb26', 'sc25', 'sk48', 'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30']
SMOKE_GAMES = ['lf52', 's5i5', 'ft09']
HOOK = "demo_excluded_games = []  # CONTROL: all 25 public games\n"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(run, notebook, slug, title):
    d = HERE / run / 'kernel'; d.mkdir(parents=True, exist_ok=True)
    (d / f'{slug}.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    meta = json.loads((CONTROL / 'kernel-metadata.json').read_text())
    meta.update(id=f'pranavagrawai/{slug}', title=title, code_file=f'{slug}.ipynb', is_private=True)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return {'run': run, 'slug': slug, 'notebook_sha256': sha(d / f'{slug}.ipynb'), 'metadata': meta}

def main():
    treatment = json.loads((SOLVER / 'kernel/m2-evidence.ipynb').read_text())
    control = json.loads((CONTROL / 'dfranzen-m2-control.ipynb').read_text())
    for cell in control['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None; cell['outputs'] = []
    smoke = json.loads(json.dumps(treatment))
    hook = [c for c in smoke['cells'] if HOOK in ''.join(c['source'])]
    assert len(hook) == 1
    excluded = [g for g in ALL_GAMES if g not in SMOKE_GAMES]
    hook[0]['source'] = ''.join(hook[0]['source']).replace(HOOK, f"demo_excluded_games = {excluded!r}  # SMOKE: {SMOKE_GAMES}\n").splitlines(True)
    previous = {r['run']: r for r in json.loads((HERE / 'runs.json').read_text())['runs']} if (HERE / 'runs.json').exists() else {}
    def build(run, nb, slug, title):
        # a launched run's notebook is evidence: never rebuild it
        if (HERE / run / 'launch-attempt.json').exists():
            return previous[run]
        return write(run, nb, slug, title)
    runs = [build('smoke-1', smoke, 'm2-evidence-smoke-1', 'M2 evidence smoke 1'),
            build('pair-1/control', control, 'm2-evidence-pair1-control', 'M2 evidence pair1 control'),
            build('pair-1/treatment', treatment, 'm2-evidence-pair1-treatment', 'M2 evidence pair1 treatment')]
    # The two pair arms may differ only by the inserted overlay cell and its title cell.
    t_code = [''.join(c['source']) for c in treatment['cells'] if c['cell_type'] == 'code']
    c_code = [''.join(c['source']) for c in control['cells'] if c['cell_type'] == 'code']
    assert [s for s in t_code if not s.startswith('# M2 evidence: install')] == c_code, 'arms differ beyond the overlay'
    (HERE / 'runs.json').write_text(json.dumps({'control_source': str(CONTROL / 'dfranzen-m2-control.ipynb'),
        'control_source_sha256': sha(CONTROL / 'dfranzen-m2-control.ipynb'),
        'treatment_source_sha256': sha(SOLVER / 'kernel/m2-evidence.ipynb'), 'runs': runs}, indent=2) + '\n')
    print(json.dumps([(r['run'], r['slug']) for r in runs]))

if __name__ == '__main__':
    main()
