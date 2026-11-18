"""Build the B1 run directories from frozen inputs. Never uploads or runs anything.
smoke-1      : B1 notebook, games restricted to lf52/s5i5/ft09 (as m2_evidence smoke-1), and the analyzer
               context window cut to 52K in the overlay cell so live trims, and so the pin, actually fire.
pair-1/b1-r1 : kernel/m2-b1.ipynb as built and qualified.
pair-1/b1-r2 : the same notebook; only the kernel id and title differ (an independent replicate).
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
OVERLAY_TIMEOUT = 'os.environ["LOCAL_ANALYZER_TOOL_TIMEOUT"] = "60"  # B1: harness reads it at import; ceiling raised to 60 s'
SMOKE_WINDOW = 'os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = str((40 + 12) * 1024)  # SMOKE ONLY: force live trims'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(run, notebook, slug, title):
    d = HERE / run / 'kernel'; d.mkdir(parents=True, exist_ok=True)
    (d / f'{slug}.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    meta = json.loads((CONTROL / 'kernel-metadata.json').read_text())
    meta.update(id=f'pranavagrawai/{slug}', title=title, code_file=f'{slug}.ipynb', is_private=True)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return {'run': run, 'slug': slug, 'notebook_sha256': sha(d / f'{slug}.ipynb'), 'metadata': meta}


def main():
    b1 = json.loads((SOLVER / 'kernel/m2-b1.ipynb').read_text())
    smoke = json.loads(json.dumps(b1))
    hook = [c for c in smoke['cells'] if HOOK in ''.join(c['source'])]
    overlay = [c for c in smoke['cells'] if ''.join(c['source']).startswith('# M2 B1: install')]
    assert len(hook) == 1 and len(overlay) == 1
    excluded = [g for g in ALL_GAMES if g not in SMOKE_GAMES]
    hook[0]['source'] = ''.join(hook[0]['source']).replace(HOOK, f"demo_excluded_games = {excluded!r}  # SMOKE: {SMOKE_GAMES}\n").splitlines(True)
    src = ''.join(overlay[0]['source'])
    assert src.count(OVERLAY_TIMEOUT) == 1
    overlay[0]['source'] = src.replace(OVERLAY_TIMEOUT, OVERLAY_TIMEOUT + '\n' + SMOKE_WINDOW).splitlines(True)
    previous = {r['run']: r for r in json.loads((HERE / 'runs.json').read_text())['runs']} if (HERE / 'runs.json').exists() else {}

    def build(run, nb, slug, title):
        # a launched run's notebook is evidence: never rebuild it
        if (HERE / run / 'launch-attempt.json').exists():
            return previous[run]
        return write(run, nb, slug, title)

    runs = [build('smoke-1', smoke, 'm2-b1-smoke-1', 'M2 B1 smoke 1'),
            build('pair-1/b1-r1', b1, 'm2-b1-pair1-r1', 'M2 B1 pair1 r1'),
            build('pair-1/b1-r2', b1, 'm2-b1-pair1-r2', 'M2 B1 pair1 r2')]
    # The replicas must be the qualified notebook byte for byte, and differ from the control only by overlay cells.
    for r in runs[1:]:
        assert r['notebook_sha256'] == sha(SOLVER / 'kernel/m2-b1.ipynb')
    control = json.loads((CONTROL / 'dfranzen-m2-control.ipynb').read_text())
    b_code = [''.join(c['source']) for c in b1['cells'] if c['cell_type'] == 'code']
    c_code = [''.join(c['source']) for c in control['cells'] if c['cell_type'] == 'code']
    assert [s for s in b_code if not s.startswith('# M2 B1: install')] == c_code, 'B1 differs from control beyond the overlay'
    (HERE / 'runs.json').write_text(json.dumps({'b1_source_sha256': sha(SOLVER / 'kernel/m2-b1.ipynb'),
        'manifest_sha256': sha(SOLVER / 'manifest.json'), 'runs': runs}, indent=2) + '\n')
    print(json.dumps([(r['run'], r['slug']) for r in runs]))


if __name__ == '__main__':
    main()
