"""Freeze a three-arm wave (control, port, lean) from one shared budget block; no launch.

port = hybrid candidate as built (orchestrator mode, control treatment: Actor + Builder + Scribe).
lean = the same notebook with Builder and Scribe removed (no_world_model mode, no_scribe treatment).
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_pair import HERE, CONTROL, PAIR_SOURCE, sha, text, arm_notebook  # noqa: E402

PORT_ENV = "'TYCHO_MODE': 'orchestrator', 'TYCHO_NOOA_TREATMENT': 'control',"
LEAN_ENV = "'TYCHO_MODE': 'no_world_model', 'TYCHO_NOOA_TREATMENT': 'no_scribe',"
HYBRID_ARMS = ('port', 'lean')
# Donor cells a hybrid arm may change: title, submission guard, solver swap, viewer.
ALLOWED_SOLVER_CELLS = {0, 4, 20, 'last'}


def normalized(notebook, arm):
    cells = [text(c) for c in notebook['cells']]
    cells[0] = cells[0].replace(f"'{arm}'", "'ARM'").replace(f"-{arm}'", "-ARM'")
    return cells


def donor_cells_differing(control, hybrid, arm):
    """Donor-numbered cells where a hybrid arm differs from control (install cell excluded)."""
    a = normalized(control, 'control')
    b = normalized(hybrid, arm)
    b = b[:10] + b[11:]
    assert len(a) == len(b), 'cell count drift beyond the install cell'
    last = len(a) - 2
    return {('last' if i == last else i - 1) for i, (x, y) in enumerate(zip(a, b)) if x != y}


def lean_from_port(port_base):
    lean = json.loads(json.dumps(port_base))
    hits = [c for c in lean['cells'] if PORT_ENV in text(c)]
    assert len(hits) == 1 and text(hits[0]).count(PORT_ENV) == 1, 'port env seam changed'
    hits[0]['source'] = text(hits[0]).replace(PORT_ENV, LEAN_ENV).splitlines(True)
    return lean


def main():
    import nbformat
    sys.path.insert(0, str(HERE/'src'))
    import hybrid.pair as pair
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave-id', default='wave-1')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=None, help='arm-order draw; recorded')
    args = parser.parse_args()
    out = args.out or HERE/'experiment'/args.wave_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert sha((HERE/'kernel/tycho-dfranzen.ipynb').read_bytes()) == manifest['notebook_sha256'], 'rebuild first'
    assert manifest['source_files']['hybrid/pair.py'] == sha(PAIR_SOURCE.read_bytes()), 'payload lacks current pair.py'
    port_base = json.loads((HERE/'kernel/tycho-dfranzen.ipynb').read_text())
    bases = {'control': json.loads((CONTROL/'dfranzen-m2-control.ipynb').read_text()),
             'port': port_base, 'lean': lean_from_port(port_base)}
    hybrid_meta = json.loads((HERE/'kernel/kernel-metadata.json').read_text())
    metas = {'control': json.loads((CONTROL/'kernel-metadata.json').read_text()),
             'port': hybrid_meta, 'lean': hybrid_meta}
    notebooks = {arm: arm_notebook(bases[arm], arm, args.wave_id) for arm in bases}
    differing = {}
    for arm in HYBRID_ARMS:
        differing[arm] = donor_cells_differing(notebooks['control'], notebooks[arm], arm)
        assert differing[arm] <= ALLOWED_SOLVER_CELLS, f'{arm}: non-solver cells differ {differing[arm]}'
    port_cells, lean_cells = normalized(notebooks['port'], 'port'), normalized(notebooks['lean'], 'lean')
    lean_only = [i for i, (x, y) in enumerate(zip(port_cells, lean_cells)) if x != y]
    assert len(lean_only) == 1 and LEAN_ENV in lean_cells[lean_only[0]], 'lean must differ from port only by mode'
    for key in ('enable_gpu', 'enable_internet', 'machine_shape', 'docker_image', 'competition_sources', 'model_sources'):
        assert metas['control'].get(key) == hybrid_meta.get(key), f'metadata {key} differs'
    assert set(metas['control']['dataset_sources']) <= set(hybrid_meta['dataset_sources'])
    identities = {}
    for arm, notebook in notebooks.items():
        directory = out/arm; (directory/'kernel').mkdir(parents=True)
        nbformat.validate(nbformat.from_dict(notebook))
        for i, c in enumerate(notebook['cells']):
            source = text(c)
            if c['cell_type'] == 'code' and not any(l.startswith(('!', '%')) for l in source.splitlines()):
                compile(source, f'{arm}:{i}', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        path = directory/'kernel/notebook.ipynb'
        path.write_text(json.dumps(notebook, indent=1)+'\n')
        assert path.stat().st_size < 1_000_000
        meta = dict(metas[arm], id=f'pranavagrawai/tycho-dfranzen-{args.wave_id}-{arm}',
                    title=f'Tycho dfranzen {args.wave_id} {arm}', code_file=path.name)
        (directory/'kernel/kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
        if arm in HYBRID_ARMS:
            for name in ('manifest.json', 'canonical-candidate.patch'): shutil.copy2(HERE/name, directory/name)
            shutil.copytree(HERE/'source-bundle', directory/'source-bundle')
        identities[arm] = {'notebook_sha256': sha(path.read_bytes()),
            'metadata_sha256': sha((directory/'kernel/kernel-metadata.json').read_bytes()),
            'kernel': meta['id'], 'gpu_count': 1}
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    order = random.Random(seed).sample(sorted(bases), len(bases))
    identity = {'pair_id': args.wave_id, 'arms': identities, 'pair_source_sha256': sha(PAIR_SOURCE.read_bytes()),
        'source_payload_sha256': manifest['source_payload_sha256'],
        'differing_donor_cells': {arm: sorted(map(str, cells)) for arm, cells in differing.items()},
        'hard_timeout_seconds': pair.SESSION_SECONDS, 'arm_order_seed': seed, 'launch_order': order,
        'reserved_gpu_hours': len(bases)*pair.SESSION_SECONDS/3600}
    (out/'pair-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
