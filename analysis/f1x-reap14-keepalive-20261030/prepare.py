"""Build the authorized D21 identity with exactly one inserted keepalive cell."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'solvers/m2_f1x_reap/experiment/screen-1/merge/kernel/m2-f1x-reap14-screen1.ipynb'
BUILDER = ROOT / 'analysis/hidden-zero-epidemiology-20261027/build_keepalive.py'
PARENT_SHA = '9d634d6accb024d53b875106c4d77ad12948d895ea485cd4be44b7b4f83f56c0'
SLUG = 'm2-f1x-reap14-keepalive-d21'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected():
    assert digest(PARENT) == PARENT_SHA, 'parent identity changed'
    spec = importlib.util.spec_from_file_location('keepalive_builder', BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = json.loads(PARENT.read_text())
    candidate = module.build(parent)
    index = next(i for i, c in enumerate(candidate['cells']) if ''.join(c['source']).startswith('# M2 KEEPALIVE'))
    assert candidate['cells'][:index] + candidate['cells'][index + 1:] == parent['cells']
    assert "print('Starting benchmark...')" in ''.join(candidate['cells'][index + 1]['source'])
    return candidate


if __name__ == '__main__':
    if (HERE / 'launch-attempt.json').exists():
        raise SystemExit('launch attempt exists; frozen candidate must not be rebuilt')
    folder = HERE / 'kernel'
    folder.mkdir(exist_ok=True)
    notebook = folder / (SLUG + '.ipynb')
    notebook.write_text(json.dumps(expected(), indent=1) + '\n')
    metadata = json.loads((PARENT.parent / 'kernel-metadata.json').read_text())
    metadata.update(id='pranavagrawai/' + SLUG, title='M2 F1x REAP14 keepalive D21', code_file=notebook.name)
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'notebook': str(notebook), 'sha256': digest(notebook), 'bytes': notebook.stat().st_size}))
