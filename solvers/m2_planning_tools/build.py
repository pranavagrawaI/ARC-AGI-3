"""Compose a local source tree from the hash-pinned parent and sparse overlay."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'm2_b1_fixed' / 'src'


def prepare(destination):
    destination = Path(destination)
    provenance = json.loads((HERE / 'provenance.json').read_text())
    for relative, expected in provenance['parent_files'].items():
        if hashlib.sha256((PARENT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Parent identity changed: ' + relative)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(PARENT, destination, ignore=shutil.ignore_patterns('__pycache__'))
    for path in (HERE / 'overlay').rglob('*.py'):
        compile(path.read_bytes(), str(path), 'exec')
        target = destination / path.relative_to(HERE / 'overlay')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination


if __name__ == '__main__':
    import sys
    print(prepare(sys.argv[1]))
