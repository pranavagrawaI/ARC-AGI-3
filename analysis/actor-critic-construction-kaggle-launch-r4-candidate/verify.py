#!/usr/bin/env python3
"""Read-only package/notebook/origin reconciliation; never a launch gate."""
from __future__ import annotations
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    assert (HERE / 'MANIFEST.json').read_bytes() == (PACKAGE / 'MANIFEST.json').read_bytes()
    assert manifest['builder_sha256'] == sha(HERE / 'build.py')
    expected_files = set(manifest['assets']) | set(manifest['files']) | {'MANIFEST.json'}
    actual_files = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*') if p.is_file()}
    assert actual_files == expected_files, sorted(actual_files ^ expected_files)
    mismatches = []
    for name, digest in {**manifest['assets'], **manifest['files']}.items():
        if sha(PACKAGE / name) != digest:
            mismatches.append(name)
    for name, digest in manifest['origins'].items():
        if sha(ROOT / name) != digest:
            mismatches.append(name)
    nb = json.loads((PACKAGE / 'construction.ipynb').read_text())
    tree = ast.parse(''.join(nb['cells'][1]['source']))
    assets = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ASSETS' for t in n.targets)))
    assert set(assets) == set(manifest['assets'])
    python_sources = 0
    for name, data in assets.items():
        raw = base64.b64decode(data, validate=True)
        if name.endswith('.py'):
            compile(raw, name, 'exec')
            python_sources += 1
        if raw != (PACKAGE / name).read_bytes():
            mismatches.append('notebook:' + name)
        assert not name.endswith(('case.json', '.zip', '.env')) and 'remote_entry' not in name
    protected_path = ROOT / 'analysis/actor-critic-construction-openrouter-20260914T154902Z/PRELAUNCH-MANIFEST.json'
    assert sha(protected_path) == manifest['openrouter_manifest_sha256']
    protected = json.loads(protected_path.read_text())['sources']
    assert len(protected) == 422
    protected_drift = [name for name, digest in protected.items() if sha(ROOT / name) != digest]
    assert not mismatches and not protected_drift, (mismatches, protected_drift)
    predecessor = json.loads((HERE / 'PREDECESSOR-HASHES.json').read_text())
    predecessor_drift = [name for name, digest in predecessor.items() if sha(ROOT / name) != digest]
    assert not predecessor_drift, predecessor_drift
    public = json.loads((HERE / 'SOURCE-IDENTITY.json').read_text())['sources']
    public_drift = [name for name, digest in public.items() if sha(ROOT / name) != digest]
    assert not public_drift, public_drift
    return {'status': 'passed', 'scope': 'local source reconciliation, not remote qualification or authorization',
            'public_identity_sources': len(public), 'public_identity_drift': public_drift,
            'assets': len(assets), 'compiled_python_sources': python_sources, 'protected_sources': len(protected), 'mismatches': mismatches,
            'protected_drift': protected_drift, 'predecessor_files': len(predecessor), 'predecessor_drift': predecessor_drift, 'manifest_sha256': sha(HERE / 'MANIFEST.json'),
            'notebook_sha256': sha(PACKAGE / 'construction.ipynb'), 'builder_sha256': sha(HERE / 'build.py'),
            'metadata_sha256': sha(PACKAGE / 'kernel-metadata.json'), 'card_sha256': sha(HERE / 'EXPERIMENT-CARD.md')}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
