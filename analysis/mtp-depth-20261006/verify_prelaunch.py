"""Verify immutable code, local payload, uploaded readback, and the prelaunch tag."""
import hashlib
import json
import subprocess
from pathlib import Path

import build_notebook

HERE = Path(__file__).resolve().parent
TAG = build_notebook.TAG


def git(*args):
    return subprocess.run(['git', *args], cwd=HERE, capture_output=True,
                          text=True, check=True).stdout.strip()


def verify(require_tag=True):
    dirty = git('status', '--porcelain', '--', '.')
    if dirty:
        raise ValueError(f'experiment differs from its committed state:\n{dirty}')
    dataset = HERE / 'kaggle/dataset'
    manifest_path = dataset / 'manifest.json'
    digest = build_notebook.sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    bad = [name for name, sha in manifest.items()
           if not (dataset / name).is_file() or build_notebook.sha256(dataset / name) != sha]
    if bad:
        raise ValueError(f'local dataset hash mismatch: {bad}')
    prov = json.loads((dataset / 'provenance.json').read_text())
    if prov['code_dirty'] or prov['prelaunch_tag'] != TAG:
        raise ValueError('payload provenance is dirty or names another tag')
    prefix = Path(git('rev-parse', '--show-prefix'))
    for name in build_notebook.CODE:
        blob = subprocess.run(['git', 'show', f"{prov['code_commit']}:{prefix / name}"],
                              cwd=HERE, capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != manifest[name]:
            raise ValueError(f'{name} is not the code from the recorded commit')
    readback = json.loads((HERE / 'evidence/dataset-readback.json').read_text())
    if (readback.get('mismatches') or readback['manifest_sha256'] != digest
            or readback['files'] != manifest):
        raise ValueError('uploaded dataset readback does not match the frozen payload')
    nb_path = HERE / f'kaggle/notebook/{build_notebook.KERNEL_SLUG}.ipynb'
    nb = json.loads(nb_path.read_text())
    setup = ''.join(next(c for c in nb['cells'] if c['cell_type'] == 'code')['source'])
    if digest not in setup or prov['code_commit'] not in setup:
        raise ValueError('notebook is not bound to the recorded dataset and commit')
    result = {'prelaunch_tag': TAG, 'provenance': prov,
              'notebook': nb_path.name, 'notebook_sha256': build_notebook.sha256(nb_path),
              'kernel_metadata': json.loads((nb_path.parent / 'kernel-metadata.json').read_text()),
              'dataset_manifest_sha256': digest, 'dataset_version': readback['version'],
              'alpha': json.loads((HERE / 'alpha.json').read_text())['alpha']}
    if require_tag:
        result['tag_commit'] = git('rev-parse', f'{TAG}^{{commit}}')
        if git('diff', result['tag_commit'], '--', '.'):
            raise ValueError('committed experiment differs from the prelaunch tag')
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
