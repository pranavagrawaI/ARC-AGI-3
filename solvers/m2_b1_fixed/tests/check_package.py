"""Execute only the generated source installer against a temporary donor tree."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]/'analysis/m2-winners-20261001/audit/dfranzen-repo/ARC3-Inference'
notebook=json.loads((ROOT/'kernel/m2-b1-fixed.ipynb').read_text())
manifest=json.loads((ROOT/'manifest.json').read_text())
assert hashlib.sha256((ROOT/'kernel/m2-b1-fixed.ipynb').read_bytes()).hexdigest()==manifest['candidate_notebook_sha256']
install=next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('# M2 B1: install'))
for index,cell in enumerate(notebook['cells']):
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        if not source.startswith('%%writefile'):
            # The donor has two IPython !shell escapes. Compile their Python
            # call equivalents only; never execute these commands locally.
            source='\n'.join('get_ipython().system('+repr(line[1:])+')' if line.startswith('!')
                             else line for line in source.splitlines())
            compile(source,f'notebook-cell-{index}','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
for path in (ROOT/'src').rglob('*.py'):
    compile(path.read_text(),str(path),'exec')
with tempfile.TemporaryDirectory() as directory:
    bundle=Path(directory)
    target=bundle/'src/ARC3-Inference'
    shutil.copytree(BASE/'inference',target/'inference',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    ns={'BUNDLE_DIR':bundle}
    exec(compile(install,'installer','exec'),ns)
    exec(compile(install,'installer-idempotent','exec'),ns)
    for rel,item in manifest['changes'].items():
        assert hashlib.sha256((target/rel).read_bytes()).hexdigest()==item['after']
    # Exercise the installed artifact, not only the source tree's import path.
    runtime_python = ROOT.parent/'tycho_nooa/.venv/bin/python'
    subprocess.run([str(runtime_python), '-B', str(ROOT/'tests/gate_bugfix.py')],
                   env={**os.environ, 'M2_BUGFIX_SOURCE': str(target)}, check=True)
    last=target/next(reversed(manifest['changes']))
    last.write_text(last.read_text()+'\n# unexpected local edit\n')
    before={str(p):p.read_bytes() for p in target.rglob('*.py')}
    try:
        exec(compile(install,'installer-rejects-drift','exec'),ns)
    except RuntimeError as error:
        assert 'Unexpected donor source' in str(error)
    else:
        raise AssertionError('installer accepted source drift')
    assert before=={str(p):p.read_bytes() for p in target.rglob('*.py')}
print('Package PASS: notebook/source compile, exact overlay, idempotence, drift rejected before mutation')
