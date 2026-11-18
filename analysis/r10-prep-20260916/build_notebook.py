"""Embed reviewed R10a candidate files, without freezing a launch manifest."""
import ast
import base64
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = HERE / 'r10a-package'
DONOR = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r9-cd82'

def build():
    excluded = {'construction.ipynb', 'kernel-metadata.json', 'ASSET-HASHES.json'}
    assets = {str(p.relative_to(PACKAGE)):p.read_bytes() for p in sorted(PACKAGE.rglob('*')) if p.is_file() and str(p.relative_to(PACKAGE)) not in excluded}
    assert not any('__pycache__' in p or p.endswith('.pyc') for p in assets)
    hashes = {n:hashlib.sha256(b).hexdigest() for n,b in assets.items()}
    raw = (json.dumps(hashes,indent=2,sort_keys=True)+'\n').encode()
    (PACKAGE/'ASSET-HASHES.json').write_bytes(raw)
    assets['ASSET-HASHES.json'] = raw
    notebook = json.loads((DONOR/'construction.ipynb').read_text())
    tree = ast.parse(''.join(notebook['cells'][1]['source']))
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id == 'ASSETS':
                node.value = ast.parse(repr({n:base64.b64encode(b).decode() for n,b in assets.items()}),mode='eval').body
            if node.targets[0].id == 'EXPECTED':
                node.value = ast.parse(repr({n:hashlib.sha256(b).hexdigest() for n,b in assets.items()}),mode='eval').body
    notebook['cells'][1]['source'] = (ast.unparse(tree)+'\n').splitlines(keepends=True)
    notebook['cells'][0]['source'] = ['# R10a CD82 constructor repair candidate\nUnfrozen. Local preparation only. Independent review and Hermes gates pending.\n']
    for i,cell in enumerate(notebook['cells']): cell['id'] = f'r10a-{i}'
    (PACKAGE/'construction.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    meta=json.loads((DONOR/'kernel-metadata.json').read_text())
    meta.update(id='pranavagrawai/arc3-r10a-cd82-20260916', title='R10a CD82 constructor repair')
    (PACKAGE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    inventory={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.rglob('*')) if p.is_file()}
    (HERE/'R10A-PACKAGE-INVENTORY.json').write_text(json.dumps({'status':'UNFROZEN_CANDIDATE','files':inventory},indent=2)+'\n')
    print(json.dumps({'assets':len(assets),'package_files':len(inventory),'frozen':False,'launched':False}))
if __name__=='__main__': build()
