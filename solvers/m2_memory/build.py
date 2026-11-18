"""Build local candidate notebook and donor fixes. Never uploads or runs them."""
import ast
import base64
import difflib
import hashlib
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AUDIT = REPO / 'analysis/m2-winners-20261001/audit'
BASE = AUDIT / 'dfranzen-repo/ARC3-Inference'
CONTROL = AUDIT.parent / 'runs/control'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def diff(old, new, name):
    return ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                      fromfile='a/'+name, tofile='b/'+name))


def receipt_fix(source):
    before = '"action_results": list(message.get("action_results") or host_action_results),'
    assert source.count(before) == 1
    source = source.replace(before, '"action_results": list(host_action_results),')
    source = source.replace('import inspect\n', 'import inspect\nfrom copy import deepcopy\n', 1)
    before = 'host_action_results.append(dict(raw_action_result))'
    assert source.count(before) == 1
    return source.replace(before, 'host_action_results.append(deepcopy(raw_action_result))')


def build():
    changes, patch_text = {}, ''
    for path in sorted((HERE/'src').rglob('*.py')):
        rel = path.relative_to(HERE/'src').as_posix()
        old = (BASE/rel).read_bytes()
        new = path.read_bytes()
        if old != new:
            compile(new, str(path), 'exec')
            changes[rel] = {'before': sha(old), 'after': sha(new), 'source': new.decode()}
            patch_text += diff(old.decode(), new.decode(), 'ARC3-Inference/'+rel)
    (HERE/'candidate.patch').write_text(patch_text)
    packed = base64.b64encode(zlib.compress(json.dumps(changes).encode(), 9)).decode()
    install = f'''# M2 memory: install verified source before importing the harness.
import base64, hashlib, json, sys, zlib
from pathlib import Path
if any(n == "inference" or n.startswith("inference.") for n in sys.modules):
    raise RuntimeError("M2 memory overlay must be installed before harness imports")
_m2_files = json.loads(zlib.decompress(base64.b64decode({packed!r})))
_m2_root = BUNDLE_DIR / "src" / "ARC3-Inference"
for _m2_rel, _m2_item in _m2_files.items():
    _m2_path = _m2_root / _m2_rel
    _m2_hash = hashlib.sha256(_m2_path.read_bytes()).hexdigest()
    if _m2_hash not in (_m2_item["before"], _m2_item["after"]):
        raise RuntimeError("Unexpected donor source: " + _m2_rel)
for _m2_rel, _m2_item in _m2_files.items():
    _m2_path = _m2_root / _m2_rel
    _m2_tmp = _m2_path.with_suffix(".m2-tmp")
    _m2_tmp.write_text(_m2_item["source"], encoding="utf-8")
    _m2_tmp.replace(_m2_path)
    assert hashlib.sha256(_m2_path.read_bytes()).hexdigest() == _m2_item["after"]
print("M2 memory installed", {{k: v["after"] for k, v in _m2_files.items()}})
'''
    nb_path = CONTROL/'dfranzen-m2-control.ipynb'
    notebook = json.loads(nb_path.read_text())
    assert 'harness patch applied successfully' in ''.join(notebook['cells'][4]['source'])
    notebook['cells'].insert(5, {'cell_type':'code', 'metadata':{}, 'execution_count':None,
                               'outputs':[], 'source':install.splitlines(True)})
    notebook['cells'].insert(0, {'cell_type':'markdown','metadata':{},'source':[
        '# M2 memory candidate\n',
        'Dfranzen baseline with verified correctness repairs, bounded sandbox transport adapted from lordhansolo, '
        'per-action animation history inspired by sirikilohit M111, explicit per-game saved data, '
        'and cached frame views.\n',
        'Local CPU qualification only; no gameplay improvement is claimed. Original donor attribution follows.\n']})
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            cell['execution_count']=None;cell['outputs']=[]
    target=HERE/'kernel';target.mkdir(exist_ok=True)
    notebook_path=target/'m2-memory.ipynb'
    notebook_path.write_text(json.dumps(notebook,indent=1)+'\n')
    metadata=json.loads((CONTROL/'kernel-metadata.json').read_text())
    metadata.update(id='pranavagrawai/m2-memory',title='M2 memory',code_file='m2-memory.ipynb',is_private=True)
    (target/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    donors={}
    paths={'lordhansolo':AUDIT/'lord-bundle/src/ARC3-Inference/inference/agent/python_tool_sandbox.py',
           'sirikilohit':AUDIT/'bug-audit-20261001/m90_wang_sandbox.py'}
    for name,path in paths.items():
        old=path.read_text();new=receipt_fix(old)
        compile(new,str(path),'exec')
        rel='inference/agent/python_tool_sandbox.py' if name=='lordhansolo' else 'm90_wang_sandbox.py'
        (HERE/'donor_fixes'/f'{name}.patch').write_text(diff(old,new,rel))
        (HERE/'donor_fixes'/f'{name}_sandbox.py').write_text(new)
        donors[name]={'source':str(path.relative_to(REPO)), 'before':sha(old.encode()),'after':sha(new.encode())}
    manifest={'baseline_notebook':str(nb_path.relative_to(REPO)), 'baseline_sha256':sha(nb_path.read_bytes()),
              'candidate_notebook_sha256':sha(notebook_path.read_bytes()),
              'changes':{k:{n:v for n,v in item.items() if n!='source'} for k,item in changes.items()},
              'donor_fixes':donors, 'qualification':'CPU only; not submitted or gameplay-qualified'}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'changed_source_files':list(changes),'notebook':str(notebook_path)},indent=2))


if __name__=='__main__':
    build()
