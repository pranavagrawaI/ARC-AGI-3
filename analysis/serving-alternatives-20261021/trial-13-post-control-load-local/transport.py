"""Lossless notebook transport only. No allocation, authority, or API calls."""
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cell_source(notebook):
    cells = notebook['cells']
    if len(cells) != 1 or cells[0]['cell_type'] != 'code':
        raise ValueError('expected reviewed single code cell')
    source = cells[0]['source']
    return ''.join(source) if isinstance(source, list) else source


def literals(source):
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ('PAYLOAD', 'CONFIG', 'ASSIGNMENT')}


def wrap(notebook_bytes):
    """Wrap exact notebook bytes; future arming MUST supply newly reviewed bytes."""
    notebook = json.loads(notebook_bytes)
    source = cell_source(notebook)
    blob = base64.b64encode(lzma.compress(notebook_bytes, preset=6)).decode('ascii')
    wrapper_source = f'''# Lossless transport; inner notebook retains its own authority gate.
import base64, hashlib, json, lzma
NOTEBOOK_SHA256 = {sha(notebook_bytes)!r}
SOURCE_SHA256 = {sha(source.encode())!r}
BLOB = {blob!r}
raw = lzma.decompress(base64.b64decode(BLOB, validate=True))
if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
    raise RuntimeError('transport notebook digest mismatch')
notebook = json.loads(raw)
if len(notebook['cells']) != 1 or notebook['cells'][0]['cell_type'] != 'code':
    raise RuntimeError('transport notebook structure mismatch')
source = notebook['cells'][0]['source']
source = ''.join(source) if isinstance(source, list) else source
if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError('transport source digest mismatch')
exec(compile(source, '<reviewed-lossless-notebook>', 'exec'))
'''
    result = dict(nbformat=4, nbformat_minor=5, metadata=notebook['metadata'], cells=[dict(
        cell_type='code', execution_count=None, metadata={}, outputs=[], source=wrapper_source.splitlines(keepends=True))])
    return (json.dumps(result, separators=(',', ':')) + '\n').encode()


def decode(wrapper_bytes):
    source = cell_source(json.loads(wrapper_bytes))
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('BLOB','NOTEBOOK_SHA256','SOURCE_SHA256')}
    raw = lzma.decompress(base64.b64decode(values['BLOB'], validate=True))
    if sha(raw) != values['NOTEBOOK_SHA256'] or sha(cell_source(json.loads(raw)).encode()) != values['SOURCE_SHA256']:
        raise ValueError('transport digest mismatch')
    return raw


def build_unassigned(original_path, destination):
    raw = Path(original_path).read_bytes()
    values = literals(cell_source(json.loads(raw)))
    if values.get('ASSIGNMENT', 'missing') is not None:
        raise ValueError('local proposal must originate from UNASSIGNED notebook')
    wrapped = wrap(raw)
    if decode(wrapped) != raw:
        raise ValueError('lossless roundtrip failed')
    Path(destination).write_bytes(wrapped)
    return dict(original_notebook_sha256=sha(raw), original_bytes=len(raw),
                original_source_sha256=sha(cell_source(json.loads(raw)).encode()),
                wrapper_sha256=sha(wrapped), wrapper_bytes=len(wrapped), assignment=None)
