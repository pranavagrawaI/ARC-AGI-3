"""Verify notebook payload, donor preservation and local dependency receipts."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.metadata
import json
import zlib
from pathlib import Path

import nbformat

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    path=HERE/'kernel/tycho-dfranzen.ipynb'
    assert sha(path.read_bytes())==manifest['notebook_sha256']
    assert sha((HERE/'build.py').read_bytes())==manifest['build_script_sha256']
    assert sha((ROOT/'solvers/tycho_nooa/uv.lock').read_bytes())==manifest['canonical_lock_sha256']
    assert sha((ROOT/'deployment/kaggle/tycho-nooa-full-wheelhouse-v11/wheelhouse-manifest.json').read_bytes())==manifest['wheelhouse_manifest_sha256']
    assert sha((ROOT/manifest['donor_notebook']).read_bytes())==manifest['donor_notebook_sha256']
    notebook=json.loads(path.read_text())
    nbformat.validate(nbformat.from_dict(notebook))
    install=''.join(notebook['cells'][9]['source'])
    tree=ast.parse(install)
    assignments={node.targets[0].id:node.value for node in tree.body
                 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
    hashes=ast.literal_eval(assignments['_hybrid_hashes'])
    payload=(HERE/'source-bundle/hybrid-runtime.zlib').read_bytes()
    assert sha(payload)==manifest['source_payload_sha256']
    assert repr(sha(payload)) in install
    files=json.loads(zlib.decompress(payload))
    assert len(path.read_bytes()) < 1_000_000, 'Kaggle notebook source cap'
    assert set(files)==set(hashes)==set(manifest['source_files'])
    for name,data in files.items():
        decoded=base64.b64decode(data)
        assert sha(decoded)==hashes[name]==manifest['source_files'][name],name
        assert (HERE/'runtime'/name).read_bytes()==decoded,name
    actual={p.relative_to(HERE/'runtime').as_posix() for p in (HERE/'runtime').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
    assert actual==set(files),(actual-set(files),set(files)-actual)
    for item in manifest['source_provenance']:
        assert sha((ROOT/item['source']).read_bytes())==item['before'],item['source']
    base=json.loads((ROOT/manifest['donor_notebook']).read_text())
    assert len(notebook['cells'])==len(base['cells'])+1
    changed={0,4,20,len(base['cells'])-1}
    preserved=[]
    for index,cell in enumerate(base['cells']):
        successor=notebook['cells'][index+(index>=9)]
        if index not in changed:
            assert ''.join(cell['source'])==''.join(successor['source']),index
            preserved.append(index)
    for index in (9,21):
        compile(''.join(notebook['cells'][index]['source']),f'cell_{index}','exec',
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    metadata=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
    donor_metadata=json.loads((ROOT/manifest['donor_notebook']).with_name('kernel-metadata.json').read_text())
    for key in ('docker_image','machine_shape','model_sources','competition_sources',
                'enable_gpu','enable_internet'):
        assert metadata[key]==donor_metadata[key],key
    extras=json.loads((HERE/'runtime-deps/hybrid-deps-manifest.json').read_text())
    assert sha((HERE/'runtime-deps/hybrid-deps-manifest.json').read_bytes())==manifest['extra_deps_manifest_sha256']
    for item in extras['wheels']:
        assert sha((HERE/'runtime-deps'/item['filename']).read_bytes())==item['sha256']
    wheelhouse=json.loads((ROOT/'deployment/kaggle/tycho-nooa-full-wheelhouse-v11/wheelhouse-manifest.json').read_text())
    differences=[]
    versions={}
    for item in wheelhouse['wheels']:
        if item['name']=='arc-agi-3-tycho-nooa':
            continue  # Candidate source is intentionally supplied by its payload.
        try:
            version=importlib.metadata.version(item['name'])
        except importlib.metadata.PackageNotFoundError:
            version=None
        versions[item['name']]=version
        if version!=item['version']:
            differences.append({'package':item['name'],'wheelhouse':item['version'],'local':version})
    report={'notebook_sha256':manifest['notebook_sha256'],'payload_files':len(files),
        'unchanged_donor_cell_indices':preserved,'modified_donor_cell_indices':sorted(changed),
        'canonical_and_donor_sources_unchanged':True,'notebook_schema_and_modified_code_valid':True,
        'model_serving_and_hardware_preserved':True,'extra_dependency_wheels_verified':True,
        'local_versions':versions,'local_vs_kaggle_dependency_differences':differences,
        'kaggle_offline_install':'not executed; exact v11 wheel files absent from local artifact directory',
        'gpu_qualification':False}
    (HERE/'evidence/packaging.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('notebook_sha256','payload_files',
        'canonical_and_donor_sources_unchanged','local_vs_kaggle_dependency_differences')}))


if __name__=='__main__':
    main()
