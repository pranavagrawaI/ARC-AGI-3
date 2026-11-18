"""Read-only candidate qualification; fixtures are disposable, no provider calls."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREP = REPO / 'analysis/actor-critic-construction-kaggle-launch'
PACKAGE = REPO / 'deployment/kaggle/actor-critic-construction-kaggle-smoke'
OR = REPO / 'analysis/actor-critic-construction-openrouter-20260914T154902Z'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    return {str(p.relative_to(REPO)): sha(p) for root in (PREP, PACKAGE) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(command, cwd):
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(cwd), 'PYTHONDONTWRITEBYTECODE': '1', 'ARC_DISABLE_DOTENV': '1'}
    proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    return {'command': command, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}


def main():
    before = snapshot()
    result = {'schema': 'kaggle-package-readonly-verification-v1', 'evidence_kind': 'local fixtures and source reconciliation, not remote GPU/model evidence', 'source_hashes': before, 'provider_calls': 0, 'pushes': 0}
    manifest = json.loads((OR / 'PRELAUNCH-MANIFEST.json').read_text())
    review = json.loads((OR / 'REVIEW.json').read_text())
    result['openrouter'] = {'manifest_sha256': sha(OR / 'PRELAUNCH-MANIFEST.json'), 'source_count': len(manifest['sources']), 'source_drift': [name for name, digest in manifest['sources'].items() if not (REPO / name).is_file() or sha(REPO / name) != digest], 'review_status': review['status'], 'review_digest_matches': review['manifest_sha256'] == sha(OR / 'PRELAUNCH-MANIFEST.json')}
    audit = load('candidate_audit', PACKAGE / 'audit.py')
    result['openrouter']['recomputed_audit'] = audit.classify(OR / 'raw')
    pm = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    result['package_manifest'] = {'complete': pm['complete'], 'push': pm['push'], 'asset_drift': [name for name, digest in {**pm['assets'], **pm['files']}.items() if not (PACKAGE / name).is_file() or sha(PACKAGE / name) != digest]}
    nb = json.loads((PACKAGE / 'construction.ipynb').read_text())
    result['notebook_cells'] = [{'index': i, 'type': c['cell_type'], 'source_bytes': len(''.join(c['source']).encode())} for i, c in enumerate(nb['cells'])]
    bootstrap = ast.parse((PACKAGE / 'donor_bootstrap.py').read_text())
    for node in bootstrap.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ASSETS' for t in node.targets):
            assets = ast.literal_eval(node.value)
            result['donor_embedded_asset_names'] = sorted(assets)
            result['forbidden_case_asset_present'] = any(Path(name).name == 'case.json' for name in assets)
    with tempfile.TemporaryDirectory(prefix='kaggle-package-verification-') as tmp:
        tmp = Path(tmp)
        empty = tmp / 'empty'; empty.mkdir()
        working = tmp / 'working'; working.mkdir()
        preamble = ''.join(nb['cells'][1]['source']).replace('/kaggle/working', str(working))
        result['notebook_clean_working_dir'] = run([sys.executable, '-I', '-c', preamble], empty)
        # Also exercise the prepared package with adjacent files available, not only the notebook.
        isolated_package = tmp / 'deployment/kaggle/smoke'
        shutil.copytree(PACKAGE, isolated_package, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        result['packaged_worker_help'] = run([sys.executable, '-I', str(isolated_package / 'headless.py'), '--help'], empty)
        receipts = tmp / 'receipts'; receipts.mkdir()
        request = {'call': 1, 'request': {'model': '/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot', 'messages': [], 'max_tokens': 192}}
        response = {'call': 1, 'response': {'model': request['request']['model'], 'choices': [{'message': {'role': 'assistant', 'content': 'synthetic fixture'}}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}}
        (receipts / 'requests.jsonl').write_text(json.dumps(request) + '\n')
        (receipts / 'responses.jsonl').write_text(json.dumps(response) + '\n')
        result['valid_local_receipt_accepted_by_packaged_audit'] = audit.request_response_accounted(receipts)
        # Positive control only: synthetic gateway receipt, never a local-provider result.
        response['response'].update(model='qwen/qwen3.8-27b', provider='CoreWeave')
        (receipts / 'responses.jsonl').write_text(json.dumps(response) + '\n')
        result['synthetic_gateway_positive_control_accepted'] = audit.request_response_accounted(receipts)
        suite_code = "import importlib.util,sys,pathlib; p=pathlib.Path(sys.argv[1]); s=importlib.util.spec_from_file_location('build',p/'build.py'); m=importlib.util.module_from_spec(s); sys.modules['build']=m; s.loader.exec_module(m); m.PACKAGE=pathlib.Path(sys.argv[2]); m.HERE=pathlib.Path(sys.argv[3]); m.HERE.mkdir(); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',str(p/'tests/test_package.py')]))"
        result['existing_tests_disposable_build'] = run([sys.executable, '-I', '-c', suite_code, str(PREP), str(tmp / 'rebuilt'), str(tmp / 'build-output')], empty)
        result['launch_refusal'] = run([sys.executable, '-I', str(PREP / 'launch_once.py')], empty)
    result['candidate_source_drift_during_checks'] = [name for name, digest in before.items() if not (REPO / name).is_file() or sha(REPO / name) != digest]
    result['verdict'] = 'HOLD'
    result['unexercised'] = ['remote containment', 'local HTTP continuation through actual packaged worker', 'model-backed local GPU integration', 'production supervisor final audit', 'quota/target absence/cancellation prelaunch gates']
    (OUT / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_hashes'}, indent=2))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
