from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

HERE = Path(__file__).resolve().parents[1]
PACKAGE = HERE.parents[1] / 'deployment/kaggle/actor-critic-construction-kaggle-r2'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def wire():
    return module(HERE / 'local_wire.py', 'test_local_wire')


def test_wire_continuation_identity_and_no_response_metadata(wire):
    model = wire.LOCAL_MODEL
    body = {'model': 'qwen/qwen3.8-27b', 'messages': [
        {'role': 'user', 'content': 'x'},
        {'role': 'assistant', 'content': None, 'reasoning_content': None, 'refusal': None,
         'tool_calls': [{'id': 'opaque/A:8', 'index': 9, 'type': 'function', 'function': {'name': 'execute_python', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'opaque/A:8', 'content': 'ok'}],
        'max_tokens': 192, 'parallel_tool_calls': True,
        'provider': {'only': ['coreweave/fp8']}, 'reasoning': {'enabled': False}}
    original = copy.deepcopy(body)
    result = wire.project_local(body, local_model=model, mounted_weights=model, startup_argv=['--model-path', model])
    assert body == original
    assert result['max_tokens'] == 192
    assert result['messages'][-1]['tool_call_id'] == 'opaque/A:8'
    assert not {'provider', 'parallel_tool_calls', 'reasoning', 'extra_body'} & set(result)
    assert 'reasoning_content' not in result['messages'][1]
    assert 'index' not in result['messages'][1]['tool_calls'][0]
    body['messages'][-1]['tool_call_id'] = 'repair-would-be-wrong'
    with pytest.raises(ValueError, match='unmatched'):
        wire.project_local(body, local_model=model, mounted_weights=model, startup_argv=['--model-path', model])


@pytest.mark.parametrize('argv', [[], ['--model-path', '/wrong'], ['--model-path=' + '/fake'], ['--model-path'], ['x', 'prefix/kaggle/input/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot']])
def test_identity_never_invented_from_substring(wire, argv):
    with pytest.raises(ValueError):
        wire.validate_identity(wire.LOCAL_MODEL, wire.LOCAL_MODEL, argv)


def test_build_is_deterministic_and_assets_reconcile(tmp_path):
    builder = module(HERE / 'build.py', 'disposable_builder')
    manifest = builder.build(tmp_path / 'package', write_manifest=False)
    frozen = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    assert manifest == frozen
    for name, digest in {**manifest['assets'], **manifest['files']}.items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest['origins'].items():
        assert hashlib.sha256((builder.REPO / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((HERE / 'build.py').read_bytes()).hexdigest() == manifest['builder_sha256']
    meta = json.loads((PACKAGE / 'kernel-metadata.json').read_text())
    assert meta['is_private'] is True and meta['enable_internet'] is False
    assert meta['competition_sources'] == ['arc-prize-2026-arc-agi-3']
    assert meta['dataset_sources'][0] == 'jakobbrggen/qwen3-8-27b-fp8-hf-snapshot'
    assert manifest['complete'] is False and manifest['push'] is False


def test_launch_refuses_even_with_pass(tmp_path):
    # The packaging launcher has no provider invocation path.
    (tmp_path / 'launch_once.py').write_bytes((HERE / 'launch_once.py').read_bytes())
    for pass_record in ({}, {'status': 'PASS'}):
        (tmp_path / 'OPENROUTER-PASS.json').write_text(json.dumps(pass_record))
        proc = subprocess.run([sys.executable, str(tmp_path / 'launch_once.py')], cwd=tmp_path, capture_output=True, text=True)
        assert proc.returncode != 0
        assert json.loads(proc.stdout)['pushed'] is False
