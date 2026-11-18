"""Exercise the exact notebook-embedded worker and concurrent summarizer path."""
import ast
import base64
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


@pytest.fixture
def embedded(monkeypatch, tmp_path):
    import build_notebook
    import freeze_candidate
    import test_reasoning

    notebook = build_notebook.build(discovery_arm='d')
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    payloads = []
    extracted = tmp_path / 'embedded'
    extracted.mkdir()
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'b64decode' and node.args
                and isinstance(node.args[0], ast.Constant)):
            raw = base64.b64decode(node.args[0].value)
            payloads.append(raw)
            if raw.startswith(b'PK'):
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for name in archive.namelist():
                        assert not Path(name).is_absolute() and '..' not in Path(name).parts
                        path = extracted / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(archive.read(name))
    for name in ('adapter.py', 'shared_local_solver.py'):
        raw = (HERE / name).read_bytes()
        assert raw in payloads
        (extracted / name).write_bytes(raw)
    for name, digest in freeze_candidate.verify()['files'].items():
        assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == digest
    monkeypatch.setattr(test_reasoning, 'HERE', extracted)
    return test_reasoning


@pytest.mark.parametrize('deep', [False, True])
def test_embedded_worker_reasoning_gate(embedded, monkeypatch, tmp_path, deep):
    embedded.test_actual_worker_records_reasoning_and_commits_tool_action(monkeypatch, tmp_path / 'run', deep)


@pytest.mark.asyncio
async def test_embedded_concurrent_summary(embedded, monkeypatch):
    await embedded.test_deep_permission_does_not_leak_to_concurrent_summary(monkeypatch)
