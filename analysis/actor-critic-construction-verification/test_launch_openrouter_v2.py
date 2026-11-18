"""Paid-launch gate tests; subprocess calls are recorded, never executed."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def setup_gate(monkeypatch, tmp_path):
    path = Path(__file__).with_name("launch_openrouter_v2.py")
    spec = importlib.util.spec_from_file_location("coordinator_launch_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    source = tmp_path / "source.py"
    source.write_text("pass\n")
    command = ["unshare", "--user", "--map-auto", "--map-root-user", str(tmp_path / "deployment/kaggle/.venv/bin/python"), str(tmp_path / "analysis/actor-critic-construction-live-launch-v2/harness.py"), "--execute", "--out", str(run / "raw")]
    manifest = run / "PRELAUNCH-MANIFEST.json"
    manifest.write_text(json.dumps({"sources": {"source.py": module.digest(source)}, "command": command, "run_dir": str(run)}))
    monkeypatch.setattr(sys, "argv", [str(path), str(run)])
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: (calls.append(a), SimpleNamespace(returncode=0))[1])
    return module, run, source, calls


@pytest.mark.parametrize("review", [{}, [], {"status": "HOLD"}, {"status": "APPROVED", "manifest_sha256": "wrong"}])
def test_review_must_approve_exact_manifest(monkeypatch, tmp_path, review):
    module, run, _, calls = setup_gate(monkeypatch, tmp_path)
    (run / "REVIEW.json").write_text(json.dumps(review))
    with pytest.raises(RuntimeError):
        module.main()
    assert calls == []


def test_drift_and_consumed_identity_block_before_execution(monkeypatch, tmp_path):
    module, run, source, calls = setup_gate(monkeypatch, tmp_path)
    (run / "REVIEW.json").write_text(json.dumps({"status": "APPROVED", "manifest_sha256": module.digest(run / "PRELAUNCH-MANIFEST.json")}))
    source.write_text("changed\n")
    with pytest.raises(RuntimeError, match="Source drift"):
        module.main()
    assert calls == []
    source.write_text("pass\n")
    assert module.main() == 0
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        module.main()
    assert len(calls) == 1

