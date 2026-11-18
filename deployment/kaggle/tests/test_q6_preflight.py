import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "q6-control-preflight"


def module():
    spec = importlib.util.spec_from_file_location("q6_prepare", HERE / "prepare.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_preflight_matches_qualified_control_and_launch_caps():
    prepare = module()
    launch = prepare.prepare(check=True)
    assert launch["maximum_pushes"] == 1
    assert launch["automatic_retries"] == 0
    assert launch["argv"][-4:] == [
        "--timeout",
        "14400",
        "--accelerator",
        "NvidiaRtxPro6000",
    ]
    assert (HERE / "preflight.ipynb").read_bytes() == (
        prepare.CANDIDATE / "control.ipynb"
    ).read_bytes()


def test_card_drift_blocks_preflight(monkeypatch):
    prepare = module()
    original_digest = prepare.digest
    monkeypatch.setattr(
        prepare,
        "digest",
        lambda p: "changed-card-hash" if p == prepare.CARD else original_digest(p),
    )
    with pytest.raises(RuntimeError, match="launch/card drift"):
        prepare.prepare(check=True)
