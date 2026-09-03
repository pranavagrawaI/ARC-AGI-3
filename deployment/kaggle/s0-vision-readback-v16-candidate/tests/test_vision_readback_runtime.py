"""Checks on the outer runtime's V16 vision-readback mode.

The mode reuses the R8-L3B lifecycle wholesale — model hash verification, SGLang launch,
warmup, teardown classification — and only swaps what runs against a ready server. These
tests pin the seams where that reuse could go wrong: the mode must not be usable without
its lifecycle owner, must not co-exist with the game launcher, and must declare a claim
ceiling that refuses gameplay.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CANDIDATE = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def runtime():
    spec = importlib.util.spec_from_file_location(
        "v16_outer_runtime", CANDIDATE / "exact_request_outer_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE_ARGS = [
    "runtime",
    "--output-dir", "/tmp/out",
    "--attention-backend", "triton",
    "--sampling-backend", "pytorch",
    "--startup-timeout-s", "1200",
    "--request-timeout-s", "180",
]


def parse(runtime, extra: list[str], monkeypatch):
    monkeypatch.setattr(sys, "argv", BASE_ARGS + extra)
    return runtime.parse_args()


def vision_args(runtime) -> list[str]:
    return [
        "--r8-l3b",
        "--port", str(runtime.R8_L3B_PORT),
        "--vision-readback-probe", "/tmp/probe.py",
        "--vision-frames", "/tmp/frames.json",
    ]


def test_vision_mode_parses_with_its_lifecycle_owner(runtime, monkeypatch):
    args = parse(runtime, vision_args(runtime), monkeypatch)
    assert args.vision_readback_probe == Path("/tmp/probe.py")
    assert args.vision_frames == Path("/tmp/frames.json")
    assert args.vision_probe_timeout_s == 5400.0


def test_probe_and_frames_are_required_together(runtime, monkeypatch):
    with pytest.raises(SystemExit):
        parse(runtime, [
            "--r8-l3b", "--port", str(runtime.R8_L3B_PORT),
            "--vision-readback-probe", "/tmp/probe.py",
        ], monkeypatch)
    with pytest.raises(SystemExit):
        parse(runtime, [
            "--r8-l3b", "--port", str(runtime.R8_L3B_PORT),
            "--vision-frames", "/tmp/frames.json",
        ], monkeypatch)


def test_vision_mode_requires_the_r8_l3b_lifecycle(runtime, monkeypatch):
    """Without --r8-l3b there is no teardown/containment gate to inherit."""
    with pytest.raises(SystemExit):
        parse(runtime, [
            "--vision-readback-probe", "/tmp/probe.py",
            "--vision-frames", "/tmp/frames.json",
        ], monkeypatch)


def test_vision_mode_cannot_run_alongside_the_game_launcher(runtime, monkeypatch):
    with pytest.raises(SystemExit):
        parse(runtime, vision_args(runtime) + [
            "--official-launcher-probe", "/tmp/launcher.py",
            "--launcher-python", "/tmp/python",
        ], monkeypatch)


def test_vision_probe_timeout_must_be_positive(runtime, monkeypatch):
    with pytest.raises(SystemExit):
        parse(runtime, vision_args(runtime) + ["--vision-probe-timeout-s", "0"], monkeypatch)


def test_vision_mode_keeps_the_qualified_backend_requirement(runtime, monkeypatch):
    """R8-L3B pinned Triton/PyTorch; the probe must measure that same route."""
    monkeypatch.setattr(sys, "argv", [
        "runtime", "--output-dir", "/tmp/out",
        "--attention-backend", "flashinfer",
        "--startup-timeout-s", "1200", "--request-timeout-s", "180",
    ] + vision_args(runtime))
    with pytest.raises(SystemExit):
        runtime.parse_args()


def test_claim_ceiling_refuses_gameplay_and_rhae(runtime):
    source = (CANDIDATE / "exact_request_outer_runtime.py").read_text()
    start = source.index('if args.vision_readback_probe:\n        claim_ceiling = (')
    ceiling = source[start : source.index("elif args.official_launcher_probe:", start)]
    assert "plays no game" in ceiling
    assert "commits no environment action" in ceiling
    for refused in ("RHAE", "level completion", "gameplay"):
        assert refused in ceiling


def test_mode_name_is_distinct(runtime):
    source = (CANDIDATE / "exact_request_outer_runtime.py").read_text()
    assert '"s0-vision-readback"' in source
    assert source.count('"s0-vision-readback"') == 1


def test_request_reconciliation_is_enforced_not_merely_recorded(runtime):
    """A count that does not match SGLang's access log must abort, not be written."""
    source = (CANDIDATE / "exact_request_outer_runtime.py").read_text()
    block_start = source.index('stage = "vision_readback"')
    block = source[block_start : source.index('if args.official_launcher_probe:', block_start)]
    assert 'if not request_count_facts["verified"]:' in block
    assert "raise RuntimeError(" in block.split('if not request_count_facts["verified"]:')[1]
    # the probe's own success is a precondition, not an outcome to summarise
    assert 'vision_result.get("status") != "passed"' in block
